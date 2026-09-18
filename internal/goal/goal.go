package goal

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/google/uuid"
)

type GoalStatus string

const (
	GoalActive   GoalStatus = "active"
	GoalPaused   GoalStatus = "paused"
	GoalComplete GoalStatus = "complete"
)

type contextKey struct{ name string }

// continuationKey marks a context as belonging to a continuation turn: a
// synthetic turn the runtime starts on behalf of a goal.
var continuationKey = contextKey{"goal_continuation"}

// WithContinuation returns a context for a continuation turn driving goalID.
func WithContinuation(ctx context.Context, goalID string) context.Context {
	return context.WithValue(ctx, continuationKey, goalID)
}

// ContinuationOf reports the goal a continuation turn is driving, or false
// when ctx belongs to an ordinary turn.
func ContinuationOf(ctx context.Context) (goalID string, ok bool) {
	goalID, ok = ctx.Value(continuationKey).(string)
	return goalID, ok
}

type Goal struct {
	SessionID     string     `json:"session_id"`
	GoalID        string     `json:"goal_id"`
	Objective     string     `json:"objective"`
	Status        GoalStatus `json:"status"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
	ActiveSeconds int64      `json:"active_seconds"`
}

// isActiveGoal reports whether g is an active goal. When goalID is non-empty
// the goal must also be the one the caller observed, so a goal replaced
// mid-turn is not mistaken for its predecessor. A nil g is never active.
func (g *Goal) isActiveGoal(goalID string) bool {
	if g == nil || g.Status != GoalActive {
		return false
	}
	return goalID == "" || g.GoalID == goalID
}

// transitions lists, for every target status, the statuses a goal may move
// from. UpdateStatus applies them as compare-and-set conditions so that two
// racing writers can never both observe a successful transition.
var transitions = map[GoalStatus][]GoalStatus{
	GoalActive:   {GoalPaused},
	GoalPaused:   {GoalActive},
	GoalComplete: {GoalActive, GoalPaused},
}

type Service interface {
	pubsub.Subscriber[Goal]
	Get(ctx context.Context, sessionID string) (*Goal, error)
	Create(ctx context.Context, sessionID string, objective string) (*Goal, error)
	// UpdateStatus atomically moves the goal to status. It succeeds only
	// when the goal currently holds a status listed in transitions; when
	// the goal already holds the requested status, or a pause targets a
	// completed goal, the current row is returned unchanged.
	UpdateStatus(ctx context.Context, sessionID string, goalID string, status GoalStatus) (*Goal, error)
	Clear(ctx context.Context, sessionID string, goalID string) (*Goal, error)
	// PauseAllActive moves every active goal to paused. It is meant for
	// startup, where any goal still active was left behind by a process
	// that did not shut down cleanly.
	PauseAllActive(ctx context.Context) (int64, error)
}

type service struct {
	*pubsub.Broker[Goal]
	db *sql.DB
	q  *db.Queries
}

func NewService(q *db.Queries, conn *sql.DB) Service {
	broker := pubsub.NewBroker[Goal]()
	return &service{
		Broker: broker,
		db:     conn,
		q:      q,
	}
}

func (s *service) Get(ctx context.Context, sessionID string) (*Goal, error) {
	dbGoal, err := s.q.GetGoalBySessionID(ctx, sessionID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("getting goal: %w", err)
	}
	return s.fromDBItem(dbGoal), nil
}

func (s *service) Create(ctx context.Context, sessionID string, objective string) (*Goal, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	qtx := s.q.WithTx(tx)

	existing, err := qtx.GetGoalBySessionID(ctx, sessionID)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("getting goal: %w", err)
	}

	if err == nil {
		existingGoal := s.fromDBItem(existing)
		if existingGoal.Status != GoalComplete {
			return nil, fmt.Errorf("session already has an active goal")
		}
		rows, deleteErr := qtx.DeleteGoal(ctx, db.DeleteGoalParams{SessionID: sessionID, GoalID: existing.GoalID})
		if deleteErr != nil {
			return nil, fmt.Errorf("clearing completed goal: %w", deleteErr)
		}
		if rows == 0 {
			return nil, fmt.Errorf("clearing completed goal: goal not found or stale goal ID")
		}
	}

	goalID := uuid.New().String()
	dbGoal, err := qtx.CreateGoal(ctx, db.CreateGoalParams{
		SessionID: sessionID,
		GoalID:    goalID,
		Objective: objective,
		Status:    string(GoalActive),
	})
	if err != nil {
		return nil, fmt.Errorf("creating goal: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("committing transaction: %w", err)
	}

	goal := s.fromDBItem(dbGoal)
	s.Publish(pubsub.UpdatedEvent, *goal)
	return goal, nil
}

func (s *service) UpdateStatus(ctx context.Context, sessionID string, goalID string, status GoalStatus) (*Goal, error) {
	froms, ok := transitions[status]
	if !ok {
		return nil, fmt.Errorf("unknown goal status %q", status)
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	qtx := s.q.WithTx(tx)

	// Only active rows accumulate time, so this is a no-op unless the goal
	// is actually leaving the active state below.
	if status != GoalActive {
		if err := qtx.AccumulateActiveTime(ctx, sessionID); err != nil {
			return nil, fmt.Errorf("accumulating active time: %w", err)
		}
	}

	var dbGoal db.Goal
	for _, from := range froms {
		dbGoal, err = qtx.UpdateGoalStatus(ctx, db.UpdateGoalStatusParams{
			Status:     string(status),
			SessionID:  sessionID,
			GoalID:     goalID,
			FromStatus: string(from),
		})
		if err == nil {
			break
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("updating goal status: %w", err)
		}
	}
	if err != nil {
		// No row matched the compare-and-set: the goal ID is stale or the
		// goal already sits in a state this transition does not start from.
		current, getErr := qtx.GetGoalBySessionID(ctx, sessionID)
		if getErr != nil {
			if errors.Is(getErr, sql.ErrNoRows) {
				return nil, fmt.Errorf("goal not found or stale goal ID")
			}
			return nil, fmt.Errorf("getting goal: %w", getErr)
		}
		if current.GoalID != goalID {
			return nil, fmt.Errorf("goal not found or stale goal ID")
		}
		currentStatus := GoalStatus(current.Status)
		// Repeating a transition is harmless, and a cancellation racing the
		// completion tool must not reopen a finished goal.
		if currentStatus == status || (status == GoalPaused && currentStatus == GoalComplete) {
			return s.fromDBItem(current), nil
		}
		return nil, fmt.Errorf("goal is %s and cannot become %s", currentStatus, status)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("committing transaction: %w", err)
	}

	goal := s.fromDBItem(dbGoal)
	s.Publish(pubsub.UpdatedEvent, *goal)
	return goal, nil
}

func (s *service) Clear(ctx context.Context, sessionID string, goalID string) (*Goal, error) {
	goal, err := s.Get(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if goal == nil {
		return nil, nil
	}
	if goal.GoalID != goalID {
		return nil, fmt.Errorf("goal not found or stale goal ID")
	}
	rows, err := s.q.DeleteGoal(ctx, db.DeleteGoalParams{SessionID: sessionID, GoalID: goalID})
	if err != nil {
		return nil, fmt.Errorf("deleting goal: %w", err)
	}
	if rows == 0 {
		return nil, fmt.Errorf("goal not found or stale goal ID")
	}
	s.Publish(pubsub.DeletedEvent, *goal)
	return goal, nil
}

func (s *service) PauseAllActive(ctx context.Context) (int64, error) {
	rows, err := s.q.PauseActiveGoals(ctx)
	if err != nil {
		return 0, fmt.Errorf("pausing active goals: %w", err)
	}
	return rows, nil
}

func (s *service) fromDBItem(item db.Goal) *Goal {
	return &Goal{
		SessionID:     item.SessionID,
		GoalID:        item.GoalID,
		Objective:     item.Objective,
		Status:        GoalStatus(item.Status),
		CreatedAt:     time.Unix(item.CreatedAt, 0),
		UpdatedAt:     time.Unix(item.UpdatedAt, 0),
		ActiveSeconds: item.ActiveSeconds,
	}
}
