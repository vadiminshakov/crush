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
	GoalActive    GoalStatus = "active"
	GoalPaused    GoalStatus = "paused"
	GoalComplete  GoalStatus = "complete"
)

type Goal struct {
	ScopeID   string     `json:"scope_id"`
	GoalID    string     `json:"goal_id"`
	Objective string     `json:"objective"`
	Status    GoalStatus `json:"status"`
	Version   int64      `json:"version"`
	CreatedAt time.Time  `json:"created_at"`
	UpdatedAt time.Time  `json:"updated_at"`
}

type Service interface {
	pubsub.Subscriber[Goal]
	Get(ctx context.Context, scopeID string) (*Goal, error)
	CreateOrReplace(ctx context.Context, scopeID string, objective string) (*Goal, error)
	UpdateStatus(ctx context.Context, scopeID string, goalID string, status GoalStatus) (*Goal, error)
	Clear(ctx context.Context, scopeID string) error
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

func (s *service) Get(ctx context.Context, scopeID string) (*Goal, error) {
	dbGoal, err := s.q.GetGoalByScopeID(ctx, scopeID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("getting goal: %w", err)
	}
	return s.fromDBItem(dbGoal), nil
}

func (s *service) CreateOrReplace(ctx context.Context, scopeID string, objective string) (*Goal, error) {
	goalID := uuid.New().String()
	dbGoal, err := s.q.CreateOrReplaceGoal(ctx, db.CreateOrReplaceGoalParams{
		ScopeID:   scopeID,
		GoalID:    goalID,
		Objective: objective,
		Status:    string(GoalActive),
	})
	if err != nil {
		return nil, fmt.Errorf("creating goal: %w", err)
	}
	goal := s.fromDBItem(dbGoal)
	s.Publish(pubsub.UpdatedEvent, *goal)
	return goal, nil
}

func (s *service) UpdateStatus(ctx context.Context, scopeID string, goalID string, status GoalStatus) (*Goal, error) {
	dbGoal, err := s.q.UpdateGoalStatus(ctx, db.UpdateGoalStatusParams{
		ScopeID: scopeID,
		GoalID:  goalID,
		Status:  string(status),
	})
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("goal not found or stale goal ID")
		}
		return nil, fmt.Errorf("updating goal status: %w", err)
	}
	goal := s.fromDBItem(dbGoal)
	s.Publish(pubsub.UpdatedEvent, *goal)
	return goal, nil
}

func (s *service) Clear(ctx context.Context, scopeID string) error {
	goal, err := s.Get(ctx, scopeID)
	if err != nil {
		return err
	}
	if goal == nil {
		return nil
	}
	err = s.q.DeleteGoal(ctx, scopeID)
	if err != nil {
		return fmt.Errorf("deleting goal: %w", err)
	}
	s.Publish(pubsub.DeletedEvent, *goal)
	return nil
}

func (s *service) fromDBItem(item db.Goal) *Goal {
	return &Goal{
		ScopeID:   item.ScopeID,
		GoalID:    item.GoalID,
		Objective: item.Objective,
		Status:    GoalStatus(item.Status),
		Version:   item.Version,
		CreatedAt: time.Unix(item.CreatedAt, 0),
		UpdatedAt: time.Unix(item.UpdatedAt, 0),
	}
}
