package goal

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/agent/notify"
	"github.com/charmbracelet/crush/internal/pubsub"
)

// pauseTimeout bounds the goal pause that follows a failed turn.
const pauseTimeout = 5 * time.Second

// Agent is the port through which the runtime starts continuation turns.
type Agent interface {
	// RunContinuation runs prompt as a continuation turn and reports it to
	// Runtime.TurnFinished like any other turn, handing ctx, which
	// WithContinuation marks, on to the turn's tools. It only starts on an
	// idle session: when the session already has work (running, dispatched
	// or queued), it returns a nil result and error without running or
	// queueing anything, and that work tries to continue the goal when it
	// finishes.
	RunContinuation(ctx context.Context, sessionID, prompt string) (*fantasy.AgentResult, error)
}

// Runtime keeps a session's goal moving. It is a post-turn policy on top of
// the agent, not a scheduler: the agent reports every finished turn to
// TurnFinished, which applies the turn's outcome to the goal and, while the
// goal stays active, asks for one more continuation turn. The agent starts
// a continuation only on an idle session, which keeps continuations of one
// session from running in parallel; a continuation that finds the session
// busy is dropped, and the turn keeping it busy asks again when it
// finishes.
//
// The goal store is the source of truth: every continuation re-reads the
// goal right before it starts, and status writes are compare-and-set, so
// the runtime only has to serialize that read against status changes. A
// continuation runs under its own context, which the status change that
// stops the goal (Pause, Clear, Stop) cancels.
//
// A continuation turn that does no tool work suppresses the next
// automatic continuation, so a goal the model only talks about does not
// spin; a user turn, Set or Resume lifts the suppression.
type Runtime struct {
	goals  Service
	agent  Agent
	notify pubsub.Publisher[notify.Notification]

	// stopped is set by Stop and refuses continuations from then on.
	stopped atomic.Bool

	// mu serializes goal reads against status writes and the start of a
	// continuation, so a continuation never starts for a goal that has
	// just been paused, cleared or replaced. It is never held across calls
	// into the agent.
	mu       sync.Mutex
	sessions map[string]*sessionState
}

// sessionState is what the runtime remembers about a session it has
// driven.
type sessionState struct {
	// running is the continuation the session is running, if any.
	running *continuation
	// suppressed is set when the last continuation did no tool work.
	suppressed bool
}

// continuation is one continuation turn in flight.
type continuation struct {
	goalID string
	ctx    context.Context
	cancel context.CancelFunc
}

func NewRuntime(goals Service, agent Agent, notify pubsub.Publisher[notify.Notification]) *Runtime {
	return &Runtime{
		goals:    goals,
		agent:    agent,
		notify:   notify,
		sessions: make(map[string]*sessionState),
	}
}

// TryContinueGoal starts, in the background, one continuation turn for the
// session's goal if the goal is active and the session idle, and does
// nothing otherwise. A nil Runtime does nothing.
func (r *Runtime) TryContinueGoal(sessionID string) {
	if r == nil {
		return
	}
	go func() {
		if err := r.continueGoal(sessionID); err != nil {
			slog.Error("Goal continuation failed", "session_id", sessionID, "error", err)
		}
	}()
}

// continueGoal starts at most one continuation turn for the session's
// active goal. The next one is asked for when that turn finishes.
func (r *Runtime) continueGoal(sessionID string) error {
	c, prompt, err := r.startContinuation(sessionID)
	if err != nil || c == nil {
		return err
	}
	defer r.finishContinuation(sessionID, c)
	slog.Info("Starting goal continuation turn", "session_id", sessionID, "goal_id", c.goalID)
	if r.notify != nil {
		r.notify.Publish(pubsub.CreatedEvent, notify.Notification{SessionID: sessionID, Type: notify.TypeGoalContinue})
	}
	_, err = r.agent.RunContinuation(WithContinuation(c.ctx, c.goalID), sessionID, prompt)
	if c.ctx.Err() != nil {
		// A status change cancelled the turn, which is not a failure.
		return nil
	}
	return err
}

// startContinuation reads the goal and, when it is active and nothing
// holds the session back, registers a continuation for it. It returns a
// nil continuation when there is nothing to start.
func (r *Runtime) startContinuation(sessionID string) (*continuation, string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped.Load() {
		return nil, "", nil
	}
	s := r.session(sessionID)
	if s.running != nil || s.suppressed {
		return nil, "", nil
	}
	g, err := r.goals.Get(context.Background(), sessionID)
	if err != nil || g == nil || g.Status != GoalActive {
		return nil, "", err
	}
	prompt, err := renderContinuationPrompt(g)
	if err != nil {
		return nil, "", err
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.running = &continuation{goalID: g.GoalID, ctx: ctx, cancel: cancel}
	return s.running, prompt, nil
}

// finishContinuation releases the continuation once its turn returned.
func (r *Runtime) finishContinuation(sessionID string, c *continuation) {
	r.mu.Lock()
	defer r.mu.Unlock()
	c.cancel()
	if s := r.sessions[sessionID]; s != nil && s.running == c {
		s.running = nil
	}
}

// TurnFinished applies the outcome of a finished turn, ordinary or
// continuation alike, to the session's goal, given the turn's context and
// what the agent returned for it:
//   - an answered turn asks for the next continuation, unless it was a
//     continuation that did no tool work, which suppresses the next one;
//   - a failed turn pauses the goal;
//   - a cancelled turn leaves the goal's status to whoever cancelled it
//     (Pause, Clear, Stop, or an agent cancel, which pauses first) and only
//     tries to continue the goal, as a Resume may have asked for while the
//     turn kept the session busy;
//   - a turn whose prompt was only queued changes nothing, as the turn that
//     runs the queue reports its own finish.
//
// A nil Runtime does nothing.
func (r *Runtime) TurnFinished(ctx context.Context, sessionID string, result *fantasy.AgentResult, err error) {
	if r == nil {
		return
	}
	outcome := classifyTurn(ctx, result, err)
	if outcome == turnQueued {
		return
	}
	_, isContinuation := ContinuationOf(ctx)
	// A user turn lifts the suppression; an idle continuation sets it.
	r.setSuppressed(sessionID, isContinuation && outcome == turnAnswered && !hasToolCalls(result))
	switch outcome {
	case turnFailed:
		pauseCtx, cancel := context.WithTimeout(context.Background(), pauseTimeout)
		defer cancel()
		if _, err := r.Pause(pauseCtx, sessionID); err != nil {
			slog.Error("Failed to pause goal after a failed turn", "session_id", sessionID, "error", err)
		}
	case turnAnswered, turnCancelled:
		r.TryContinueGoal(sessionID)
	}
}

func (r *Runtime) setSuppressed(sessionID string, suppressed bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.session(sessionID).suppressed = suppressed
	if suppressed {
		slog.Info("Goal continuation did no tool work; waiting for the user", "session_id", sessionID)
	}
}

// session returns the session's state, creating it on first use. Callers
// must hold mu.
func (r *Runtime) session(sessionID string) *sessionState {
	s, ok := r.sessions[sessionID]
	if !ok {
		s = &sessionState{}
		r.sessions[sessionID] = s
	}
	return s
}

// cancelRunning cancels the session's running continuation, if any.
// Callers must hold mu.
func (r *Runtime) cancelRunning(sessionID string) {
	if s := r.sessions[sessionID]; s != nil && s.running != nil {
		s.running.cancel()
	}
}
