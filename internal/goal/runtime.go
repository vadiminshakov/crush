package goal

import (
	"context"
	"log/slog"
	"sync"
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
// goal stays active, asks for one more continuation turn. Since the agent
// starts a continuation only on an idle session, the runtime never tracks
// who occupies a session: a continuation that finds it busy is dropped, and
// the turn keeping it busy tries again when it finishes. Continuation
// attempts of one session run one at a time, so a goal read and the
// continuation it starts never race with another attempt's.
//
// Continuations run under a per-session context that every status change
// the runtime makes cancels and replaces. A change thereby stops them,
// including one whose goal read has not reached the agent yet.
type Runtime struct {
	goals  Service
	agent  Agent
	notify pubsub.Publisher[notify.Notification]

	// root is cancelled by Stop, and every continuation context with it.
	root context.Context
	stop context.CancelFunc

	// statusMu serializes status changes, so that a status write and the
	// continuation cancel following it are observed by others as one step.
	statusMu sync.Mutex

	// mu guards continuationCtx and attempting. It is never held across
	// calls out.
	mu sync.Mutex
	// continuationCtx holds the context each session's continuations run
	// under.
	continuationCtx map[string]cancelableCtx
	// attempting holds the sessions with a continuation attempt running,
	// each with whether another attempt was requested meanwhile.
	attempting map[string]bool
}

type cancelableCtx struct {
	ctx    context.Context
	cancel context.CancelFunc
}

func NewRuntime(goals Service, agent Agent, notify pubsub.Publisher[notify.Notification]) *Runtime {
	root, stop := context.WithCancel(context.Background())
	return &Runtime{
		goals:           goals,
		agent:           agent,
		notify:          notify,
		root:            root,
		stop:            stop,
		continuationCtx: make(map[string]cancelableCtx),
		attempting:      make(map[string]bool),
	}
}

// TryContinueGoal starts a continuation turn in the background if the
// session's goal is active and the session idle, and does nothing
// otherwise. Attempts requested while one runs collapse into a single
// attempt after it. A nil Runtime does nothing.
func (r *Runtime) TryContinueGoal(sessionID string) {
	if r == nil || !r.startAttempt(sessionID) {
		return
	}
	go func() {
		for again := true; again; again = r.finishAttempt(sessionID) {
			if err := r.continueGoal(sessionID); err != nil {
				slog.Error("Goal continuation failed", "session_id", sessionID, "error", err)
			}
		}
	}()
}

// startAttempt reports whether the caller should run the session's
// continuation attempt. If one is already running, it asks that one to
// make one more attempt.
func (r *Runtime) startAttempt(sessionID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, running := r.attempting[sessionID]; running {
		r.attempting[sessionID] = true
		return false
	}
	r.attempting[sessionID] = false
	return true
}

// finishAttempt ends the session's continuation attempt and reports whether
// another one was requested meanwhile.
func (r *Runtime) finishAttempt(sessionID string) (again bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.attempting[sessionID] {
		r.attempting[sessionID] = false
		return true
	}
	delete(r.attempting, sessionID)
	return false
}

// continueGoal starts at most one continuation turn for the session's
// active goal. The next one is asked for when that turn finishes.
func (r *Runtime) continueGoal(sessionID string) error {
	// Take the context before reading the goal: a status change landing in
	// between either cancels it or shows up in the read.
	ctx := r.continuationContext(sessionID)
	if ctx.Err() != nil {
		return nil
	}
	g, err := r.goals.Get(context.Background(), sessionID)
	if err != nil || g == nil || g.Status != GoalActive {
		return err
	}
	prompt, err := renderContinuationPrompt(g)
	if err != nil {
		return err
	}
	slog.Info("Starting goal continuation turn", "session_id", sessionID, "goal_id", g.GoalID)
	if r.notify != nil {
		r.notify.Publish(pubsub.CreatedEvent, notify.Notification{SessionID: sessionID, Type: notify.TypeGoalContinue})
	}
	_, err = r.agent.RunContinuation(WithContinuation(ctx, g.GoalID), sessionID, prompt)
	if ctx.Err() != nil {
		// A status change cancelled the turn, which is not a failure.
		return nil
	}
	return err
}

// TurnFinished applies the outcome of a finished turn, ordinary or
// continuation alike, to the session's goal, given the turn's context and
// what the agent returned for it:
//   - an answered turn asks for the next continuation;
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
	switch classifyTurn(ctx, result, err) {
	case turnQueued:
	case turnFailed:
		r.pauseAfterFailure(sessionID)
	case turnAnswered, turnCancelled:
		r.TryContinueGoal(sessionID)
	}
}

// pauseAfterFailure pauses the session's goal after a failed turn.
func (r *Runtime) pauseAfterFailure(sessionID string) {
	ctx, cancel := context.WithTimeout(context.Background(), pauseTimeout)
	defer cancel()
	if _, err := r.Pause(ctx, sessionID); err != nil {
		slog.Error("Failed to pause goal after a failed turn", "session_id", sessionID, "error", err)
	}
}

// continuationContext returns the context the session's continuations run
// under.
func (r *Runtime) continuationContext(sessionID string) context.Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	c, ok := r.continuationCtx[sessionID]
	if !ok {
		c = r.newContinuationCtx()
		r.continuationCtx[sessionID] = c
	}
	return c.ctx
}

// cancelContinuations cancels the session's continuations, running or about
// to start, and gives later ones a fresh context.
func (r *Runtime) cancelContinuations(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if c, ok := r.continuationCtx[sessionID]; ok {
		c.cancel()
	}
	r.continuationCtx[sessionID] = r.newContinuationCtx()
}

func (r *Runtime) newContinuationCtx() cancelableCtx {
	ctx, cancel := context.WithCancel(r.root)
	return cancelableCtx{ctx: ctx, cancel: cancel}
}

// sessionIDs returns every session the runtime has seen.
func (r *Runtime) sessionIDs() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.continuationCtx))
	for id := range r.continuationCtx {
		ids = append(ids, id)
	}
	return ids
}
