package goal

import (
	"context"
	"errors"
	"log/slog"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/agent/notify"
	"github.com/charmbracelet/crush/internal/pubsub"
)

// MaybeContinue drives synthetic continuation turns for the session's active
// goal until the session becomes busy, the goal leaves the active state, or a
// turn stops.
func (r *Runtime) MaybeContinue(ctx context.Context, sessionID string) error {
	runCtx, ok := r.sessions.acquireContinuation(ctx, sessionID)
	if !ok {
		return nil
	}
	// releaseContinuation must be deferred directly so its recover() observes
	// panics.
	defer r.releaseContinuation(ctx, sessionID)
	return r.runContinuations(ctx, runCtx, sessionID)
}

// releaseContinuation gives up continuation ownership and kicks the goal
// again if another request arrived meanwhile. A panic pauses the goal and is
// then re-raised.
func (r *Runtime) releaseContinuation(ctx context.Context, sessionID string) {
	failure := recover()
	if failure != nil {
		if err := r.pauseDetached(ctx, sessionID, ""); err != nil {
			slog.Error("Failed to pause goal", "reason", "panic", "session_id", sessionID, "error", err)
		}
	}
	recheck := r.sessions.releaseContinuation(sessionID)
	if failure != nil {
		panic(failure)
	}
	if recheck {
		r.Kick(sessionID)
	}
}

// runContinuations runs synthetic turns back to back while the goal stays
// active and the session stays free.
func (r *Runtime) runContinuations(ctx, runCtx context.Context, sessionID string) error {
	for {
		if r.sessionOccupied(sessionID) {
			return nil
		}
		g, err := r.goals.Get(context.WithoutCancel(ctx), sessionID)
		if err != nil {
			return err
		}
		if !g.isActiveGoal("") {
			return nil
		}
		result, err := r.runTurn(runCtx, sessionID, g)
		switch classifyTurn(result, err).reaction() {
		case pauseGoal:
			return errors.Join(err, r.pauseDetached(ctx, sessionID, g.GoalID))
		case keepWaiting:
			// The prompt was queued behind a user prompt; that turn's lease
			// kicks the goal when it finishes.
			return nil
		case continueGoal:
		}
	}
}

// sessionOccupied reports whether a coordinator turn, a busy agent or a
// queued prompt currently owns the session.
func (r *Runtime) sessionOccupied(sessionID string) bool {
	return r.sessions.turnInFlight(sessionID) ||
		r.agent.IsSessionBusy(sessionID) ||
		r.agent.QueuedPrompts(sessionID) > 0
}

// runTurn executes one synthetic continuation turn for g. Cancellation of
// runCtx is always folded into the returned error.
func (r *Runtime) runTurn(runCtx context.Context, sessionID string, g *Goal) (result *fantasy.AgentResult, err error) {
	defer func() { err = errors.Join(err, runCtx.Err()) }()
	prompt, err := renderContinuationPrompt(g)
	if err != nil {
		return nil, err
	}
	if err := runCtx.Err(); err != nil {
		return nil, err
	}
	slog.Info("Starting synthetic continuation turn", "session_id", sessionID, "goal_id", g.GoalID)
	if r.notify != nil {
		r.notify.Publish(pubsub.CreatedEvent, notify.Notification{SessionID: sessionID, Type: notify.TypeGoalContinue})
	}
	return r.agent.Run(WithContinuation(runCtx, g.GoalID), sessionID, prompt)
}
