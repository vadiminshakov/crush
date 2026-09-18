package goal

import (
	"context"
	"log/slog"
)

// Pause persists the pause before cancelling a pending continuation. Completed
// goals keep their terminal status.
func (r *Runtime) Pause(ctx context.Context, sessionID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	return r.pauseLocked(ctx, sessionID, "")
}

// Resume admits another continuation only after the paused turn has unwound.
func (r *Runtime) Resume(ctx context.Context, sessionID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	if err := r.sessions.admitResume(sessionID); err != nil {
		return nil, err
	}
	g, err := r.goals.Get(ctx, sessionID)
	if err != nil || g == nil || g.Status != GoalPaused {
		return g, err
	}
	g, err = r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalActive)
	if err != nil {
		return g, err
	}
	r.Kick(sessionID)
	return g, nil
}

// Clear removes the goal and cancels any admitted continuation.
func (r *Runtime) Clear(ctx context.Context, sessionID string, goalID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	g, err := r.goals.Clear(ctx, sessionID, goalID)
	if err == nil && g != nil {
		r.sessions.cancelContinuation(sessionID)
	}
	return g, err
}

// Stop prevents new continuations and pauses known goals before shutdown.
func (r *Runtime) Stop(ctx context.Context) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	for _, sessionID := range r.sessions.shutdown() {
		if _, err := r.pauseLocked(ctx, sessionID, ""); err != nil {
			slog.Error("Failed to pause goal during shutdown", "session_id", sessionID, "error", err)
		}
	}
}

// pauseLocked pauses the session's goal if it is still active (and, when
// expectGoalID is set, still the observed one), then interrupts any running
// continuation. Callers must hold statusMu.
func (r *Runtime) pauseLocked(ctx context.Context, sessionID string, expectGoalID string) (*Goal, error) {
	g, err := r.goals.Get(ctx, sessionID)
	if err == nil && g.isActiveGoal(expectGoalID) {
		g, err = r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalPaused)
	}
	r.sessions.cancelContinuation(sessionID)
	return g, err
}

// pauseDetached pauses the goal on a context that survives the caller's
// cancellation, bounded by pauseTimeout.
func (r *Runtime) pauseDetached(ctx context.Context, sessionID string, expectGoalID string) error {
	pauseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), pauseTimeout)
	defer cancel()
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	_, err := r.pauseLocked(pauseCtx, sessionID, expectGoalID)
	return err
}
