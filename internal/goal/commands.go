package goal

import (
	"context"
	"errors"
	"log/slog"
)

var errStopped = errors.New("goal runtime is shutting down")

// Set creates the session's goal and starts driving it.
func (r *Runtime) Set(ctx context.Context, sessionID, objective string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	if r.root.Err() != nil {
		return nil, errStopped
	}
	g, err := r.goals.Create(ctx, sessionID, objective)
	if err != nil {
		return nil, err
	}
	r.cancelContinuations(sessionID)
	r.TryContinueGoal(sessionID)
	return g, nil
}

// Pause pauses an active goal and cancels its continuation. A completed
// goal keeps its status.
func (r *Runtime) Pause(ctx context.Context, sessionID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	return r.pauseLocked(ctx, sessionID)
}

// Resume reactivates a paused goal and starts driving it again.
func (r *Runtime) Resume(ctx context.Context, sessionID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	if r.root.Err() != nil {
		return nil, errStopped
	}
	g, err := r.goals.Get(ctx, sessionID)
	if err != nil || g == nil || g.Status != GoalPaused {
		return g, err
	}
	g, err = r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalActive)
	if err != nil {
		return g, err
	}
	r.cancelContinuations(sessionID)
	r.TryContinueGoal(sessionID)
	return g, nil
}

// Clear removes the goal and cancels its continuation.
func (r *Runtime) Clear(ctx context.Context, sessionID string, goalID string) (*Goal, error) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	g, err := r.goals.Clear(ctx, sessionID, goalID)
	if err == nil && g != nil {
		r.cancelContinuations(sessionID)
	}
	return g, err
}

// Stop cancels every continuation, refuses new ones and pauses the goals of
// the sessions the runtime has seen, before shutdown.
func (r *Runtime) Stop(ctx context.Context) {
	r.statusMu.Lock()
	defer r.statusMu.Unlock()
	r.stop()
	for _, sessionID := range r.sessionIDs() {
		if _, err := r.pauseLocked(ctx, sessionID); err != nil {
			slog.Error("Failed to pause goal during shutdown", "session_id", sessionID, "error", err)
		}
	}
}

// pauseLocked pauses the session's goal if it is active. Callers must hold
// statusMu.
func (r *Runtime) pauseLocked(ctx context.Context, sessionID string) (*Goal, error) {
	g, err := r.goals.Get(ctx, sessionID)
	if err != nil || g == nil || g.Status != GoalActive {
		return g, err
	}
	g, err = r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalPaused)
	r.cancelContinuations(sessionID)
	return g, err
}
