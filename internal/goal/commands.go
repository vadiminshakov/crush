package goal

import (
	"context"
	"errors"
	"log/slog"
)

var errStopped = errors.New("goal runtime is shutting down")

// Set creates the session's goal and starts driving it.
func (r *Runtime) Set(ctx context.Context, sessionID, objective string) (*Goal, error) {
	return r.activate(sessionID, func() (*Goal, error) {
		return r.goals.Create(ctx, sessionID, objective)
	})
}

// Resume reactivates a paused goal and starts driving it again.
func (r *Runtime) Resume(ctx context.Context, sessionID string) (*Goal, error) {
	return r.activate(sessionID, func() (*Goal, error) {
		g, err := r.goals.Get(ctx, sessionID)
		if err != nil || g == nil || g.Status != GoalPaused {
			return g, err
		}
		return r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalActive)
	})
}

// activate applies write, which is expected to leave the session with an
// active goal, lifts the session's continuation suppression and asks for a
// continuation.
func (r *Runtime) activate(sessionID string, write func() (*Goal, error)) (*Goal, error) {
	g, err := r.activateLocked(sessionID, write)
	if err != nil {
		return g, err
	}
	r.TryContinueGoal(sessionID)
	return g, nil
}

func (r *Runtime) activateLocked(sessionID string, write func() (*Goal, error)) (*Goal, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped.Load() {
		return nil, errStopped
	}
	g, err := write()
	if err == nil {
		r.session(sessionID).suppressed = false
	}
	return g, err
}

// Pause pauses an active goal and cancels its continuation. A completed
// goal keeps its status.
func (r *Runtime) Pause(ctx context.Context, sessionID string) (*Goal, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pauseLocked(ctx, sessionID)
}

// Clear removes the goal and cancels its continuation.
func (r *Runtime) Clear(ctx context.Context, sessionID string, goalID string) (*Goal, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	g, err := r.goals.Clear(ctx, sessionID, goalID)
	if err == nil && g != nil {
		r.cancelRunning(sessionID)
	}
	return g, err
}

// Stop cancels every continuation, refuses new ones and pauses the goals of
// the sessions the runtime has seen, before shutdown.
func (r *Runtime) Stop(ctx context.Context) {
	r.stopped.Store(true)
	r.mu.Lock()
	defer r.mu.Unlock()
	for sessionID := range r.sessions {
		if _, err := r.pauseLocked(ctx, sessionID); err != nil {
			slog.Error("Failed to pause goal during shutdown", "session_id", sessionID, "error", err)
		}
	}
}

// pauseLocked pauses the session's goal if it is active and cancels the
// session's running continuation. Callers must hold mu.
func (r *Runtime) pauseLocked(ctx context.Context, sessionID string) (*Goal, error) {
	g, err := r.goals.Get(ctx, sessionID)
	if err != nil || g == nil || g.Status != GoalActive {
		return g, err
	}
	g, err = r.goals.UpdateStatus(ctx, sessionID, g.GoalID, GoalPaused)
	r.cancelRunning(sessionID)
	return g, err
}
