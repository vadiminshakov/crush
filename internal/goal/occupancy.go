package goal

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"sync/atomic"

	"charm.land/fantasy"
)

// TurnLease marks a session as occupied by a coordinator turn, from setup
// through finalization, including the interval after the session agent
// releases its own busy flag. While any lease is held, continuations yield
// and Resume is refused.
//
// Every lease must be closed exactly once: with Finish when the work was an
// agent turn, or with Release when it produced no agent result. Closing is
// what re-checks the goal, since continuations that yielded to the lease are
// not retried by anyone else.
type TurnLease struct {
	runtime   *Runtime
	sessionID string
	// continuation is set when the turn is a synthetic continuation turn.
	// Its outcome belongs to the continuation loop that started it.
	continuation bool
	closed       atomic.Bool
}

// Occupy takes a lease on the session for a coordinator turn. A nil Runtime
// returns a nil lease, whose methods do nothing.
func (r *Runtime) Occupy(ctx context.Context, sessionID string) *TurnLease {
	if r == nil {
		return nil
	}
	r.sessions.beginTurn(sessionID)
	_, continuation := ContinuationOf(ctx)
	return &TurnLease{runtime: r, sessionID: sessionID, continuation: continuation}
}

// Finish closes a lease held by an agent turn and applies the turn's
// outcome to the goal: a stopped turn pauses it, an answered one lets it
// continue. ctx is the turn's context; its cancellation counts as a stop.
func (l *TurnLease) Finish(ctx context.Context, result *fantasy.AgentResult, err error) {
	if l == nil || !l.closed.CompareAndSwap(false, true) {
		return
	}
	reaction := keepWaiting
	if !l.continuation {
		reaction = classifyTurn(result, errors.Join(err, ctx.Err())).reaction()
	}
	// Pause before freeing the session so Resume cannot slip in between.
	if reaction == pauseGoal {
		if err := l.runtime.pauseDetached(ctx, l.sessionID, ""); err != nil {
			slog.Error("Failed to pause goal", "reason", "agent stopped", "session_id", l.sessionID, "error", err)
		}
	}
	l.runtime.sessions.endTurn(l.sessionID)
	if reaction == continueGoal {
		l.runtime.Kick(l.sessionID)
	}
}

// Release closes a lease held by work that produces no agent result, such
// as summarization, and re-checks the goal.
func (l *TurnLease) Release() {
	if l == nil || !l.closed.CompareAndSwap(false, true) {
		return
	}
	l.runtime.sessions.endTurn(l.sessionID)
	l.runtime.Kick(l.sessionID)
}

// sessionState is the runtime's in-memory view of one session: how many
// coordinator turns occupy it and whether a goal continuation owns it.
type sessionState struct {
	// turns counts coordinator turns in flight. A turn spans setup and
	// finalization, including the interval after the session agent releases
	// its own busy flag.
	turns int
	// cancelContinuation is non-nil while a continuation owns the session.
	cancelContinuation context.CancelFunc
	// recheck records that a continuation request arrived while another one
	// owned the session and must be replayed once that owner releases it.
	recheck bool
}

// occupied reports whether any turn or continuation is using the session.
func (s *sessionState) occupied() bool {
	return s.turns > 0 || s.cancelContinuation != nil
}

func (s *sessionState) cancel() {
	if s.cancelContinuation != nil {
		s.cancelContinuation()
	}
}

// sessionRegistry tracks every session the runtime has touched. It is the
// only synchronization point for in-memory runtime state and never calls
// out, so it can be used from any other lock.
type sessionRegistry struct {
	mu       sync.Mutex
	sessions map[string]*sessionState
	stopped  bool
}

func newSessionRegistry() *sessionRegistry {
	return &sessionRegistry{sessions: make(map[string]*sessionState)}
}

// ensure returns the state for sessionID, registering the session when it
// is seen for the first time. Callers must hold mu.
func (r *sessionRegistry) ensure(sessionID string) *sessionState {
	s := r.sessions[sessionID]
	if s == nil {
		s = &sessionState{}
		r.sessions[sessionID] = s
	}
	return s
}

// beginTurn records a coordinator turn occupying the session.
func (r *sessionRegistry) beginTurn(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ensure(sessionID).turns++
}

// endTurn releases a turn recorded by beginTurn.
func (r *sessionRegistry) endTurn(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if s := r.sessions[sessionID]; s != nil && s.turns > 0 {
		s.turns--
	}
}

// turnInFlight reports whether a coordinator turn occupies the session.
func (r *sessionRegistry) turnInFlight(sessionID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.sessions[sessionID]
	return s != nil && s.turns > 0
}

// acquireContinuation makes the caller the session's sole continuation
// owner and returns the context governing its turns. It reports false when
// the caller must yield: the registry is stopped, a turn is in flight, or
// another continuation already owns the session (in which case a re-check
// is left pending for it).
func (r *sessionRegistry) acquireContinuation(ctx context.Context, sessionID string) (context.Context, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped {
		return nil, false
	}
	s := r.ensure(sessionID)
	if s.turns > 0 {
		return nil, false
	}
	if s.cancelContinuation != nil {
		s.recheck = true
		return nil, false
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancelContinuation = cancel
	return runCtx, true
}

// releaseContinuation gives up continuation ownership and reports whether a
// re-check request arrived while the caller owned the session.
func (r *sessionRegistry) releaseContinuation(sessionID string) (recheck bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.sessions[sessionID]
	if s == nil {
		return false
	}
	s.cancel()
	s.cancelContinuation = nil
	recheck = s.recheck && !r.stopped
	s.recheck = false
	return recheck
}

// cancelContinuation interrupts the continuation owning the session, if any.
// Ownership is released by the owner itself via releaseContinuation.
func (r *sessionRegistry) cancelContinuation(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if s := r.sessions[sessionID]; s != nil {
		s.cancel()
	}
}

// admitResume reports whether the session may accept a resumed goal: the
// registry must not be stopping and the previous turn must have unwound.
func (r *sessionRegistry) admitResume(sessionID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopped {
		return errors.New("goal runtime is shutting down")
	}
	if s := r.sessions[sessionID]; s != nil && s.occupied() {
		return errors.New("goal turn is still stopping; try resume again shortly")
	}
	return nil
}

// shutdown prevents new continuations and returns every known session so
// the caller can pause their goals.
func (r *sessionRegistry) shutdown() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stopped = true
	return slices.Sorted(maps.Keys(r.sessions))
}
