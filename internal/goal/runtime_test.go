package goal

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

const waitFor = 5 * time.Second

// fakeStore is an in-memory Service holding one session's goal.
type fakeStore struct {
	Service
	mu   sync.Mutex
	goal *Goal
}

func newFakeStore(status GoalStatus) *fakeStore {
	return &fakeStore{goal: &Goal{SessionID: "session", GoalID: "goal", Status: status}}
}

func (s *fakeStore) Get(context.Context, string) (*Goal, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal == nil {
		return nil, nil
	}
	g := *s.goal
	return &g, nil
}

// Create names the goal after its objective.
func (s *fakeStore) Create(_ context.Context, sessionID, objective string) (*Goal, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal != nil && s.goal.Status != GoalComplete {
		return nil, errors.New("session already has an active goal")
	}
	s.goal = &Goal{SessionID: sessionID, GoalID: objective, Objective: objective, Status: GoalActive}
	g := *s.goal
	return &g, nil
}

func (s *fakeStore) UpdateStatus(_ context.Context, _, goalID string, status GoalStatus) (*Goal, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal == nil || s.goal.GoalID != goalID {
		return nil, errors.New("stale goal")
	}
	if slices.Contains(transitions[status], s.goal.Status) {
		s.goal.Status = status
	}
	g := *s.goal
	return &g, nil
}

func (s *fakeStore) Clear(_ context.Context, _, goalID string) (*Goal, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal == nil || s.goal.GoalID != goalID {
		return nil, errors.New("stale goal")
	}
	g := s.goal
	s.goal = nil
	return g, nil
}

func (s *fakeStore) status() GoalStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.goal == nil {
		return ""
	}
	return s.goal.Status
}

// complete stands in for update_goal(status="complete").
func (s *fakeStore) complete() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.goal.Status = GoalComplete
}

// fakeAgent stands in for the coordinator: it reports every turn to
// Runtime.TurnFinished after freeing the session, and starts a
// continuation only on an idle session.
type fakeAgent struct {
	runtime       *Runtime
	continuation  func(ctx context.Context, goalID string) (*fantasy.AgentResult, error)
	continuations atomic.Int32

	mu   sync.Mutex
	busy bool
}

func newRuntime(store Service, continuation func(ctx context.Context, goalID string) (*fantasy.AgentResult, error)) (*Runtime, *fakeAgent) {
	agent := &fakeAgent{continuation: continuation}
	agent.runtime = NewRuntime(store, agent, nil)
	return agent.runtime, agent
}

func (a *fakeAgent) RunContinuation(ctx context.Context, _, _ string) (*fantasy.AgentResult, error) {
	if !a.claim() {
		return nil, nil
	}
	a.continuations.Add(1)
	goalID, _ := ContinuationOf(ctx)
	return a.turn(ctx, func() (*fantasy.AgentResult, error) { return a.continuation(ctx, goalID) })
}

// claim occupies the session, reporting false if it is already busy.
func (a *fakeAgent) claim() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.busy {
		return false
	}
	a.busy = true
	return true
}

// turn runs fn as a turn on a session claimed beforehand.
func (a *fakeAgent) turn(ctx context.Context, fn func() (*fantasy.AgentResult, error)) (*fantasy.AgentResult, error) {
	result, err := fn()
	a.mu.Lock()
	a.busy = false
	a.mu.Unlock()
	a.runtime.TurnFinished(ctx, "session", result, err)
	return result, err
}

func answered() (*fantasy.AgentResult, error) {
	return &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonStop}}, nil
}

// blockUntilCancelled is a continuation that runs until a status change
// cancels it.
func blockUntilCancelled(ctx context.Context, _ string) (*fantasy.AgentResult, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestClassifyTurn(t *testing.T) {
	t.Parallel()
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tt := range []struct {
		name    string
		ctx     context.Context
		result  *fantasy.AgentResult
		err     error
		outcome turnOutcome
	}{
		{name: "queued", outcome: turnQueued},
		{name: "setup error", err: errors.New("model unavailable"), outcome: turnFailed},
		{name: "provider timeout", err: context.DeadlineExceeded, outcome: turnFailed},
		{name: "agent cancel", err: fmt.Errorf("stream: %w", context.Canceled), outcome: turnCancelled},
		{name: "context cancel", ctx: cancelled, err: errors.New("stream closed"), outcome: turnCancelled},
		{name: "normal answer", result: &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonStop}}, outcome: turnAnswered},
		{name: "token limit", result: &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonLength}}, outcome: turnFailed},
		{name: "refusal", result: &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonContentFilter}}, outcome: turnFailed},
		{name: "loop guard", result: &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonStop}, Steps: []fantasy.StepResult{{Response: fantasy.Response{FinishReason: fantasy.FinishReasonToolCalls}}}}, outcome: turnFailed},
		{name: "tool halt", result: &fantasy.AgentResult{Response: fantasy.Response{FinishReason: fantasy.FinishReasonStop, Content: fantasy.ResponseContent{fantasy.ToolResultContent{StopTurn: true}}}}, outcome: turnFailed},
		{name: "unknown stop", result: &fantasy.AgentResult{}, outcome: turnFailed},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			ctx := cmp.Or(tt.ctx, context.Background())
			require.Equal(t, tt.outcome, classifyTurn(ctx, tt.result, tt.err))
		})
	}
}

func TestAnsweredTurnsContinueUntilGoalComplete(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	var agent *fakeAgent
	runtime, agent := newRuntime(store, func(context.Context, string) (*fantasy.AgentResult, error) {
		if agent.continuations.Load() == 3 {
			store.complete()
		}
		return answered()
	})
	runtime.TryContinueGoal("session")
	require.Eventually(t, func() bool { return store.status() == GoalComplete }, waitFor, time.Millisecond)
	require.Never(t, func() bool { return agent.continuations.Load() > 3 }, 50*time.Millisecond, time.Millisecond)
}

func TestFailedTurnPausesGoal(t *testing.T) {
	t.Parallel()
	for _, err := range []error{errors.New("provider unavailable"), context.DeadlineExceeded} {
		t.Run(err.Error(), func(t *testing.T) {
			t.Parallel()
			store := newFakeStore(GoalActive)
			runtime, agent := newRuntime(store, func(context.Context, string) (*fantasy.AgentResult, error) {
				return nil, err
			})
			runtime.TryContinueGoal("session")
			require.Eventually(t, func() bool { return store.status() == GoalPaused }, waitFor, time.Millisecond)
			runtime.TryContinueGoal("session")
			require.Never(t, func() bool { return agent.continuations.Load() > 1 }, 50*time.Millisecond, time.Millisecond)
		})
	}
}

// Whoever cancels a turn decides the goal's status; the runtime only
// re-checks the goal afterwards.
func TestCancelledTurnLeavesStatusToCanceller(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	runtime, agent := newRuntime(store, func(context.Context, string) (*fantasy.AgentResult, error) {
		store.complete()
		return answered()
	})
	runtime.TurnFinished(t.Context(), "session", nil, context.Canceled)
	require.Eventually(t, func() bool { return store.status() == GoalComplete }, waitFor, time.Millisecond)
	require.EqualValues(t, 1, agent.continuations.Load())
}

func TestQueuedTurnLeavesGoalAlone(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	runtime, agent := newRuntime(store, nil)
	runtime.TurnFinished(t.Context(), "session", nil, nil)
	require.Equal(t, GoalActive, store.status())
	require.Never(t, func() bool { return agent.continuations.Load() > 0 }, 50*time.Millisecond, time.Millisecond)
}

// A continuation that finds the session busy is dropped; the turn keeping
// the session busy asks for it again when it finishes.
func TestBusySessionDefersContinuationToItsTurn(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	runtime, agent := newRuntime(store, func(context.Context, string) (*fantasy.AgentResult, error) {
		store.complete()
		return answered()
	})
	require.True(t, agent.claim())
	require.NoError(t, runtime.continueGoal("session"))
	require.Zero(t, agent.continuations.Load())

	_, err := agent.turn(t.Context(), answered)
	require.NoError(t, err)
	require.Eventually(t, func() bool { return store.status() == GoalComplete }, waitFor, time.Millisecond)
	require.EqualValues(t, 1, agent.continuations.Load())
}

func TestPauseAndClearCancelRunningContinuation(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name   string
		change func(*Runtime) error
		status GoalStatus
	}{
		{name: "pause", status: GoalPaused, change: func(r *Runtime) error {
			_, err := r.Pause(context.Background(), "session")
			return err
		}},
		{name: "clear", change: func(r *Runtime) error {
			_, err := r.Clear(context.Background(), "session", "goal")
			return err
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := newFakeStore(GoalActive)
			stopped := make(chan error, 1)
			runtime, agent := newRuntime(store, func(ctx context.Context, goalID string) (*fantasy.AgentResult, error) {
				result, err := blockUntilCancelled(ctx, goalID)
				stopped <- err
				return result, err
			})
			runtime.TryContinueGoal("session")
			require.Eventually(t, func() bool { return agent.continuations.Load() == 1 }, waitFor, time.Millisecond)

			require.NoError(t, tt.change(runtime))
			require.ErrorIs(t, <-stopped, context.Canceled)
			require.Equal(t, tt.status, store.status())
			require.Never(t, func() bool { return agent.continuations.Load() > 1 }, 50*time.Millisecond, time.Millisecond)
		})
	}
}

func TestStaleClearKeepsReplacementRunning(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	running := make(chan context.Context, 1)
	runtime, _ := newRuntime(store, func(ctx context.Context, goalID string) (*fantasy.AgentResult, error) {
		running <- ctx
		return blockUntilCancelled(ctx, goalID)
	})
	runtime.TryContinueGoal("session")
	ctx := <-running

	_, err := runtime.Clear(t.Context(), "session", "old")
	require.ErrorContains(t, err, "stale goal")
	require.NoError(t, ctx.Err())

	_, err = runtime.Pause(t.Context(), "session")
	require.NoError(t, err)
}

// A turn cancelled by a pause may still be unwinding when the user
// resumes. It must not pause the resumed goal, and since it kept the
// session busy when Resume asked to continue, it must continue the goal.
func TestResumeWhileStoppedTurnUnwinds(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	runtime, agent := newRuntime(store, func(context.Context, string) (*fantasy.AgentResult, error) {
		store.complete()
		return answered()
	})

	require.True(t, agent.claim())
	started, release := make(chan struct{}), make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		agent.turn(context.Background(), func() (*fantasy.AgentResult, error) { //nolint:errcheck
			close(started)
			<-release
			return nil, context.Canceled
		})
	}()
	<-started

	_, err := runtime.Pause(t.Context(), "session")
	require.NoError(t, err)
	_, err = runtime.Resume(t.Context(), "session")
	require.NoError(t, err)
	close(release)
	<-done

	require.Eventually(t, func() bool { return store.status() == GoalComplete }, waitFor, time.Millisecond)
	require.EqualValues(t, 1, agent.continuations.Load())
}

// A continuation of a cleared goal that fails afterwards must not pause the
// goal that replaced it.
func TestOldContinuationFailureKeepsReplacementGoal(t *testing.T) {
	t.Parallel()
	store := newFakeStore(GoalActive)
	release := make(chan struct{})
	ran := make(chan string, 2)
	runtime, _ := newRuntime(store, func(_ context.Context, goalID string) (*fantasy.AgentResult, error) {
		ran <- goalID
		if goalID == "goal" {
			<-release
			return nil, errors.New("old turn failed")
		}
		store.complete()
		return answered()
	})
	runtime.TryContinueGoal("session")
	require.Equal(t, "goal", <-ran)

	_, err := runtime.Clear(t.Context(), "session", "goal")
	require.NoError(t, err)
	_, err = runtime.Set(t.Context(), "session", "replacement")
	require.NoError(t, err)
	close(release)

	require.Equal(t, "replacement", <-ran)
	require.Eventually(t, func() bool { return store.status() == GoalComplete }, waitFor, time.Millisecond)
}

func TestStopPausesGoalsAndRefusesContinuation(t *testing.T) {
	t.Parallel()
	store := &fakeStore{}
	runtime, agent := newRuntime(store, blockUntilCancelled)
	_, err := runtime.Set(t.Context(), "session", "goal")
	require.NoError(t, err)
	require.Eventually(t, func() bool { return agent.continuations.Load() == 1 }, waitFor, time.Millisecond)

	runtime.Stop(t.Context())
	require.Equal(t, GoalPaused, store.status())
	_, err = runtime.Resume(t.Context(), "session")
	require.ErrorIs(t, err, errStopped)
	require.NoError(t, runtime.continueGoal("session"))
	require.EqualValues(t, 1, agent.continuations.Load())
}

func TestNilRuntime(t *testing.T) {
	t.Parallel()
	var runtime *Runtime
	runtime.TryContinueGoal("session")
	runtime.TurnFinished(t.Context(), "session", nil, errors.New("failed"))
}
