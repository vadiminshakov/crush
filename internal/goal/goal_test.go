package goal

import (
	"sync"
	"testing"

	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/stretchr/testify/require"
)

func newTestService(t *testing.T) (Service, string) {
	t.Helper()
	dataDir := t.TempDir()
	t.Cleanup(func() {
		require.NoError(t, db.Release(dataDir))
	})

	conn, err := db.Connect(t.Context(), dataDir)
	require.NoError(t, err)

	queries := db.New(conn)
	sessions := session.NewService(queries, conn)
	createdSession, err := sessions.Create(t.Context(), "Goal test")
	require.NoError(t, err)
	return NewService(queries, conn), createdSession.ID
}

func TestUpdateStatusTransitions(t *testing.T) {
	goals, sessionID := newTestService(t)
	g, err := goals.Create(t.Context(), sessionID, "Objective")
	require.NoError(t, err)

	// Activating an already active goal is a harmless no-op.
	same, err := goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalActive)
	require.NoError(t, err)
	require.Equal(t, GoalActive, same.Status)

	paused, err := goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalPaused)
	require.NoError(t, err)
	require.Equal(t, GoalPaused, paused.Status)

	// Pausing twice is idempotent.
	paused, err = goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalPaused)
	require.NoError(t, err)
	require.Equal(t, GoalPaused, paused.Status)

	completed, err := goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalComplete)
	require.NoError(t, err)
	require.Equal(t, GoalComplete, completed.Status)

	// A pause racing the completion must keep the terminal status.
	still, err := goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalPaused)
	require.NoError(t, err)
	require.Equal(t, GoalComplete, still.Status)

	// A completed goal can never be reopened.
	_, err = goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalActive)
	require.ErrorContains(t, err, "cannot become active")

	_, err = goals.UpdateStatus(t.Context(), sessionID, "stale", GoalPaused)
	require.ErrorContains(t, err, "stale goal ID")

	_, err = goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalStatus("bogus"))
	require.ErrorContains(t, err, "unknown goal status")
}

func TestUpdateStatusConcurrentPauseAndComplete(t *testing.T) {
	goals, sessionID := newTestService(t)
	g, err := goals.Create(t.Context(), sessionID, "Objective")
	require.NoError(t, err)

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			_, _ = goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalPaused)
		})
		wg.Go(func() {
			_, _ = goals.UpdateStatus(t.Context(), sessionID, g.GoalID, GoalComplete)
		})
	}
	wg.Wait()

	current, err := goals.Get(t.Context(), sessionID)
	require.NoError(t, err)
	require.Equal(t, GoalComplete, current.Status)
}

func TestPauseAllActive(t *testing.T) {
	goals, sessionID := newTestService(t)
	g, err := goals.Create(t.Context(), sessionID, "Objective")
	require.NoError(t, err)

	paused, err := goals.PauseAllActive(t.Context())
	require.NoError(t, err)
	require.EqualValues(t, 1, paused)

	current, err := goals.Get(t.Context(), sessionID)
	require.NoError(t, err)
	require.Equal(t, g.GoalID, current.GoalID)
	require.Equal(t, GoalPaused, current.Status)

	paused, err = goals.PauseAllActive(t.Context())
	require.NoError(t, err)
	require.Zero(t, paused)
}

func TestStaleClearDoesNotDeleteReplacementGoal(t *testing.T) {
	dataDir := t.TempDir()
	t.Cleanup(func() {
		require.NoError(t, db.Release(dataDir))
	})

	conn, err := db.Connect(t.Context(), dataDir)
	require.NoError(t, err)

	queries := db.New(conn)
	sessions := session.NewService(queries, conn)
	createdSession, err := sessions.Create(t.Context(), "Goal clear test")
	require.NoError(t, err)

	goals := NewService(queries, conn)
	oldGoal, err := goals.Create(t.Context(), createdSession.ID, "Old objective")
	require.NoError(t, err)
	_, err = goals.UpdateStatus(t.Context(), createdSession.ID, oldGoal.GoalID, GoalComplete)
	require.NoError(t, err)

	replacement, err := goals.Create(t.Context(), createdSession.ID, "Replacement objective")
	require.NoError(t, err)

	_, err = goals.Clear(t.Context(), createdSession.ID, oldGoal.GoalID)
	require.ErrorContains(t, err, "stale goal ID")

	current, err := goals.Get(t.Context(), createdSession.ID)
	require.NoError(t, err)
	require.Equal(t, replacement.GoalID, current.GoalID)
}

func TestContinuationContext(t *testing.T) {
	t.Parallel()
	_, ok := ContinuationOf(t.Context())
	require.False(t, ok)
	goalID, ok := ContinuationOf(WithContinuation(t.Context(), "goal"))
	require.True(t, ok)
	require.Equal(t, "goal", goalID)
}
