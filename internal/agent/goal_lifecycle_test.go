package agent

import (
	"context"
	"errors"
	"testing"

	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/goal"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/stretchr/testify/require"
)

func coordinatorWithGoal(t *testing.T) (*coordinator, goal.Service, string, *mockSessionAgent) {
	t.Helper()
	conn, err := db.Connect(t.Context(), t.TempDir())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, conn.Close()) })
	queries := db.New(conn)
	sessions := session.NewService(queries, conn)
	s, err := sessions.Create(t.Context(), "Goal test")
	require.NoError(t, err)
	store := goal.NewService(queries, conn)
	_, err = store.Create(t.Context(), s.ID, "Fix tests")
	require.NoError(t, err)
	a := &mockSessionAgent{}
	c := &coordinator{mainAgent: a, goalService: store}
	c.goalRuntime = goal.NewRuntime(store, c, nil)
	return c, store, s.ID, a
}

func TestCoordinatorCancelPausesGoal(t *testing.T) {
	t.Parallel()
	c, store, sessionID, a := coordinatorWithGoal(t)
	c.Cancel(sessionID)
	g, err := store.Get(t.Context(), sessionID)
	require.NoError(t, err)
	require.Equal(t, goal.GoalPaused, g.Status)
	require.Equal(t, []string{sessionID}, a.cancelled)
}

func TestCoordinatorInitializationFailurePausesGoal(t *testing.T) {
	t.Parallel()
	c, store, sessionID, _ := coordinatorWithGoal(t)
	failure := errors.New("provider initialization failed")
	c.readyWg.Go(func() error { return failure })
	_, err := c.Run(context.Background(), sessionID, "Continue")
	require.ErrorIs(t, err, failure)
	g, err := store.Get(t.Context(), sessionID)
	require.NoError(t, err)
	require.Equal(t, goal.GoalPaused, g.Status)
}

func TestCoordinatorCancelPreservesCompletedGoal(t *testing.T) {
	t.Parallel()
	c, store, sessionID, _ := coordinatorWithGoal(t)
	g, err := store.Get(t.Context(), sessionID)
	require.NoError(t, err)
	_, err = store.UpdateStatus(t.Context(), sessionID, g.GoalID, goal.GoalComplete)
	require.NoError(t, err)
	c.Cancel(sessionID)
	g, err = store.Get(t.Context(), sessionID)
	require.NoError(t, err)
	require.Equal(t, goal.GoalComplete, g.Status)
	// A completion that wins the race with a previously-read pause stays complete.
	g, err = store.UpdateStatus(t.Context(), sessionID, g.GoalID, goal.GoalPaused)
	require.NoError(t, err)
	require.Equal(t, goal.GoalComplete, g.Status)
}
