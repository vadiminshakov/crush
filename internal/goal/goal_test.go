package goal

import (
	"testing"

	"github.com/charmbracelet/crush/internal/db"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/stretchr/testify/require"
)

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
