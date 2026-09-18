package goal

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSessionRegistryContinuationOwnership(t *testing.T) {
	t.Parallel()
	reg := newSessionRegistry()

	runCtx, ok := reg.acquireContinuation(t.Context(), "session")
	require.True(t, ok)

	// A second request while owned must yield and leave a re-check behind.
	_, ok = reg.acquireContinuation(t.Context(), "session")
	require.False(t, ok)
	require.ErrorContains(t, reg.admitResume("session"), "still stopping")

	reg.cancelContinuation("session")
	require.Error(t, runCtx.Err())

	require.True(t, reg.releaseContinuation("session"))
	require.False(t, reg.releaseContinuation("session"))
	require.NoError(t, reg.admitResume("session"))
}

func TestSessionRegistryTurnsBlockContinuations(t *testing.T) {
	t.Parallel()
	reg := newSessionRegistry()

	reg.beginTurn("session")
	reg.beginTurn("session")
	require.True(t, reg.turnInFlight("session"))
	_, ok := reg.acquireContinuation(t.Context(), "session")
	require.False(t, ok)

	reg.endTurn("session")
	require.True(t, reg.turnInFlight("session"))
	reg.endTurn("session")
	require.False(t, reg.turnInFlight("session"))
	// Releasing more turns than were begun never goes negative.
	reg.endTurn("session")
	require.False(t, reg.turnInFlight("session"))

	_, ok = reg.acquireContinuation(t.Context(), "session")
	require.True(t, ok)
}

func TestSessionRegistryShutdown(t *testing.T) {
	t.Parallel()
	reg := newSessionRegistry()
	reg.beginTurn("b")
	_, ok := reg.acquireContinuation(t.Context(), "a")
	require.True(t, ok)

	require.Equal(t, []string{"a", "b"}, reg.shutdown())

	// Stopped registries admit nothing and drop pending re-checks.
	_, ok = reg.acquireContinuation(t.Context(), "c")
	require.False(t, ok)
	require.ErrorContains(t, reg.admitResume("c"), "shutting down")
	require.False(t, reg.releaseContinuation("a"))
}
