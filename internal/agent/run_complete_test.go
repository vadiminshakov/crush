package agent

import (
	"context"
	"testing"
	"time"

	"github.com/charmbracelet/crush/internal/agent/notify"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/stretchr/testify/require"
)

// TestSessionAgentRun_QueueStripsOnComplete verifies that when a Run
// call is enqueued (because the session is already busy), the
// OnComplete hook is NOT propagated onto the queued copy. The hook
// belongs to the caller's retry/coalesce scope (typically
// coordinator.Run) which has already returned by the time the queue
// drains; carrying it forward would silently funnel the terminal
// event into a closure nobody reads, and subscribers (`crush run`)
// would hang waiting for a RunComplete that never publishes.
func TestSessionAgentRun_QueueStripsOnComplete(t *testing.T) {
	t.Parallel()

	env := testEnv(t)
	a := NewSessionAgent(SessionAgentOptions{
		Sessions: env.sessions,
		Messages: env.messages,
	}).(*sessionAgent)

	const sessionID = "queued-session"
	// Mark the session as busy so Run takes the queue branch
	// without needing a real model.
	a.activeRequests.Set(sessionID, func() {})

	var called bool
	hook := func(notify.RunComplete) { called = true }

	res, err := a.Run(t.Context(), SessionAgentCall{
		SessionID:  sessionID,
		RunID:      "run-xyz",
		Prompt:     "queued prompt",
		OnComplete: hook,
	})
	require.NoError(t, err)
	require.Nil(t, res, "queued Run must return (nil, nil)")
	require.False(t, called,
		"OnComplete must not fire on the enqueue path; the caller's scope is still live")

	queued, ok := a.messageQueue.Get(sessionID)
	require.True(t, ok)
	require.Len(t, queued, 1)
	require.Nil(t, queued[0].OnComplete,
		"queued SessionAgentCall must have OnComplete stripped so the drain falls back to the default broker publish")
	require.Equal(t, "queued prompt", queued[0].Prompt,
		"all other fields must be preserved on the queued copy")
	require.Equal(t, "run-xyz", queued[0].RunID,
		"RunID must be preserved on the queued copy so the drained turn's "+
			"RunComplete still correlates with the originating SendMessage")
}

// TestDrainUncanceledQueue_FiltersUnderDispatchLock verifies that the
// queue drain evaluates the per-session cancel mark while holding the
// dispatch mutex (canceledBySeq's documented precondition). Queued calls
// at or below the cancel high-water mark are dropped, calls queued after
// the cancel (higher seq) survive, untracked enqueues (seq == 0) are
// dropped whenever any mark is present, and the queue is cleared.
func TestDrainUncanceledQueue_FiltersUnderDispatchLock(t *testing.T) {
	t.Parallel()

	env := testEnv(t)
	a := NewSessionAgent(SessionAgentOptions{
		Sessions: env.sessions,
		Messages: env.messages,
	}).(*sessionAgent)

	const sessionID = "drain-session"
	a.messageQueue.Set(sessionID, []SessionAgentCall{
		{SessionID: sessionID, Prompt: "below", acceptSeq: 1},
		{SessionID: sessionID, Prompt: "at-mark", acceptSeq: 2},
		{SessionID: sessionID, Prompt: "after", acceptSeq: 3},
		{SessionID: sessionID, Prompt: "untracked", acceptSeq: 0},
	})
	// Cancel high-water mark at seq 2: seq <= 2 and seq == 0 are covered.
	a.cancelMark.Set(sessionID, 2)

	survivors := a.drainUncanceledQueue(sessionID)

	require.Len(t, survivors, 1,
		"only the follow-up queued after the cancel (seq > mark) must survive")
	require.Equal(t, "after", survivors[0].Prompt)

	_, ok := a.messageQueue.Get(sessionID)
	require.False(t, ok, "drain must clear the session message queue")
}

// TestDrainUncanceledQueue_NoMarkKeepsAll verifies that with no cancel
// mark recorded, every queued call survives the drain.
func TestDrainUncanceledQueue_NoMarkKeepsAll(t *testing.T) {
	t.Parallel()

	env := testEnv(t)
	a := NewSessionAgent(SessionAgentOptions{
		Sessions: env.sessions,
		Messages: env.messages,
	}).(*sessionAgent)

	const sessionID = "drain-nomark"
	a.messageQueue.Set(sessionID, []SessionAgentCall{
		{SessionID: sessionID, Prompt: "a", acceptSeq: 0},
		{SessionID: sessionID, Prompt: "b", acceptSeq: 5},
	})

	survivors := a.drainUncanceledQueue(sessionID)
	require.Len(t, survivors, 2, "no cancel mark means all queued calls survive")
}

// TestRunCompletePublisher_MustDeliverOverTakesPublish exercises the
// pubsub.Publisher interface change end-to-end: a Broker is the only
// concrete Publisher implementation and must satisfy both Publish and
// PublishMustDeliver. The coordinator's final RunComplete emit relies
// on PublishMustDeliver to apply bounded-blocking semantics so a
// momentarily-full subscriber buffer can't silently drop the
// authoritative end-of-run event.
func TestRunCompletePublisher_MustDeliverOverTakesPublish(t *testing.T) {
	t.Parallel()

	broker := pubsub.NewBroker[notify.RunComplete]()
	t.Cleanup(broker.Shutdown)

	// Subscribe before publishing so the event is delivered.
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	ch := broker.Subscribe(ctx)

	rc := notify.RunComplete{SessionID: "S", MessageID: "m", Text: "ok"}
	var pub pubsub.Publisher[notify.RunComplete] = broker
	pub.PublishMustDeliver(t.Context(), pubsub.UpdatedEvent, rc)

	select {
	case got := <-ch:
		require.Equal(t, rc, got.Payload)
	case <-time.After(time.Second):
		t.Fatal("PublishMustDeliver did not deliver event")
	}
}
