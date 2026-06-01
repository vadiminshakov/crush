package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// finishStreamModel is a minimal fantasy.LanguageModel that streams a
// single text part followed by a normal (FinishReasonStop) finish. It
// is enough to drive sessionAgent.Run through PrepareStep and a clean
// completion without a recorded provider cassette.
type finishStreamModel struct {
	text string
}

func (m *finishStreamModel) Provider() string { return "fake" }
func (m *finishStreamModel) Model() string    { return "fake-model" }

func (m *finishStreamModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	return &fantasy.Response{
		Content:      fantasy.ResponseContent{fantasy.TextContent{Text: m.text}},
		FinishReason: fantasy.FinishReasonStop,
	}, nil
}

func (m *finishStreamModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	text := m.text
	return func(yield func(fantasy.StreamPart) bool) {
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: "1"}) {
			return
		}
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextDelta, ID: "1", Delta: text}) {
			return
		}
		if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: "1"}) {
			return
		}
		yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeFinish, FinishReason: fantasy.FinishReasonStop})
	}, nil
}

func (m *finishStreamModel) GenerateObject(ctx context.Context, call fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *finishStreamModel) StreamObject(ctx context.Context, call fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, errors.New("not implemented")
}

func newStreamTestAgent(t *testing.T) (*sessionAgent, fakeEnv) {
	t.Helper()
	env := testEnv(t)
	model := &finishStreamModel{text: "done"}
	sa := testSessionAgent(env, model, model, "system").(*sessionAgent)
	return sa, env
}

// TestCancel_ActiveAndAcceptedFiresBothBranches covers the case where a
// session is actively running (activeRequests set) AND a follow-up has
// been accepted (acceptedRuns > 0). A single Cancel must fire both: it
// invokes the active cancel func and records a pending cancel for the
// accepted follow-up.
func TestCancel_ActiveAndAcceptedFiresBothBranches(t *testing.T) {
	t.Parallel()
	sa, _ := newCancelTestAgent(t)

	const sid = "sid"
	var activeCanceled atomic.Bool
	sa.activeRequests.Set(sid, func() { activeCanceled.Store(true) })

	accept := sa.BeginAccepted(sid)
	defer accept.Close()

	sa.Cancel(sid)

	require.True(t, activeCanceled.Load(), "active cancel func must fire")
	require.True(t, sa.hasPendingCancel(sid), "accepted follow-up must record a pending cancel")
}

// TestRun_BusyWithPendingCancelTakesCancelOnEntry covers the busy-queue
// branch consulting pendingCancels: when the session is busy AND a
// cancel has been recorded for an accepted follow-up, Run must take the
// cancel-on-entry path (persist a canceled turn) instead of enqueueing
// the call behind the active run.
func TestRun_BusyWithPendingCancelTakesCancelOnEntry(t *testing.T) {
	t.Parallel()
	sa, env := newCancelTestAgent(t)

	sess, err := env.sessions.Create(t.Context(), "session")
	require.NoError(t, err)

	// Make the session look busy: an earlier prompt is active.
	sa.activeRequests.Set(sess.ID, func() {})

	accept := sa.BeginAccepted(sess.ID)
	// A cancel arrives while this follow-up is accepted-but-not-active.
	sa.Cancel(sess.ID)
	require.True(t, sa.hasPendingCancel(sess.ID))

	result, err := sa.Run(t.Context(), SessionAgentCall{
		SessionID: sess.ID,
		Prompt:    "follow-up",
		Accepted:  accept,
	})
	require.NoError(t, err)
	require.Nil(t, result)

	// The follow-up was canceled on entry, not enqueued.
	require.Equal(t, 0, sa.QueuedPrompts(sess.ID),
		"cancel-on-entry must not enqueue the follow-up behind the active run")
	require.False(t, sa.hasPendingCancel(sess.ID), "pending cancel must be consumed")
	require.Equal(t, 0, sa.acceptedCount(sess.ID), "accept reservation must be released")

	msgs, err := env.messages.List(t.Context(), sess.ID)
	require.NoError(t, err)
	require.Len(t, msgs, 2)
	assert.Equal(t, message.User, msgs[0].Role)
	assert.Equal(t, message.Assistant, msgs[1].Role)
	assert.Equal(t, message.FinishReasonCanceled, msgs[1].FinishReason())
}

// TestRun_PrepareStepDrainSkipsQueuedOnPendingCancel verifies that the
// queue drain inside PrepareStep skips queued follow-up prompts when a
// cancel has been recorded for the session: the queued prompt must not
// be folded into the active turn as an extra user message.
func TestRun_PrepareStepDrainSkipsQueuedOnPendingCancel(t *testing.T) {
	t.Parallel()
	sa, env := newStreamTestAgent(t)

	sess, err := env.sessions.Create(t.Context(), "session")
	require.NoError(t, err)

	// A follow-up prompt sits queued for the session.
	sa.enqueueCall(SessionAgentCall{SessionID: sess.ID, Prompt: "queued-followup"})
	// A cancel was recorded for the session while it sat in the queue.
	sa.pendingCancels.Set(sess.ID, struct{}{})

	result, err := sa.Run(t.Context(), SessionAgentCall{
		SessionID: sess.ID,
		Prompt:    "main",
	})
	require.NoError(t, err)
	require.NotNil(t, result)

	// Only the main prompt produced a user message; the queued
	// follow-up was skipped, not folded into the turn.
	msgs, err := env.messages.List(t.Context(), sess.ID)
	require.NoError(t, err)
	var userMsgs []message.Message
	for _, m := range msgs {
		if m.Role == message.User {
			userMsgs = append(userMsgs, m)
		}
	}
	require.Len(t, userMsgs, 1, "queued follow-up must not create a user message")
	assert.Equal(t, "main", userMsgs[0].Content().String())

	// The queue was drained and the pending cancel consumed.
	require.Equal(t, 0, sa.QueuedPrompts(sess.ID))
	require.False(t, sa.hasPendingCancel(sess.ID))
}

// TestRun_NormalCompletionClearsStalePendingCancel verifies that a Run
// which completes normally clears any stale pending-cancel entry for the
// session, so it cannot catch a future run.
func TestRun_NormalCompletionClearsStalePendingCancel(t *testing.T) {
	t.Parallel()
	sa, env := newStreamTestAgent(t)

	sess, err := env.sessions.Create(t.Context(), "session")
	require.NoError(t, err)

	// A stale pending cancel lingers (no queued work, no accepted run).
	sa.pendingCancels.Set(sess.ID, struct{}{})

	result, err := sa.Run(t.Context(), SessionAgentCall{
		SessionID: sess.ID,
		Prompt:    "main",
	})
	require.NoError(t, err)
	require.NotNil(t, result)

	require.False(t, sa.hasPendingCancel(sess.ID),
		"normal completion must clear the stale pending cancel")

	msgs, err := env.messages.List(t.Context(), sess.ID)
	require.NoError(t, err)
	require.Len(t, msgs, 2)
	assert.Equal(t, message.Assistant, msgs[1].Role)
	assert.Equal(t, message.FinishReasonEndTurn, msgs[1].FinishReason())
}
