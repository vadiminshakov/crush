package goal

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/agent/notify"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/crush/internal/pubsub"
)

// pauseTimeout bounds goal pauses that run on a detached context after the
// originating turn has already been cancelled or failed.
const pauseTimeout = 5 * time.Second

// AgentRunner is the port through which the runtime drives the session
// agent and observes whether a session is already occupied by it.
type AgentRunner interface {
	Run(ctx context.Context, sessionID string, prompt string, attachments ...message.Attachment) (*fantasy.AgentResult, error)
	IsSessionBusy(sessionID string) bool
	QueuedPrompts(sessionID string) int
}

// Runtime is the application service that keeps a session's goal moving.
// It coordinates three collaborators: the goal store (persisted status),
// the agent runner (turn execution) and the session registry (in-memory
// occupancy). Goal status rules live in Goal and turnOutcome; the runtime
// only sequences them.
//
// The runtime is split by role:
//   - commands.go: user commands that change goal status;
//   - occupancy.go: turns that occupy a session, announced via TurnLease;
//   - continuation.go: synthetic turns that drive an active goal.
//
// One rule ties them together: a continuation always yields to an occupied
// session, so whoever frees a session must Kick the goal again.
type Runtime struct {
	goals    Service
	agent    AgentRunner
	notify   pubsub.Publisher[notify.Notification]
	sessions *sessionRegistry

	// statusMu serializes status commands (pause, resume, clear, stop) so
	// that a status write and its matching continuation cancel or admission
	// are observed by other commands as a single step.
	statusMu sync.Mutex
}

func NewRuntime(store Service, agent AgentRunner, notify pubsub.Publisher[notify.Notification]) *Runtime {
	return &Runtime{
		goals:    store,
		agent:    agent,
		notify:   notify,
		sessions: newSessionRegistry(),
	}
}

// Kick re-checks the session's goal in the background and continues it if
// it is active and the session is free.
func (r *Runtime) Kick(sessionID string) {
	go func() {
		if err := r.MaybeContinue(context.Background(), sessionID); err != nil {
			slog.Error("Goal runtime continuation failed", "session_id", sessionID, "error", err)
		}
	}()
}
