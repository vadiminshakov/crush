package question

import (
	"context"
	"sync"

	"github.com/charmbracelet/crush/internal/csync"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/google/uuid"
)

// QuestionRequest is the event published to the UI when the model wants
// to ask the user a clarifying question.
type QuestionRequest struct {
	ID       string   `json:"id"`
	Question string   `json:"question"`
	Options  []string `json:"options"`
}

// Service is the interface the tool and UI both depend on.
type Service interface {
	pubsub.Subscriber[QuestionRequest]
	// Ask blocks until the user responds or ctx is cancelled.
	Ask(ctx context.Context, question string, options []string) (string, error)
	// Respond unblocks a pending Ask call with the given answer.
	Respond(id, answer string)
}

type questionService struct {
	*pubsub.Broker[QuestionRequest]

	pendingRequests *csync.Map[string, chan string]
	// serialises concurrent Ask calls — only one question shown at a time
	requestMu sync.Mutex
}

func NewService() Service {
	return &questionService{
		Broker:          pubsub.NewBroker[QuestionRequest](),
		pendingRequests: csync.NewMap[string, chan string](),
	}
}

func (s *questionService) Ask(ctx context.Context, question string, options []string) (string, error) {
	s.requestMu.Lock()
	defer s.requestMu.Unlock()

	req := QuestionRequest{
		ID:       uuid.New().String(),
		Question: question,
		Options:  options,
	}

	respCh := make(chan string, 1)
	s.pendingRequests.Set(req.ID, respCh)
	defer s.pendingRequests.Del(req.ID)

	s.Publish(pubsub.CreatedEvent, req)

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case answer := <-respCh:
		return answer, nil
	}
}

func (s *questionService) Respond(id, answer string) {
	ch, ok := s.pendingRequests.Get(id)
	if ok {
		ch <- answer
	}
}
