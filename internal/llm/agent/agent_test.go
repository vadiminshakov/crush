package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/csync"
	"github.com/charmbracelet/crush/internal/llm/provider"
	"github.com/charmbracelet/crush/internal/llm/tools"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/charmbracelet/crush/mocks"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Test_Agent_Run_With_Tools tests the agent's ability to execute real tools
// in an isolated filesystem environment. All dependencies are mocked except for
// the tools themselves. This test verifies:
// 1. Agent can process provider responses that request tool execution
// 2. Tools are actually executed by the agent
// 3. Tool results are captured and returned properly
// 4. The complete agent workflow (provider -> tools -> provider) works end-to-end
func Test_Agent_Run_With_Tools(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	_, err := config.Init(tempDir, tempDir, false)
	require.NoError(t, err)

	testFiles := map[string]string{
		"src/main.go":     "package main\n\nfunc main() {\n\tprintln(\"Hello, World!\")\n}",
		"src/utils.go":    "package main\n\nfunc helper() string {\n\treturn \"test\"\n}",
		"README.md":       "# Test Project\n\nThis is a test project.",
		"config/app.json": "{\"name\": \"test-app\", \"version\": \"1.0.0\"}",
	}

	for filePath, content := range testFiles {
		fullPath := filepath.Join(tempDir, filePath)
		require.NoError(t, os.MkdirAll(filepath.Dir(fullPath), 0o755))
		require.NoError(t, os.WriteFile(fullPath, []byte(content), 0o644))
	}

	mockProvider := mocks.NewMockProvider(t)
	mockSessionSvc := mocks.NewMockSessionService(t)
	mockMessageSvc := mocks.NewMockMessageService(t)

	realTools := func() []tools.BaseTool {
		return []tools.BaseTool{
			tools.NewLsTool(nil, tempDir),
			tools.NewGrepTool(tempDir),
			tools.NewGlobTool(tempDir),
		}
	}

	// setup agent with real tools but mocked services
	agent := &agent{
		Broker:         pubsub.NewBroker[AgentEvent](),
		provider:       mockProvider,
		providerID:     "test-provider",
		sessions:       mockSessionSvc,
		messages:       mockMessageSvc,
		agentCfg:       config.Agent{Model: config.SelectedModelTypeLarge, ID: "test-agent"},
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
		tools:          csync.NewLazySlice(realTools),
	}

	testSessionID := "test-session"
	testContent := "List files in src directory and search for 'main' in the code"

	testModel := catwalk.Model{
		ID:             "test-model",
		SupportsImages: true,
	}
	mockProvider.EXPECT().Model().Return(testModel).Maybe()

	testSession := session.Session{
		ID:               testSessionID,
		SummaryMessageID: "",
	}
	mockSessionSvc.EXPECT().Get(mock.Anything, testSessionID).Return(testSession, nil).Maybe()
	mockSessionSvc.EXPECT().Save(mock.Anything, mock.AnythingOfType("session.Session")).Return(session.Session{}, nil).Maybe()

	mockMessageSvc.EXPECT().List(mock.Anything, testSessionID).Return([]message.Message{}, nil)

	userMsg := message.Message{
		ID:        "user-msg-1",
		SessionID: testSessionID,
		Role:      message.User,
		Parts:     []message.ContentPart{message.TextContent{Text: testContent}},
	}
	mockMessageSvc.EXPECT().Create(mock.Anything, testSessionID, mock.MatchedBy(func(params message.CreateMessageParams) bool {
		return params.Role == message.User
	})).Return(userMsg, nil)

	assistantMsg := message.Message{
		ID:        "assistant-msg-1",
		SessionID: testSessionID,
		Role:      message.Assistant,
		Parts:     []message.ContentPart{},
	}
	mockMessageSvc.EXPECT().Create(mock.Anything, testSessionID, mock.MatchedBy(func(params message.CreateMessageParams) bool {
		return params.Role == message.Assistant
	})).Return(assistantMsg, nil)

	// create tool message mock
	toolMsg := message.Message{
		ID:        "tool-msg-1",
		SessionID: testSessionID,
		Role:      message.Tool,
		Parts:     []message.ContentPart{},
	}
	var capturedToolResults []message.ToolResult
	mockMessageSvc.EXPECT().Create(mock.Anything, testSessionID, mock.MatchedBy(func(params message.CreateMessageParams) bool {
		if params.Role == message.Tool {
			for _, part := range params.Parts {
				if tr, ok := part.(message.ToolResult); ok {
					capturedToolResults = append(capturedToolResults, tr)
				}
			}
			return true
		}
		return false
	})).Return(toolMsg, nil).Once()

	mockMessageSvc.EXPECT().Update(mock.Anything, mock.AnythingOfType("message.Message")).Return(nil).Maybe()

	// create event channel with tool execution flow
	// first response: Agent decides to use tools
	eventChan := make(chan provider.ProviderEvent, 2)
	eventChan <- provider.ProviderEvent{
		Type:    provider.EventContentDelta,
		Content: "I'll list the src directory and search for 'main'.",
	}
	eventChan <- provider.ProviderEvent{
		Type: provider.EventComplete,
		Response: &provider.ProviderResponse{
			Content:      "I'll list the src directory and search for 'main'.",
			FinishReason: message.FinishReasonToolUse,
			Usage:        provider.TokenUsage{InputTokens: 50, OutputTokens: 10},
			ToolCalls: []message.ToolCall{
				{
					ID:    "tool-call-1",
					Name:  "ls",
					Input: `{"path": "src"}`,
				},
				{
					ID:    "tool-call-2",
					Name:  "grep",
					Input: `{"pattern": "main", "path": "."}`,
				},
			},
		},
	}
	close(eventChan)

	// second response: after tools executed, provide final answer
	followUpChan := make(chan provider.ProviderEvent, 2)
	followUpChan <- provider.ProviderEvent{
		Type:    provider.EventContentDelta,
		Content: "I found the files in src/ directory and located 'main' in the code.",
	}
	followUpChan <- provider.ProviderEvent{
		Type: provider.EventComplete,
		Response: &provider.ProviderResponse{
			Content:      "I found the files in src/ directory and located 'main' in the code.",
			FinishReason: message.FinishReasonEndTurn,
			Usage:        provider.TokenUsage{InputTokens: 75, OutputTokens: 25},
			ToolCalls:    []message.ToolCall{},
		},
	}
	close(followUpChan)

	mockProvider.EXPECT().StreamResponse(mock.Anything, mock.AnythingOfType("[]message.Message"), mock.AnythingOfType("[]tools.BaseTool")).Return(eventChan).Once()
	mockProvider.EXPECT().StreamResponse(mock.Anything, mock.AnythingOfType("[]message.Message"), mock.AnythingOfType("[]tools.BaseTool")).Return(followUpChan).Once()

	ctx := context.Background()

	// execute the test
	resultChan, err := agent.Run(ctx, testSessionID, testContent)
	require.NoError(t, err)
	require.NotNil(t, resultChan)

	// wait for completion
	timeout := time.After(5 * time.Second)
	var finalResult AgentEvent

	for {
		select {
		case result, ok := <-resultChan:
			require.Truef(t, ok, "Channel closed without receiving final result")

			if result.Done {
				finalResult = result
				goto done
			}
			if result.Type == AgentEventTypeError {
				require.FailNowf(t, "Received error", "%v", result.Error)
			}
		case <-timeout:
			require.FailNow(t, "Test timed out waiting for result")
		}
	}

done:
	// verify the final result
	require.Equal(t, AgentEventTypeResponse, finalResult.Type)
	require.True(t, finalResult.Done)
	require.NotNil(t, finalResult.Message)
	require.Equal(t, message.Assistant, finalResult.Message.Role)

	// verify that tools were actually executed and results captured
	require.NotEmpty(t, capturedToolResults, "Tools should have been executed and results captured")
	require.Len(t, capturedToolResults, 2, "Should have results from both ls and grep tools")

	// check that we have results from the expected tools
	var lsResultFound, grepResultFound bool
	for _, result := range capturedToolResults {
		switch result.ToolCallID {
		case "tool-call-1": // ls tool
			lsResultFound = true
			require.False(t, result.IsError, "ls tool should execute successfully")
			require.Contains(t, result.Content, "main.go", "ls should find main.go file")
			require.Contains(t, result.Content, "utils.go", "ls should find utils.go file")
		case "tool-call-2": // grep tool
			grepResultFound = true
			require.False(t, result.IsError, "grep tool should execute successfully")
			require.Contains(t, result.Content, "main", "grep should find 'main' pattern in the files")
		}
	}
	require.True(t, lsResultFound, "ls tool result should be present")
	require.True(t, grepResultFound, "grep tool result should be present")
}

func Test_Agent_IsBusy(t *testing.T) {
	t.Parallel()

	agent := &agent{
		activeRequests: csync.NewMap[string, context.CancelFunc](),
	}

	require.False(t, agent.IsBusy())
	require.False(t, agent.IsSessionBusy("test-session"))

	// set agent as busy
	cancel := func() {}
	agent.activeRequests.Set("test-session", cancel)

	require.True(t, agent.IsBusy())
	require.True(t, agent.IsSessionBusy("test-session"))
}

func Test_Agent_Cancel(t *testing.T) {
	t.Parallel()

	agent := &agent{
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
	}

	cancelled := false
	originalCancel := func() { cancelled = true }
	agent.activeRequests.Set("test-session", originalCancel)
	agent.promptQueue.Set("test-session", []string{"prompt1", "prompt2"})

	require.True(t, agent.IsSessionBusy("test-session"))
	require.Equal(t, 2, agent.QueuedPrompts("test-session"))

	agent.Cancel("test-session")

	require.True(t, cancelled)
	require.False(t, agent.IsSessionBusy("test-session"))
	require.Equal(t, 0, agent.QueuedPrompts("test-session"))
}

func Test_Agent_CancelAll(t *testing.T) {
	t.Parallel()

	agent := &agent{
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
	}

	var wg sync.WaitGroup
	cancelled := make([]bool, 2)

	cancel1 := func() {
		cancelled[0] = true
		wg.Done()
	}
	cancel2 := func() {
		cancelled[1] = true
		wg.Done()
	}

	wg.Add(2)
	agent.activeRequests.Set("session1", cancel1)
	agent.activeRequests.Set("session2", cancel2)

	require.True(t, agent.IsBusy())

	agent.CancelAll()

	// wait for cancellations to complete
	wg.Wait()

	require.True(t, cancelled[0])
	require.True(t, cancelled[1])
	require.False(t, agent.IsBusy())
}

func Test_Agent_QueuedPrompts(t *testing.T) {
	t.Parallel()

	agent := &agent{
		promptQueue: csync.NewMap[string, []string](),
	}

	require.Equal(t, 0, agent.QueuedPrompts("test-session"))

	agent.promptQueue.Set("test-session", []string{"test prompt"})
	require.Equal(t, 1, agent.QueuedPrompts("test-session"))

	agent.promptQueue.Set("test-session", []string{"prompt1", "prompt2"})
	require.Equal(t, 2, agent.QueuedPrompts("test-session"))

	agent.promptQueue.Set("test-session", nil)
	require.Equal(t, 0, agent.QueuedPrompts("test-session"))
}

func Test_Agent_ClearQueue(t *testing.T) {
	t.Parallel()

	agent := &agent{
		promptQueue: csync.NewMap[string, []string](),
	}

	agent.promptQueue.Set("test-session", []string{"prompt1", "prompt2"})
	require.Equal(t, 2, agent.QueuedPrompts("test-session"))

	agent.ClearQueue("test-session")
	require.Equal(t, 0, agent.QueuedPrompts("test-session"))
}

func Test_Agent_Run_SessionBusyQueue(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	_, err := config.Init(tempDir, tempDir, false)
	require.NoError(t, err)

	mockProvider := mocks.NewMockProvider(t)
	mockSessionSvc := mocks.NewMockSessionService(t)
	mockMessageSvc := mocks.NewMockMessageService(t)

	agent := &agent{
		Broker:         pubsub.NewBroker[AgentEvent](),
		provider:       mockProvider,
		sessions:       mockSessionSvc,
		messages:       mockMessageSvc,
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
		agentCfg:       config.Agent{Model: config.SelectedModelTypeLarge},
	}

	testModel := catwalk.Model{
		ID:             "test-model",
		SupportsImages: false,
	}
	mockProvider.EXPECT().Model().Return(testModel).Maybe()

	// make session busy
	testSessionID := "busy-session"
	ctx, cancel := context.WithCancel(context.Background())
	agent.activeRequests.Set(testSessionID, cancel)
	defer cancel()

	resultChan, err := agent.Run(ctx, testSessionID, "queued content")
	require.NoError(t, err)
	require.Nil(t, resultChan)

	// verify content was queued
	require.Equal(t, 1, agent.QueuedPrompts(testSessionID))
}

func Test_Agent_Run_CancelledContext(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	_, err := config.Init(tempDir, tempDir, false)
	require.NoError(t, err)

	mockProvider := mocks.NewMockProvider(t)
	mockSessionSvc := mocks.NewMockSessionService(t)
	mockMessageSvc := mocks.NewMockMessageService(t)

	agent := &agent{
		Broker:         pubsub.NewBroker[AgentEvent](),
		provider:       mockProvider,
		sessions:       mockSessionSvc,
		messages:       mockMessageSvc,
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
		agentCfg:       config.Agent{Model: config.SelectedModelTypeLarge},
	}

	testModel := catwalk.Model{
		ID:             "test-model",
		SupportsImages: false,
	}
	mockProvider.EXPECT().Model().Return(testModel).Maybe()

	mockMessageSvc.EXPECT().List(mock.Anything, "test-session").Return([]message.Message{}, nil).Maybe()
	testSession := session.Session{
		ID:               "test-session",
		SummaryMessageID: "",
	}
	mockSessionSvc.EXPECT().Get(mock.Anything, "test-session").Return(testSession, nil).Maybe()

	userMsg := message.Message{
		ID:        "user-msg-1",
		SessionID: "test-session",
		Role:      message.User,
		Parts:     []message.ContentPart{message.TextContent{Text: "test content"}},
	}
	mockMessageSvc.EXPECT().Create(mock.Anything, "test-session", mock.MatchedBy(func(params message.CreateMessageParams) bool {
		return params.Role == message.User
	})).Return(userMsg, nil).Maybe()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	// run with cancelled context
	resultChan, err := agent.Run(ctx, "test-session", "test content")
	require.NoError(t, err)
	require.NotNil(t, resultChan)

	select {
	case result := <-resultChan:
		require.Equal(t, AgentEventTypeError, result.Type)
		require.ErrorIs(t, result.Error, context.Canceled)
	case <-time.After(2 * time.Second):
		require.Fail(t, "Test timed out waiting for error result")
	}
}

func Test_Agent_Summarize_SessionBusy(t *testing.T) {
	t.Parallel()

	mockProvider := mocks.NewMockProvider(t)

	agent := &agent{
		Broker:            pubsub.NewBroker[AgentEvent](),
		activeRequests:    csync.NewMap[string, context.CancelFunc](),
		promptQueue:       csync.NewMap[string, []string](),
		summarizeProvider: mockProvider,
	}

	// make session busy
	testSessionID := "busy-session"
	_, cancel := context.WithCancel(context.Background())
	agent.activeRequests.Set(testSessionID, cancel)
	defer cancel()

	// try to summarize busy session
	err := agent.Summarize(context.Background(), testSessionID)

	// should return ErrSessionBusy
	require.Equal(t, ErrSessionBusy, err)
}

func Test_Agent_Summarize_NoProvider(t *testing.T) {
	t.Parallel()

	agent := &agent{
		Broker:            pubsub.NewBroker[AgentEvent](),
		activeRequests:    csync.NewMap[string, context.CancelFunc](),
		promptQueue:       csync.NewMap[string, []string](),
		summarizeProvider: nil, // no summarize provider
	}

	testSessionID := "test-session"

	// try to summarize without provider
	err := agent.Summarize(context.Background(), testSessionID)
	require.Error(t, err)
	require.Contains(t, err.Error(), "summarize provider not available")
}

func Test_Agent_IsBusy_MultipleRequests(t *testing.T) {
	t.Parallel()

	agent := &agent{
		activeRequests: csync.NewMap[string, context.CancelFunc](),
	}

	require.False(t, agent.IsBusy())

	// add multiple active requests
	cancel1 := func() {}
	cancel2 := func() {}
	agent.activeRequests.Set("session1", cancel1)
	agent.activeRequests.Set("session2", cancel2)

	require.True(t, agent.IsBusy())
	require.True(t, agent.IsSessionBusy("session1"))
	require.True(t, agent.IsSessionBusy("session2"))
	require.False(t, agent.IsSessionBusy("session3"))
}

func Test_Agent_Run_ProviderError(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	_, err := config.Init(tempDir, tempDir, false)
	require.NoError(t, err)

	mockProvider := mocks.NewMockProvider(t)
	mockSessionSvc := mocks.NewMockSessionService(t)
	mockMessageSvc := mocks.NewMockMessageService(t)

	agent := &agent{
		Broker:         pubsub.NewBroker[AgentEvent](),
		provider:       mockProvider,
		providerID:     "test-provider",
		sessions:       mockSessionSvc,
		messages:       mockMessageSvc,
		agentCfg:       config.Agent{Model: config.SelectedModelTypeLarge, ID: "test-agent"},
		activeRequests: csync.NewMap[string, context.CancelFunc](),
		promptQueue:    csync.NewMap[string, []string](),
		tools:          csync.NewLazySlice(func() []tools.BaseTool { return []tools.BaseTool{} }),
	}

	testSessionID := "test-session"
	testContent := "test content"

	testModel := catwalk.Model{
		ID:             "test-model",
		SupportsImages: true,
	}
	mockProvider.EXPECT().Model().Return(testModel).Maybe()

	testSession := session.Session{
		ID:               testSessionID,
		SummaryMessageID: "",
	}
	mockSessionSvc.EXPECT().Get(mock.Anything, testSessionID).Return(testSession, nil).Maybe()

	mockMessageSvc.EXPECT().List(mock.Anything, testSessionID).Return([]message.Message{}, nil)

	userMsg := message.Message{
		ID:        "user-msg-1",
		SessionID: testSessionID,
		Role:      message.User,
		Parts:     []message.ContentPart{message.TextContent{Text: testContent}},
	}
	mockMessageSvc.EXPECT().Create(mock.Anything, testSessionID, mock.MatchedBy(func(params message.CreateMessageParams) bool {
		return params.Role == message.User
	})).Return(userMsg, nil)

	assistantMsg := message.Message{
		ID:        "assistant-msg-1",
		SessionID: testSessionID,
		Role:      message.Assistant,
		Parts:     []message.ContentPart{},
	}
	mockMessageSvc.EXPECT().Create(mock.Anything, testSessionID, mock.MatchedBy(func(params message.CreateMessageParams) bool {
		return params.Role == message.Assistant
	})).Return(assistantMsg, nil)

	mockMessageSvc.EXPECT().Update(mock.Anything, mock.AnythingOfType("message.Message")).Return(nil).Maybe()
	mockSessionSvc.EXPECT().Save(mock.Anything, mock.AnythingOfType("session.Session")).Return(session.Session{}, nil).Maybe()

	// create event channel with an error
	eventChan := make(chan provider.ProviderEvent, 1)
	eventChan <- provider.ProviderEvent{
		Type:  provider.EventError,
		Error: fmt.Errorf("provider API error"),
	}
	close(eventChan)

	mockProvider.EXPECT().StreamResponse(mock.Anything, mock.AnythingOfType("[]message.Message"), mock.AnythingOfType("[]tools.BaseTool")).Return(eventChan).Maybe()

	ctx := context.Background()
	resultChan, err := agent.Run(ctx, testSessionID, testContent)
	require.NoError(t, err)
	require.NotNil(t, resultChan)

	timeout := time.After(3 * time.Second)

	select {
	case result, ok := <-resultChan:
		require.True(t, ok, "Channel should not be closed without result")
		require.Equal(t, AgentEventTypeError, result.Type)
		require.NotNil(t, result.Error)
		require.Contains(t, result.Error.Error(), "provider API error")
	case <-timeout:
		require.FailNow(t, "Test timed out waiting for error result")
	}

	// channel should be closed after result
	select {
	case _, ok := <-resultChan:
		require.False(t, ok, "Channel should be closed after result")
	case <-time.After(1 * time.Second):
		require.FailNow(t, "Channel was not closed after result")
	}
}
