package model

import (
	"context"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/crush/internal/goal"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/charmbracelet/crush/internal/session"
	"github.com/charmbracelet/crush/internal/ui/util"
)

type (
	loadedGoalMsg         struct{ goal *goal.Goal }
	goalSessionCreatedMsg struct {
		session   session.Session
		objective string
		err       error
	}
)

type goalCreatedMsg struct {
	goal  *goal.Goal
	input string
	err   error
}

func (m *UI) needsGoalObjective() bool {
	return m.currentGoal == nil || m.currentGoal.Status == goal.GoalComplete
}

func (m *UI) submitGoal(input string) tea.Cmd {
	if m.modeSwitching || m.isAgentBusy() {
		return util.ReportWarn("Agent is busy, please wait before setting a goal...")
	}
	if len(m.attachments.List()) > 0 {
		return util.ReportWarn("Enter a text goal without attachments. You can send attachments after the goal starts.")
	}
	if strings.TrimSpace(input) == "" {
		return util.ReportWarn("Please provide an objective for the goal.")
	}
	if err := m.com.Workspace.AgentReadyErr(); err != nil {
		return util.ReportError(err)
	}
	m.modeSwitching = true
	if m.hasSession() {
		return m.createGoal(m.session.ID, input)
	}
	return func() tea.Msg {
		s, err := m.com.Workspace.CreateSession(context.Background(), "New Session")
		return goalSessionCreatedMsg{session: s, objective: input, err: err}
	}
}

func (m *UI) createGoal(sessionID, input string) tea.Cmd {
	return func() tea.Msg {
		g, err := m.com.Workspace.GoalSet(context.Background(), sessionID, strings.TrimSpace(input))
		return goalCreatedMsg{goal: g, input: input, err: err}
	}
}

// applyLoadedGoal restores active goals only after selecting their session.
func (m *UI) applyLoadedGoal(msg loadedGoalMsg) []tea.Cmd {
	if !m.hasSession() || msg.goal.SessionID != m.session.ID {
		return nil
	}
	m.adoptGoalSnapshot(msg.goal)
	m.randomizePlaceholders()
	if m.currentGoal.Status != goal.GoalActive {
		return nil
	}
	return []tea.Cmd{m.setInputModeWithGoalResume(uiInputModeGoal, true, false), goalTimerTickCmd()}
}

func (m *UI) applyGoalCreated(msg goalCreatedMsg) []tea.Cmd {
	m.modeSwitching = false
	if msg.err != nil {
		return []tea.Cmd{util.ReportError(msg.err)}
	}
	if !m.hasSession() || msg.goal.SessionID != m.session.ID {
		return nil
	}
	m.adoptGoalSnapshot(msg.goal)
	m.historyReset()
	var cmds []tea.Cmd
	if m.textarea.Value() == msg.input {
		prevHeight := m.textarea.Height()
		m.textarea.Reset()
		if cmd := m.handleTextareaHeightChange(prevHeight); cmd != nil {
			cmds = append(cmds, cmd)
		}
	}
	m.randomizePlaceholders()
	m.invalidateBusyCaches()
	return append(cmds, goalTimerTickCmd())
}

func (m *UI) pauseGoal() tea.Cmd { return m.changeGoal(false) }
func (m *UI) clearGoal() tea.Cmd { return m.changeGoal(true) }

func (m *UI) changeGoal(clear bool) tea.Cmd {
	if !m.hasSession() {
		return util.ReportWarn("No goal for this session.")
	}
	if m.modeSwitching {
		return util.ReportWarn("Please wait for the input mode to finish switching...")
	}
	sessionID := m.session.ID
	return func() tea.Msg {
		var g *goal.Goal
		var err error
		eventType := pubsub.UpdatedEvent
		if clear {
			g, err = m.com.Workspace.GoalClear(context.Background(), sessionID)
			eventType = pubsub.DeletedEvent
		} else {
			g, err = m.com.Workspace.GoalPause(context.Background(), sessionID)
		}
		if err != nil {
			return util.ReportError(err)()
		}
		if g == nil {
			return util.NewInfoMsg("No goal for this session.")
		}
		return pubsub.Event[goal.Goal]{Type: eventType, Payload: *g}
	}
}

func (m *UI) resumeGoal() tea.Cmd {
	if m.currentGoal == nil || m.currentGoal.Status != goal.GoalPaused {
		return util.ReportWarn("No paused goal to resume.")
	}
	return m.switchInputMode(uiInputModeGoal, true)
}

// adoptGoalSnapshot never overwrites an event already received for this goal.
// A fast failure can publish its pause before GoalSet returns its active snapshot.
func (m *UI) adoptGoalSnapshot(g *goal.Goal) {
	if m.currentGoal == nil || m.currentGoal.GoalID != g.GoalID {
		m.currentGoal = g
	}
}
