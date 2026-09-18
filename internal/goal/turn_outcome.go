package goal

import (
	"slices"

	"charm.land/fantasy"
)

// turnOutcome classifies how an agent turn ended from the goal's point of
// view. It is the single rule deciding whether a goal may keep driving its
// session.
type turnOutcome int

const (
	// turnQueued means the prompt was queued behind other work and never
	// executed. The goal neither advances nor stops.
	turnQueued turnOutcome = iota
	// turnAnswered means the agent produced a normal answer. This is the
	// only finish that may start another goal turn.
	turnAnswered
	// turnStopped means the turn failed or the agent stopped on purpose.
	// The goal must pause.
	turnStopped
)

// classifyTurn derives the outcome of a turn from the agent result and error
// returned for it.
func classifyTurn(result *fantasy.AgentResult, err error) turnOutcome {
	if err != nil {
		return turnStopped
	}
	if result == nil {
		return turnQueued
	}
	response := result.Response
	if len(result.Steps) > 0 {
		response = result.Steps[len(result.Steps)-1].Response
	}
	if response.FinishReason != fantasy.FinishReasonStop {
		return turnStopped
	}
	haltedByTool := slices.ContainsFunc(response.Content.ToolResults(), func(tr fantasy.ToolResultContent) bool {
		return tr.StopTurn
	})
	if haltedByTool {
		return turnStopped
	}
	return turnAnswered
}

// goalReaction is what an active goal does in response to a turn outcome.
type goalReaction int

const (
	// keepWaiting leaves the goal as it is: the turn never ran, and whoever
	// runs the queued work re-checks the goal when it finishes.
	keepWaiting goalReaction = iota
	// continueGoal lets the goal drive another continuation turn.
	continueGoal
	// pauseGoal pauses the goal until the user resumes it.
	pauseGoal
)

// reaction is the single policy mapping a turn outcome to the goal's
// response. Whether the goal is still active is checked where the reaction
// is applied.
func (o turnOutcome) reaction() goalReaction {
	switch o {
	case turnAnswered:
		return continueGoal
	case turnStopped:
		return pauseGoal
	default:
		return keepWaiting
	}
}
