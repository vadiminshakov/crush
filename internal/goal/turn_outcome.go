package goal

import (
	"context"
	"errors"
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
	// turnFailed means the turn failed or the agent stopped on purpose.
	// The goal must pause.
	turnFailed
	// turnCancelled means someone cancelled the turn and, with it, took
	// over the goal's status.
	turnCancelled
)

// classifyTurn derives the outcome of a turn from its context and the agent
// result and error returned for it.
func classifyTurn(ctx context.Context, result *fantasy.AgentResult, err error) turnOutcome {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) {
		return turnCancelled
	}
	if err != nil {
		return turnFailed
	}
	if result == nil {
		return turnQueued
	}
	response := result.Response
	if len(result.Steps) > 0 {
		response = result.Steps[len(result.Steps)-1].Response
	}
	if response.FinishReason != fantasy.FinishReasonStop {
		return turnFailed
	}
	haltedByTool := slices.ContainsFunc(response.Content.ToolResults(), func(tr fantasy.ToolResultContent) bool {
		return tr.StopTurn
	})
	if haltedByTool {
		return turnFailed
	}
	return turnAnswered
}

// hasToolCalls reports whether the turn called any tool.
func hasToolCalls(result *fantasy.AgentResult) bool {
	if result == nil {
		return false
	}
	if len(result.Response.Content.ToolCalls()) > 0 {
		return true
	}
	return slices.ContainsFunc(result.Steps, func(step fantasy.StepResult) bool {
		return len(step.Content.ToolCalls()) > 0
	})
}
