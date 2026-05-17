package tools

import (
	"context"
	"fmt"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/goal"
)

const GetGoalToolName = "get_goal"

func NewGetGoalTool(goalService goal.Service) fantasy.AgentTool {
	return fantasy.NewAgentTool(GetGoalToolName, "Get the current active goal for the session.", func(ctx context.Context, _ any, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
		sessionID, ok := ctx.Value(SessionIDContextKey).(string)
		if !ok {
			return fantasy.ToolResponse{}, fmt.Errorf("session id not found in context")
		}

		g, err := goalService.Get(ctx, sessionID)
		if err != nil {
			return fantasy.ToolResponse{}, err
		}

		if g == nil {
			return fantasy.NewTextResponse("No active goal for this session."), nil
		}

		return fantasy.NewTextResponse(fmt.Sprintf("Objective: %s\nStatus: %s\nGoalID: %s", g.Objective, g.Status, g.GoalID)), nil
	})
}

const UpdateGoalToolName = "update_goal"

type UpdateGoalInput struct {
	Status goal.GoalStatus `json:"status"`
}

func NewUpdateGoalTool(goalService goal.Service) fantasy.AgentTool {
	return fantasy.NewAgentTool(UpdateGoalToolName, "Update the status of the current goal. Currently only supports setting status to 'complete'.", func(ctx context.Context, input UpdateGoalInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
		sessionID, ok := ctx.Value(SessionIDContextKey).(string)
		if !ok {
			return fantasy.ToolResponse{}, fmt.Errorf("session id not found in context")
		}

		if input.Status != goal.GoalComplete {
			return fantasy.NewTextErrorResponse("update_goal only supports status='complete'"), nil
		}

		g, err := goalService.Get(ctx, sessionID)
		if err != nil {
			return fantasy.ToolResponse{}, err
		}
		if g == nil {
			return fantasy.NewTextErrorResponse("No active goal found to update."), nil
		}

		// Stale update protection: verify goal ID from context if present.
		if expectedGoalID, ok := ctx.Value("goal_id").(string); ok {
			if g.GoalID != expectedGoalID {
				return fantasy.NewTextErrorResponse("Goal ID mismatch: you are trying to update a goal that has been replaced."), nil
			}
		}

		updated, err := goalService.UpdateStatus(ctx, sessionID, g.GoalID, input.Status)
		if err != nil {
			return fantasy.ToolResponse{}, err
		}

		return fantasy.NewTextResponse(fmt.Sprintf("Goal '%s' marked as complete.", updated.Objective)), nil
	})
}
