package tools

import (
	"context"
	_ "embed"

	"charm.land/fantasy"
	"github.com/charmbracelet/crush/internal/question"
)

const QuestionToolName = "question"

//go:embed question.md.tpl
var questionDescription string

// QuestionParams defines the parameters the model passes when calling this tool.
type QuestionParams struct {
	Question string   `json:"question" description:"The clarifying question to ask the user"`
	Options  []string `json:"options,omitempty" description:"Up to 4 suggested answers; the user may also type a custom answer"`
}

func NewQuestionTool(svc question.Service) fantasy.AgentTool {
	return fantasy.NewAgentTool(
		QuestionToolName,
		questionDescription,
		func(ctx context.Context, params QuestionParams, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if params.Question == "" {
				return fantasy.NewTextErrorResponse("question parameter is required"), nil
			}
			answer, err := svc.Ask(ctx, params.Question, params.Options)
			if err != nil {
				return fantasy.NewTextErrorResponse("question cancelled: " + err.Error()), nil
			}
			return fantasy.NewTextResponse(answer), nil
		},
	)
}
