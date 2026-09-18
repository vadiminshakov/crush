package goal

import (
	_ "embed"
	"strings"
	"text/template"
)

//go:embed continuation_prompt.md.tpl
var continuationPromptTmpl []byte

var continuationTpl = template.Must(
	template.New("continuation").Parse(string(continuationPromptTmpl)),
)

// renderContinuationPrompt builds the synthetic user prompt that asks the
// agent to keep working toward g.
func renderContinuationPrompt(g *Goal) (string, error) {
	var sb strings.Builder
	if err := continuationTpl.Execute(&sb, g); err != nil {
		return "", err
	}
	return sb.String(), nil
}
