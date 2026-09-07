package dialog

import (
	"cmp"

	"charm.land/bubbles/v2/help"
	"charm.land/bubbles/v2/key"
	tea "charm.land/bubbletea/v2"
	"charm.land/catwalk/pkg/catwalk"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/ui/common"
	uv "github.com/charmbracelet/ultraviolet"
)

// AuthMethodID is the identifier for the auth method selection dialog.
const AuthMethodID = "auth_method"

const defaultAuthMethodDialogMaxWidth = 64

// AuthMethod asks how to authenticate a provider that supports both
// OAuth and an API key, before starting either flow.
type AuthMethod struct {
	com          *common.Common
	isOnboarding bool
	provider     catwalk.Provider
	model        config.SelectedModel
	modelType    config.SelectedModelType

	selected int
	help     help.Model
	keyMap   struct {
		UpDown key.Binding
		Select key.Binding
		Close  key.Binding
	}
}

var _ Dialog = (*AuthMethod)(nil)

// NewAuthMethod creates a dialog that chooses between OAuth and API key
// authentication for the given provider.
func NewAuthMethod(
	com *common.Common,
	isOnboarding bool,
	provider catwalk.Provider,
	model config.SelectedModel,
	modelType config.SelectedModelType,
) *AuthMethod {
	m := &AuthMethod{
		com:          com,
		isOnboarding: isOnboarding,
		provider:     provider,
		model:        model,
		modelType:    modelType,
	}

	m.help = help.New()
	m.help.Styles = com.Styles.DialogHelpStyles()

	m.keyMap.UpDown = key.NewBinding(
		key.WithKeys("up", "down", "tab", "shift+tab"),
		key.WithHelp("↑/↓", "choose"),
	)
	m.keyMap.Select = key.NewBinding(
		key.WithKeys("enter", "ctrl+y"),
		key.WithHelp("enter", "confirm"),
	)
	m.keyMap.Close = CloseKey

	return m
}

// ID implements Dialog.
func (m *AuthMethod) ID() string {
	return AuthMethodID
}

// HandleMsg implements Dialog.
func (m *AuthMethod) HandleMsg(msg tea.Msg) Action {
	keyMsg, ok := msg.(tea.KeyPressMsg)
	if !ok {
		return nil
	}

	switch {
	case key.Matches(keyMsg, m.keyMap.Close):
		return ActionClose{}
	case key.Matches(keyMsg, m.keyMap.UpDown):
		m.selected = 1 - m.selected
		return nil
	case key.Matches(keyMsg, m.keyMap.Select):
		return ActionSelectAuthMethod{
			Provider:  m.provider,
			Model:     m.model,
			ModelType: m.modelType,
			UseOAuth:  m.selected == 0,
		}
	}
	return nil
}

// Draw implements Dialog.
func (m *AuthMethod) Draw(scr uv.Screen, area uv.Rectangle) *tea.Cursor {
	t := m.com.Styles
	width := max(0, min(defaultAuthMethodDialogMaxWidth, area.Dx()-t.Dialog.View.GetHorizontalBorderSize()))
	innerWidth := width - t.Dialog.View.GetHorizontalFrameSize()

	rc := NewRenderContext(t, width)
	rc.Title = "Authenticate with " + cmp.Or(m.provider.Name, string(m.provider.ID))

	oauthLabel := "Sign in with ChatGPT"
	apiKeyLabel := "Enter an OpenAI API key"
	options := []string{oauthLabel, apiKeyLabel}

	rows := make([]string, 0, len(options))
	for i, label := range options {
		style := t.Dialog.NormalItem
		prefix := "  "
		if i == m.selected {
			style = t.Dialog.SelectedItem
			prefix = "> "
		}
		rows = append(rows, style.Width(innerWidth).Render(prefix+label))
	}
	rc.AddPart(lipgloss.JoinVertical(lipgloss.Left, rows...))

	rc.Help = renderDialogHelp(t, &m.help, m, innerWidth)

	view := rc.Render()
	if m.isOnboarding {
		rc.Title = ""
		rc.IsOnboarding = true
		view = rc.Render()
		DrawOnboardingCursor(scr, area, view, nil)
	} else {
		DrawCenter(scr, area, view)
	}
	return nil
}

// FullHelp implements help.KeyMap.
func (m *AuthMethod) FullHelp() [][]key.Binding {
	return [][]key.Binding{m.ShortHelp()}
}

// ShortHelp implements help.KeyMap.
func (m *AuthMethod) ShortHelp() []key.Binding {
	return []key.Binding{
		m.keyMap.UpDown,
		m.keyMap.Select,
		m.keyMap.Close,
	}
}
