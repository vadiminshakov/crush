package dialog

import (
	"cmp"
	"strings"

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

const (
	defaultAuthMethodDialogMaxWidth = 72
	authMethodCardGap               = 2
	// authMethodCardHeight is the total card height, border included. The
	// odd content height lets the one-line "API Key" label center exactly;
	// the two-line OAuth label lands within half a row of center.
	authMethodCardHeight = 13
	// authMethodMinCardWidth is the smallest card width that keeps the
	// two-line OAuth label legible; below it the cards stack vertically.
	authMethodMinCardWidth = 20
)

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
		Choose key.Binding
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

	m.keyMap.Choose = key.NewBinding(
		key.WithKeys("left", "right", "up", "down", "tab", "shift+tab"),
		key.WithHelp("←/→", "choose"),
	)
	m.keyMap.Select = key.NewBinding(
		key.WithKeys("enter", "ctrl+y"),
		key.WithHelp("enter", "accept"),
	)
	m.keyMap.Close = key.NewBinding(
		key.WithKeys("esc", "alt+esc"),
		key.WithHelp("esc", "back"),
	)

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
	case key.Matches(keyMsg, m.keyMap.Choose):
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
	rc.Title = "Let’s Auth " + cmp.Or(m.provider.Name, string(m.provider.ID))
	rc.Gap = 1

	rc.AddPart(t.Dialog.AuthMethod.Prompt.Width(innerWidth).Render("How would you like to authenticate?"))

	cardWidth := max(0, (innerWidth-authMethodCardGap)/2)
	sideBySide := cardWidth >= authMethodMinCardWidth
	if !sideBySide {
		cardWidth = innerWidth
	}
	oauthCard := m.renderCard("ChatGPT Account\nwith Subscription", m.selected == 0, cardWidth)
	apiKeyCard := m.renderCard("API Key", m.selected == 1, cardWidth)

	var cards string
	if sideBySide {
		cards = lipgloss.JoinHorizontal(lipgloss.Top, oauthCard, strings.Repeat(" ", authMethodCardGap), apiKeyCard)
	} else {
		cards = lipgloss.JoinVertical(lipgloss.Left, oauthCard, "", apiKeyCard)
	}
	rc.AddPart(cards)

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

// renderCard renders one auth option as a bordered card with its label
// centered both ways. The selected card gets the focused frame.
func (m *AuthMethod) renderCard(label string, focused bool, width int) string {
	t := m.com.Styles
	style := t.Dialog.AuthMethod.CardBlurred
	if focused {
		style = t.Dialog.AuthMethod.CardFocused
	}
	return style.
		Width(width).
		Height(authMethodCardHeight).
		Align(lipgloss.Center, lipgloss.Center).
		Render(label)
}

// FullHelp implements help.KeyMap.
func (m *AuthMethod) FullHelp() [][]key.Binding {
	return [][]key.Binding{m.ShortHelp()}
}

// ShortHelp implements help.KeyMap.
func (m *AuthMethod) ShortHelp() []key.Binding {
	return []key.Binding{
		m.keyMap.Choose,
		m.keyMap.Select,
		m.keyMap.Close,
	}
}
