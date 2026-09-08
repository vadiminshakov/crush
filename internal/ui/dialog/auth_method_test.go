package dialog

import (
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/ui/common"
	"github.com/charmbracelet/crush/internal/ui/styles"
	"github.com/stretchr/testify/require"
)

func newTestAuthMethod() *AuthMethod {
	s := styles.CharmtonePantera()
	return NewAuthMethod(
		&common.Common{Styles: &s},
		false,
		catwalk.Provider{ID: catwalk.InferenceProviderOpenAI, Name: "OpenAI"},
		config.SelectedModel{Provider: "openai", Model: "gpt-5.1"},
		config.SelectedModelTypeLarge,
	)
}

func TestAuthMethodDefaultsToOAuth(t *testing.T) {
	t.Parallel()

	m := newTestAuthMethod()
	require.Equal(t, AuthMethodID, m.ID())

	action := m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	selected, ok := action.(ActionSelectAuthMethod)
	require.True(t, ok)
	require.True(t, selected.UseOAuth)
	require.Equal(t, "openai", string(selected.Provider.ID))
	require.Equal(t, "gpt-5.1", selected.Model.Model)
	require.Equal(t, config.SelectedModelTypeLarge, selected.ModelType)
}

func TestAuthMethodTogglesToAPIKey(t *testing.T) {
	t.Parallel()

	m := newTestAuthMethod()
	m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyDown})

	action := m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	selected, ok := action.(ActionSelectAuthMethod)
	require.True(t, ok)
	require.False(t, selected.UseOAuth)

	// Up toggles back to OAuth.
	m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyUp})
	action = m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	selected, ok = action.(ActionSelectAuthMethod)
	require.True(t, ok)
	require.True(t, selected.UseOAuth)

	// Left/right toggle as well, since the cards sit side by side.
	m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyRight})
	action = m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	selected, ok = action.(ActionSelectAuthMethod)
	require.True(t, ok)
	require.False(t, selected.UseOAuth)

	m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyLeft})
	action = m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEnter})
	selected, ok = action.(ActionSelectAuthMethod)
	require.True(t, ok)
	require.True(t, selected.UseOAuth)
}

func TestAuthMethodClose(t *testing.T) {
	t.Parallel()

	m := newTestAuthMethod()
	action := m.HandleMsg(tea.KeyPressMsg{Code: tea.KeyEscape})
	_, ok := action.(ActionClose)
	require.True(t, ok)
}
