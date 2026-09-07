package dialog

import (
	"testing"

	"charm.land/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/oauth"
	"github.com/charmbracelet/crush/internal/ui/common"
	"github.com/charmbracelet/crush/internal/ui/styles"
	"github.com/stretchr/testify/require"
)

func TestOpenAIAPSection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		providerConfig   bool
		hasKey, hasOAuth bool
		wantTitle        string
		wantConfigured   bool
	}{
		{
			name:           "no credentials: single plain section",
			providerConfig: false,
			wantTitle:      "OpenAI",
			wantConfigured: false,
		},
		{
			name:           "api key only: split, api side configured",
			providerConfig: true,
			hasKey:         true,
			wantTitle:      "OpenAI (API)",
			wantConfigured: true,
		},
		{
			name:           "oauth only: split, api side not configured",
			providerConfig: true,
			hasOAuth:       true,
			wantTitle:      "OpenAI (API)",
			wantConfigured: false,
		},
		{
			name:           "both: split, api side configured",
			providerConfig: true,
			hasKey:         true,
			hasOAuth:       true,
			wantTitle:      "OpenAI (API)",
			wantConfigured: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			title, configured := openAIAPSection("OpenAI", tt.providerConfig, tt.hasKey, tt.hasOAuth)
			require.Equal(t, tt.wantTitle, title)
			require.Equal(t, tt.wantConfigured, configured)
		})
	}
}

func newTestModels() *Models {
	s := styles.CharmtonePantera()
	return &Models{
		com:       &common.Common{Styles: &s},
		modelType: ModelTypeLarge,
	}
}

func TestAppendOpenAIOAuthGroup(t *testing.T) {
	t.Parallel()

	provider := catwalk.Provider{ID: catwalk.InferenceProviderOpenAI, Name: "OpenAI"}

	t.Run("signed in lists the subscription catalog", func(t *testing.T) {
		t.Parallel()

		m := newTestModels()
		pc := config.ProviderConfig{
			OAuthToken: &oauth.Token{AccessToken: "at"},
			ChatGPTModels: []catwalk.Model{
				{ID: "gpt-5.6-luna", Name: "GPT-5.6-Luna"},
				{ID: "gpt-5.5", Name: "GPT-5.5"},
			},
		}

		var groups []ModelGroup
		items := map[string]*ModelItem{}
		selected := ""
		m.appendOpenAIOAuthGroup(provider, pc, config.SelectedModel{Provider: "openai", Model: "gpt-5.5"}, items, &selected, &groups)

		require.Len(t, groups, 1)
		require.Equal(t, "OpenAI (OAuth)", groups[0].Title)
		require.True(t, groups[0].configured, "signed in shows the configured badge")
		require.Len(t, groups[0].Items, 2)
		require.Contains(t, items, "openai:gpt-5.6-luna")
		require.Equal(t, "openai:gpt-5.5", selected, "the current model stays selected")
	})

	t.Run("signed out shows the sign-in placeholder", func(t *testing.T) {
		t.Parallel()

		m := newTestModels()
		var groups []ModelGroup
		m.appendOpenAIOAuthGroup(provider, config.ProviderConfig{}, config.SelectedModel{}, map[string]*ModelItem{}, new(string), &groups)

		require.Len(t, groups, 1)
		require.Equal(t, "OpenAI (OAuth)", groups[0].Title)
		require.False(t, groups[0].configured)
		require.Len(t, groups[0].Items, 1)
		require.Equal(t, signInChatGPTLabel, groups[0].Items[0].model.Name)
	})

	t.Run("signed in without a catalog is skipped", func(t *testing.T) {
		t.Parallel()

		m := newTestModels()
		pc := config.ProviderConfig{OAuthToken: &oauth.Token{AccessToken: "at"}}
		var groups []ModelGroup
		m.appendOpenAIOAuthGroup(provider, pc, config.SelectedModel{}, map[string]*ModelItem{}, new(string), &groups)

		require.Empty(t, groups, "an empty section would only confuse")
	})
}
