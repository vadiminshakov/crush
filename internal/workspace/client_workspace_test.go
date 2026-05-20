package workspace

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/charmbracelet/crush/internal/client"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/crush/internal/permission"
	"github.com/charmbracelet/crush/internal/proto"
	"github.com/stretchr/testify/require"
)

// TestProtoToMessageToolResult ensures that ToolResult metadata,
// data, and MIME type survive the conversion from proto on the
// client. Without these fields the TUI cannot render rich tool
// output (e.g. syntax-highlighted code from view, diffs from edit,
// images, etc.) and falls back to the raw LLM-facing string.
func TestProtoToMessageToolResult(t *testing.T) {
	t.Parallel()

	src := proto.Message{
		ID:   "m1",
		Role: proto.Tool,
		Parts: []proto.ContentPart{
			proto.ToolResult{
				ToolCallID: "call-1",
				Name:       "view",
				Content:    "<file>\n  1| hi\n</file>",
				Data:       "base64data",
				MIMEType:   "image/png",
				Metadata:   `{"file_path":"/tmp/x","content":"hi"}`,
				IsError:    false,
			},
		},
	}

	got := protoToMessage(src)
	require.Len(t, got.Parts, 1)
	tr, ok := got.Parts[0].(message.ToolResult)
	require.True(t, ok, "expected message.ToolResult, got %T", got.Parts[0])
	require.Equal(t, "call-1", tr.ToolCallID)
	require.Equal(t, "view", tr.Name)
	require.Equal(t, "<file>\n  1| hi\n</file>", tr.Content)
	require.Equal(t, "base64data", tr.Data)
	require.Equal(t, "image/png", tr.MIMEType)
	require.Equal(t, `{"file_path":"/tmp/x","content":"hi"}`, tr.Metadata)
	require.False(t, tr.IsError)
}

// TestClientWorkspace_PermissionGrantMapping verifies that
// PermissionGrant on the ClientWorkspace serializes a one-time grant
// (proto.PermissionAllow) and PermissionGrantPersistent serializes a
// persistent grant (proto.PermissionAllowForSession). A swap between
// these two would silently flip "allow once" into "remember for the
// session", and vice versa, so we pin the wire mapping here.
func TestClientWorkspace_PermissionGrantMapping(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		call func(*ClientWorkspace, permission.PermissionRequest)
		want proto.PermissionAction
	}{
		{
			name: "Grant -> PermissionAllow",
			call: func(w *ClientWorkspace, p permission.PermissionRequest) {
				w.PermissionGrant(p)
			},
			want: proto.PermissionAllow,
		},
		{
			name: "GrantPersistent -> PermissionAllowForSession",
			call: func(w *ClientWorkspace, p permission.PermissionRequest) {
				w.PermissionGrantPersistent(p)
			},
			want: proto.PermissionAllowForSession,
		},
		{
			name: "Deny -> PermissionDeny",
			call: func(w *ClientWorkspace, p permission.PermissionRequest) {
				w.PermissionDeny(p)
			},
			want: proto.PermissionDeny,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var got proto.PermissionGrant
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.Equal(t, http.MethodPost, r.Method)
				require.Equal(t, "/v1/workspaces/ws-1/permissions/grant", r.URL.Path)
				body, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				require.NoError(t, json.Unmarshal(body, &got))
				require.NoError(t, json.NewEncoder(w).Encode(proto.PermissionGrantResponse{Resolved: true}))
			}))
			defer srv.Close()

			u, err := url.Parse(srv.URL)
			require.NoError(t, err)
			c, err := client.NewClient(t.TempDir(), "tcp", u.Host)
			require.NoError(t, err)

			ws := NewClientWorkspace(c, proto.Workspace{ID: "ws-1"})

			perm := permission.PermissionRequest{
				ID:          "req-1",
				SessionID:   "sess-1",
				ToolCallID:  "tc-1",
				ToolName:    "tool",
				Description: "do thing",
				Action:      "act",
				Path:        "/tmp/p",
			}
			tc.call(ws, perm)

			require.Equal(t, tc.want, got.Action)
			require.Equal(t, "req-1", got.Permission.ID)
			require.Equal(t, "sess-1", got.Permission.SessionID)
			require.Equal(t, "tc-1", got.Permission.ToolCallID)
			require.Equal(t, "tool", got.Permission.ToolName)
			require.Equal(t, "act", got.Permission.Action)
			require.Equal(t, "/tmp/p", got.Permission.Path)
		})
	}
}
