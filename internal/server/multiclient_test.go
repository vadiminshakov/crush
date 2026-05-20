package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/charmbracelet/crush/internal/backend"
	"github.com/charmbracelet/crush/internal/proto"
	"github.com/stretchr/testify/require"
)

// newTestController builds a controllerV1 around a backend without a
// real config store, suitable for handler-level 400 tests.
func newTestController() *controllerV1 {
	s := &Server{}
	s.backend = backend.New(context.Background(), nil, nil)
	return &controllerV1{backend: s.backend, server: s}
}

func TestPostWorkspaces_RejectsMissingClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	body, err := json.Marshal(proto.Workspace{Path: t.TempDir()})
	require.NoError(t, err)
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/workspaces", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	c.handlePostWorkspaces(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	var perr proto.Error
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &perr))
	require.Contains(t, perr.Message, "client_id")
}

func TestPostWorkspaces_RejectsMalformedClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	body, err := json.Marshal(proto.Workspace{Path: t.TempDir(), ClientID: "not-a-uuid"})
	require.NoError(t, err)
	req := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/v1/workspaces", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	c.handlePostWorkspaces(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestDeleteWorkspace_RejectsMissingClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	req := httptest.NewRequestWithContext(t.Context(), http.MethodDelete, "/v1/workspaces/abc", nil)
	req.SetPathValue("id", "abc")
	rec := httptest.NewRecorder()

	c.handleDeleteWorkspaces(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestDeleteWorkspace_RejectsMalformedClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	req := httptest.NewRequestWithContext(t.Context(), http.MethodDelete, "/v1/workspaces/abc?client_id=nope", nil)
	req.SetPathValue("id", "abc")
	rec := httptest.NewRecorder()

	c.handleDeleteWorkspaces(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestSubscribeEvents_RejectsMissingClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/v1/workspaces/abc/events", nil)
	req.SetPathValue("id", "abc")
	rec := httptest.NewRecorder()

	c.handleGetWorkspaceEvents(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestSubscribeEvents_RejectsMalformedClientID(t *testing.T) {
	t.Parallel()
	c := newTestController()

	req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/v1/workspaces/abc/events?client_id=nope", nil)
	req.SetPathValue("id", "abc")
	rec := httptest.NewRecorder()

	c.handleGetWorkspaceEvents(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
}
