package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
)

const (
	testEnterpriseID = "test-enterprise"
	testUserID       = "user1"
	testPassword     = "password123"
)

// testServer holds a configured Server and auth token for tests.
type testServer struct {
	srv   *Server
	token string
}

func setupTestServer(t *testing.T) *testServer {
	t.Helper()

	ctx := context.Background()

	// Enterprise store + config
	entStore := enterprise.NewMemoryStore()
	entCfg, err := enterprise.NewConfig(testEnterpriseID, 128, 64)
	if err != nil {
		t.Fatalf("failed to create enterprise config: %v", err)
	}
	if err := entStore.Put(ctx, entCfg); err != nil {
		t.Fatalf("failed to store enterprise config: %v", err)
	}

	// Auth service + test user
	authCfg := auth.DefaultServiceConfig()
	authSvc := auth.NewService(authCfg, entStore)
	if err := authSvc.RegisterUser(ctx, testUserID, testEnterpriseID, []byte(testPassword), []string{auth.ScopeSearch}); err != nil {
		t.Fatalf("failed to register user: %v", err)
	}

	// Blob store with test data in super-bucket 5
	blobStore := blob.NewMemoryStore()
	testBlob := blob.NewBlob("blob-1", "5", []byte("encrypted-data"), 128)
	if err := blobStore.Put(ctx, testBlob); err != nil {
		t.Fatalf("failed to store blob: %v", err)
	}

	srv := New(DefaultConfig(), blobStore, authSvc, entStore)

	// Authenticate to get a token
	creds, err := authSvc.Authenticate(ctx, testUserID, []byte(testPassword))
	if err != nil {
		t.Fatalf("failed to authenticate: %v", err)
	}

	return &testServer{srv: srv, token: creds.Token}
}

func (ts *testServer) doRequest(t *testing.T, method, path string, body any, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()

	var reqBody *bytes.Buffer
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("failed to marshal body: %v", err)
		}
		reqBody = bytes.NewBuffer(b)
	} else {
		reqBody = &bytes.Buffer{}
	}

	req := httptest.NewRequest(method, path, reqBody)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	rr := httptest.NewRecorder()
	ts.srv.mux.ServeHTTP(rr, req)
	return rr
}

func TestHandleHealth(t *testing.T) {
	ts := setupTestServer(t)
	rr := ts.doRequest(t, "GET", "/health", nil, nil)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}

	var resp map[string]string
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp["status"] != "healthy" {
		t.Errorf("expected status=healthy, got %q", resp["status"])
	}
}

func TestHandleLogin_Success(t *testing.T) {
	ts := setupTestServer(t)
	rr := ts.doRequest(t, "POST", "/api/v1/auth/login", LoginRequest{
		UserID:       testUserID,
		EnterpriseID: testEnterpriseID,
		Password:     testPassword,
	}, nil)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp LoginResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Token == "" {
		t.Error("expected non-empty token")
	}
	if resp.Dimension != 128 {
		t.Errorf("expected dimension=128, got %d", resp.Dimension)
	}
}

func TestHandleLogin_InvalidCredentials(t *testing.T) {
	ts := setupTestServer(t)
	rr := ts.doRequest(t, "POST", "/api/v1/auth/login", LoginRequest{
		UserID:       testUserID,
		EnterpriseID: testEnterpriseID,
		Password:     "wrong-password",
	}, nil)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestHandleLogin_BadBody(t *testing.T) {
	ts := setupTestServer(t)
	req := httptest.NewRequest("POST", "/api/v1/auth/login", bytes.NewBufferString("not-json"))
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	ts.srv.mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestHandleRefresh_Success(t *testing.T) {
	// Use a short-TTL auth service so the token is immediately refreshable.
	ctx := context.Background()

	entStore := enterprise.NewMemoryStore()
	entCfg, _ := enterprise.NewConfig(testEnterpriseID, 128, 64)
	entStore.Put(ctx, entCfg)

	authCfg := auth.DefaultServiceConfig()
	authCfg.TokenTTL = 10 * time.Minute      // Short TTL
	authCfg.RefreshWindow = 15 * time.Minute  // Window > TTL means always refreshable
	authSvc := auth.NewService(authCfg, entStore)
	authSvc.RegisterUser(ctx, testUserID, testEnterpriseID, []byte(testPassword), []string{auth.ScopeSearch})

	blobStore := blob.NewMemoryStore()
	srv := New(DefaultConfig(), blobStore, authSvc, entStore)

	creds, _ := authSvc.Authenticate(ctx, testUserID, []byte(testPassword))

	ts := &testServer{srv: srv, token: creds.Token}
	rr := ts.doRequest(t, "POST", "/api/v1/auth/refresh", RefreshRequest{
		Token: ts.token,
	}, nil)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp RefreshResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Token == "" {
		t.Error("expected non-empty token")
	}
}

func TestHandleRefresh_InvalidToken(t *testing.T) {
	ts := setupTestServer(t)
	rr := ts.doRequest(t, "POST", "/api/v1/auth/refresh", RefreshRequest{
		Token: "invalid-token",
	}, nil)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestHandleGetBuckets_Unauthorized(t *testing.T) {
	ts := setupTestServer(t)

	// No auth header
	rr := ts.doRequest(t, "GET", "/api/v1/buckets/"+testEnterpriseID+"/5", nil, nil)
	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestHandleGetBuckets_InvalidAuthFormat(t *testing.T) {
	ts := setupTestServer(t)

	tests := []struct {
		name   string
		header string
	}{
		{"no Bearer prefix", "Token abc123"},
		{"empty bearer", "Bearer"},
		{"basic auth", "Basic dXNlcjpwYXNz"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := ts.doRequest(t, "GET", "/api/v1/buckets/"+testEnterpriseID+"/5", nil, map[string]string{
				"Authorization": tt.header,
			})
			if rr.Code != http.StatusUnauthorized {
				t.Errorf("expected 401, got %d", rr.Code)
			}
		})
	}
}

func TestHandleGetBuckets_EnterpriseMismatch(t *testing.T) {
	ts := setupTestServer(t)

	rr := ts.doRequest(t, "GET", "/api/v1/buckets/wrong-enterprise/5", nil, map[string]string{
		"Authorization": "Bearer " + ts.token,
	})
	if rr.Code != http.StatusForbidden {
		t.Errorf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleGetBuckets_Success(t *testing.T) {
	ts := setupTestServer(t)

	rr := ts.doRequest(t, "GET", "/api/v1/buckets/"+testEnterpriseID+"/5", nil, map[string]string{
		"Authorization": "Bearer " + ts.token,
	})
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp GetBucketsResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(resp.Blobs) != 1 {
		t.Errorf("expected 1 blob, got %d", len(resp.Blobs))
	}
	if len(resp.Blobs) > 0 && resp.Blobs[0].ID != "blob-1" {
		t.Errorf("expected blob ID=blob-1, got %q", resp.Blobs[0].ID)
	}
}

func TestHandleGetBuckets_InvalidSuperID(t *testing.T) {
	ts := setupTestServer(t)

	rr := ts.doRequest(t, "GET", "/api/v1/buckets/"+testEnterpriseID+"/notanumber", nil, map[string]string{
		"Authorization": "Bearer " + ts.token,
	})
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestHandleGetBuckets_EmptyBucket(t *testing.T) {
	ts := setupTestServer(t)

	rr := ts.doRequest(t, "GET", "/api/v1/buckets/"+testEnterpriseID+"/999", nil, map[string]string{
		"Authorization": "Bearer " + ts.token,
	})
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp GetBucketsResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if len(resp.Blobs) != 0 {
		t.Errorf("expected 0 blobs, got %d", len(resp.Blobs))
	}
}
