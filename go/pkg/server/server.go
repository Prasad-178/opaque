// Package server provides the REST API server for privacy-preserving vector search.
//
// Architecture (simplified for local deployment):
//   - Client authenticates and gets credentials (centroids, AES key)
//   - Client does HE scoring LOCALLY (no query leaves client!)
//   - Client requests blobs from selected clusters + decoys
//   - Server returns encrypted blobs (doesn't know which are real vs decoy)
//   - Client decrypts and scores locally
//
// This maintains privacy because:
//   - Query NEVER leaves the client - all HE ops are local
//   - Cluster selection is based on locally decrypted HE scores
//   - Blob access is hidden via decoy requests
//   - Final scoring is client-side with AES-decrypted vectors
//
// Note: This is simpler than having the server do HE operations.
// For a full production deployment, you might want server-side HE
// to avoid sending centroids to client, but that requires more complex
// key management (Galois keys transfer, etc.)
package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
)

// Server handles REST API requests for private vector search.
type Server struct {
	// Storage for encrypted vectors
	blobStore blob.Store

	// Authentication service
	authService *auth.Service

	// Enterprise config store
	enterpriseStore enterprise.Store

	// HTTP server
	httpServer *http.Server
	mux        *http.ServeMux
}

// Config holds server configuration.
type Config struct {
	// Address to listen on (e.g., ":8080")
	Address string

	// Read/write timeouts
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

// DefaultConfig returns sensible defaults for local development.
func DefaultConfig() Config {
	return Config{
		Address:      ":8080",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second, // Longer for HE operations
	}
}

// New creates a new server instance.
func New(cfg Config, blobStore blob.Store, authService *auth.Service, enterpriseStore enterprise.Store) *Server {
	s := &Server{
		blobStore:       blobStore,
		authService:     authService,
		enterpriseStore: enterpriseStore,
		mux:             http.NewServeMux(),
	}

	// Register routes
	s.registerRoutes()

	// Create HTTP server
	s.httpServer = &http.Server{
		Addr:         cfg.Address,
		Handler:      s.mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	return s
}

// registerRoutes sets up all API endpoints.
func (s *Server) registerRoutes() {
	// Health check
	s.mux.HandleFunc("GET /health", s.handleHealth)

	// Authentication
	s.mux.HandleFunc("POST /api/v1/auth/login", s.handleLogin)
	s.mux.HandleFunc("POST /api/v1/auth/refresh", s.handleRefresh)

	// Blob access (HE scoring is done client-side)
	s.mux.HandleFunc("GET /api/v1/buckets/{enterpriseID}/{superID}", s.withAuth(s.handleGetBuckets))
}

// Start starts the HTTP server.
func (s *Server) Start() error {
	log.Printf("Server starting on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

// withAuth wraps a handler with authentication.
func (s *Server) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Get token from Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			writeError(w, http.StatusUnauthorized, "missing authorization header")
			return
		}

		// Expect "Bearer <token>"
		parts := strings.SplitN(authHeader, " ", 2)
		if len(parts) != 2 || parts[0] != "Bearer" {
			writeError(w, http.StatusUnauthorized, "invalid authorization header format")
			return
		}
		token := parts[1]

		// Validate token
		creds, err := s.authService.ValidateToken(r.Context(), token)
		if err != nil {
			writeError(w, http.StatusUnauthorized, "invalid or expired token")
			return
		}

		// Add credentials to context
		ctx := context.WithValue(r.Context(), credentialsKey, creds)
		next(w, r.WithContext(ctx))
	}
}

// Context key for credentials
type contextKey string

const credentialsKey contextKey = "credentials"

// getToken retrieves the validated auth token from context.
func getToken(ctx context.Context) *auth.Token {
	tok, _ := ctx.Value(credentialsKey).(*auth.Token)
	return tok
}

// Health check handler
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "healthy",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// Login request/response
type LoginRequest struct {
	UserID       string `json:"user_id"`
	EnterpriseID string `json:"enterprise_id"`
	Password     string `json:"password"`
}

type LoginResponse struct {
	Token      string      `json:"token"`
	ExpiresAt  time.Time   `json:"expires_at"`
	AESKey     string      `json:"aes_key"`      // Base64
	Centroids  [][]float64 `json:"centroids"`
	Dimension  int         `json:"dimension"`
	NumClusters int        `json:"num_clusters"`
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	creds, err := s.authService.Authenticate(r.Context(), req.UserID, []byte(req.Password))
	if err != nil {
		writeError(w, http.StatusUnauthorized, "authentication failed")
		return
	}

	writeJSON(w, http.StatusOK, LoginResponse{
		Token:       creds.Token,
		ExpiresAt:   creds.TokenExpiry,
		AESKey:      base64.StdEncoding.EncodeToString(creds.AESKey),
		Centroids:   creds.Centroids,
		Dimension:   creds.Dimension,
		NumClusters: creds.NumSuperBuckets,
	})
}

// Refresh token handler
type RefreshRequest struct {
	Token string `json:"token"`
}

type RefreshResponse struct {
	Token     string    `json:"token"`
	ExpiresAt time.Time `json:"expires_at"`
}

func (s *Server) handleRefresh(w http.ResponseWriter, r *http.Request) {
	var req RefreshRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	creds, err := s.authService.RefreshToken(r.Context(), req.Token)
	if err != nil {
		writeError(w, http.StatusUnauthorized, "token refresh failed")
		return
	}

	writeJSON(w, http.StatusOK, RefreshResponse{
		Token:     creds.Token,
		ExpiresAt: creds.TokenExpiry,
	})
}

// Get buckets handler
type GetBucketsResponse struct {
	Blobs []BlobData `json:"blobs"`
}

type BlobData struct {
	ID         string `json:"id"`
	Ciphertext string `json:"ciphertext"` // Base64
	Dimension  int    `json:"dimension"`
}

func (s *Server) handleGetBuckets(w http.ResponseWriter, r *http.Request) {
	enterpriseID := r.PathValue("enterpriseID")
	superIDStr := r.PathValue("superID")

	superID, err := strconv.Atoi(superIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid super bucket ID")
		return
	}

	// Verify credentials match enterprise
	tok := getToken(r.Context())
	if tok == nil || tok.EnterpriseID != enterpriseID {
		writeError(w, http.StatusForbidden, "enterprise mismatch")
		return
	}

	// Get blobs from store
	blobs, err := s.blobStore.GetSuperBuckets(r.Context(), []int{superID})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to fetch blobs")
		return
	}

	// Convert to response format
	blobData := make([]BlobData, len(blobs))
	for i, b := range blobs {
		blobData[i] = BlobData{
			ID:         b.ID,
			Ciphertext: base64.StdEncoding.EncodeToString(b.Ciphertext),
			Dimension:  b.Dimension,
		}
	}

	writeJSON(w, http.StatusOK, GetBucketsResponse{
		Blobs: blobData,
	})
}

// Helper functions
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
