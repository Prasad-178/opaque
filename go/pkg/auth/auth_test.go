package auth

import (
	"context"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/enterprise"
)

func setupTestService(t *testing.T) (*Service, *enterprise.Config) {
	// Create enterprise config
	enterpriseCfg, err := enterprise.NewConfig("test-enterprise", 128, 64)
	if err != nil {
		t.Fatalf("failed to create enterprise config: %v", err)
	}

	// Create enterprise store and add config
	store := enterprise.NewMemoryStore()
	if err := store.Put(context.Background(), enterpriseCfg); err != nil {
		t.Fatalf("failed to store enterprise config: %v", err)
	}

	// Create auth service
	cfg := DefaultServiceConfig()
	service := NewService(cfg, store)

	return service, enterpriseCfg
}

func TestRegisterUser(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)

	// Register user
	err := service.RegisterUser(ctx, "user1", "test-enterprise", []byte("password123"), []string{ScopeSearch})
	if err != nil {
		t.Fatalf("failed to register user: %v", err)
	}

	// Try to register same user again
	err = service.RegisterUser(ctx, "user1", "test-enterprise", []byte("password456"), []string{ScopeSearch})
	if err != ErrUserExists {
		t.Errorf("expected ErrUserExists, got %v", err)
	}

	// Try to register for non-existent enterprise
	err = service.RegisterUser(ctx, "user2", "non-existent", []byte("password"), []string{ScopeSearch})
	if err != enterprise.ErrEnterpriseNotFound {
		t.Errorf("expected ErrEnterpriseNotFound, got %v", err)
	}
}

func TestAuthenticate(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)

	// Register user
	password := []byte("password123")
	err := service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch, ScopeIndex})
	if err != nil {
		t.Fatalf("failed to register user: %v", err)
	}

	// Authenticate with correct password
	creds, err := service.Authenticate(ctx, "user1", password)
	if err != nil {
		t.Fatalf("failed to authenticate: %v", err)
	}

	// Verify credentials
	if creds.Token == "" {
		t.Error("token should not be empty")
	}
	if creds.EnterpriseID != "test-enterprise" {
		t.Errorf("enterprise ID mismatch: got %s", creds.EnterpriseID)
	}
	if len(creds.AESKey) != 32 {
		t.Errorf("AES key should be 32 bytes, got %d", len(creds.AESKey))
	}
	if len(creds.LSHHyperplanes) == 0 {
		t.Error("LSH hyperplanes should not be empty")
	}
	if creds.TokenExpiry.Before(time.Now()) {
		t.Error("token should not be expired")
	}

	// Authenticate with wrong password
	_, err = service.Authenticate(ctx, "user1", []byte("wrongpassword"))
	if err != ErrInvalidCredentials {
		t.Errorf("expected ErrInvalidCredentials, got %v", err)
	}

	// Authenticate with non-existent user
	_, err = service.Authenticate(ctx, "non-existent", password)
	if err != ErrUserNotFound {
		t.Errorf("expected ErrUserNotFound, got %v", err)
	}
}

func TestValidateToken(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)

	// Register and authenticate
	password := []byte("password123")
	service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch})
	creds, _ := service.Authenticate(ctx, "user1", password)

	// Validate token
	token, err := service.ValidateToken(ctx, creds.Token)
	if err != nil {
		t.Fatalf("failed to validate token: %v", err)
	}
	if token.UserID != "user1" {
		t.Errorf("user ID mismatch: got %s", token.UserID)
	}
	if !token.HasScope(ScopeSearch) {
		t.Error("token should have search scope")
	}

	// Validate invalid token
	_, err = service.ValidateToken(ctx, "invalid-token")
	if err != ErrInvalidToken {
		t.Errorf("expected ErrInvalidToken, got %v", err)
	}
}

func TestRefreshToken(t *testing.T) {
	ctx := context.Background()

	// Create service with short TTL for testing
	enterpriseCfg, _ := enterprise.NewConfig("test-enterprise", 128, 64)
	store := enterprise.NewMemoryStore()
	store.Put(ctx, enterpriseCfg)

	cfg := ServiceConfig{
		TokenTTL:      2 * time.Second,
		RefreshWindow: 1 * time.Second,
		LSHBits:       8,
		Dimension:     128,
	}
	service := NewService(cfg, store)

	// Register and authenticate
	password := []byte("password123")
	service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch})
	creds, _ := service.Authenticate(ctx, "user1", password)

	// Try to refresh immediately (not in refresh window)
	_, err := service.RefreshToken(ctx, creds.Token)
	if err != ErrTokenNotRefreshable {
		t.Errorf("expected ErrTokenNotRefreshable, got %v", err)
	}

	// Wait until in refresh window
	time.Sleep(1100 * time.Millisecond)

	// Now refresh should work
	newCreds, err := service.RefreshToken(ctx, creds.Token)
	if err != nil {
		t.Fatalf("failed to refresh token: %v", err)
	}
	if newCreds.Token == creds.Token {
		t.Error("new token should be different from old token")
	}

	// Old token should be invalid
	_, err = service.ValidateToken(ctx, creds.Token)
	if err != ErrInvalidToken {
		t.Errorf("old token should be invalid, got %v", err)
	}

	// New token should be valid
	_, err = service.ValidateToken(ctx, newCreds.Token)
	if err != nil {
		t.Errorf("new token should be valid, got %v", err)
	}
}

func TestRevokeToken(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)

	// Register and authenticate
	password := []byte("password123")
	service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch})
	creds, _ := service.Authenticate(ctx, "user1", password)

	// Revoke token
	err := service.RevokeToken(ctx, creds.Token)
	if err != nil {
		t.Fatalf("failed to revoke token: %v", err)
	}

	// Token should be invalid
	_, err = service.ValidateToken(ctx, creds.Token)
	if err != ErrInvalidToken {
		t.Errorf("token should be invalid after revocation, got %v", err)
	}
}

func TestDisableUser(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)

	// Register and authenticate
	password := []byte("password123")
	service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch})
	creds, _ := service.Authenticate(ctx, "user1", password)

	// Disable user
	err := service.DisableUser(ctx, "user1")
	if err != nil {
		t.Fatalf("failed to disable user: %v", err)
	}

	// Token should be revoked
	_, err = service.ValidateToken(ctx, creds.Token)
	if err != ErrInvalidToken {
		t.Errorf("token should be invalid after user disabled, got %v", err)
	}

	// Should not be able to authenticate
	_, err = service.Authenticate(ctx, "user1", password)
	if err != ErrUserDisabled {
		t.Errorf("expected ErrUserDisabled, got %v", err)
	}

	// Re-enable user
	err = service.EnableUser(ctx, "user1")
	if err != nil {
		t.Fatalf("failed to enable user: %v", err)
	}

	// Should be able to authenticate again
	_, err = service.Authenticate(ctx, "user1", password)
	if err != nil {
		t.Errorf("should be able to authenticate after re-enabling, got %v", err)
	}
}

func TestTokenScopes(t *testing.T) {
	token := &Token{
		TokenID:   "test",
		Scopes:    []string{ScopeSearch, ScopeIndex},
		ExpiresAt: time.Now().Add(time.Hour),
	}

	if !token.HasScope(ScopeSearch) {
		t.Error("should have search scope")
	}
	if !token.HasScope(ScopeIndex) {
		t.Error("should have index scope")
	}
	if token.HasScope(ScopeAdmin) {
		t.Error("should not have admin scope")
	}

	// Admin scope should grant all
	token.Scopes = []string{ScopeAdmin}
	if !token.HasScope(ScopeSearch) {
		t.Error("admin should have search scope")
	}
	if !token.HasScope(ScopeIndex) {
		t.Error("admin should have index scope")
	}
}

func TestCleanupExpiredTokens(t *testing.T) {
	ctx := context.Background()

	// Create service with very short TTL
	enterpriseCfg, _ := enterprise.NewConfig("test-enterprise", 128, 64)
	store := enterprise.NewMemoryStore()
	store.Put(ctx, enterpriseCfg)

	cfg := ServiceConfig{
		TokenTTL:      100 * time.Millisecond,
		RefreshWindow: 50 * time.Millisecond,
		LSHBits:       8,
		Dimension:     128,
	}
	service := NewService(cfg, store)

	// Register and authenticate
	password := []byte("password123")
	service.RegisterUser(ctx, "user1", "test-enterprise", password, []string{ScopeSearch})
	service.Authenticate(ctx, "user1", password)
	service.Authenticate(ctx, "user1", password)

	if service.ActiveTokenCount() != 2 {
		t.Errorf("expected 2 active tokens, got %d", service.ActiveTokenCount())
	}

	// Wait for expiry
	time.Sleep(150 * time.Millisecond)

	// Cleanup
	cleaned := service.CleanupExpiredTokens()
	if cleaned != 2 {
		t.Errorf("expected to cleanup 2 tokens, got %d", cleaned)
	}

	if service.ActiveTokenCount() != 0 {
		t.Errorf("expected 0 active tokens, got %d", service.ActiveTokenCount())
	}
}
