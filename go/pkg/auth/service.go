package auth

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/lsh"
)

var (
	// ErrInvalidToken is returned when a token is invalid or not found.
	ErrInvalidToken = errors.New("invalid or expired token")
	// ErrUnauthorized is returned when access is denied.
	ErrUnauthorized = errors.New("unauthorized access")
	// ErrTokenExpired is returned when a token has expired.
	ErrTokenExpired = errors.New("token has expired")
	// ErrUserNotFound is returned when a user is not found.
	ErrUserNotFound = errors.New("user not found")
	// ErrInvalidCredentials is returned when credentials are incorrect.
	ErrInvalidCredentials = errors.New("invalid credentials")
	// ErrUserExists is returned when trying to create an existing user.
	ErrUserExists = errors.New("user already exists")
	// ErrUserDisabled is returned when a user account is disabled.
	ErrUserDisabled = errors.New("user account is disabled")
	// ErrTokenNotRefreshable is returned when a token is not yet eligible for refresh.
	ErrTokenNotRefreshable = errors.New("token not yet eligible for refresh")
)

// Service handles authentication and key distribution.
type Service struct {
	config      ServiceConfig
	configStore enterprise.Store
	tokens      map[string]*Token
	users       map[string]*User
	mu          sync.RWMutex
}

// NewService creates a new authentication service.
func NewService(cfg ServiceConfig, configStore enterprise.Store) *Service {
	cfg.Validate()
	return &Service{
		config:      cfg,
		configStore: configStore,
		tokens:      make(map[string]*Token),
		users:       make(map[string]*User),
	}
}

// RegisterUser registers a new user for an enterprise.
// In production, use bcrypt for password hashing.
func (s *Service) RegisterUser(ctx context.Context, userID, enterpriseID string, passwordHash []byte, scopes []string) error {
	// Verify enterprise exists
	if !s.configStore.Exists(ctx, enterpriseID) {
		return enterprise.ErrEnterpriseNotFound
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.users[userID]; exists {
		return ErrUserExists
	}

	s.users[userID] = &User{
		UserID:       userID,
		EnterpriseID: enterpriseID,
		PasswordHash: passwordHash,
		Scopes:       scopes,
		Enabled:      true,
		CreatedAt:    time.Now(),
	}
	return nil
}

// Authenticate authenticates a user and returns credentials.
func (s *Service) Authenticate(ctx context.Context, userID string, password []byte) (*ClientCredentials, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	user, ok := s.users[userID]
	if !ok {
		return nil, ErrUserNotFound
	}

	if !user.Enabled {
		return nil, ErrUserDisabled
	}

	// Constant-time password comparison
	if subtle.ConstantTimeCompare(user.PasswordHash, password) != 1 {
		return nil, ErrInvalidCredentials
	}

	// Update last login
	user.LastLogin = time.Now()

	// Get enterprise configuration
	enterpriseCfg, err := s.configStore.Get(ctx, user.EnterpriseID)
	if err != nil {
		return nil, fmt.Errorf("failed to get enterprise config: %w", err)
	}

	// Generate token
	tokenID := generateTokenID()
	now := time.Now()
	token := &Token{
		TokenID:      tokenID,
		EnterpriseID: user.EnterpriseID,
		UserID:       userID,
		IssuedAt:     now,
		ExpiresAt:    now.Add(s.config.TokenTTL),
		Scopes:       user.Scopes,
	}
	s.tokens[tokenID] = token

	// Generate LSH hyperplanes from the secret seed
	hyperplanes := lsh.GenerateHyperplanes(
		enterpriseCfg.GetLSHSeedAsInt64(),
		s.config.LSHBits,
		enterpriseCfg.Dimension,
	)

	return &ClientCredentials{
		Token:           tokenID,
		TokenExpiry:     token.ExpiresAt,
		AESKey:          enterpriseCfg.AESKey,
		LSHHyperplanes:  hyperplanes,
		Centroids:       enterpriseCfg.Centroids,
		EnterpriseID:    user.EnterpriseID,
		Dimension:       enterpriseCfg.Dimension,
		NumSuperBuckets: enterpriseCfg.NumSuperBuckets,
	}, nil
}

// ValidateToken validates a token and returns the associated Token.
func (s *Service) ValidateToken(ctx context.Context, tokenID string) (*Token, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	token, ok := s.tokens[tokenID]
	if !ok {
		return nil, ErrInvalidToken
	}

	if token.IsExpired() {
		return nil, ErrTokenExpired
	}

	return token, nil
}

// RefreshToken creates a new token if the current one is within the refresh window.
func (s *Service) RefreshToken(ctx context.Context, tokenID string) (*ClientCredentials, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	token, ok := s.tokens[tokenID]
	if !ok {
		return nil, ErrInvalidToken
	}

	if token.IsExpired() {
		delete(s.tokens, tokenID)
		return nil, ErrTokenExpired
	}

	// Check if within refresh window
	if time.Until(token.ExpiresAt) > s.config.RefreshWindow {
		return nil, ErrTokenNotRefreshable
	}

	// Get user
	user, ok := s.users[token.UserID]
	if !ok {
		return nil, ErrUserNotFound
	}

	if !user.Enabled {
		return nil, ErrUserDisabled
	}

	// Get enterprise config
	enterpriseCfg, err := s.configStore.Get(ctx, token.EnterpriseID)
	if err != nil {
		return nil, fmt.Errorf("failed to get enterprise config: %w", err)
	}

	// Invalidate old token
	delete(s.tokens, tokenID)

	// Create new token
	newTokenID := generateTokenID()
	now := time.Now()
	newToken := &Token{
		TokenID:      newTokenID,
		EnterpriseID: token.EnterpriseID,
		UserID:       token.UserID,
		IssuedAt:     now,
		ExpiresAt:    now.Add(s.config.TokenTTL),
		Scopes:       user.Scopes,
	}
	s.tokens[newTokenID] = newToken

	// Generate fresh hyperplanes
	hyperplanes := lsh.GenerateHyperplanes(
		enterpriseCfg.GetLSHSeedAsInt64(),
		s.config.LSHBits,
		enterpriseCfg.Dimension,
	)

	return &ClientCredentials{
		Token:           newTokenID,
		TokenExpiry:     newToken.ExpiresAt,
		AESKey:          enterpriseCfg.AESKey,
		LSHHyperplanes:  hyperplanes,
		Centroids:       enterpriseCfg.Centroids,
		EnterpriseID:    token.EnterpriseID,
		Dimension:       enterpriseCfg.Dimension,
		NumSuperBuckets: enterpriseCfg.NumSuperBuckets,
	}, nil
}

// RevokeToken invalidates a token.
func (s *Service) RevokeToken(ctx context.Context, tokenID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.tokens, tokenID)
	return nil
}

// RevokeAllUserTokens invalidates all tokens for a user.
func (s *Service) RevokeAllUserTokens(ctx context.Context, userID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for tokenID, token := range s.tokens {
		if token.UserID == userID {
			delete(s.tokens, tokenID)
		}
	}
	return nil
}

// DisableUser disables a user account.
func (s *Service) DisableUser(ctx context.Context, userID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	user, ok := s.users[userID]
	if !ok {
		return ErrUserNotFound
	}

	user.Enabled = false

	// Revoke all tokens
	for tokenID, token := range s.tokens {
		if token.UserID == userID {
			delete(s.tokens, tokenID)
		}
	}

	return nil
}

// EnableUser enables a user account.
func (s *Service) EnableUser(ctx context.Context, userID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	user, ok := s.users[userID]
	if !ok {
		return ErrUserNotFound
	}

	user.Enabled = true
	return nil
}

// GetUser returns a user by ID.
func (s *Service) GetUser(ctx context.Context, userID string) (*User, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	user, ok := s.users[userID]
	if !ok {
		return nil, ErrUserNotFound
	}

	// Return a copy
	return &User{
		UserID:       user.UserID,
		EnterpriseID: user.EnterpriseID,
		Scopes:       user.Scopes,
		Enabled:      user.Enabled,
		CreatedAt:    user.CreatedAt,
		LastLogin:    user.LastLogin,
	}, nil
}

// ListUsers returns all users for an enterprise.
func (s *Service) ListUsers(ctx context.Context, enterpriseID string) ([]*User, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var users []*User
	for _, user := range s.users {
		if user.EnterpriseID == enterpriseID {
			users = append(users, &User{
				UserID:       user.UserID,
				EnterpriseID: user.EnterpriseID,
				Scopes:       user.Scopes,
				Enabled:      user.Enabled,
				CreatedAt:    user.CreatedAt,
				LastLogin:    user.LastLogin,
			})
		}
	}
	return users, nil
}

// CleanupExpiredTokens removes expired tokens from memory.
func (s *Service) CleanupExpiredTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	count := 0
	for tokenID, token := range s.tokens {
		if token.IsExpired() {
			delete(s.tokens, tokenID)
			count++
		}
	}
	return count
}

// ActiveTokenCount returns the number of active tokens.
func (s *Service) ActiveTokenCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	count := 0
	for _, token := range s.tokens {
		if !token.IsExpired() {
			count++
		}
	}
	return count
}

// generateTokenID creates a cryptographically secure token ID.
func generateTokenID() string {
	b := make([]byte, 32)
	rand.Read(b)
	return hex.EncodeToString(b)
}
