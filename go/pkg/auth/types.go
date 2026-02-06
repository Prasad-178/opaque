// Package auth provides token-based authentication and key distribution
// for Tier 2.5 hierarchical private search (Option B).
//
// The authentication flow:
//  1. User authenticates with credentials
//  2. Auth service validates and returns:
//     - Session token (1-hour TTL)
//     - AES key (for vector decryption)
//     - LSH hyperplanes (for bucket computation)
//     - Centroids (for local HE scoring cache)
//  3. Client uses credentials for searches
//  4. Token can be refreshed before expiry
package auth

import (
	"time"
)

// Scopes for authorization
const (
	// ScopeSearch allows performing searches
	ScopeSearch = "search"
	// ScopeIndex allows adding/updating vectors
	ScopeIndex = "index"
	// ScopeAdmin allows full access including key management
	ScopeAdmin = "admin"
)

// Token represents an authentication token with associated metadata.
type Token struct {
	// TokenID is the unique identifier for this token
	TokenID string

	// EnterpriseID identifies which enterprise this token grants access to
	EnterpriseID string

	// UserID identifies the authenticated user
	UserID string

	// IssuedAt is when the token was created
	IssuedAt time.Time

	// ExpiresAt is when the token expires
	ExpiresAt time.Time

	// Scopes defines what operations this token allows
	Scopes []string
}

// IsExpired returns true if the token has expired.
func (t *Token) IsExpired() bool {
	return time.Now().After(t.ExpiresAt)
}

// IsValid returns true if the token is valid and not expired.
func (t *Token) IsValid() bool {
	return t.TokenID != "" && !t.IsExpired()
}

// HasScope checks if the token has a specific scope.
func (t *Token) HasScope(scope string) bool {
	for _, s := range t.Scopes {
		if s == scope || s == ScopeAdmin {
			return true
		}
	}
	return false
}

// TimeUntilExpiry returns the duration until the token expires.
func (t *Token) TimeUntilExpiry() time.Duration {
	return time.Until(t.ExpiresAt)
}

// ClientCredentials contains the secrets distributed to an authenticated client.
// These credentials enable the client to perform Tier 2.5 searches.
type ClientCredentials struct {
	// Token for subsequent API calls
	Token string

	// TokenExpiry when the token expires
	TokenExpiry time.Time

	// AESKey for vector decryption (32 bytes, AES-256)
	AESKey []byte

	// LSHHyperplanes derived from the enterprise LSH seed
	// Distributed as pre-computed planes instead of the seed for security
	LSHHyperplanes [][]float64

	// Centroids for local caching (used in HE scoring)
	Centroids [][]float64

	// EnterpriseID for reference
	EnterpriseID string

	// Dimension of vectors
	Dimension int

	// NumSuperBuckets for bucket computation
	NumSuperBuckets int
}

// IsExpired returns true if the credentials have expired.
func (c *ClientCredentials) IsExpired() bool {
	return time.Now().After(c.TokenExpiry)
}

// TimeUntilExpiry returns the duration until the credentials expire.
func (c *ClientCredentials) TimeUntilExpiry() time.Duration {
	return time.Until(c.TokenExpiry)
}

// User represents an authenticated user.
type User struct {
	// UserID uniquely identifies the user
	UserID string

	// EnterpriseID identifies which enterprise this user belongs to
	EnterpriseID string

	// PasswordHash is the hashed password (use bcrypt in production)
	PasswordHash []byte

	// Scopes defines what operations this user can perform
	Scopes []string

	// Enabled indicates if the user is active
	Enabled bool

	// CreatedAt is when the user was created
	CreatedAt time.Time

	// LastLogin is when the user last authenticated
	LastLogin time.Time
}

// ServiceConfig holds configuration for the auth service.
type ServiceConfig struct {
	// TokenTTL is how long tokens are valid (default: 1 hour)
	TokenTTL time.Duration

	// RefreshWindow is how long before expiry a token can be refreshed
	RefreshWindow time.Duration

	// LSHBits is the number of LSH hyperplanes to generate
	LSHBits int

	// Dimension is the vector dimension
	Dimension int
}

// DefaultServiceConfig returns sensible defaults for the auth service.
func DefaultServiceConfig() ServiceConfig {
	return ServiceConfig{
		TokenTTL:      time.Hour,
		RefreshWindow: 15 * time.Minute,
		LSHBits:       8,
		Dimension:     128,
	}
}

// Validate checks that the service configuration is valid.
func (c *ServiceConfig) Validate() error {
	if c.TokenTTL <= 0 {
		c.TokenTTL = time.Hour
	}
	if c.RefreshWindow <= 0 {
		c.RefreshWindow = 15 * time.Minute
	}
	if c.LSHBits <= 0 {
		c.LSHBits = 8
	}
	if c.Dimension <= 0 {
		c.Dimension = 128
	}
	return nil
}
