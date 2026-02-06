// Package enterprise provides per-enterprise configuration and secret management
// for Tier 2.5 hierarchical private search.
//
// Each enterprise has its own:
//   - AES-256 key for vector encryption
//   - LSH seed for secret hyperplane generation
//   - Centroids for HE scoring
//
// These secrets are distributed to authenticated users via the auth service,
// ensuring that the server cannot map bucket IDs to query regions.
package enterprise

import (
	"crypto/rand"
	"encoding/gob"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"time"
)

// Config holds all enterprise-specific secrets and configuration.
// This is distributed to authenticated users via the auth service.
type Config struct {
	// EnterpriseID uniquely identifies the enterprise
	EnterpriseID string

	// AESKey is the 256-bit key for vector encryption (32 bytes)
	AESKey []byte

	// LSHSeed is the cryptographically random seed for LSH hyperplanes
	// This is SECRET - unlike the current public seeds (42, 137)
	LSHSeed []byte // 32 bytes

	// Centroids are the super-bucket centroids for HE scoring
	// These can be cached client-side after authentication
	Centroids [][]float64

	// Dimension is the vector dimension
	Dimension int

	// NumSuperBuckets is the number of super-buckets
	NumSuperBuckets int

	// NumSubBuckets is the number of sub-buckets (within each super-bucket)
	NumSubBuckets int

	// Metadata
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Version     int
	Description string
}

// NewConfig creates a new enterprise configuration with fresh secrets.
// Uses default of 64 sub-buckets.
func NewConfig(enterpriseID string, dimension int, numSuperBuckets int) (*Config, error) {
	return NewConfigWithSubBuckets(enterpriseID, dimension, numSuperBuckets, 64)
}

// NewConfigWithSubBuckets creates a new enterprise configuration with custom sub-bucket count.
func NewConfigWithSubBuckets(enterpriseID string, dimension int, numSuperBuckets int, numSubBuckets int) (*Config, error) {
	if enterpriseID == "" {
		return nil, fmt.Errorf("enterprise ID is required")
	}
	if dimension <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if numSuperBuckets <= 0 {
		return nil, fmt.Errorf("numSuperBuckets must be positive")
	}
	if numSubBuckets <= 0 {
		numSubBuckets = 64 // Default
	}

	// Generate cryptographically secure AES key
	aesKey := make([]byte, 32)
	if _, err := rand.Read(aesKey); err != nil {
		return nil, fmt.Errorf("failed to generate AES key: %w", err)
	}

	// Generate cryptographically secure LSH seed
	lshSeed := make([]byte, 32)
	if _, err := rand.Read(lshSeed); err != nil {
		return nil, fmt.Errorf("failed to generate LSH seed: %w", err)
	}

	// Initialize empty centroids (will be computed during indexing)
	centroids := make([][]float64, numSuperBuckets)
	for i := range centroids {
		centroids[i] = make([]float64, dimension)
	}

	now := time.Now()
	return &Config{
		EnterpriseID:    enterpriseID,
		AESKey:          aesKey,
		LSHSeed:         lshSeed,
		Centroids:       centroids,
		Dimension:       dimension,
		NumSuperBuckets: numSuperBuckets,
		NumSubBuckets:   numSubBuckets,
		CreatedAt:       now,
		UpdatedAt:       now,
		Version:         1,
	}, nil
}

// GetLSHSeedAsInt64 returns the LSH seed as int64 for compatibility
// with the existing lsh.Index which uses int64 seeds.
func (c *Config) GetLSHSeedAsInt64() int64 {
	if len(c.LSHSeed) < 8 {
		return 0
	}
	// Use first 8 bytes of the 32-byte seed
	var seed int64
	for i := 0; i < 8; i++ {
		seed |= int64(c.LSHSeed[i]) << (uint(i) * 8)
	}
	return seed
}

// GetSubLSHSeedAsInt64 returns a derived seed for sub-bucket LSH.
// Uses XOR with a constant to derive a different but deterministic seed.
func (c *Config) GetSubLSHSeedAsInt64() int64 {
	return c.GetLSHSeedAsInt64() ^ 0x12345678DEADBEEF
}

// GetSubLSHBits returns the number of LSH bits for sub-buckets.
// This must match what the builder uses for consistent hashing.
func (c *Config) GetSubLSHBits() int {
	numSubBuckets := c.NumSubBuckets
	if numSubBuckets <= 0 {
		numSubBuckets = 64 // Default
	}
	bits := int(math.Ceil(math.Log2(float64(numSubBuckets)))) + 2
	if bits < 6 {
		bits = 6
	}
	return bits
}

// Fingerprint returns a safe identifier for logging (not the full key).
func (c *Config) Fingerprint() string {
	if len(c.AESKey) < 4 {
		return "invalid"
	}
	return hex.EncodeToString(c.AESKey[:4]) + "..."
}

// Validate checks that the configuration is complete and valid.
func (c *Config) Validate() error {
	if c.EnterpriseID == "" {
		return fmt.Errorf("enterprise ID is required")
	}
	if len(c.AESKey) != 32 {
		return fmt.Errorf("AES key must be 32 bytes, got %d", len(c.AESKey))
	}
	if len(c.LSHSeed) < 8 {
		return fmt.Errorf("LSH seed must be at least 8 bytes")
	}
	if c.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive")
	}
	if c.NumSuperBuckets <= 0 {
		return fmt.Errorf("numSuperBuckets must be positive")
	}
	return nil
}

// SetCentroids updates the centroids (called after index building).
func (c *Config) SetCentroids(centroids [][]float64) {
	c.Centroids = centroids
	c.UpdatedAt = time.Now()
	c.Version++
}

// Clone creates a deep copy of the configuration.
func (c *Config) Clone() *Config {
	clone := &Config{
		EnterpriseID:    c.EnterpriseID,
		AESKey:          make([]byte, len(c.AESKey)),
		LSHSeed:         make([]byte, len(c.LSHSeed)),
		Centroids:       make([][]float64, len(c.Centroids)),
		Dimension:       c.Dimension,
		NumSuperBuckets: c.NumSuperBuckets,
		NumSubBuckets:   c.NumSubBuckets,
		CreatedAt:       c.CreatedAt,
		UpdatedAt:       c.UpdatedAt,
		Version:         c.Version,
		Description:     c.Description,
	}
	copy(clone.AESKey, c.AESKey)
	copy(clone.LSHSeed, c.LSHSeed)
	for i, centroid := range c.Centroids {
		clone.Centroids[i] = make([]float64, len(centroid))
		copy(clone.Centroids[i], centroid)
	}
	return clone
}

// Save serializes the configuration to a writer.
func (c *Config) Save(w io.Writer) error {
	enc := gob.NewEncoder(w)
	return enc.Encode(c)
}

// SaveToFile saves the configuration to a file.
func (c *Config) SaveToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return c.Save(f)
}

// LoadConfig deserializes a configuration from a reader.
func LoadConfig(r io.Reader) (*Config, error) {
	var cfg Config
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// LoadConfigFromFile loads a configuration from a file.
func LoadConfigFromFile(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LoadConfig(f)
}
