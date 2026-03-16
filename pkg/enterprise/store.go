package enterprise

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

var (
	// ErrEnterpriseNotFound is returned when an enterprise configuration is not found.
	ErrEnterpriseNotFound = errors.New("enterprise not found")
	// ErrEnterpriseExists is returned when trying to create an enterprise that already exists.
	ErrEnterpriseExists = errors.New("enterprise already exists")
)

// Store is the interface for enterprise configuration storage.
type Store interface {
	// Get retrieves an enterprise configuration by ID.
	Get(ctx context.Context, enterpriseID string) (*Config, error)

	// Put stores or updates an enterprise configuration.
	Put(ctx context.Context, cfg *Config) error

	// Delete removes an enterprise configuration.
	Delete(ctx context.Context, enterpriseID string) error

	// List returns all enterprise IDs.
	List(ctx context.Context) ([]string, error)

	// Exists checks if an enterprise exists.
	Exists(ctx context.Context, enterpriseID string) bool

	// Close closes the store.
	Close() error
}

// MemoryStore implements Store using in-memory storage.
// For production, implement a secure vault-backed store.
type MemoryStore struct {
	configs map[string]*Config
	mu      sync.RWMutex
}

// NewMemoryStore creates an in-memory configuration store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		configs: make(map[string]*Config),
	}
}

// Get retrieves an enterprise configuration by ID.
func (s *MemoryStore) Get(ctx context.Context, enterpriseID string) (*Config, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg, ok := s.configs[enterpriseID]
	if !ok {
		return nil, ErrEnterpriseNotFound
	}
	// Return a clone to prevent mutation
	return cfg.Clone(), nil
}

// Put stores or updates an enterprise configuration.
func (s *MemoryStore) Put(ctx context.Context, cfg *Config) error {
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Store a clone to prevent external mutation
	s.configs[cfg.EnterpriseID] = cfg.Clone()
	return nil
}

// Delete removes an enterprise configuration.
func (s *MemoryStore) Delete(ctx context.Context, enterpriseID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.configs[enterpriseID]; !ok {
		return ErrEnterpriseNotFound
	}

	delete(s.configs, enterpriseID)
	return nil
}

// List returns all enterprise IDs.
func (s *MemoryStore) List(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids := make([]string, 0, len(s.configs))
	for id := range s.configs {
		ids = append(ids, id)
	}
	return ids, nil
}

// Exists checks if an enterprise exists.
func (s *MemoryStore) Exists(ctx context.Context, enterpriseID string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	_, ok := s.configs[enterpriseID]
	return ok
}

// Close closes the store.
func (s *MemoryStore) Close() error {
	return nil
}

var _ Store = (*MemoryStore)(nil)

// FileStore implements Store using filesystem storage.
// Each enterprise config is stored as a separate file.
type FileStore struct {
	baseDir string
	mu      sync.RWMutex
}

// NewFileStore creates a file-based configuration store.
func NewFileStore(baseDir string) (*FileStore, error) {
	if err := os.MkdirAll(baseDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}
	return &FileStore{baseDir: baseDir}, nil
}

func (s *FileStore) configPath(enterpriseID string) string {
	return filepath.Join(s.baseDir, enterpriseID+".gob")
}

// Get retrieves an enterprise configuration by ID.
func (s *FileStore) Get(ctx context.Context, enterpriseID string) (*Config, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := s.configPath(enterpriseID)
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrEnterpriseNotFound
		}
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer f.Close()

	var cfg Config
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("failed to decode config: %w", err)
	}

	return &cfg, nil
}

// Put stores or updates an enterprise configuration.
func (s *FileStore) Put(ctx context.Context, cfg *Config) error {
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid config: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.configPath(cfg.EnterpriseID)
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(cfg); err != nil {
		return fmt.Errorf("failed to encode config: %w", err)
	}

	// Set restrictive permissions (owner read/write only)
	if err := os.Chmod(path, 0600); err != nil {
		return fmt.Errorf("failed to set permissions: %w", err)
	}

	return nil
}

// Delete removes an enterprise configuration.
func (s *FileStore) Delete(ctx context.Context, enterpriseID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := s.configPath(enterpriseID)
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return ErrEnterpriseNotFound
		}
		return fmt.Errorf("failed to delete config file: %w", err)
	}
	return nil
}

// List returns all enterprise IDs.
func (s *FileStore) List(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}

	ids := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if filepath.Ext(name) == ".gob" {
			ids = append(ids, name[:len(name)-4])
		}
	}
	return ids, nil
}

// Exists checks if an enterprise exists.
func (s *FileStore) Exists(ctx context.Context, enterpriseID string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	path := s.configPath(enterpriseID)
	_, err := os.Stat(path)
	return err == nil
}

// Close closes the store.
func (s *FileStore) Close() error {
	return nil
}

var _ Store = (*FileStore)(nil)
