// Package store provides vector storage backends.
package store

import (
	"context"
	"errors"
	"sync"
)

var (
	ErrNotFound = errors.New("vector not found")
)

// VectorStore is the interface for vector storage backends.
type VectorStore interface {
	// GetByIDs retrieves vectors by their IDs.
	GetByIDs(ctx context.Context, ids []string) ([][]float64, error)

	// GetByID retrieves a single vector by ID.
	GetByID(ctx context.Context, id string) ([]float64, error)

	// Add adds vectors to the store.
	Add(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error

	// Delete removes vectors by ID.
	Delete(ctx context.Context, ids []string) error

	// Count returns total vector count.
	Count(ctx context.Context) (int64, error)

	// Close closes the connection.
	Close() error
}

// MemoryStore is an in-memory vector store for testing.
type MemoryStore struct {
	vectors  map[string][]float64
	metadata map[string]map[string]any
	mu       sync.RWMutex
}

// NewMemoryStore creates a new in-memory vector store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		vectors:  make(map[string][]float64),
		metadata: make(map[string]map[string]any),
	}
}

// GetByIDs retrieves vectors by their IDs.
func (s *MemoryStore) GetByIDs(ctx context.Context, ids []string) ([][]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	vectors := make([][]float64, len(ids))
	for i, id := range ids {
		vec, ok := s.vectors[id]
		if !ok {
			return nil, ErrNotFound
		}
		vectors[i] = vec
	}

	return vectors, nil
}

// GetByID retrieves a single vector by ID.
func (s *MemoryStore) GetByID(ctx context.Context, id string) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	vec, ok := s.vectors[id]
	if !ok {
		return nil, ErrNotFound
	}

	return vec, nil
}

// Add adds vectors to the store.
func (s *MemoryStore) Add(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error {
	if len(ids) != len(vectors) {
		return errors.New("ids and vectors length mismatch")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for i, id := range ids {
		s.vectors[id] = vectors[i]
		if metadata != nil && i < len(metadata) {
			s.metadata[id] = metadata[i]
		}
	}

	return nil
}

// Delete removes vectors by ID.
func (s *MemoryStore) Delete(ctx context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.vectors, id)
		delete(s.metadata, id)
	}

	return nil
}

// Count returns total vector count.
func (s *MemoryStore) Count(ctx context.Context) (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return int64(len(s.vectors)), nil
}

// Close closes the store.
func (s *MemoryStore) Close() error {
	return nil
}

// GetMetadata retrieves metadata for a vector.
func (s *MemoryStore) GetMetadata(id string) (map[string]any, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	meta, ok := s.metadata[id]
	return meta, ok
}

// GetAll returns all vectors (for testing).
func (s *MemoryStore) GetAll() map[string][]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string][]float64, len(s.vectors))
	for id, vec := range s.vectors {
		result[id] = vec
	}
	return result
}
