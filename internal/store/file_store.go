package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// FileStore is a durable VectorStore implementation backed by a JSON file.
//
// It is optimized for operational simplicity and small-to-medium datasets,
// not high write throughput.
type FileStore struct {
	path     string
	vectors  map[string][]float64
	metadata map[string]map[string]any
	mu       sync.RWMutex
}

type fileStoreState struct {
	Vectors  map[string][]float64      `json:"vectors"`
	Metadata map[string]map[string]any `json:"metadata,omitempty"`
}

// NewFileStore creates (or opens) a file-backed vector store.
func NewFileStore(path string) (*FileStore, error) {
	if path == "" {
		return nil, fmt.Errorf("file store path is required")
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("create store directory: %w", err)
	}

	s := &FileStore{
		path:     path,
		vectors:  make(map[string][]float64),
		metadata: make(map[string]map[string]any),
	}

	if err := s.load(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *FileStore) load() error {
	data, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read store file: %w", err)
	}

	if len(data) == 0 {
		return nil
	}

	var state fileStoreState
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("parse store file: %w", err)
	}

	if state.Vectors != nil {
		s.vectors = state.Vectors
	}
	if state.Metadata != nil {
		s.metadata = state.Metadata
	}

	return nil
}

func (s *FileStore) flushLocked() error {
	state := fileStoreState{
		Vectors:  s.vectors,
		Metadata: s.metadata,
	}

	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal store state: %w", err)
	}

	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("write temp store file: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("commit store file: %w", err)
	}

	return nil
}

// GetByIDs retrieves vectors by their IDs.
func (s *FileStore) GetByIDs(ctx context.Context, ids []string) ([][]float64, error) {
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
func (s *FileStore) GetByID(ctx context.Context, id string) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	vec, ok := s.vectors[id]
	if !ok {
		return nil, ErrNotFound
	}

	return vec, nil
}

// Add adds vectors to the store.
func (s *FileStore) Add(ctx context.Context, ids []string, vectors [][]float64, metadata []map[string]any) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids and vectors length mismatch")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for i, id := range ids {
		s.vectors[id] = vectors[i]
		if metadata != nil && i < len(metadata) {
			s.metadata[id] = metadata[i]
		}
	}

	return s.flushLocked()
}

// Delete removes vectors by ID.
func (s *FileStore) Delete(ctx context.Context, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.vectors, id)
		delete(s.metadata, id)
	}

	return s.flushLocked()
}

// Count returns total vector count.
func (s *FileStore) Count(ctx context.Context) (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return int64(len(s.vectors)), nil
}

// Close closes the store.
func (s *FileStore) Close() error {
	return nil
}
