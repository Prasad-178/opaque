package blob

import (
	"context"
	"sync"
)

// MemoryStore implements Store using in-memory maps.
// Useful for testing, demos, and small-scale deployments.
// Not persistent - data is lost on restart.
type MemoryStore struct {
	// blobs maps ID -> Blob
	blobs map[string]*Blob

	// buckets maps bucket -> list of blob IDs
	buckets map[string][]string

	mu sync.RWMutex
}

// NewMemoryStore creates a new in-memory blob store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		blobs:   make(map[string]*Blob),
		buckets: make(map[string][]string),
	}
}

// Put stores a blob in memory.
func (s *MemoryStore) Put(ctx context.Context, blob *Blob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.blobs[blob.ID]; exists {
		return ErrBlobExists
	}

	// Store blob
	s.blobs[blob.ID] = blob

	// Add to bucket index
	s.buckets[blob.LSHBucket] = append(s.buckets[blob.LSHBucket], blob.ID)

	return nil
}

// PutBatch stores multiple blobs.
func (s *MemoryStore) PutBatch(ctx context.Context, blobs []*Blob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check for duplicates first
	for _, blob := range blobs {
		if _, exists := s.blobs[blob.ID]; exists {
			return ErrBlobExists
		}
	}

	// Store all blobs
	for _, blob := range blobs {
		s.blobs[blob.ID] = blob
		s.buckets[blob.LSHBucket] = append(s.buckets[blob.LSHBucket], blob.ID)
	}

	return nil
}

// Get retrieves a blob by ID.
func (s *MemoryStore) Get(ctx context.Context, id string) (*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	blob, exists := s.blobs[id]
	if !exists {
		return nil, ErrBlobNotFound
	}

	return blob, nil
}

// GetBatch retrieves multiple blobs by ID.
func (s *MemoryStore) GetBatch(ctx context.Context, ids []string) ([]*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	blobs := make([]*Blob, len(ids))
	for i, id := range ids {
		blobs[i] = s.blobs[id] // nil if not found
	}

	return blobs, nil
}

// GetBucket retrieves all blobs in a bucket.
func (s *MemoryStore) GetBucket(ctx context.Context, bucket string) ([]*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids, exists := s.buckets[bucket]
	if !exists {
		return []*Blob{}, nil
	}

	blobs := make([]*Blob, 0, len(ids))
	for _, id := range ids {
		if blob, ok := s.blobs[id]; ok {
			blobs = append(blobs, blob)
		}
	}

	return blobs, nil
}

// GetBuckets retrieves blobs from multiple buckets.
func (s *MemoryStore) GetBuckets(ctx context.Context, buckets []string) ([]*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var allBlobs []*Blob
	seen := make(map[string]bool)

	for _, bucket := range buckets {
		ids := s.buckets[bucket]
		for _, id := range ids {
			if !seen[id] {
				seen[id] = true
				if blob, ok := s.blobs[id]; ok {
					allBlobs = append(allBlobs, blob)
				}
			}
		}
	}

	return allBlobs, nil
}

// Delete removes a blob by ID.
func (s *MemoryStore) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	blob, exists := s.blobs[id]
	if !exists {
		return nil // No error if not found
	}

	// Remove from bucket index
	bucket := blob.LSHBucket
	ids := s.buckets[bucket]
	for i, bid := range ids {
		if bid == id {
			s.buckets[bucket] = append(ids[:i], ids[i+1:]...)
			break
		}
	}

	// Remove empty bucket
	if len(s.buckets[bucket]) == 0 {
		delete(s.buckets, bucket)
	}

	// Remove blob
	delete(s.blobs, id)

	return nil
}

// DeleteBatch removes multiple blobs by ID.
func (s *MemoryStore) DeleteBatch(ctx context.Context, ids []string) error {
	for _, id := range ids {
		if err := s.Delete(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// ListBuckets returns all bucket identifiers.
func (s *MemoryStore) ListBuckets(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	buckets := make([]string, 0, len(s.buckets))
	for bucket := range s.buckets {
		buckets = append(buckets, bucket)
	}

	return buckets, nil
}

// BucketInfo returns information about a specific bucket.
func (s *MemoryStore) BucketInfo(ctx context.Context, bucket string) (*BucketInfo, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids, exists := s.buckets[bucket]
	if !exists {
		return nil, ErrBucketNotFound
	}

	var totalSize int64
	for _, id := range ids {
		if blob, ok := s.blobs[id]; ok {
			totalSize += int64(len(blob.Ciphertext))
		}
	}

	return &BucketInfo{
		Bucket:    bucket,
		Count:     len(ids),
		TotalSize: totalSize,
	}, nil
}

// Stats returns overall store statistics.
func (s *MemoryStore) Stats(ctx context.Context) (*StoreStats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var totalSize int64
	for _, blob := range s.blobs {
		totalSize += int64(len(blob.Ciphertext))
	}

	avgBlobsPerBucket := 0.0
	if len(s.buckets) > 0 {
		avgBlobsPerBucket = float64(len(s.blobs)) / float64(len(s.buckets))
	}

	return &StoreStats{
		TotalBlobs:        int64(len(s.blobs)),
		TotalBuckets:      int64(len(s.buckets)),
		TotalSize:         totalSize,
		AvgBlobsPerBucket: avgBlobsPerBucket,
	}, nil
}

// Close is a no-op for memory store.
func (s *MemoryStore) Close() error {
	return nil
}

// Clear removes all blobs (useful for testing).
func (s *MemoryStore) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.blobs = make(map[string]*Blob)
	s.buckets = make(map[string][]string)
}

// Ensure MemoryStore implements Store interface.
var _ Store = (*MemoryStore)(nil)
