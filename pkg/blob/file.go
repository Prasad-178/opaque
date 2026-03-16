package blob

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// FileStore implements Store using the local filesystem.
// Each blob is stored as a JSON file. Good for local development and testing.
//
// Directory structure:
//
//	basePath/
//	├── index.json          # Bucket -> blob ID mappings
//	└── blobs/
//	    ├── blob_id_1.json
//	    ├── blob_id_2.json
//	    └── ...
type FileStore struct {
	basePath  string
	blobsPath string

	// In-memory index (loaded from disk)
	buckets map[string][]string // bucket -> blob IDs

	mu sync.RWMutex
}

// NewFileStore creates a new file-based blob store.
// Creates the directory structure if it doesn't exist.
func NewFileStore(basePath string) (*FileStore, error) {
	blobsPath := filepath.Join(basePath, "blobs")

	// Create directories
	if err := os.MkdirAll(blobsPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create blobs directory: %w", err)
	}

	store := &FileStore{
		basePath:  basePath,
		blobsPath: blobsPath,
		buckets:   make(map[string][]string),
	}

	// Load existing index
	if err := store.loadIndex(); err != nil {
		return nil, err
	}

	return store, nil
}

// indexPath returns the path to the index file.
func (s *FileStore) indexPath() string {
	return filepath.Join(s.basePath, "index.json")
}

// blobPath returns the path to a blob file.
func (s *FileStore) blobPath(id string) string {
	// Sanitize ID to prevent path traversal
	safeID := strings.ReplaceAll(id, "/", "_")
	safeID = strings.ReplaceAll(safeID, "\\", "_")
	safeID = strings.ReplaceAll(safeID, "..", "_")
	return filepath.Join(s.blobsPath, safeID+".json")
}

// loadIndex loads the bucket index from disk.
func (s *FileStore) loadIndex() error {
	data, err := os.ReadFile(s.indexPath())
	if os.IsNotExist(err) {
		return nil // No index yet, start fresh
	}
	if err != nil {
		return fmt.Errorf("failed to read index: %w", err)
	}

	if err := json.Unmarshal(data, &s.buckets); err != nil {
		return fmt.Errorf("failed to parse index: %w", err)
	}

	return nil
}

// saveIndex saves the bucket index to disk.
func (s *FileStore) saveIndex() error {
	data, err := json.MarshalIndent(s.buckets, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal index: %w", err)
	}

	if err := os.WriteFile(s.indexPath(), data, 0644); err != nil {
		return fmt.Errorf("failed to write index: %w", err)
	}

	return nil
}

// Put stores a blob to disk.
func (s *FileStore) Put(ctx context.Context, blob *Blob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if exists
	if _, err := os.Stat(s.blobPath(blob.ID)); err == nil {
		return ErrBlobExists
	}

	// Serialize blob
	data, err := blob.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize blob: %w", err)
	}

	// Write blob file
	if err := os.WriteFile(s.blobPath(blob.ID), data, 0644); err != nil {
		return fmt.Errorf("failed to write blob: %w", err)
	}

	// Update index
	s.buckets[blob.LSHBucket] = append(s.buckets[blob.LSHBucket], blob.ID)

	// Save index
	if err := s.saveIndex(); err != nil {
		// Rollback: delete blob file
		os.Remove(s.blobPath(blob.ID))
		return err
	}

	return nil
}

// PutBatch stores multiple blobs.
func (s *FileStore) PutBatch(ctx context.Context, blobs []*Blob) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check for duplicates first
	for _, blob := range blobs {
		if _, err := os.Stat(s.blobPath(blob.ID)); err == nil {
			return ErrBlobExists
		}
	}

	// Write all blobs
	writtenIDs := make([]string, 0, len(blobs))
	for _, blob := range blobs {
		data, err := blob.Serialize()
		if err != nil {
			// Rollback
			for _, id := range writtenIDs {
				os.Remove(s.blobPath(id))
			}
			return fmt.Errorf("failed to serialize blob %s: %w", blob.ID, err)
		}

		if err := os.WriteFile(s.blobPath(blob.ID), data, 0644); err != nil {
			// Rollback
			for _, id := range writtenIDs {
				os.Remove(s.blobPath(id))
			}
			return fmt.Errorf("failed to write blob %s: %w", blob.ID, err)
		}

		writtenIDs = append(writtenIDs, blob.ID)
		s.buckets[blob.LSHBucket] = append(s.buckets[blob.LSHBucket], blob.ID)
	}

	// Save index
	if err := s.saveIndex(); err != nil {
		// Rollback
		for _, id := range writtenIDs {
			os.Remove(s.blobPath(id))
		}
		return err
	}

	return nil
}

// Get retrieves a blob from disk.
func (s *FileStore) Get(ctx context.Context, id string) (*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, err := os.ReadFile(s.blobPath(id))
	if os.IsNotExist(err) {
		return nil, ErrBlobNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read blob: %w", err)
	}

	return Deserialize(data)
}

// GetBatch retrieves multiple blobs.
func (s *FileStore) GetBatch(ctx context.Context, ids []string) ([]*Blob, error) {
	blobs := make([]*Blob, len(ids))

	for i, id := range ids {
		blob, err := s.Get(ctx, id)
		if err == ErrBlobNotFound {
			blobs[i] = nil
			continue
		}
		if err != nil {
			return nil, err
		}
		blobs[i] = blob
	}

	return blobs, nil
}

// GetBucket retrieves all blobs in a bucket.
func (s *FileStore) GetBucket(ctx context.Context, bucket string) ([]*Blob, error) {
	s.mu.RLock()
	ids := s.buckets[bucket]
	s.mu.RUnlock()

	if len(ids) == 0 {
		return []*Blob{}, nil
	}

	blobs := make([]*Blob, 0, len(ids))
	for _, id := range ids {
		blob, err := s.Get(ctx, id)
		if err != nil {
			continue // Skip failed reads
		}
		blobs = append(blobs, blob)
	}

	return blobs, nil
}

// GetBuckets retrieves blobs from multiple buckets.
func (s *FileStore) GetBuckets(ctx context.Context, buckets []string) ([]*Blob, error) {
	s.mu.RLock()
	var allIDs []string
	seen := make(map[string]bool)
	for _, bucket := range buckets {
		for _, id := range s.buckets[bucket] {
			if !seen[id] {
				seen[id] = true
				allIDs = append(allIDs, id)
			}
		}
	}
	s.mu.RUnlock()

	blobs := make([]*Blob, 0, len(allIDs))
	for _, id := range allIDs {
		blob, err := s.Get(ctx, id)
		if err != nil {
			continue
		}
		blobs = append(blobs, blob)
	}

	return blobs, nil
}

// Delete removes a blob from disk.
func (s *FileStore) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Read blob to get bucket
	data, err := os.ReadFile(s.blobPath(id))
	if os.IsNotExist(err) {
		return nil // Already deleted
	}
	if err != nil {
		return fmt.Errorf("failed to read blob: %w", err)
	}

	blob, err := Deserialize(data)
	if err != nil {
		return fmt.Errorf("failed to parse blob: %w", err)
	}

	// Remove from index
	ids := s.buckets[blob.LSHBucket]
	for i, bid := range ids {
		if bid == id {
			s.buckets[blob.LSHBucket] = append(ids[:i], ids[i+1:]...)
			break
		}
	}
	if len(s.buckets[blob.LSHBucket]) == 0 {
		delete(s.buckets, blob.LSHBucket)
	}

	// Delete file
	if err := os.Remove(s.blobPath(id)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete blob file: %w", err)
	}

	// Save index
	return s.saveIndex()
}

// DeleteBatch removes multiple blobs.
func (s *FileStore) DeleteBatch(ctx context.Context, ids []string) error {
	for _, id := range ids {
		if err := s.Delete(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// ListBuckets returns all bucket identifiers.
func (s *FileStore) ListBuckets(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	buckets := make([]string, 0, len(s.buckets))
	for bucket := range s.buckets {
		buckets = append(buckets, bucket)
	}
	return buckets, nil
}

// BucketInfo returns information about a bucket.
func (s *FileStore) BucketInfo(ctx context.Context, bucket string) (*BucketInfo, error) {
	s.mu.RLock()
	ids, exists := s.buckets[bucket]
	s.mu.RUnlock()

	if !exists {
		return nil, ErrBucketNotFound
	}

	var totalSize int64
	for _, id := range ids {
		info, err := os.Stat(s.blobPath(id))
		if err == nil {
			totalSize += info.Size()
		}
	}

	return &BucketInfo{
		Bucket:    bucket,
		Count:     len(ids),
		TotalSize: totalSize,
	}, nil
}

// Stats returns store statistics.
func (s *FileStore) Stats(ctx context.Context) (*StoreStats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var totalBlobs int64
	var totalSize int64

	for _, ids := range s.buckets {
		totalBlobs += int64(len(ids))
		for _, id := range ids {
			info, err := os.Stat(s.blobPath(id))
			if err == nil {
				totalSize += info.Size()
			}
		}
	}

	avgBlobsPerBucket := 0.0
	if len(s.buckets) > 0 {
		avgBlobsPerBucket = float64(totalBlobs) / float64(len(s.buckets))
	}

	return &StoreStats{
		TotalBlobs:        totalBlobs,
		TotalBuckets:      int64(len(s.buckets)),
		TotalSize:         totalSize,
		AvgBlobsPerBucket: avgBlobsPerBucket,
	}, nil
}

// Close is a no-op for file store.
func (s *FileStore) Close() error {
	return nil
}

// GetSuperBuckets retrieves all blobs from the specified super-bucket IDs.
func (s *FileStore) GetSuperBuckets(ctx context.Context, superBucketIDs []int) ([]*Blob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var allBlobs []*Blob
	seen := make(map[string]bool)

	for _, superID := range superBucketIDs {
		// Find all buckets that belong to this super-bucket
		// Bucket keys are either "XX" (super only) or "XX_YY" (super + sub)
		superPrefix := fmt.Sprintf("%02d", superID)
		for bucket, ids := range s.buckets {
			// Check if bucket belongs to this super-bucket
			if bucket == superPrefix || strings.HasPrefix(bucket, superPrefix+"_") {
				for _, id := range ids {
					if !seen[id] {
						seen[id] = true
						blob, err := s.Get(ctx, id)
						if err == nil && blob != nil {
							allBlobs = append(allBlobs, blob)
						}
					}
				}
			}
		}
	}

	return allBlobs, nil
}

// Ensure FileStore implements Store interface.
var _ Store = (*FileStore)(nil)
