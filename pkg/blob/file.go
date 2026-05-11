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
//
// # Concurrency
//
// PutBatch is safe for concurrent use from many goroutines. The bulk of the
// work — serializing blobs and writing them to disk — happens WITHOUT holding
// the mutex; only the in-memory buckets-map update is locked, and that lock
// is held briefly. This is what makes the parallel-encrypt phase in
// hierarchical.KMeansBuilder actually parallel on file-backed storage; the
// prior implementation serialised all 16 workers on a single lock held for
// the entire batch (~1024 file writes each), turning a 10-minute build into
// hours at 1M × 1536-dim.
//
// # Index persistence
//
// The on-disk index.json is NOT rewritten on every PutBatch — that turned
// out to be the worst offender during the DBpedia 1M @ 1536-dim build, where
// the index file grows to ~20 MB JSON and was being marshaled + rewritten
// ~1000 times per build (~20 GB cumulative disk writes for the index alone).
// Instead, we mark the in-memory index "dirty" on every mutation, and
// persist on Close() or via an explicit Flush() call. If the process crashes
// mid-build, the index can be rebuilt by scanning the blobs/ directory.
type FileStore struct {
	basePath  string
	blobsPath string

	// In-memory index (loaded from disk)
	buckets map[string][]string // bucket -> blob IDs

	mu    sync.RWMutex
	dirty bool // index has unsaved changes; flushed on Close/Flush
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
	// Use filepath.Base to strip any directory components, preventing path traversal.
	// This handles /, \, .., null bytes, and any other path separator tricks.
	safeID := filepath.Base(id)
	if safeID == "." || safeID == "/" || safeID == "\\" {
		safeID = "_invalid_"
	}
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

// saveIndex saves the bucket index to disk atomically via write-then-rename.
func (s *FileStore) saveIndex() error {
	data, err := json.MarshalIndent(s.buckets, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal index: %w", err)
	}

	// Write to temp file, then rename for atomic update.
	tmpPath := s.indexPath() + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0600); err != nil {
		return fmt.Errorf("failed to write index: %w", err)
	}
	if err := os.Rename(tmpPath, s.indexPath()); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename index: %w", err)
	}

	return nil
}

// Put stores a single blob to disk. The on-disk index is persisted eagerly
// here so single-Put callers (e.g. incremental adds) keep the historical
// "Put = durable" contract. Use PutBatch + Flush for the bulk path.
func (s *FileStore) Put(ctx context.Context, blob *Blob) error {
	if _, err := os.Stat(s.blobPath(blob.ID)); err == nil {
		return ErrBlobExists
	}

	data, err := blob.Serialize()
	if err != nil {
		return fmt.Errorf("failed to serialize blob: %w", err)
	}

	if err := os.WriteFile(s.blobPath(blob.ID), data, 0600); err != nil {
		return fmt.Errorf("failed to write blob: %w", err)
	}

	s.mu.Lock()
	s.buckets[blob.LSHBucket] = append(s.buckets[blob.LSHBucket], blob.ID)
	s.dirty = true
	s.mu.Unlock()

	if err := s.Flush(); err != nil {
		os.Remove(s.blobPath(blob.ID))
		return err
	}
	return nil
}

// PutBatch stores multiple blobs. Safe for concurrent use across goroutines.
//
// Bulk disk writes happen WITHOUT the global mutex (each blob has a unique
// filename), so multiple goroutines calling PutBatch in parallel can saturate
// EBS bandwidth instead of serialising on one lock. The buckets-map update
// is the only thing that takes the lock, and it does so briefly.
//
// The on-disk index is NOT saved here — see the FileStore type comment for
// the rationale. Call Flush() or Close() to persist.
//
// NOTE: existence checks are skipped on the bulk path. Callers (the build
// pipeline) generate fresh unique IDs and don't need overwrite protection.
// Use Put() for the single-blob path which does check.
func (s *FileStore) PutBatch(ctx context.Context, blobs []*Blob) error {
	if len(blobs) == 0 {
		return nil
	}

	for _, b := range blobs {
		data, err := b.Serialize()
		if err != nil {
			return fmt.Errorf("failed to serialize blob %s: %w", b.ID, err)
		}
		if err := os.WriteFile(s.blobPath(b.ID), data, 0600); err != nil {
			return fmt.Errorf("failed to write blob %s: %w", b.ID, err)
		}
	}

	s.mu.Lock()
	for _, b := range blobs {
		s.buckets[b.LSHBucket] = append(s.buckets[b.LSHBucket], b.ID)
	}
	s.dirty = true
	s.mu.Unlock()

	return nil
}

// Flush persists the in-memory bucket index to disk. Safe to call from any
// goroutine. Idempotent: a no-op when there are no unsaved changes.
func (s *FileStore) Flush() error {
	s.mu.Lock()
	if !s.dirty {
		s.mu.Unlock()
		return nil
	}
	// Snapshot under the lock so we can release it before the slow JSON
	// marshal + disk write. Index file is single-writer (we hold the snapshot
	// in a local map) but the FileStore can keep accepting PutBatch in the
	// meantime; dirty stays true on next change.
	snapshot := make(map[string][]string, len(s.buckets))
	for k, v := range s.buckets {
		cp := make([]string, len(v))
		copy(cp, v)
		snapshot[k] = cp
	}
	s.dirty = false
	s.mu.Unlock()

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		s.markDirty()
		return fmt.Errorf("failed to marshal index: %w", err)
	}

	tmpPath := s.indexPath() + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0600); err != nil {
		s.markDirty()
		return fmt.Errorf("failed to write index: %w", err)
	}
	if err := os.Rename(tmpPath, s.indexPath()); err != nil {
		os.Remove(tmpPath)
		s.markDirty()
		return fmt.Errorf("failed to rename index: %w", err)
	}
	return nil
}

// markDirty re-sets the dirty flag — used when a flush attempt fails so the
// next Flush retries persistence.
func (s *FileStore) markDirty() {
	s.mu.Lock()
	s.dirty = true
	s.mu.Unlock()
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

	s.dirty = true
	return nil
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

// Close flushes any unsaved in-memory index to disk and shuts down.
func (s *FileStore) Close() error {
	return s.Flush()
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
