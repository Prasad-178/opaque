package opaque

import (
	"context"
	"fmt"
)

// Metadata is a map of key-value pairs attached to a vector.
// Keys are strings; values can be string, int, float64, or bool.
//
// Metadata is stored encrypted alongside vectors and can be used
// for filtered search via [DB.SearchWithFilter].
type Metadata map[string]any

// Filter specifies criteria for filtered search.
type Filter struct {
	// Where contains exact-match conditions. A result must match ALL conditions.
	// Supported value types: string, int, float64, bool.
	Where map[string]any
}

// AddWithMetadata buffers a vector with associated metadata for indexing.
//
// Metadata is encrypted alongside the vector and can be used for filtered
// search with [DB.SearchWithFilter]. The id must be unique within the DB.
//
// Both the vector and metadata are copied internally.
func (db *DB) AddWithMetadata(ctx context.Context, id string, vector []float64, meta Metadata) error {
	if len(vector) != db.cfg.Dimension {
		return fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(vector), db.cfg.Dimension)
	}
	if id == "" {
		return ErrEmptyID
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	v32 := make([]float32, len(vector))
	for i, x := range vector {
		v32[i] = float32(x)
	}

	db.pendingIDs = append(db.pendingIDs, id)
	db.pendingVectors = append(db.pendingVectors, v32)

	// If the index is already built, assign to nearest centroid and store immediately.
	if db.state == stateReady {
		if err := db.addToIndexLocked(ctx, id, vector); err != nil {
			return fmt.Errorf("opaque: incremental add failed: %w", err)
		}
		db.refreshCentroidCaches()
	}

	if db.state == stateEmpty {
		db.state = stateBuffered
	}
	db.dataVersion++

	// Store metadata.
	if meta != nil {
		if db.metadata == nil {
			db.metadata = make(map[string]Metadata)
		}
		m := make(Metadata, len(meta))
		for k, v := range meta {
			m[k] = v
		}
		db.metadata[id] = m
	}

	return nil
}

// AddBatchWithMetadata buffers multiple vectors with associated metadata.
// The metadatas slice must have the same length as ids and vectors.
// Use nil for vectors without metadata.
func (db *DB) AddBatchWithMetadata(ctx context.Context, ids []string, vectors [][]float64, metadatas []Metadata) error {
	if len(ids) != len(vectors) {
		return fmt.Errorf("opaque: ids length %d does not match vectors length %d", len(ids), len(vectors))
	}
	if len(metadatas) != len(ids) {
		return fmt.Errorf("opaque: metadatas length %d does not match ids length %d", len(metadatas), len(ids))
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	for i, v := range vectors {
		if len(v) != db.cfg.Dimension {
			return fmt.Errorf("%w: vector %d (id=%q) has %d, want %d", ErrDimensionMismatch, i, ids[i], len(v), db.cfg.Dimension)
		}
		if ids[i] == "" {
			return fmt.Errorf("%w: vector %d", ErrEmptyID, i)
		}
	}

	ready := db.state == stateReady
	for i, v := range vectors {
		v32 := make([]float32, len(v))
		for j, x := range v {
			v32[j] = float32(x)
		}
		db.pendingIDs = append(db.pendingIDs, ids[i])
		db.pendingVectors = append(db.pendingVectors, v32)

		// If the index is already built, assign to nearest centroid and store immediately.
		if ready {
			if err := db.addToIndexLocked(ctx, ids[i], v); err != nil {
				return fmt.Errorf("opaque: incremental add failed for %q: %w", ids[i], err)
			}
		}

		if metadatas[i] != nil {
			if db.metadata == nil {
				db.metadata = make(map[string]Metadata)
			}
			m := make(Metadata, len(metadatas[i]))
			for k, val := range metadatas[i] {
				m[k] = val
			}
			db.metadata[ids[i]] = m
		}
	}

	// Refresh centroid caches once after all vectors are added (not per-vector).
	if ready {
		db.refreshCentroidCaches()
	}

	if db.state == stateEmpty {
		db.state = stateBuffered
	}
	db.dataVersion += uint64(len(ids))
	return nil
}

// SearchWithFilter returns the topK most similar vectors matching the filter.
//
// This runs a normal search and then post-filters results by metadata.
// Filtered-out results are not replaced, so fewer than topK results may
// be returned. For better recall with filters, increase topK.
//
// All conditions in [Filter.Where] must match (AND logic). Matching uses
// exact equality for string, int, float64, and bool values.
func (db *DB) SearchWithFilter(ctx context.Context, query []float64, topK int, filter Filter) ([]Result, error) {
	// Fetch more candidates to compensate for filtering.
	fetchK := topK * 3
	if fetchK < 50 {
		fetchK = 50
	}

	results, err := db.Search(ctx, query, fetchK)
	if err != nil {
		return nil, err
	}

	// Apply metadata filter.
	db.mu.RLock()
	defer db.mu.RUnlock()

	filtered := make([]Result, 0, topK)
	for _, r := range results {
		if matchesFilter(db.metadata[r.ID], filter) {
			filtered = append(filtered, r)
			if len(filtered) >= topK {
				break
			}
		}
	}
	return filtered, nil
}

// GetMetadata retrieves the metadata for a vector by ID.
//
// Returns nil if the vector has no metadata, or [ErrNotFound] if the ID
// does not exist.
func (db *DB) GetMetadata(ctx context.Context, id string) (Metadata, error) {
	if id == "" {
		return nil, ErrEmptyID
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.deletedIDs[id] {
		return nil, ErrNotFound
	}

	// Check existence.
	found := false
	for _, pid := range db.pendingIDs {
		if pid == id {
			found = true
			break
		}
	}
	if !found && db.blobStore != nil {
		b, err := db.blobStore.Get(ctx, id)
		if err == nil && b != nil {
			found = true
		}
	}
	if !found {
		return nil, ErrNotFound
	}

	meta := db.metadata[id]
	if meta == nil {
		return nil, nil
	}

	// Return a copy.
	result := make(Metadata, len(meta))
	for k, v := range meta {
		result[k] = v
	}
	return result, nil
}

// matchesFilter checks if metadata satisfies all filter conditions.
// A nil/empty filter matches everything. A nil metadata only matches
// an empty filter.
func matchesFilter(meta Metadata, filter Filter) bool {
	if len(filter.Where) == 0 {
		return true
	}
	if meta == nil {
		return false
	}
	for key, want := range filter.Where {
		got, ok := meta[key]
		if !ok {
			return false
		}
		if !valuesEqual(got, want) {
			return false
		}
	}
	return true
}

// valuesEqual compares two metadata values for equality.
// Handles type coercion between int/float64 for JSON roundtrip compatibility.
func valuesEqual(a, b any) bool {
	// Direct equality.
	if a == b {
		return true
	}

	// Handle numeric type coercion (JSON unmarshals numbers as float64).
	switch av := a.(type) {
	case int:
		if bv, ok := b.(float64); ok {
			return float64(av) == bv
		}
	case float64:
		if bv, ok := b.(int); ok {
			return av == float64(bv)
		}
	}

	return false
}
