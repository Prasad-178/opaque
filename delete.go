package opaque

import (
	"context"
	"fmt"
)

// Delete soft-deletes a vector by ID. The vector is excluded from future
// [DB.Search] results immediately. The underlying storage is reclaimed on
// the next [DB.Rebuild].
//
// Returns [ErrEmptyID] if the ID is empty, or [ErrNotFound] if the ID
// does not exist in either the pending vectors or the built index.
//
// Delete is safe for concurrent use with [DB.Search].
func (db *DB) Delete(ctx context.Context, id string) error {
	if id == "" {
		return ErrEmptyID
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Check that the ID exists somewhere.
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
		return ErrNotFound
	}

	if db.deletedIDs == nil {
		db.deletedIDs = make(map[string]bool)
	}
	if db.deletedIDs[id] {
		return nil
	}
	db.deletedIDs[id] = true
	db.dataVersion++
	return nil
}

// Update replaces a vector's data. This is equivalent to [DB.Delete] followed
// by [DB.Add] — the old vector is soft-deleted and the new one is buffered
// for the next [DB.Rebuild].
//
// The updated vector takes effect in search results after Rebuild.
// Until then, the old vector is excluded from search (soft-deleted) and
// the new one is pending.
//
// Returns [ErrEmptyID] if the ID is empty, [ErrNotFound] if the ID does
// not exist, or [ErrDimensionMismatch] if the vector has the wrong length.
func (db *DB) Update(ctx context.Context, id string, vector []float64) error {
	if id == "" {
		return ErrEmptyID
	}
	if len(vector) != db.cfg.Dimension {
		return fmt.Errorf("%w: got %d, want %d", ErrDimensionMismatch, len(vector), db.cfg.Dimension)
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Verify the ID exists.
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
		return ErrNotFound
	}

	// Soft-delete the old vector.
	if db.deletedIDs == nil {
		db.deletedIDs = make(map[string]bool)
	}
	db.deletedIDs[id] = true

	// Buffer the new vector — downcast to float32 for the storage tier.
	v32 := make([]float32, len(vector))
	for i, x := range vector {
		v32[i] = float32(x)
	}
	db.pendingIDs = append(db.pendingIDs, id)
	db.pendingVectors = append(db.pendingVectors, v32)
	db.dataVersion++

	return nil
}
