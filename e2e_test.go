package opaque

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"
)

// TestE2E is a comprehensive end-to-end test that exercises every major feature
// of the Opaque library in a single flow:
//
//   - DB lifecycle: NewDB → Add → Build → Search → Close
//   - Incremental indexing: Add after Build (train-once-assign-many)
//   - Batch operations: AddBatch, AddBatchWithMetadata
//   - Mutations: Delete, Update, Rebuild
//   - Metadata: AddWithMetadata, GetMetadata, SearchWithFilter
//   - Queries: Search, Size, Count, Has, Get, List, Stats, ClusterStats, GetConfig
//   - Persistence: Save → Load → Search → Add → Rebuild
//   - Concurrent search safety
//   - Error handling: sentinel errors for invalid operations
//   - PCA dimensionality reduction
//   - File storage backend
func TestE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	ctx := context.Background()

	const (
		dim      = 128
		clusters = 8
		topK     = 5
	)

	// ── Phase 1: DB Creation & Config ─────────────────────────────────

	t.Run("Phase1_Creation", func(t *testing.T) {
		// Valid creation.
		db, err := NewDB(Config{Dimension: dim})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		cfg := db.GetConfig()
		if cfg.Dimension != dim {
			t.Errorf("Dimension = %d, want %d", cfg.Dimension, dim)
		}
		if cfg.NumClusters != 64 {
			t.Errorf("default NumClusters = %d, want 64", cfg.NumClusters)
		}

		// Error: zero dimension.
		_, err = NewDB(Config{})
		if err == nil {
			t.Error("expected error for zero dimension")
		}

		// Error: invalid config.
		_, err = NewDB(Config{Dimension: dim, NumClusters: 1})
		if err == nil {
			t.Error("expected error for NumClusters < 2")
		}
	})

	// ── Phase 2: Full Lifecycle ───────────────────────────────────────

	t.Run("Phase2_FullLifecycle", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		// ── State: Empty ──

		if db.IsReady() {
			t.Error("IsReady should be false before Build")
		}
		if db.Size() != 0 {
			t.Errorf("Size = %d, want 0", db.Size())
		}

		// Error: search before build.
		_, err = db.Search(ctx, make([]float64, dim), topK)
		if err == nil {
			t.Error("Search before Build should return error")
		}

		// Error: build with no vectors.
		err = db.Build(ctx)
		if !errors.Is(err, ErrNoVectors) {
			t.Errorf("Build with no vectors: got %v, want ErrNoVectors", err)
		}

		// ── Add vectors ──

		ids, vectors := generateTestVectors(100, dim, 42)

		// Error: wrong dimension.
		err = db.Add(ctx, "bad", make([]float64, dim+1))
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("Add wrong dim: got %v, want ErrDimensionMismatch", err)
		}

		// Error: empty ID.
		err = db.Add(ctx, "", vectors[0])
		if !errors.Is(err, ErrEmptyID) {
			t.Errorf("Add empty ID: got %v, want ErrEmptyID", err)
		}

		// Single add.
		if err := db.Add(ctx, ids[0], vectors[0]); err != nil {
			t.Fatalf("Add: %v", err)
		}
		if db.Size() != 1 {
			t.Errorf("Size after 1 Add = %d, want 1", db.Size())
		}

		// Batch add (remaining 99 vectors).
		if err := db.AddBatch(ctx, ids[1:], vectors[1:]); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if db.Size() != 100 {
			t.Errorf("Size after AddBatch = %d, want 100", db.Size())
		}

		// ── Build ──

		var progressPhases []string
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		if !db.IsReady() {
			t.Error("IsReady should be true after Build")
		}

		// ── ClusterStats ──

		stats := db.ClusterStats()
		if stats.NumClusters != clusters {
			t.Errorf("ClusterStats.NumClusters = %d, want %d", stats.NumClusters, clusters)
		}
		if stats.EmptyClusters != 0 {
			t.Errorf("ClusterStats.EmptyClusters = %d, want 0", stats.EmptyClusters)
		}
		if stats.MinSize <= 0 {
			t.Errorf("ClusterStats.MinSize = %d, want > 0", stats.MinSize)
		}
		if stats.Iterations <= 0 {
			t.Errorf("ClusterStats.Iterations = %d, want > 0", stats.Iterations)
		}

		// ── Search ──

		results, err := db.Search(ctx, vectors[0], topK)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search returned no results")
		}

		// Self-match should be the top result (or very close).
		if results[0].ID != ids[0] {
			// Due to normalized storage, the top result might not be exact self-match
			// but it should be in the results.
			found := false
			for _, r := range results {
				if r.ID == ids[0] {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Self-match %q not found in top-%d results", ids[0], topK)
			}
		}

		// Scores should be descending.
		for i := 1; i < len(results); i++ {
			if results[i].Score > results[i-1].Score+1e-9 {
				t.Errorf("Results not sorted: score[%d]=%.4f > score[%d]=%.4f",
					i, results[i].Score, i-1, results[i-1].Score)
			}
		}

		// Error: search with wrong dimension.
		_, err = db.Search(ctx, make([]float64, dim+1), topK)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("Search wrong dim: got %v, want ErrDimensionMismatch", err)
		}

		// ── Info methods ──

		if !db.Has(ctx, ids[0]) {
			t.Error("Has(ids[0]) = false, want true")
		}
		if db.Has(ctx, "nonexistent") {
			t.Error("Has(nonexistent) = true, want false")
		}

		vec, err := db.Get(ctx, ids[0])
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if len(vec) != dim {
			t.Errorf("Get returned dim %d, want %d", len(vec), dim)
		}

		count := db.Count(ctx)
		if count != 100 {
			t.Errorf("Count = %d, want 100", count)
		}

		listed, err := db.List(ctx, 0, 10)
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		if len(listed) != 10 {
			t.Errorf("List(0,10) returned %d items, want 10", len(listed))
		}

		dbStats := db.Stats(ctx)
		if !dbStats.IsReady {
			t.Error("Stats.IsReady = false")
		}
		if dbStats.IndexedVectors != 100 {
			t.Errorf("Stats.IndexedVectors = %d, want 100", dbStats.IndexedVectors)
		}

		// ── Incremental Add (post-Build) ──

		newVec := make([]float64, dim)
		copy(newVec, vectors[0])
		newVec[0] += 0.01 // slight perturbation
		if err := db.Add(ctx, "incremental-1", newVec); err != nil {
			t.Fatalf("Add after Build: %v", err)
		}

		// Should be immediately searchable.
		results, err = db.Search(ctx, newVec, topK)
		if err != nil {
			t.Fatalf("Search after incremental add: %v", err)
		}
		found := false
		for _, r := range results {
			if r.ID == "incremental-1" {
				found = true
				break
			}
		}
		if !found {
			t.Error("Incrementally added vector not found in search results")
		}

		// Incremental batch add.
		batchIDs := []string{"inc-batch-1", "inc-batch-2"}
		batchVecs := [][]float64{vectors[1], vectors[2]}
		if err := db.AddBatch(ctx, batchIDs, batchVecs); err != nil {
			t.Fatalf("AddBatch after Build: %v", err)
		}
		// Verify one is searchable.
		results, err = db.Search(ctx, vectors[1], topK)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		found = false
		for _, r := range results {
			if r.ID == "inc-batch-1" {
				found = true
				break
			}
		}
		if !found {
			t.Error("Batch-added vector 'inc-batch-1' not found after incremental add")
		}

		// ── Delete ──

		if err := db.Delete(ctx, ids[1]); err != nil {
			t.Fatalf("Delete: %v", err)
		}

		// Deleted vector should not appear in search.
		results, err = db.Search(ctx, vectors[1], 50)
		if err != nil {
			t.Fatalf("Search after delete: %v", err)
		}
		for _, r := range results {
			if r.ID == ids[1] {
				t.Errorf("Deleted vector %q still in search results", ids[1])
			}
		}

		// Has should return false for deleted.
		if db.Has(ctx, ids[1]) {
			t.Error("Has(deleted) = true, want false")
		}

		// Delete nonexistent.
		err = db.Delete(ctx, "nonexistent")
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("Delete nonexistent: got %v, want ErrNotFound", err)
		}

		// Delete empty ID.
		err = db.Delete(ctx, "")
		if !errors.Is(err, ErrEmptyID) {
			t.Errorf("Delete empty ID: got %v, want ErrEmptyID", err)
		}

		// ── Update ──

		updatedVec := make([]float64, dim)
		copy(updatedVec, vectors[2])
		updatedVec[0] += 0.5 // bigger perturbation
		if err := db.Update(ctx, ids[2], updatedVec); err != nil {
			t.Fatalf("Update: %v", err)
		}

		// The old vector for ids[2] should be soft-deleted.
		// It won't appear in search because it's in deletedIDs.

		// Update nonexistent.
		err = db.Update(ctx, "nonexistent", vectors[0])
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("Update nonexistent: got %v, want ErrNotFound", err)
		}

		// ── Rebuild (re-clusters everything) ──

		if err := db.Rebuild(ctx); err != nil {
			t.Fatalf("Rebuild: %v", err)
		}
		if !db.IsReady() {
			t.Error("IsReady after Rebuild = false")
		}

		// After rebuild, deleted vector should still be gone.
		results, err = db.Search(ctx, vectors[1], 50)
		if err != nil {
			t.Fatalf("Search after Rebuild: %v", err)
		}
		for _, r := range results {
			if r.ID == ids[1] {
				t.Error("Deleted vector still in results after Rebuild")
			}
		}

		// Incremental vectors should still be findable.
		results, err = db.Search(ctx, newVec, topK)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		found = false
		for _, r := range results {
			if r.ID == "incremental-1" {
				found = true
				break
			}
		}
		if !found {
			t.Error("Incremental vector lost after Rebuild")
		}

		// ── Concurrent Search ──

		var wg sync.WaitGroup
		errs := make(chan error, 10)
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				_, err := db.Search(ctx, vectors[i%len(vectors)], topK)
				if err != nil {
					errs <- fmt.Errorf("concurrent search %d: %w", i, err)
				}
			}(i)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}

		_ = progressPhases // used above
	})

	// ── Phase 3: Metadata ─────────────────────────────────────────────

	t.Run("Phase3_Metadata", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		// Add vectors with metadata.
		ids, vectors := generateTestVectors(60, dim, 77)

		metadatas := make([]Metadata, len(ids))
		for i := range ids {
			category := "science"
			if i%3 == 0 {
				category = "math"
			} else if i%3 == 1 {
				category = "history"
			}
			metadatas[i] = Metadata{
				"category": category,
				"index":    i,
				"active":   true,
			}
		}

		if err := db.AddBatchWithMetadata(ctx, ids, vectors, metadatas); err != nil {
			t.Fatalf("AddBatchWithMetadata: %v", err)
		}

		// Also add a single vector with metadata.
		singleMeta := Metadata{"category": "math", "special": true}
		singleVec := make([]float64, dim)
		copy(singleVec, vectors[0])
		singleVec[0] += 0.01
		if err := db.AddWithMetadata(ctx, "meta-single", singleVec, singleMeta); err != nil {
			t.Fatalf("AddWithMetadata: %v", err)
		}

		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		// GetMetadata.
		meta, err := db.GetMetadata(ctx, ids[0])
		if err != nil {
			t.Fatalf("GetMetadata: %v", err)
		}
		if meta["category"] != "math" {
			t.Errorf("GetMetadata category = %v, want 'math'", meta["category"])
		}

		// GetMetadata for single-added vector.
		meta, err = db.GetMetadata(ctx, "meta-single")
		if err != nil {
			t.Fatalf("GetMetadata(meta-single): %v", err)
		}
		if meta["special"] != true {
			t.Errorf("GetMetadata special = %v, want true", meta["special"])
		}

		// GetMetadata for nonexistent.
		_, err = db.GetMetadata(ctx, "nonexistent")
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("GetMetadata nonexistent: got %v, want ErrNotFound", err)
		}

		// SearchWithFilter — exact match.
		results, err := db.SearchWithFilter(ctx, vectors[0], 10, Filter{
			Where: map[string]any{"category": "math"},
		})
		if err != nil {
			t.Fatalf("SearchWithFilter: %v", err)
		}

		// All results should have category == "math".
		for _, r := range results {
			meta, err := db.GetMetadata(ctx, r.ID)
			if err != nil {
				t.Errorf("GetMetadata(%s): %v", r.ID, err)
				continue
			}
			if meta["category"] != "math" {
				t.Errorf("SearchWithFilter result %s has category=%v, want 'math'", r.ID, meta["category"])
			}
		}

		// SearchWithFilter — multiple conditions (AND).
		results, err = db.SearchWithFilter(ctx, vectors[0], 10, Filter{
			Where: map[string]any{"category": "math", "active": true},
		})
		if err != nil {
			t.Fatalf("SearchWithFilter multi: %v", err)
		}
		// Should still work — all math vectors have active=true.
		if len(results) == 0 {
			t.Error("SearchWithFilter with AND conditions returned no results")
		}

		// SearchWithFilter — no matching filter.
		results, err = db.SearchWithFilter(ctx, vectors[0], 10, Filter{
			Where: map[string]any{"category": "nonexistent-category"},
		})
		if err != nil {
			t.Fatalf("SearchWithFilter no match: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("SearchWithFilter with impossible filter returned %d results", len(results))
		}

		// SearchWithFilter — empty filter returns all.
		results, err = db.SearchWithFilter(ctx, vectors[0], topK, Filter{})
		if err != nil {
			t.Fatalf("SearchWithFilter empty: %v", err)
		}
		if len(results) == 0 {
			t.Error("SearchWithFilter with empty filter returned no results")
		}
	})

	// ── Phase 4: Persistence (Save/Load) ──────────────────────────────

	t.Run("Phase4_Persistence", func(t *testing.T) {
		saveDir := t.TempDir()
		savePath := saveDir + "/opaque-save"

		ids, vectors := generateTestVectors(80, dim, 55)

		// Build and save.
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}

		// Add some with metadata for persistence test.
		metas := make([]Metadata, len(ids))
		for i := range ids {
			metas[i] = Metadata{"source": "test", "idx": i}
		}
		if err := db.AddBatchWithMetadata(ctx, ids, vectors, metas); err != nil {
			t.Fatalf("AddBatchWithMetadata: %v", err)
		}

		// Delete one before saving.
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}
		if err := db.Delete(ctx, ids[0]); err != nil {
			t.Fatalf("Delete: %v", err)
		}

		// Save.
		if err := db.Save(savePath); err != nil {
			t.Fatalf("Save: %v", err)
		}

		// Save to same path again should fail.
		err = db.Save(savePath)
		if err == nil {
			t.Error("expected error saving to existing path")
		}

		origStats := db.ClusterStats()
		db.Close()

		// Load.
		loaded, err := Load(savePath)
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer loaded.Close()

		if !loaded.IsReady() {
			t.Error("Loaded DB not ready")
		}

		// ClusterStats should be preserved.
		loadedStats := loaded.ClusterStats()
		if loadedStats.NumClusters != origStats.NumClusters {
			t.Errorf("Loaded NumClusters = %d, want %d", loadedStats.NumClusters, origStats.NumClusters)
		}

		// Search should work.
		results, err := loaded.Search(ctx, vectors[5], topK)
		if err != nil {
			t.Fatalf("Search on loaded DB: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search on loaded DB returned no results")
		}

		// Deleted vector should still be excluded.
		if loaded.Has(ctx, ids[0]) {
			t.Error("Deleted vector exists after Load")
		}

		// Metadata should survive persistence.
		meta, err := loaded.GetMetadata(ctx, ids[5])
		if err != nil {
			t.Fatalf("GetMetadata on loaded DB: %v", err)
		}
		if meta["source"] != "test" {
			t.Errorf("Loaded metadata source = %v, want 'test'", meta["source"])
		}

		// Add new vectors to loaded DB and rebuild.
		newIDs, newVecs := generateTestVectors(20, dim, 200)
		for i := range newIDs {
			newIDs[i] = fmt.Sprintf("post-load-%d", i)
		}
		if err := loaded.AddBatch(ctx, newIDs, newVecs); err != nil {
			t.Fatalf("AddBatch on loaded: %v", err)
		}
		if err := loaded.Rebuild(ctx); err != nil {
			t.Fatalf("Rebuild on loaded: %v", err)
		}

		// New vectors should be searchable.
		results, err = loaded.Search(ctx, newVecs[0], topK)
		if err != nil {
			t.Fatalf("Search after Rebuild on loaded: %v", err)
		}
		found := false
		for _, r := range results {
			if r.ID == "post-load-0" {
				found = true
				break
			}
		}
		if !found {
			t.Error("Post-load vector not found after Rebuild")
		}

		// Load from nonexistent path.
		_, err = Load("/nonexistent/path")
		if err == nil {
			t.Error("expected error loading from nonexistent path")
		}
	})

	// ── Phase 5: File Storage ─────────────────────────────────────────

	t.Run("Phase5_FileStorage", func(t *testing.T) {
		tmpDir := t.TempDir()

		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
			Storage:        File,
			StoragePath:    tmpDir + "/vectors",
		})
		if err != nil {
			t.Fatalf("NewDB with file storage: %v", err)
		}
		defer db.Close()

		ids, vectors := generateTestVectors(50, dim, 123)
		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		// Verify files exist on disk.
		entries, err := os.ReadDir(tmpDir + "/vectors")
		if err != nil {
			t.Fatalf("ReadDir: %v", err)
		}
		if len(entries) == 0 {
			t.Error("No files in storage directory")
		}

		// Search should work with file backend.
		results, err := db.Search(ctx, vectors[0], topK)
		if err != nil {
			t.Fatalf("Search with file storage: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search with file storage returned no results")
		}

		// Get should decrypt from file.
		vec, err := db.Get(ctx, ids[0])
		if err != nil {
			t.Fatalf("Get with file storage: %v", err)
		}
		if len(vec) != dim {
			t.Errorf("Get returned dim %d, want %d", len(vec), dim)
		}
	})

	// ── Phase 6: PCA Dimensionality Reduction ─────────────────────────

	t.Run("Phase6_PCA", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
			PCADimension:   64, // reduce 128 → 64
		})
		if err != nil {
			t.Fatalf("NewDB with PCA: %v", err)
		}
		defer db.Close()

		ids, vectors := generateTestVectors(200, dim, 88)
		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build with PCA: %v", err)
		}

		dbStats := db.Stats(ctx)
		if !dbStats.HasPCA {
			t.Error("Stats.HasPCA = false after PCA build")
		}

		// Search should still work (query is 128-dim, PCA transforms internally).
		results, err := db.Search(ctx, vectors[0], topK)
		if err != nil {
			t.Fatalf("Search with PCA: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search with PCA returned no results")
		}

		// Self-match should be findable.
		found := false
		for _, r := range results {
			if r.ID == ids[0] {
				found = true
				break
			}
		}
		if !found {
			t.Error("Self-match not found with PCA enabled")
		}
	})

	// ── Phase 7: Progress Callback ────────────────────────────────────

	t.Run("Phase7_ProgressCallback", func(t *testing.T) {
		var mu sync.Mutex
		phases := make(map[string]bool)

		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
			OnBuildProgress: func(phase string, pct float64) {
				mu.Lock()
				phases[phase] = true
				mu.Unlock()
				if pct < 0 || pct > 1 {
					t.Errorf("Progress pct out of range: %f", pct)
				}
			},
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		ids, vectors := generateTestVectors(50, dim, 99)
		db.AddBatch(ctx, ids, vectors)
		db.Build(ctx)

		mu.Lock()
		if !phases["clustering"] {
			t.Error("Missing 'clustering' progress phase")
		}
		if !phases["encrypting"] {
			t.Error("Missing 'encrypting' progress phase")
		}
		if !phases["indexing"] {
			t.Error("Missing 'indexing' progress phase")
		}
		mu.Unlock()
	})

	// ── Phase 8: Auto-Index Lifecycle ─────────────────────────────────

	t.Run("Phase8_AutoIndex", func(t *testing.T) {
		var autoErr error
		var errMu sync.Mutex

		db, err := NewDB(Config{
			Dimension:        dim,
			NumClusters:      clusters,
			TopClusters:      4,
			NumDecoys:        2,
			WorkerPoolSize:   2,
			AutoIndexEnabled: true,
			AutoIndexInterval: 500 * time.Millisecond,
			AutoIndexMinChanges: 1,
			OnAutoIndexError: func(err error) {
				errMu.Lock()
				autoErr = err
				errMu.Unlock()
			},
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		ids, vectors := generateTestVectors(50, dim, 111)
		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}

		// Wait for auto-build to trigger.
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if db.IsReady() {
				break
			}
			time.Sleep(200 * time.Millisecond)
		}

		if !db.IsReady() {
			t.Fatal("Auto-index did not build within timeout")
		}

		// Search should work.
		results, err := db.Search(ctx, vectors[0], topK)
		if err != nil {
			t.Fatalf("Search after auto-build: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search after auto-build returned no results")
		}

		errMu.Lock()
		if autoErr != nil {
			t.Errorf("Auto-index error: %v", autoErr)
		}
		errMu.Unlock()
	})

	// ── Phase 9: Redundant Assignments ────────────────────────────────

	t.Run("Phase9_RedundantAssignments", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:            dim,
			NumClusters:          clusters,
			TopClusters:          4,
			NumDecoys:            2,
			WorkerPoolSize:       2,
			RedundantAssignments: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		ids, vectors := generateTestVectors(80, dim, 66)
		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		// With redundant assignments, Count should be > number of unique vectors
		// (each vector stored in 2 clusters).
		count := db.Count(ctx)
		if count <= len(ids) {
			t.Errorf("Count with redundant=2 should be > %d, got %d", len(ids), count)
		}

		// Search should still work fine.
		results, err := db.Search(ctx, vectors[0], topK)
		if err != nil {
			t.Fatalf("Search with redundant: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("Search with redundant returned no results")
		}
	})

	// ── Phase 10: Edge Cases ──────────────────────────────────────────

	t.Run("Phase10_EdgeCases", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    4,
			NumDecoys:      2,
			WorkerPoolSize: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}

		// Get before build.
		_, err = db.Get(ctx, "foo")
		if !errors.Is(err, ErrNotReady) {
			t.Errorf("Get before Build: got %v, want ErrNotReady", err)
		}

		// List before build.
		_, err = db.List(ctx, 0, 10)
		if !errors.Is(err, ErrNotReady) {
			t.Errorf("List before Build: got %v, want ErrNotReady", err)
		}

		// Save before build.
		err = db.Save(t.TempDir())
		if !errors.Is(err, ErrNotReady) {
			t.Errorf("Save before Build: got %v, want ErrNotReady", err)
		}

		// Build, then search with topK=0.
		ids, vectors := generateTestVectors(30, dim, 222)
		db.AddBatch(ctx, ids, vectors)
		db.Build(ctx)
		_, err = db.Search(ctx, vectors[0], 0)
		if err == nil {
			t.Error("expected error for topK=0")
		}

		// Close and verify operations fail.
		db.Close()
		if db.IsReady() {
			t.Error("IsReady after Close = true")
		}
	})

	// ── Phase 11: Search Quality Sanity Check ─────────────────────────

	t.Run("Phase11_SearchQuality", func(t *testing.T) {
		db, err := NewDB(Config{
			Dimension:      dim,
			NumClusters:    clusters,
			TopClusters:    6, // probe more clusters for better recall
			NumDecoys:      2,
			WorkerPoolSize: 2,
		})
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}
		defer db.Close()

		rng := rand.New(rand.NewSource(42))

		// Create 200 vectors. Group 0-9 are very similar to each other.
		ids := make([]string, 200)
		vectors := make([][]float64, 200)

		// Base vector for the similar group.
		base := make([]float64, dim)
		for j := range base {
			base[j] = rng.Float64()
		}
		norm := 0.0
		for _, v := range base {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range base {
			base[j] /= norm
		}

		// First 10: small perturbations of base.
		for i := 0; i < 10; i++ {
			ids[i] = fmt.Sprintf("similar-%d", i)
			vec := make([]float64, dim)
			copy(vec, base)
			for j := range vec {
				vec[j] += rng.Float64() * 0.01 // tiny noise
			}
			// Re-normalize.
			norm = 0.0
			for _, v := range vec {
				norm += v * v
			}
			norm = math.Sqrt(norm)
			for j := range vec {
				vec[j] /= norm
			}
			vectors[i] = vec
		}

		// Remaining 190: random.
		for i := 10; i < 200; i++ {
			ids[i] = fmt.Sprintf("random-%d", i)
			vec := make([]float64, dim)
			for j := range vec {
				vec[j] = rng.Float64()*2 - 1
			}
			norm = 0.0
			for _, v := range vec {
				norm += v * v
			}
			norm = math.Sqrt(norm)
			for j := range vec {
				vec[j] /= norm
			}
			vectors[i] = vec
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}

		// Search for the base vector — top results should mostly be from the similar group.
		results, err := db.Search(ctx, base, 10)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}

		similarCount := 0
		for _, r := range results {
			for i := 0; i < 10; i++ {
				if r.ID == fmt.Sprintf("similar-%d", i) {
					similarCount++
					break
				}
			}
		}

		// At least 5 of the top 10 should be from the similar group.
		if similarCount < 5 {
			t.Errorf("Only %d/10 top results from similar group (expected >= 5)", similarCount)
			for _, r := range results {
				t.Logf("  %s (score=%.4f)", r.ID, r.Score)
			}
		}
	})
}
