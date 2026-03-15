//go:build integration

package test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/opaque/opaque/go"
)

// generateE2EVectors creates n normalized random vectors.
func generateE2EVectors(n, dim int, seed int64) ([]string, [][]float64) {
	rng := rand.New(rand.NewSource(seed))
	ids := make([]string, n)
	vecs := make([][]float64, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("vec-%d", i)
		v := make([]float64, dim)
		var norm float64
		for j := range v {
			v[j] = rng.Float64()*2 - 1
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		vecs[i] = v
	}
	return ids, vecs
}

// TestE2E_FullSDKLifecycle exercises the complete SDK workflow:
// NewDB → Add → Build → Search → Save → Load → Search → Delete → Update → Rebuild → Search
func TestE2E_FullSDKLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	const (
		numVectors = 500
		dim        = 128
		clusters   = 16
		topK       = 10
	)

	ctx := context.Background()
	tmpDir := t.TempDir()
	savePath := filepath.Join(tmpDir, "saved-db")

	// === Phase 1: Create, Add, Build, Search ===
	t.Log("Phase 1: Create → Add → Build → Search")

	db, err := opaque.NewDB(opaque.Config{
		Dimension:      dim,
		NumClusters:    clusters,
		TopClusters:    8,
		NumDecoys:      4,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}

	ids, vectors := generateE2EVectors(numVectors, dim, 42)

	// Add first 250 without metadata.
	if err := db.AddBatch(ctx, ids[:250], vectors[:250]); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	// Add next 250 with metadata.
	metadatas := make([]opaque.Metadata, 250)
	for i := range metadatas {
		metadatas[i] = opaque.Metadata{
			"category": fmt.Sprintf("cat-%d", i%5),
			"priority": i % 3,
		}
	}
	if err := db.AddBatchWithMetadata(ctx, ids[250:], vectors[250:], metadatas); err != nil {
		t.Fatalf("AddBatchWithMetadata: %v", err)
	}

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if !db.IsReady() {
		t.Fatal("DB should be ready after Build")
	}

	// Search: self-match should be top result.
	results, err := db.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	assertSelfMatch(t, results, "vec-42")

	t.Logf("  Search returned %d results, top: %s (%.4f)", len(results), results[0].ID, results[0].Score)

	// === Phase 2: Info APIs ===
	t.Log("Phase 2: Info APIs")

	if !db.Has(ctx, "vec-0") {
		t.Error("Has(vec-0) should be true")
	}
	if db.Has(ctx, "nonexistent") {
		t.Error("Has(nonexistent) should be false")
	}

	vec, err := db.Get(ctx, "vec-0")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(vec) != dim {
		t.Errorf("Get returned %d dims, want %d", len(vec), dim)
	}

	count := db.Count(ctx)
	if count < numVectors {
		t.Errorf("Count = %d, want >= %d", count, numVectors)
	}

	page, err := db.List(ctx, 0, 20)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(page) != 20 {
		t.Errorf("List returned %d, want 20", len(page))
	}

	stats := db.Stats(ctx)
	if !stats.IsReady {
		t.Error("Stats.IsReady should be true")
	}
	if stats.ClusterStats.NumClusters != clusters {
		t.Errorf("NumClusters = %d, want %d", stats.ClusterStats.NumClusters, clusters)
	}

	cfg := db.GetConfig()
	if cfg.Dimension != dim {
		t.Errorf("GetConfig().Dimension = %d, want %d", cfg.Dimension, dim)
	}

	t.Logf("  Count=%d, Clusters=%d, Size=%d", count, stats.ClusterStats.NumClusters, db.Size())

	// === Phase 3: Filtered Search ===
	t.Log("Phase 3: Filtered Search")

	filtered, err := db.SearchWithFilter(ctx, vectors[300], topK, opaque.Filter{
		Where: map[string]any{"category": "cat-0"},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter: %v", err)
	}
	for _, r := range filtered {
		meta, _ := db.GetMetadata(ctx, r.ID)
		if meta != nil && meta["category"] != "cat-0" {
			t.Errorf("filtered result %s has category %v, want cat-0", r.ID, meta["category"])
		}
	}

	t.Logf("  Filtered search: %d results matching category=cat-0", len(filtered))

	// === Phase 4: Save → Load → Search ===
	t.Log("Phase 4: Save → Load → Search")

	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	loaded, err := opaque.Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if !loaded.IsReady() {
		t.Fatal("loaded DB should be ready")
	}

	loadedResults, err := loaded.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search on loaded: %v", err)
	}
	assertSelfMatch(t, loadedResults, "vec-42")

	// Metadata survives Save/Load.
	meta, err := loaded.GetMetadata(ctx, ids[300])
	if err != nil {
		t.Fatalf("GetMetadata on loaded: %v", err)
	}
	if meta["category"] != "cat-0" {
		t.Errorf("loaded metadata category = %v, want cat-0", meta["category"])
	}

	t.Logf("  Loaded DB search: top=%s (%.4f), metadata preserved", loadedResults[0].ID, loadedResults[0].Score)

	// === Phase 5: Delete → Search ===
	t.Log("Phase 5: Delete → Search")

	if err := loaded.Delete(ctx, "vec-42"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if loaded.Has(ctx, "vec-42") {
		t.Error("Has(vec-42) should be false after delete")
	}

	_, err = loaded.Get(ctx, "vec-42")
	if !errors.Is(err, opaque.ErrNotFound) {
		t.Errorf("Get(vec-42) after delete: expected ErrNotFound, got %v", err)
	}

	afterDelete, err := loaded.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search after delete: %v", err)
	}
	for _, r := range afterDelete {
		if r.ID == "vec-42" {
			t.Error("vec-42 should not appear in search after delete")
		}
	}

	t.Log("  vec-42 deleted and filtered from search")

	// === Phase 6: Add + Update → Rebuild → Search ===
	t.Log("Phase 6: Add + Update → Rebuild → Search")

	// Add a new vector.
	if err := loaded.Add(ctx, "new-vec", vectors[0]); err != nil {
		t.Fatalf("Add after load: %v", err)
	}

	// Update an existing vector with a different vector.
	if err := loaded.Update(ctx, "vec-100", vectors[0]); err != nil {
		t.Fatalf("Update: %v", err)
	}

	if err := loaded.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	if !loaded.Has(ctx, "new-vec") {
		t.Error("new-vec should exist after Rebuild")
	}
	if loaded.Has(ctx, "vec-42") {
		t.Error("vec-42 should still be deleted after Rebuild")
	}
	if !loaded.Has(ctx, "vec-100") {
		t.Error("vec-100 should exist after Update+Rebuild")
	}

	rebuildResults, err := loaded.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search after Rebuild: %v", err)
	}
	if len(rebuildResults) == 0 {
		t.Fatal("no results after Rebuild")
	}

	t.Logf("  After Rebuild: %d results, top=%s (%.4f)", len(rebuildResults), rebuildResults[0].ID, rebuildResults[0].Score)

	// === Phase 7: Typed Errors ===
	t.Log("Phase 7: Typed Errors")

	err = loaded.Add(ctx, "", vectors[0])
	if !errors.Is(err, opaque.ErrEmptyID) {
		t.Errorf("expected ErrEmptyID, got %v", err)
	}

	err = loaded.Add(ctx, "x", make([]float64, 64))
	if !errors.Is(err, opaque.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}

	err = loaded.Delete(ctx, "nonexistent")
	if !errors.Is(err, opaque.ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}

	_, err = opaque.Load("/nonexistent/path")
	if err == nil {
		t.Error("Load nonexistent should fail")
	}

	t.Log("  All typed errors verified")

	loaded.Close()

	t.Log("=== E2E SDK lifecycle test passed ===")
}

// TestE2E_PCALifecycle tests the full lifecycle with PCA enabled.
func TestE2E_PCALifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	const (
		numVectors = 300
		dim        = 128
		pcaDim     = 64
		clusters   = 16
		topK       = 10
	)

	ctx := context.Background()

	db, err := opaque.NewDB(opaque.Config{
		Dimension:      dim,
		PCADimension:   pcaDim,
		NumClusters:    clusters,
		TopClusters:    8,
		NumDecoys:      4,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}

	ids, vectors := generateE2EVectors(numVectors, dim, 77)

	metadatas := make([]opaque.Metadata, numVectors)
	for i := range metadatas {
		metadatas[i] = opaque.Metadata{"source": "pca-test", "idx": i}
	}
	db.AddBatchWithMetadata(ctx, ids, vectors, metadatas)

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build with PCA: %v", err)
	}

	// Search works with original-dimension query.
	results, err := db.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}

	// Save → Load with PCA.
	savePath := filepath.Join(t.TempDir(), "pca-db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Verify pca.gob exists.
	if _, err := os.Stat(filepath.Join(savePath, "pca.gob")); err != nil {
		t.Fatalf("pca.gob not found: %v", err)
	}
	db.Close()

	loaded, err := opaque.Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	loadedResults, err := loaded.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search on loaded PCA DB: %v", err)
	}
	if len(loadedResults) == 0 {
		t.Fatal("no results on loaded PCA DB")
	}

	// Metadata survives.
	meta, err := loaded.GetMetadata(ctx, "vec-42")
	if err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if meta["source"] != "pca-test" {
		t.Errorf("metadata source = %v, want pca-test", meta["source"])
	}

	// Filtered search works.
	filtered, err := loaded.SearchWithFilter(ctx, vectors[42], topK, opaque.Filter{
		Where: map[string]any{"source": "pca-test"},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter: %v", err)
	}
	if len(filtered) == 0 {
		t.Fatal("no filtered results on PCA DB")
	}

	t.Logf("PCA e2e: %dD→%dD, search: %d results, filtered: %d results",
		dim, pcaDim, len(loadedResults), len(filtered))
}

// TestE2E_FileStorageLifecycle tests with file-backed storage.
func TestE2E_FileStorageLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	const (
		numVectors = 200
		dim        = 128
		clusters   = 8
		topK       = 5
	)

	ctx := context.Background()
	tmpDir := t.TempDir()

	db, err := opaque.NewDB(opaque.Config{
		Dimension:      dim,
		NumClusters:    clusters,
		WorkerPoolSize: 2,
		Storage:        opaque.File,
		StoragePath:    filepath.Join(tmpDir, "storage"),
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}

	ids, vectors := generateE2EVectors(numVectors, dim, 99)
	db.AddBatch(ctx, ids, vectors)

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	assertSelfMatch(t, results, "vec-0")

	// Save to a different directory.
	savePath := filepath.Join(tmpDir, "saved")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	// Load from saved directory.
	loaded, err := opaque.Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	loadedResults, err := loaded.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search on loaded: %v", err)
	}
	assertSelfMatch(t, loadedResults, "vec-0")

	t.Logf("File storage e2e: %d results before save, %d after load",
		len(results), len(loadedResults))
}

func assertSelfMatch(t *testing.T, results []opaque.Result, expectedID string) {
	t.Helper()
	if len(results) == 0 {
		t.Fatalf("no results returned")
	}
	found := false
	for _, r := range results {
		if r.ID == expectedID {
			found = true
			if r.Score < 0.95 {
				t.Errorf("self-match score for %s = %.4f, want >= 0.95", expectedID, r.Score)
			}
			break
		}
	}
	if !found {
		t.Errorf("%s not found in results", expectedID)
	}
}
