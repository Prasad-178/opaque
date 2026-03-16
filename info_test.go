package opaque

import (
	"context"
	"errors"
	"testing"
)

// --- Fast tests (no HE) ---

func TestHas_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	if db.Has(ctx, "doc-1") {
		t.Error("Has should return false for empty DB")
	}

	db.Add(ctx, "doc-1", []float64{1, 0, 0, 0})

	if !db.Has(ctx, "doc-1") {
		t.Error("Has should return true for pending vector")
	}
	if db.Has(ctx, "doc-2") {
		t.Error("Has should return false for non-existent vector")
	}
}

func TestGet_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	_, err = db.Get(ctx, "doc-1")
	if !errors.Is(err, ErrNotReady) {
		t.Fatalf("expected ErrNotReady, got %v", err)
	}
}

func TestGet_EmptyID(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	_, err = db.Get(context.Background(), "")
	if !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}
}

func TestCount_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	if c := db.Count(ctx); c != 0 {
		t.Errorf("Count before Build = %d, want 0", c)
	}
}

func TestList_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	_, err = db.List(context.Background(), 0, 10)
	if !errors.Is(err, ErrNotReady) {
		t.Fatalf("expected ErrNotReady, got %v", err)
	}
}

func TestGetConfig(t *testing.T) {
	db, err := NewDB(Config{Dimension: 128, NumClusters: 32})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	cfg := db.GetConfig()
	if cfg.Dimension != 128 {
		t.Errorf("Dimension = %d, want 128", cfg.Dimension)
	}
	if cfg.NumClusters != 32 {
		t.Errorf("NumClusters = %d, want 32", cfg.NumClusters)
	}
}

func TestStats_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	db.Add(ctx, "a", []float64{1, 0, 0, 0})
	db.Add(ctx, "b", []float64{0, 1, 0, 0})

	stats := db.Stats(ctx)
	if stats.TotalVectors != 2 {
		t.Errorf("TotalVectors = %d, want 2", stats.TotalVectors)
	}
	if stats.IndexedVectors != 0 {
		t.Errorf("IndexedVectors = %d, want 0", stats.IndexedVectors)
	}
	if stats.PendingVectors != 2 {
		t.Errorf("PendingVectors = %d, want 2", stats.PendingVectors)
	}
	if stats.IsReady {
		t.Error("IsReady should be false before Build")
	}
}

// --- Integration tests (with HE) ---

func TestInfoAPIs_AfterBuild(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 200
		dim        = 128
		clusters   = 8
	)

	ctx := context.Background()
	db, err := NewDB(Config{
		Dimension:      dim,
		NumClusters:    clusters,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Test Has
	if !db.Has(ctx, "doc-0") {
		t.Error("Has(doc-0) should be true after Build")
	}
	if db.Has(ctx, "nonexistent") {
		t.Error("Has(nonexistent) should be false")
	}

	// Test Get — retrieve and verify a vector
	vec, err := db.Get(ctx, "doc-0")
	if err != nil {
		t.Fatalf("Get(doc-0): %v", err)
	}
	if len(vec) != dim {
		t.Errorf("Get returned %d dimensions, want %d", len(vec), dim)
	}
	// Verify values match (within float precision, since we may have normalized storage)
	// With NormalizedStorage=true, the stored vector is unit-normalized.
	// The original vector is already normalized by generateTestVectors, so they should match.
	dotProduct := 0.0
	for i := range vec {
		dotProduct += vec[i] * vectors[0][i]
	}
	if dotProduct < 0.99 {
		t.Errorf("Get(doc-0) cosine similarity with original = %f, want >= 0.99", dotProduct)
	}

	// Test Get not found
	_, err = db.Get(ctx, "nonexistent")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}

	// Test Count
	count := db.Count(ctx)
	if count < numVectors {
		t.Errorf("Count = %d, want >= %d", count, numVectors)
	}

	// Test List
	page, err := db.List(ctx, 0, 10)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(page) != 10 {
		t.Errorf("List(0,10) returned %d items, want 10", len(page))
	}
	// Verify sorted order
	for i := 1; i < len(page); i++ {
		if page[i] < page[i-1] {
			t.Errorf("List not sorted: %q < %q", page[i], page[i-1])
		}
	}

	// Test List pagination
	page2, err := db.List(ctx, 10, 10)
	if err != nil {
		t.Fatalf("List(10,10): %v", err)
	}
	if len(page2) != 10 {
		t.Errorf("List(10,10) returned %d items, want 10", len(page2))
	}
	if len(page) > 0 && len(page2) > 0 && page2[0] <= page[len(page)-1] {
		t.Errorf("pages overlap: page1 ends with %q, page2 starts with %q", page[len(page)-1], page2[0])
	}

	// Test List beyond end
	beyond, err := db.List(ctx, 10000, 10)
	if err != nil {
		t.Fatalf("List(10000,10): %v", err)
	}
	if len(beyond) != 0 {
		t.Errorf("List beyond end returned %d items, want 0", len(beyond))
	}

	// Test Stats
	stats := db.Stats(ctx)
	if !stats.IsReady {
		t.Error("Stats.IsReady should be true after Build")
	}
	if stats.IndexedVectors < numVectors {
		t.Errorf("IndexedVectors = %d, want >= %d", stats.IndexedVectors, numVectors)
	}
	if stats.ClusterStats.NumClusters != clusters {
		t.Errorf("ClusterStats.NumClusters = %d, want %d", stats.ClusterStats.NumClusters, clusters)
	}

	// Test GetConfig
	cfg := db.GetConfig()
	if cfg.Dimension != dim {
		t.Errorf("GetConfig().Dimension = %d, want %d", cfg.Dimension, dim)
	}
}
