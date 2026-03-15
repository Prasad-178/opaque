package opaque

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

// --- Fast tests (no HE) ---

func TestMatchesFilter(t *testing.T) {
	meta := Metadata{"category": "tech", "year": 2024, "active": true, "score": 0.95}

	tests := []struct {
		name   string
		filter Filter
		want   bool
	}{
		{"empty filter matches all", Filter{}, true},
		{"single match", Filter{Where: map[string]any{"category": "tech"}}, true},
		{"single mismatch", Filter{Where: map[string]any{"category": "bio"}}, false},
		{"missing key", Filter{Where: map[string]any{"nonexistent": "x"}}, false},
		{"int match", Filter{Where: map[string]any{"year": 2024}}, true},
		{"int mismatch", Filter{Where: map[string]any{"year": 2025}}, false},
		{"bool match", Filter{Where: map[string]any{"active": true}}, true},
		{"bool mismatch", Filter{Where: map[string]any{"active": false}}, false},
		{"float match", Filter{Where: map[string]any{"score": 0.95}}, true},
		{"multi match", Filter{Where: map[string]any{"category": "tech", "year": 2024}}, true},
		{"partial mismatch", Filter{Where: map[string]any{"category": "tech", "year": 2025}}, false},
		{"nil metadata", Filter{Where: map[string]any{"category": "tech"}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := meta
			if tt.name == "nil metadata" {
				m = nil
			}
			if got := matchesFilter(m, tt.filter); got != tt.want {
				t.Errorf("matchesFilter() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValuesEqual_TypeCoercion(t *testing.T) {
	// JSON roundtrip converts int to float64.
	if !valuesEqual(42, float64(42)) {
		t.Error("int 42 should equal float64 42")
	}
	if !valuesEqual(float64(42), 42) {
		t.Error("float64 42 should equal int 42")
	}
	if valuesEqual(42, float64(43)) {
		t.Error("42 should not equal 43")
	}
}

func TestAddWithMetadata_Validation(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()

	// Wrong dimension.
	err = db.AddWithMetadata(ctx, "a", []float64{1, 0}, Metadata{"k": "v"})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("expected ErrDimensionMismatch, got %v", err)
	}

	// Empty ID.
	err = db.AddWithMetadata(ctx, "", []float64{1, 0, 0, 0}, Metadata{"k": "v"})
	if !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}

	// Valid add.
	err = db.AddWithMetadata(ctx, "a", []float64{1, 0, 0, 0}, Metadata{"category": "tech"})
	if err != nil {
		t.Fatalf("AddWithMetadata: %v", err)
	}

	if !db.Has(ctx, "a") {
		t.Error("Has(a) should be true")
	}
}

func TestGetMetadata_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	db.AddWithMetadata(ctx, "a", []float64{1, 0, 0, 0}, Metadata{"k": "v"})

	meta, err := db.GetMetadata(ctx, "a")
	if err != nil {
		t.Fatalf("GetMetadata: %v", err)
	}
	if meta["k"] != "v" {
		t.Errorf("metadata[k] = %v, want v", meta["k"])
	}

	// Not found.
	_, err = db.GetMetadata(ctx, "b")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestGetMetadata_Deleted(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	db.AddWithMetadata(ctx, "a", []float64{1, 0, 0, 0}, Metadata{"k": "v"})
	db.Delete(ctx, "a")

	_, err = db.GetMetadata(ctx, "a")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected ErrNotFound for deleted vector, got %v", err)
	}
}

// --- Integration tests (with HE) ---

func TestSearchWithFilter(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 200
		dim        = 128
		clusters   = 8
		topK       = 10
	)

	ctx := context.Background()
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

	ids, vectors := generateTestVectors(numVectors, dim, 42)

	// Add vectors with alternating categories.
	metadatas := make([]Metadata, numVectors)
	for i := range metadatas {
		cat := "even"
		if i%2 == 1 {
			cat = "odd"
		}
		metadatas[i] = Metadata{"category": cat, "index": i}
	}
	if err := db.AddBatchWithMetadata(ctx, ids, vectors, metadatas); err != nil {
		t.Fatalf("AddBatchWithMetadata: %v", err)
	}

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Unfiltered search.
	allResults, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(allResults) == 0 {
		t.Fatal("Search returned no results")
	}

	// Filtered search — only "even" category.
	evenResults, err := db.SearchWithFilter(ctx, vectors[0], topK, Filter{
		Where: map[string]any{"category": "even"},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter: %v", err)
	}

	// All even results should have even metadata.
	for _, r := range evenResults {
		meta, err := db.GetMetadata(ctx, r.ID)
		if err != nil {
			t.Fatalf("GetMetadata(%s): %v", r.ID, err)
		}
		if meta["category"] != "even" {
			t.Errorf("result %s has category %v, want 'even'", r.ID, meta["category"])
		}
	}

	// Filtered search — only "odd" category.
	oddResults, err := db.SearchWithFilter(ctx, vectors[0], topK, Filter{
		Where: map[string]any{"category": "odd"},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter(odd): %v", err)
	}

	for _, r := range oddResults {
		meta, err := db.GetMetadata(ctx, r.ID)
		if err != nil {
			t.Fatalf("GetMetadata(%s): %v", r.ID, err)
		}
		if meta["category"] != "odd" {
			t.Errorf("result %s has category %v, want 'odd'", r.ID, meta["category"])
		}
	}

	t.Logf("Unfiltered: %d results, Even: %d results, Odd: %d results",
		len(allResults), len(evenResults), len(oddResults))
}

func TestSearchWithFilter_NoMatch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 200
		dim        = 128
		clusters   = 8
		topK       = 5
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
	metadatas := make([]Metadata, numVectors)
	for i := range metadatas {
		metadatas[i] = Metadata{"type": "document"}
	}
	db.AddBatchWithMetadata(ctx, ids, vectors, metadatas)
	db.Build(ctx)

	// Filter for non-existent category.
	results, err := db.SearchWithFilter(ctx, vectors[0], topK, Filter{
		Where: map[string]any{"type": "image"},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for non-matching filter, got %d", len(results))
	}
}

func TestMetadata_SaveLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 200
		dim        = 128
		clusters   = 8
		topK       = 5
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

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	metadatas := make([]Metadata, numVectors)
	for i := range metadatas {
		metadatas[i] = Metadata{
			"category": "cat-" + ids[i],
			"index":    i,
			"active":   i%2 == 0,
		}
	}
	db.AddBatchWithMetadata(ctx, ids, vectors, metadatas)
	db.Build(ctx)

	savePath := filepath.Join(t.TempDir(), "db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	// Load and verify metadata survived.
	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	// Check metadata for a few vectors.
	for _, i := range []int{0, 42, 99} {
		meta, err := loaded.GetMetadata(ctx, ids[i])
		if err != nil {
			t.Fatalf("GetMetadata(%s): %v", ids[i], err)
		}
		if meta["category"] != "cat-"+ids[i] {
			t.Errorf("metadata category for %s: got %v, want %v", ids[i], meta["category"], "cat-"+ids[i])
		}
	}

	// Filtered search on loaded DB should work.
	results, err := loaded.SearchWithFilter(ctx, vectors[0], topK, Filter{
		Where: map[string]any{"active": true},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter on loaded: %v", err)
	}

	for _, r := range results {
		meta, err := loaded.GetMetadata(ctx, r.ID)
		if err != nil {
			t.Fatalf("GetMetadata(%s): %v", r.ID, err)
		}
		if meta["active"] != true {
			t.Errorf("result %s active=%v, want true", r.ID, meta["active"])
		}
	}

	t.Logf("Loaded DB filtered search: %d results", len(results))
}

func TestMixedAddWithAndWithoutMetadata(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		dim      = 128
		clusters = 8
		topK     = 5
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

	ids, vectors := generateTestVectors(200, dim, 42)

	// Add first 100 without metadata.
	db.AddBatch(ctx, ids[:100], vectors[:100])

	// Add next 100 with metadata.
	metadatas := make([]Metadata, 100)
	for i := range metadatas {
		metadatas[i] = Metadata{"has_meta": true}
	}
	db.AddBatchWithMetadata(ctx, ids[100:], vectors[100:], metadatas)

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Unfiltered search should return results from both groups.
	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}

	// Filtered search for has_meta should only return from second group.
	filtered, err := db.SearchWithFilter(ctx, vectors[150], topK, Filter{
		Where: map[string]any{"has_meta": true},
	})
	if err != nil {
		t.Fatalf("SearchWithFilter: %v", err)
	}

	for _, r := range filtered {
		meta := db.metadata[r.ID]
		if meta == nil || meta["has_meta"] != true {
			t.Errorf("result %s should have has_meta=true", r.ID)
		}
	}
}
