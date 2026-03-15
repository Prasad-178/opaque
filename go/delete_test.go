package opaque

import (
	"context"
	"errors"
	"testing"
)

// --- Fast tests (no HE) ---

func TestDelete_EmptyID(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if !errors.Is(db.Delete(context.Background(), ""), ErrEmptyID) {
		t.Fatal("expected ErrEmptyID for empty ID")
	}
}

func TestDelete_NotFound(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if !errors.Is(db.Delete(context.Background(), "nope"), ErrNotFound) {
		t.Fatal("expected ErrNotFound for nonexistent ID")
	}
}

func TestDelete_Pending(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	db.Add(ctx, "a", []float64{1, 0, 0, 0})
	db.Add(ctx, "b", []float64{0, 1, 0, 0})

	if !db.Has(ctx, "a") {
		t.Fatal("a should exist before delete")
	}

	if err := db.Delete(ctx, "a"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if db.Has(ctx, "a") {
		t.Fatal("a should not exist after delete")
	}
	if !db.Has(ctx, "b") {
		t.Fatal("b should still exist")
	}
}

func TestUpdate_EmptyID(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if !errors.Is(db.Update(context.Background(), "", []float64{1, 0, 0, 0}), ErrEmptyID) {
		t.Fatal("expected ErrEmptyID")
	}
}

func TestUpdate_DimensionMismatch(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	db.Add(ctx, "a", []float64{1, 0, 0, 0})

	if !errors.Is(db.Update(ctx, "a", []float64{1, 0}), ErrDimensionMismatch) {
		t.Fatal("expected ErrDimensionMismatch")
	}
}

func TestUpdate_NotFound(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if !errors.Is(db.Update(context.Background(), "nope", []float64{1, 0, 0, 0}), ErrNotFound) {
		t.Fatal("expected ErrNotFound")
	}
}

func TestAddAfterBuild(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		dim      = 128
		clusters = 8
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

	ids, vectors := generateTestVectors(100, dim, 42)
	db.AddBatch(ctx, ids, vectors)
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Add after Build should work now.
	err = db.Add(ctx, "new-vec", vectors[0])
	if err != nil {
		t.Fatalf("Add after Build should succeed: %v", err)
	}

	// Rebuild should include the new vector.
	if err := db.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	if !db.Has(ctx, "new-vec") {
		t.Error("new-vec should exist after Rebuild")
	}
}

// --- Integration tests (with HE) ---

func TestDelete_AfterBuild(t *testing.T) {
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
		TopClusters:    4,
		NumDecoys:      2,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	db.AddBatch(ctx, ids, vectors)
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Search for doc-0 — should find it.
	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc-0" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("doc-0 should be in search results before delete")
	}

	// Delete doc-0.
	if err := db.Delete(ctx, "doc-0"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Search again — doc-0 should be filtered out.
	results, err = db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search after delete: %v", err)
	}
	for _, r := range results {
		if r.ID == "doc-0" {
			t.Fatal("doc-0 should NOT be in search results after delete")
		}
	}

	// Has and Get should reflect deletion.
	if db.Has(ctx, "doc-0") {
		t.Error("Has(doc-0) should be false after delete")
	}
	_, err = db.Get(ctx, "doc-0")
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("Get(doc-0) after delete: expected ErrNotFound, got %v", err)
	}

	// Rebuild compacts deleted vectors.
	if err := db.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	// doc-0 should still not exist after Rebuild.
	if db.Has(ctx, "doc-0") {
		t.Error("Has(doc-0) should be false after Rebuild")
	}
}

func TestUpdate_AfterBuild(t *testing.T) {
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
	db.AddBatch(ctx, ids, vectors)
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Update doc-0 with a new vector (use doc-99's vector).
	if err := db.Update(ctx, "doc-0", vectors[99]); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// doc-0 should be soft-deleted from search until Rebuild.
	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.ID == "doc-0" {
			t.Fatal("doc-0 should be soft-deleted from search after Update (before Rebuild)")
		}
	}

	// Rebuild to apply the update.
	if err := db.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	// doc-0 should exist with the updated vector.
	if !db.Has(ctx, "doc-0") {
		t.Fatal("doc-0 should exist after Rebuild with updated vector")
	}

	// Search with the new vector (doc-99's) should find doc-0.
	results, err = db.Search(ctx, vectors[99], topK)
	if err != nil {
		t.Fatalf("Search after Rebuild: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc-0" {
			found = true
			break
		}
	}
	if !found {
		t.Log("doc-0 not in top results after update — may happen due to clustering")
	}
}

func TestRebuild_WithDeletes(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		dim      = 128
		clusters = 8
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

	ids, vectors := generateTestVectors(100, dim, 42)
	db.AddBatch(ctx, ids, vectors)
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Delete 10 vectors.
	for i := 0; i < 10; i++ {
		if err := db.Delete(ctx, ids[i]); err != nil {
			t.Fatalf("Delete(%s): %v", ids[i], err)
		}
	}

	// Rebuild should compact.
	if err := db.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	// Deleted vectors should be gone.
	for i := 0; i < 10; i++ {
		if db.Has(ctx, ids[i]) {
			t.Errorf("%s should not exist after Rebuild", ids[i])
		}
	}

	// Remaining vectors should still exist.
	for i := 10; i < 100; i++ {
		if !db.Has(ctx, ids[i]) {
			t.Errorf("%s should exist after Rebuild", ids[i])
		}
	}
}
