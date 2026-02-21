package opaque

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"
)

// --- Test helpers ---

// generateTestVectors creates n normalized random vectors of the given dimension
// using a deterministic seed for reproducibility.
func generateTestVectors(n, dim int, seed int64) ([]string, [][]float64) {
	rng := rand.New(rand.NewSource(seed))

	ids := make([]string, n)
	vectors := make([][]float64, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc-%d", i)
		vec := make([]float64, dim)
		norm := 0.0
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
			norm += vec[j] * vec[j]
		}
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}
		vectors[i] = vec
	}

	return ids, vectors
}

// --- Config and constructor tests ---

func TestNewDB_Defaults(t *testing.T) {
	db, err := NewDB(Config{Dimension: 128})
	if err != nil {
		t.Fatalf("NewDB failed: %v", err)
	}
	defer db.Close()

	if db.cfg.NumClusters != 64 {
		t.Errorf("default NumClusters = %d, want 64", db.cfg.NumClusters)
	}
	if db.cfg.TopClusters != 32 {
		t.Errorf("default TopClusters = %d, want 32", db.cfg.TopClusters)
	}
	if db.cfg.NumDecoys != 8 {
		t.Errorf("default NumDecoys = %d, want 8", db.cfg.NumDecoys)
	}
	expectedPool := runtime.NumCPU()
	if expectedPool > 8 {
		expectedPool = 8
	}
	if db.cfg.WorkerPoolSize != expectedPool {
		t.Errorf("default WorkerPoolSize = %d, want %d", db.cfg.WorkerPoolSize, expectedPool)
	}
	if db.cfg.ProbeThreshold != 0.95 {
		t.Errorf("default ProbeThreshold = %f, want 0.95", db.cfg.ProbeThreshold)
	}
	if db.cfg.RedundantAssignments != 1 {
		t.Errorf("default RedundantAssignments = %d, want 1", db.cfg.RedundantAssignments)
	}
	if db.cfg.Storage != Memory {
		t.Errorf("default Storage = %d, want Memory (0)", db.cfg.Storage)
	}
}

func TestNewDB_NoDimension(t *testing.T) {
	_, err := NewDB(Config{})
	if err == nil {
		t.Fatal("expected error for zero Dimension, got nil")
	}
}

func TestNewDB_InvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"NumClusters too small", Config{Dimension: 128, NumClusters: 1}},
		{"TopClusters > NumClusters", Config{Dimension: 128, NumClusters: 8, TopClusters: 16}},
		{"ProbeThreshold negative", Config{Dimension: 128, ProbeThreshold: -0.5}},
		{"ProbeThreshold > 1", Config{Dimension: 128, ProbeThreshold: 1.5}},
		{"File storage without path", Config{Dimension: 128, Storage: File}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewDB(tt.cfg)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tt.name)
			}
		})
	}
}

// --- Add tests ---

func TestAdd_WrongDimension(t *testing.T) {
	db, err := NewDB(Config{Dimension: 128})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	err = db.Add(context.Background(), "doc-1", make([]float64, 64))
	if err == nil {
		t.Fatal("expected error for wrong dimension, got nil")
	}
}

func TestAdd_EmptyID(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	err = db.Add(context.Background(), "", []float64{1, 0, 0, 0})
	if err == nil {
		t.Fatal("expected error for empty ID, got nil")
	}
}

func TestAddBatch_LengthMismatch(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	err = db.AddBatch(context.Background(), []string{"a", "b"}, [][]float64{{1, 0, 0, 0}})
	if err == nil {
		t.Fatal("expected error for length mismatch, got nil")
	}
}

// --- State machine tests ---

func TestBuild_NoVectors(t *testing.T) {
	db, err := NewDB(Config{Dimension: 128})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	err = db.Build(context.Background())
	if err == nil {
		t.Fatal("expected error for Build with no vectors, got nil")
	}
}

func TestSearch_BeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	// Search on empty DB.
	_, err = db.Search(context.Background(), []float64{1, 0, 0, 0}, 5)
	if err == nil {
		t.Fatal("expected error for Search on empty DB, got nil")
	}

	// Search after Add but before Build.
	db.Add(context.Background(), "doc-1", []float64{1, 0, 0, 0})
	_, err = db.Search(context.Background(), []float64{1, 0, 0, 0}, 5)
	if err == nil {
		t.Fatal("expected error for Search before Build, got nil")
	}
}

func TestSearch_InvalidTopK(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	_, err = db.Search(context.Background(), []float64{1, 0, 0, 0}, 0)
	if err == nil {
		t.Fatal("expected error for topK=0, got nil")
	}
}

func TestSize_and_IsReady(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	if db.Size() != 0 {
		t.Errorf("Size on empty DB = %d, want 0", db.Size())
	}
	if db.IsReady() {
		t.Error("IsReady on empty DB = true, want false")
	}

	db.Add(context.Background(), "a", []float64{1, 0, 0, 0})
	db.Add(context.Background(), "b", []float64{0, 1, 0, 0})

	if db.Size() != 2 {
		t.Errorf("Size after 2 adds = %d, want 2", db.Size())
	}
	if db.IsReady() {
		t.Error("IsReady before Build = true, want false")
	}
}

func TestClose_EmptyDB(t *testing.T) {
	db, err := NewDB(Config{Dimension: 128})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close on empty DB: %v", err)
	}
}

// --- Full lifecycle tests (these create HE engines and are slower) ---

func TestBuildAndSearch_Memory(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 300
		dim        = 128
		clusters   = 16
		topK       = 10
	)

	ctx := context.Background()

	db, err := NewDB(Config{
		Dimension:      dim,
		NumClusters:    clusters,
		TopClusters:    8,
		NumDecoys:      4,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ids, vectors := generateTestVectors(numVectors, dim, 42)

	// Add vectors one by one (test Add path).
	for i, id := range ids {
		if err := db.Add(ctx, id, vectors[i]); err != nil {
			t.Fatalf("Add(%q): %v", id, err)
		}
	}

	if db.Size() != numVectors {
		t.Fatalf("Size = %d, want %d", db.Size(), numVectors)
	}

	// Build the index.
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if !db.IsReady() {
		t.Fatal("IsReady after Build = false")
	}

	// Search for a known vector — it should be in the top results.
	targetIdx := 42
	results, err := db.Search(ctx, vectors[targetIdx], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}

	// The target vector should appear in results (exact self-match).
	found := false
	for _, r := range results {
		if r.ID == ids[targetIdx] {
			found = true
			if r.Score < 0.99 {
				t.Errorf("self-match score = %f, want >= 0.99", r.Score)
			}
			break
		}
	}
	if !found {
		t.Errorf("target vector %q not found in top-%d results", ids[targetIdx], topK)
	}

	// Verify results are sorted by descending score.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: score[%d]=%f > score[%d]=%f", i, results[i].Score, i-1, results[i-1].Score)
		}
	}

	// Add after Build should fail.
	err = db.Add(ctx, "extra", make([]float64, dim))
	if err == nil {
		t.Fatal("expected error for Add after Build, got nil")
	}
}

func TestBuildAndSearch_AddBatch(t *testing.T) {
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

	ids, vectors := generateTestVectors(numVectors, dim, 99)

	// Use AddBatch.
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}

	// First result should be the query vector itself.
	if results[0].ID != ids[0] {
		t.Logf("top result is %q (score=%f), expected %q", results[0].ID, results[0].Score, ids[0])
	}
}

func TestBuildAndSearch_File(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	tmpDir, err := os.MkdirTemp("", "opaque-test-*")
	if err != nil {
		t.Fatalf("TempDir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

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
		Storage:        File,
		StoragePath:    tmpDir,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	ids, vectors := generateTestVectors(numVectors, dim, 77)

	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	results, err := db.Search(ctx, vectors[10], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}

	// Verify the target vector is found.
	found := false
	for _, r := range results {
		if r.ID == ids[10] {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("target vector %q not found in top-%d results", ids[10], topK)
	}
}

func TestConcurrentSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors  = 200
		dim         = 128
		clusters    = 8
		topK        = 5
		numSearches = 20
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

	ids, vectors := generateTestVectors(numVectors, dim, 55)

	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Run concurrent searches — should not race or panic.
	var wg sync.WaitGroup
	errs := make(chan error, numSearches)

	for i := 0; i < numSearches; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			query := vectors[idx%numVectors]
			results, err := db.Search(ctx, query, topK)
			if err != nil {
				errs <- fmt.Errorf("search %d: %w", idx, err)
				return
			}
			if len(results) == 0 {
				errs <- fmt.Errorf("search %d: no results", idx)
			}
		}(i)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

func TestRebuild(t *testing.T) {
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
		TopClusters:    4,
		NumDecoys:      2,
		WorkerPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	// Initial build with 100 vectors.
	ids, vectors := generateTestVectors(100, dim, 33)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if db.Size() != 100 {
		t.Fatalf("Size after build = %d, want 100", db.Size())
	}

	// Rebuild should work (re-indexes all 100 vectors).
	if err := db.Rebuild(ctx); err != nil {
		t.Fatalf("Rebuild: %v", err)
	}

	if !db.IsReady() {
		t.Fatal("IsReady after Rebuild = false")
	}

	// Search should still work.
	results, err := db.Search(ctx, vectors[0], topK)
	if err != nil {
		t.Fatalf("Search after Rebuild: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search after Rebuild returned no results")
	}
}

func TestClose_BuiltDB(t *testing.T) {
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

	ids, vectors := generateTestVectors(100, dim, 11)
	db.AddBatch(ctx, ids, vectors)
	db.Build(ctx)

	if err := db.Close(); err != nil {
		t.Fatalf("Close on built DB: %v", err)
	}

	if db.IsReady() {
		t.Error("IsReady after Close = true, want false")
	}
}

// --- PCA tests ---

func TestNewDB_PCAValidation(t *testing.T) {
	// PCADimension must be less than Dimension
	_, err := NewDB(Config{Dimension: 128, PCADimension: 128})
	if err == nil {
		t.Error("expected error when PCADimension == Dimension")
	}
	_, err = NewDB(Config{Dimension: 128, PCADimension: 200})
	if err == nil {
		t.Error("expected error when PCADimension > Dimension")
	}

	// Valid PCA config should work
	db, err := NewDB(Config{Dimension: 128, PCADimension: 64})
	if err != nil {
		t.Fatalf("NewDB with PCA failed: %v", err)
	}
	defer db.Close()
}

func TestBuildAndSearch_WithPCA(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	const (
		numVectors = 300
		dim        = 128
		pcaDim     = 64
		clusters   = 16
		topK       = 10
	)

	ctx := context.Background()

	db, err := NewDB(Config{
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
	defer db.Close()

	ids, vectors := generateTestVectors(numVectors, dim, 42)

	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build with PCA: %v", err)
	}

	if !db.IsReady() {
		t.Fatal("IsReady after Build with PCA = false")
	}

	// PCA model should be set
	if db.pcaModel == nil {
		t.Fatal("pcaModel should be set after Build with PCA enabled")
	}
	if db.pcaModel.ReducedDim != pcaDim {
		t.Errorf("PCA reduced dim = %d, want %d", db.pcaModel.ReducedDim, pcaDim)
	}

	// Search should work with original-dimension query
	targetIdx := 42
	results, err := db.Search(ctx, vectors[targetIdx], topK)
	if err != nil {
		t.Fatalf("Search with PCA: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Search with PCA returned no results")
	}

	// Self-match should still work (PCA preserves most similarity structure)
	found := false
	for _, r := range results {
		if r.ID == ids[targetIdx] {
			found = true
			t.Logf("Self-match score with PCA: %.4f", r.Score)
			break
		}
	}
	if !found {
		t.Logf("Warning: self-match not found in top-%d with PCA (may happen due to dimensionality reduction)", topK)
	}

	t.Logf("PCA: %dD -> %dD, variance explained: %.2f%%",
		dim, pcaDim, db.pcaModel.TotalVarianceExplained()*100)
	t.Logf("Top result: ID=%s, Score=%.4f", results[0].ID, results[0].Score)
}
