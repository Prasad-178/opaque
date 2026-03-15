package opaque

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/opaque/opaque/go/pkg/pca"
)

// --- Fast tests (no HE) ---

func TestSaveBeforeBuild(t *testing.T) {
	db, err := NewDB(Config{Dimension: 4})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer db.Close()

	tmpDir := t.TempDir()
	err = db.Save(filepath.Join(tmpDir, "out"))
	if !errors.Is(err, ErrNotReady) {
		t.Fatalf("expected ErrNotReady, got %v", err)
	}

	// Also try after Add but before Build.
	db.Add(context.Background(), "a", []float64{1, 0, 0, 0})
	err = db.Save(filepath.Join(tmpDir, "out2"))
	if !errors.Is(err, ErrNotReady) {
		t.Fatalf("expected ErrNotReady (buffered state), got %v", err)
	}
}

func TestMetadataRoundtrip(t *testing.T) {
	ns := true
	cfg := Config{
		Dimension:            128,
		NumClusters:          32,
		TopClusters:          8,
		NumDecoys:            4,
		WorkerPoolSize:       2,
		Storage:              Memory,
		ProbeThreshold:       0.90,
		RedundantAssignments: 2,
		PCADimension:         64,
		NormalizedStorage:    &ns,
	}
	stats := ClusterStats{
		NumClusters:   32,
		MinSize:       5,
		MaxSize:       20,
		AvgSize:       12.5,
		EmptyClusters: 0,
		Iterations:    15,
	}
	meta := saveMetadata{
		Version:      1,
		Config:       cfg,
		ClusterStats: stats,
		HasPCA:       true,
	}

	data, err := json.Marshal(meta)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var got saveMetadata
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if got.Config.Dimension != 128 {
		t.Errorf("Dimension = %d, want 128", got.Config.Dimension)
	}
	if got.Config.NumClusters != 32 {
		t.Errorf("NumClusters = %d, want 32", got.Config.NumClusters)
	}
	if got.Config.PCADimension != 64 {
		t.Errorf("PCADimension = %d, want 64", got.Config.PCADimension)
	}
	if got.Config.NormalizedStorage == nil || !*got.Config.NormalizedStorage {
		t.Error("NormalizedStorage should be true")
	}
	if got.ClusterStats.MinSize != 5 {
		t.Errorf("MinSize = %d, want 5", got.ClusterStats.MinSize)
	}
	if got.ClusterStats.AvgSize != 12.5 {
		t.Errorf("AvgSize = %f, want 12.5", got.ClusterStats.AvgSize)
	}
	if !got.HasPCA {
		t.Error("HasPCA should be true")
	}
}

func TestPCASaveLoad(t *testing.T) {
	// Generate training data.
	rng := rand.New(rand.NewSource(42))
	const (
		n   = 50
		dim = 16
		k   = 4
	)

	vectors := make([][]float64, n)
	for i := range vectors {
		v := make([]float64, dim)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		vectors[i] = v
	}

	model, err := pca.Fit(vectors, k)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Save.
	var buf bytes.Buffer
	if err := model.Save(&buf); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load.
	loaded, err := pca.LoadPCA(&buf)
	if err != nil {
		t.Fatalf("LoadPCA: %v", err)
	}

	// Verify the loaded model produces identical transforms.
	testVec := vectors[0]
	orig, err := model.Transform(testVec)
	if err != nil {
		t.Fatalf("Transform (original): %v", err)
	}
	restored, err := loaded.Transform(testVec)
	if err != nil {
		t.Fatalf("Transform (loaded): %v", err)
	}

	if len(orig) != len(restored) {
		t.Fatalf("dimension mismatch: %d vs %d", len(orig), len(restored))
	}
	for i := range orig {
		if math.Abs(orig[i]-restored[i]) > 1e-12 {
			t.Errorf("component %d: orig=%f, restored=%f", i, orig[i], restored[i])
		}
	}
}

func TestLoadNonexistent(t *testing.T) {
	_, err := Load("/nonexistent/path/that/does/not/exist")
	if err == nil {
		t.Fatal("expected error for nonexistent path, got nil")
	}
}

func TestSaveOverwrite(t *testing.T) {
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

	savePath := filepath.Join(t.TempDir(), "db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("first Save: %v", err)
	}

	// Second save to same path should fail.
	if err := db.Save(savePath); err == nil {
		t.Fatal("expected error for overwrite Save, got nil")
	}
}

// --- Integration tests (with HE) ---

func TestSaveLoadMemory(t *testing.T) {
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

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Save.
	savePath := filepath.Join(t.TempDir(), "db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	// Load.
	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	if !loaded.IsReady() {
		t.Fatal("loaded DB is not ready")
	}

	// Search on loaded DB.
	targetIdx := 42
	results, err := loaded.Search(ctx, vectors[targetIdx], topK)
	if err != nil {
		t.Fatalf("Search on loaded DB: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search on loaded DB returned no results")
	}

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
}

func TestSaveLoadFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping HE engine test in short mode")
	}

	tmpDir := t.TempDir()
	storagePath := filepath.Join(tmpDir, "storage")
	savePath := filepath.Join(tmpDir, "saved")

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
		StoragePath:    storagePath,
	})
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}

	ids, vectors := generateTestVectors(numVectors, dim, 77)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	results, err := loaded.Search(ctx, vectors[10], topK)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

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

func TestSaveLoadWithPCA(t *testing.T) {
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

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build with PCA: %v", err)
	}

	savePath := filepath.Join(t.TempDir(), "db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Verify pca.gob exists.
	if _, err := os.Stat(filepath.Join(savePath, "pca.gob")); err != nil {
		t.Fatalf("pca.gob not found: %v", err)
	}

	db.Close()

	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	if loaded.pcaModel == nil {
		t.Fatal("loaded DB should have PCA model")
	}

	results, err := loaded.Search(ctx, vectors[42], topK)
	if err != nil {
		t.Fatalf("Search on loaded PCA DB: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search on loaded PCA DB returned no results")
	}

	t.Logf("PCA loaded: %dD -> %dD, top result: %s (%.4f)",
		dim, pcaDim, results[0].ID, results[0].Score)
}

func TestSaveLoadSearchConsistency(t *testing.T) {
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

	ids, vectors := generateTestVectors(numVectors, dim, 42)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Search before save.
	query := vectors[42]
	beforeResults, err := db.Search(ctx, query, topK)
	if err != nil {
		t.Fatalf("Search before save: %v", err)
	}

	savePath := filepath.Join(t.TempDir(), "db")
	if err := db.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	db.Close()

	loaded, err := Load(savePath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	// Search after load.
	afterResults, err := loaded.Search(ctx, query, topK)
	if err != nil {
		t.Fatalf("Search after load: %v", err)
	}

	// Top-1 result (self-match) should be identical.
	if len(beforeResults) == 0 || len(afterResults) == 0 {
		t.Fatal("expected non-empty results")
	}
	if beforeResults[0].ID != afterResults[0].ID {
		t.Errorf("top-1 ID mismatch: before=%s, after=%s", beforeResults[0].ID, afterResults[0].ID)
	}
	if math.Abs(beforeResults[0].Score-afterResults[0].Score) > 1e-6 {
		t.Errorf("top-1 score mismatch: before=%f, after=%f", beforeResults[0].Score, afterResults[0].Score)
	}

	// Remaining results may vary due to random decoy selection during search,
	// but there should be significant overlap in the result sets.
	afterIDs := make(map[string]bool)
	for _, r := range afterResults {
		afterIDs[r.ID] = true
	}
	overlap := 0
	for _, r := range beforeResults {
		if afterIDs[r.ID] {
			overlap++
		}
	}
	// Expect at least 50% overlap (decoys cause some variation).
	minOverlap := len(beforeResults) / 2
	if overlap < minOverlap {
		t.Errorf("result overlap too low: %d/%d (want >= %d)", overlap, len(beforeResults), minOverlap)
	}
}
