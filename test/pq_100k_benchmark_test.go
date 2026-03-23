package test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
)

// TestPQ100KBenchmark benchmarks PQ on 100K vectors — the sweet spot
// where local scoring dominates total latency.
//
// Run: go test -v -run TestPQ100KBenchmark ./test/ -timeout 30m
func TestPQ100KBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ 100K benchmark in short mode")
	}

	rng := rand.New(rand.NewSource(42))
	D := 128
	N := 100000
	numQueries := 20
	topK := 10

	t.Logf("=== PQ Benchmark: %dK vectors, %d-dim, %d queries ===", N/1000, D, numQueries)
	t.Log("Generating 100K clustered vectors...")

	// 50 cluster centers.
	nClusters := 50
	centers := make([][]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		v := make([]float64, D)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		centers[c] = v
	}

	ids := make([]string, N)
	vectors := make([][]float64, N)
	for i := 0; i < N; i++ {
		center := centers[rng.Intn(nClusters)]
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = center[j] + rng.NormFloat64()*0.3
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		ids[i] = fmt.Sprintf("v-%07d", i)
		vectors[i] = v
	}

	queries := make([][]float64, numQueries)
	for q := 0; q < numQueries; q++ {
		center := centers[rng.Intn(nClusters)]
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = center[j] + rng.NormFloat64()*0.2
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		queries[q] = v
	}

	t.Log("Computing brute-force ground truth (100K)...")
	groundTruth := computeGroundTruth(vectors, ids, queries, topK)

	ctx := context.Background()
	baseCfg := opaque.Config{
		Dimension:            D,
		NumClusters:          64,
		TopClusters:          8,
		NumDecoys:            8,
		RedundantAssignments: 2,
	}

	// Standard (no PQ).
	t.Log("\n--- Building standard index ---")
	dbStd, err := opaque.NewDB(baseCfg)
	if err != nil {
		t.Fatalf("NewDB: %v", err)
	}
	defer dbStd.Close()

	if err := dbStd.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	buildStart := time.Now()
	if err := dbStd.Build(ctx); err != nil {
		t.Fatalf("Build: %v", err)
	}
	stdBuildTime := time.Since(buildStart)
	stdRecall, stdLatency := benchmarkSearch(t, dbStd, queries, groundTruth, topK, "Standard")
	t.Logf("Standard: Build %v, Search %.1fms, Recall@%d %.1f%%",
		stdBuildTime.Round(time.Millisecond), stdLatency, topK, stdRecall*100)

	// PQ M=8.
	t.Log("\n--- Building PQ index (M=8) ---")
	pqCfg := baseCfg
	pqCfg.PQSubspaces = 8
	dbPQ, err := opaque.NewDB(pqCfg)
	if err != nil {
		t.Fatalf("NewDB PQ: %v", err)
	}
	defer dbPQ.Close()

	if err := dbPQ.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	buildStart = time.Now()
	if err := dbPQ.Build(ctx); err != nil {
		t.Fatalf("Build PQ: %v", err)
	}
	pqBuildTime := time.Since(buildStart)
	pqRecall, pqLatency := benchmarkSearch(t, dbPQ, queries, groundTruth, topK, "PQ-M8")
	t.Logf("PQ-M8:    Build %v, Search %.1fms, Recall@%d %.1f%%",
		pqBuildTime.Round(time.Millisecond), pqLatency, topK, pqRecall*100)

	// Summary.
	t.Logf("\n=== Summary (100K vectors, 128-dim, 64 clusters, 8 probed) ===")
	t.Logf("%-12s  Build        Avg Query    Recall@%d  Speedup", "Config", topK)
	t.Logf("%-12s  %-12v %-10.1fms  %.1f%%     1.0x", "Standard", stdBuildTime.Round(time.Millisecond), stdLatency, stdRecall*100)
	if pqLatency > 0 {
		t.Logf("%-12s  %-12v %-10.1fms  %.1f%%     %.1fx", "PQ M=8", pqBuildTime.Round(time.Millisecond), pqLatency, pqRecall*100, stdLatency/pqLatency)
	}
}
