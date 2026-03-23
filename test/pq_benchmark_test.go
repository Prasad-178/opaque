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

// TestPQBenchmark benchmarks PQ-accelerated search vs standard search
// on a 10K clustered vector dataset with full recall measurement.
//
// Run: go test -v -run TestPQBenchmark ./test/ -timeout 15m
func TestPQBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ benchmark in short mode")
	}

	rng := rand.New(rand.NewSource(42))
	D := 128
	N := 10000
	numQueries := 50
	topK := 10

	t.Logf("=== PQ Benchmark: %dK vectors, %d-dim, %d queries ===", N/1000, D, numQueries)

	// Generate clustered data (30 centers).
	nClusters := 30
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
		ids[i] = fmt.Sprintf("v-%06d", i)
		vectors[i] = v
	}

	// Generate queries.
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

	// Compute brute-force ground truth.
	groundTruth := computeGroundTruth(vectors, ids, queries, topK)

	ctx := context.Background()

	// Common config.
	baseCfg := opaque.Config{
		Dimension:            D,
		NumClusters:          32,
		TopClusters:          8,
		NumDecoys:            8,
		RedundantAssignments: 2,
	}

	// === Standard (no PQ) ===
	t.Log("\n--- Building standard index (no PQ) ---")
	dbStd, err := opaque.NewDB(baseCfg)
	if err != nil {
		t.Fatalf("NewDB standard: %v", err)
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
	t.Logf("Standard Build: %v", stdBuildTime)
	t.Logf("Standard Search: avg %.1fms, Recall@%d: %.1f%%", stdLatency, topK, stdRecall*100)

	// === PQ M=8 ===
	pqCfg8 := baseCfg
	pqCfg8.PQSubspaces = 8
	t.Log("\n--- Building PQ index (M=8) ---")
	dbPQ8, err := opaque.NewDB(pqCfg8)
	if err != nil {
		t.Fatalf("NewDB PQ8: %v", err)
	}
	defer dbPQ8.Close()

	if err := dbPQ8.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	buildStart = time.Now()
	if err := dbPQ8.Build(ctx); err != nil {
		t.Fatalf("Build PQ8: %v", err)
	}
	pq8BuildTime := time.Since(buildStart)

	pq8Recall, pq8Latency := benchmarkSearch(t, dbPQ8, queries, groundTruth, topK, "PQ-M8")
	t.Logf("PQ-M8 Build: %v (includes PQ training)", pq8BuildTime)
	t.Logf("PQ-M8 Search: avg %.1fms, Recall@%d: %.1f%%", pq8Latency, topK, pq8Recall*100)

	// === PQ M=16 ===
	pqCfg16 := baseCfg
	pqCfg16.PQSubspaces = 16
	t.Log("\n--- Building PQ index (M=16) ---")
	dbPQ16, err := opaque.NewDB(pqCfg16)
	if err != nil {
		t.Fatalf("NewDB PQ16: %v", err)
	}
	defer dbPQ16.Close()

	if err := dbPQ16.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}
	buildStart = time.Now()
	if err := dbPQ16.Build(ctx); err != nil {
		t.Fatalf("Build PQ16: %v", err)
	}
	pq16BuildTime := time.Since(buildStart)

	pq16Recall, pq16Latency := benchmarkSearch(t, dbPQ16, queries, groundTruth, topK, "PQ-M16")
	t.Logf("PQ-M16 Build: %v (includes PQ training)", pq16BuildTime)
	t.Logf("PQ-M16 Search: avg %.1fms, Recall@%d: %.1f%%", pq16Latency, topK, pq16Recall*100)

	// === Summary ===
	t.Logf("\n=== Summary (10K vectors, 128-dim) ===")
	t.Logf("%-12s  Build      Avg Query   Recall@%d   Speedup", "Config", topK)
	t.Logf("%-12s  %-10v %-10.1fms  %.1f%%       1.0x", "Standard", stdBuildTime.Round(time.Millisecond), stdLatency, stdRecall*100)
	if pq8Latency > 0 {
		t.Logf("%-12s  %-10v %-10.1fms  %.1f%%       %.1fx", "PQ M=8", pq8BuildTime.Round(time.Millisecond), pq8Latency, pq8Recall*100, stdLatency/pq8Latency)
	}
	if pq16Latency > 0 {
		t.Logf("%-12s  %-10v %-10.1fms  %.1f%%       %.1fx", "PQ M=16", pq16BuildTime.Round(time.Millisecond), pq16Latency, pq16Recall*100, stdLatency/pq16Latency)
	}
}

func benchmarkSearch(t *testing.T, db *opaque.DB, queries [][]float64, gt [][]string, topK int, label string) (recall float64, avgLatencyMs float64) {
	t.Helper()
	ctx := context.Background()
	var totalLatency time.Duration
	var totalRecall float64

	for q, query := range queries {
		start := time.Now()
		results, err := db.Search(ctx, query, topK)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("%s Search q=%d: %v", label, q, err)
		}
		totalLatency += elapsed

		gtSet := make(map[string]bool, topK)
		for _, id := range gt[q] {
			gtSet[id] = true
		}
		hits := 0
		for _, r := range results {
			if gtSet[r.ID] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(topK)
	}

	avgRecall := totalRecall / float64(len(queries))
	avgMs := float64(totalLatency.Milliseconds()) / float64(len(queries))
	return avgRecall, avgMs
}

func computeGroundTruth(vectors [][]float64, ids []string, queries [][]float64, topK int) [][]string {
	gt := make([][]string, len(queries))
	for q, query := range queries {
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(vectors))
		for i, v := range vectors {
			var dot float64
			for j := range query {
				dot += query[j] * v[j]
			}
			scores[i] = scored{id: ids[i], score: dot}
		}
		for i := 0; i < topK; i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}
		gt[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			gt[q][i] = scores[i].id
		}
	}
	return gt
}
