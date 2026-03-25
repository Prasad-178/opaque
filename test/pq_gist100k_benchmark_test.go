//go:build gist

package test

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

// TestPQ_GIST100K benchmarks PQ-accelerated search on real GIST 960-dim vectors.
// This is the critical high-dimensional case where PQ + GPU would give the
// largest combined speedup.
//
// Run: go test -tags gist -v -run TestPQ_GIST100K ./test/ -timeout 60m
func TestPQ_GIST100K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GIST 100K PQ benchmark in short mode")
	}

	dataPath := getGISTDataPath()
	if dataPath == "" {
		t.Skip("GIST dataset not found; run scripts/download_gist1m.sh first")
	}

	t.Log("Loading GIST dataset (960-dim)...")
	loadStart := time.Now()
	dataset, err := embeddings.GIST1M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	N := 100000
	if N > len(dataset.Vectors) {
		N = len(dataset.Vectors)
	}
	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]
	dim := dataset.Dimension // 960

	numQueries := 20
	topK := 10
	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	// Brute-force ground truth.
	t.Log("Computing brute-force cosine ground truth (100K × 960-dim)...")
	gtStart := time.Now()
	groundTruth := gistBruteForceTopK(queries, vectors, dim, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("     GIST 100K (960-dim) — PQ vs Standard Benchmark")
	t.Log("================================================================")
	t.Logf("Vectors:     %d (real GIST 960-dim image descriptors)", N)
	t.Logf("Clusters:    32 (~%d vectors each)", N/32)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      8 per query")
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	type benchResult struct {
		name      string
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []benchResult

	configs := []struct {
		name        string
		pqM         int
		topClusters int
		probeThresh float64
	}{
		// Standard baselines.
		{"standard-probe8", 0, 8, 0.95},
		{"standard-probe16", 0, 16, 0.95},
		// PQ configs. 960 / M must be integer. M=32 → 30-dim subspaces, M=48 → 20-dim.
		{"PQ-M32-probe8", 32, 8, 0.95},
		{"PQ-M48-probe8", 48, 8, 0.95},
		{"PQ-M32-probe16", 32, 16, 0.95},
	}

	ctx := context.Background()

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d, PQ=%d) ---", cfg.name, cfg.topClusters, cfg.pqM)

		dbCfg := opaque.Config{
			Dimension:      dim,
			NumClusters:    32,
			TopClusters:    cfg.topClusters,
			NumDecoys:      8,
			ProbeThreshold: cfg.probeThresh,
		}
		if cfg.pqM > 0 {
			dbCfg.PQSubspaces = cfg.pqM
		}

		db, err := opaque.NewDB(dbCfg)
		if err != nil {
			t.Fatalf("NewDB: %v", err)
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch: %v", err)
		}

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build: %v", err)
		}
		buildTime := time.Since(buildStart)

		// Warm-up.
		db.Search(ctx, queries[0], topK)

		latencies := make([]time.Duration, numQueries)
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			latencies[q] = time.Since(start)
			if err != nil {
				t.Fatalf("Search %d: %v", q, err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			// Recall@1.
			if resultIDs[fmt.Sprintf("gist_%d", groundTruth[q][0])] {
				recall1Sum++
			}

			// Recall@10.
			hits := 0
			for i := 0; i < topK; i++ {
				if resultIDs[fmt.Sprintf("gist_%d", groundTruth[q][i])] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		var totalLatency time.Duration
		for _, l := range latencies {
			totalLatency += l
		}
		avgLatency := totalLatency / time.Duration(numQueries)

		sorted := make([]time.Duration, numQueries)
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		p50 := sorted[numQueries/2]

		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, benchResult{
			name:      cfg.name,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			p50Query:  p50,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Build: %v, Recall@1: %.1f%%, Recall@10: %.1f%%, Avg: %v, P50: %v",
			buildTime.Round(time.Millisecond), r1*100, r10*100, avgLatency, p50)

		db.Close()
	}

	// Summary.
	t.Log("")
	t.Log("================================================================")
	t.Log("                         RESULTS")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-22s  %9s  %9s  %10s  %10s  %12s", "Config", "Recall@1", "Recall@10", "Avg Query", "P50 Query", "Build")
	t.Logf("  %-22s  %9s  %9s  %10s  %10s  %12s", "------", "--------", "---------", "---------", "---------", "-----")
	for _, r := range results {
		t.Logf("  %-22s  %7.1f%%   %7.1f%%   %9v  %9v  %11v",
			r.name,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond),
			r.buildTime.Round(time.Millisecond))
	}

	// Speedup vs first baseline.
	if len(results) >= 2 {
		std := results[0]
		t.Log("")
		t.Log("  Speedup vs standard-probe8:")
		for _, r := range results[1:] {
			speedup := float64(std.avgQuery) / float64(r.avgQuery)
			recallDelta := r.recall10 - std.recall10
			t.Logf("    %-22s  %.2fx  %+.1f pp Recall@10", r.name, speedup, recallDelta*100)
		}
	}

	t.Log("")
	t.Log("  Privacy: IDENTICAL across all configs (CKKS HE + AES-256-GCM + 8 decoys)")
	t.Log("================================================================")
}

// Uses gistBruteForceTopK and getGISTDataPath from gist_pca_benchmark_test.go (same build tag, same package).
