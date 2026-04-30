//go:build sift1m

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

// TestPQ_SIFT1M benchmarks PQ-accelerated search on 1M real SIFT vectors.
// Compares standard vs PQ across multiple probe configs.
//
// Run: go test -tags sift1m -v -run TestPQ_SIFT1M ./test/ -timeout 60m
func TestPQ_SIFT1M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M PQ benchmark in short mode")
	}

	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading SIFT1M dataset...")
	loadStart := time.Now()
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	N := 1000000
	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]

	numClusters := 128
	numQueries := 50
	topK := 10
	numDecoys := 8

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	// Brute-force ground truth on full 1M.
	t.Log("Computing brute-force cosine ground truth (1M vectors)...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("     SIFT 1M (128-dim) — PQ vs Standard Benchmark")
	t.Log("================================================================")
	t.Logf("Vectors:     %d (real SIFT 128-dim image descriptors)", N)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, N/numClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d per query", numDecoys)
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
		{"standard-strict8", 0, 8, 1.0},
		{"standard-strict16", 0, 16, 1.0},
		// PQ M=8 (best for 128-dim from SIFT100K results).
		{"PQ-M8-strict8", 8, 8, 1.0},
		{"PQ-M8-strict16", 8, 16, 1.0},
		{"PQ-M8-strict32", 8, 32, 1.0},
		// PQ with multi-probe.
		{"PQ-M8-probe16", 8, 16, 0.95},
		{"PQ-M8-probe32", 8, 32, 0.95},
	}

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d, PQ=%d, probe=%.2f) ---",
			cfg.name, cfg.topClusters, cfg.pqM, cfg.probeThresh)

		dbCfg := opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
			// All four mitigations live: σ=2^30 + DecodePublic shipped library-wide;
			// permutation π always-on at build; padding + ε opt-in here.
			PaddingMode:   opaque.PaddingBucketed,
			TargetEpsilon: 2.0,
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

			gtID := fmt.Sprintf("sift_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("sift_%d", groundTruth[q][i])
				if resultIDs[gtID] {
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
			buildTime.Round(time.Millisecond), r1*100, r10*100,
			avgLatency.Round(time.Millisecond), p50.Round(time.Millisecond))

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

	if len(results) >= 2 {
		std := results[0]
		t.Log("")
		t.Log("  Speedup vs standard-strict8:")
		for _, r := range results[1:] {
			speedup := float64(std.avgQuery) / float64(r.avgQuery)
			recallDelta := r.recall10 - std.recall10
			t.Logf("    %-22s  %.2fx  %+.1f pp Recall@10", r.name, speedup, recallDelta*100)
		}
	}

	t.Log("")
	t.Log("  Privacy: IDENTICAL across all configs (CKKS HE + AES-256-GCM + 8 decoys)")
	t.Log("  All recall measured against brute-force cosine similarity over full 1M dataset")
	t.Log("================================================================")
}
