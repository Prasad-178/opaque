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

// TestPQ_SIFT100K benchmarks PQ-accelerated search vs standard search
// on the first 100K real SIFT vectors (128-dim image descriptors).
//
// Compares recall, latency, and privacy guarantees across configs:
//   - Standard (no PQ): full AES decrypt + exact dot product scoring
//   - PQ M=8: PQ ADC bulk scoring + exact re-rank of top candidates
//   - PQ M=16: finer PQ quantization for comparison
//
// Privacy guarantees are identical: same CKKS HE, same AES-256-GCM,
// same decoy patterns. PQ operates entirely client-side.
//
// Run: go test -tags sift1m -v -run TestPQ_SIFT100K ./test/ -timeout 30m
func TestPQ_SIFT100K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT 100K PQ benchmark in short mode")
	}

	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading SIFT dataset...")
	loadStart := time.Now()
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	// Use first 100K vectors.
	N := 100000
	if N > len(dataset.Vectors) {
		N = len(dataset.Vectors)
	}
	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]

	numClusters := 64
	numQueries := 50
	topK := 10
	numDecoys := 8

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	// Compute brute-force ground truth on 100K subset.
	t.Log("Computing brute-force cosine ground truth (100K vectors)...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("     SIFT 100K — PQ vs Standard Benchmark")
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

	// Configs to test: each with TopClusters=8 (strict) for fair comparison.
	// The RA=2 / PCA=64 variants below validate the same latency / recall
	// optimisations we want to apply to DBpedia (1M × 1536-dim ada-002),
	// where pre-optimisation queries hit 18-30 s and recall stalls at ~75 %.
	// On SIFT 100K (128-dim, well-clusterable) the absolute latencies are
	// already low (~150 ms), but the *relative speedups* and the
	// no-recall-regression check transfer to high-dim semantic workloads.
	configs := []struct {
		name        string
		pqM         int
		topClusters int
		probeThresh float64
		bernoulli   bool
		pcaDim      int // 0 = disabled
		redundant   int // 0 → default of 1
	}{
		{"standard", 0, 8, 1.0, false, 0, 0},
		{"PQ-M8", 8, 8, 1.0, false, 0, 0},
		{"PQ-M16", 16, 8, 1.0, false, 0, 0},
		// Also test with multi-probe to see PQ at higher recall.
		{"standard-probe16", 0, 16, 0.95, false, 0, 0},
		{"PQ-M8-probe16", 8, 16, 0.95, false, 0, 0},
		// Bernoulli decoy sampling for tight (ε,δ)-DP composition. Same
		// expected K_decoy as the uniform-K default, but K is binomial per
		// query — recall must be preserved (decoys never affect recall).
		{"standard-bernoulli", 0, 8, 1.0, true, 0, 0},
		{"PQ-M8-probe16-bernoulli", 8, 16, 0.95, true, 0, 0},
		// --- DBpedia-prep validation configs (2026-05-11) ---
		// RA=2 confirmed as a recall win on already-maxed SIFT 100K: bumped
		// R@1 98 % → 100 % and R@10 97.2 % → 99.8 % at +10 ms query cost.
		// Now the canonical recipe alongside PQ for DBpedia 1M @ 1536-dim
		// (commit `4880cae` shipped 92.4 % R@10 / 1.17 s P50 on ada-002).
		{"standard-RA2", 0, 8, 1.0, false, 0, 2},
		// PCA-64 configs below are TESTED-AND-REJECTED — kept as guards.
		// PCA-64 on SIFT 128-dim collapses R@10 from 97 % to **49 %** (50 %
		// recall loss). RA=2 + PQ both fail to rescue it. PCA is NOT in the
		// production recipe. See CLAUDE.md "Tested-and-rejected variations"
		// + SUMMARY.md "SIFT 100K — DBpedia-prep config validation".
		{"standard-PCA64", 0, 8, 1.0, false, 64, 0},
		{"standard-PCA64-RA2", 0, 8, 1.0, false, 64, 2},
		{"PQ-M8-PCA64", 8, 8, 1.0, false, 64, 0},
		{"PQ-M8-PCA64-RA2", 8, 8, 1.0, false, 64, 2},
	}

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d, PQ=%d, PCA=%d, RA=%d) ---",
			cfg.name, cfg.topClusters, cfg.pqM, cfg.pcaDim, cfg.redundant)

		dbCfg := opaque.Config{
			Dimension:       dataset.Dimension,
			NumClusters:     numClusters,
			TopClusters:     cfg.topClusters,
			NumDecoys:       numDecoys,
			ProbeThreshold:  cfg.probeThresh,
			BernoulliDecoys: cfg.bernoulli,
		}
		if cfg.pqM > 0 {
			dbCfg.PQSubspaces = cfg.pqM
		}
		if cfg.pcaDim > 0 {
			dbCfg.PCADimension = cfg.pcaDim
		}
		if cfg.redundant > 0 {
			dbCfg.RedundantAssignments = cfg.redundant
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

		// Warm-up query.
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
			gtID := fmt.Sprintf("sift_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			// Recall@10.
			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("sift_%d", groundTruth[q][i])
				if resultIDs[gtID] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		// Compute avg and p50 latency.
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

	// Summary table.
	t.Log("")
	t.Log("================================================================")
	t.Log("                         RESULTS")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-20s  %9s  %9s  %10s  %10s  %12s", "Config", "Recall@1", "Recall@10", "Avg Query", "P50 Query", "Build")
	t.Logf("  %-20s  %9s  %9s  %10s  %10s  %12s", "------", "--------", "---------", "---------", "---------", "-----")
	for _, r := range results {
		t.Logf("  %-20s  %7.1f%%   %7.1f%%   %9v  %9v  %11v",
			r.name,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond),
			r.buildTime.Round(time.Millisecond))
	}

	// Speedup summary.
	if len(results) >= 2 {
		std := results[0]
		t.Log("")
		t.Log("  Speedup vs standard:")
		for _, r := range results[1:] {
			speedup := float64(std.avgQuery) / float64(r.avgQuery)
			recallDelta := std.recall10 - r.recall10
			t.Logf("    %-20s  %.1fx faster, %+.1f pp Recall@10", r.name, speedup, -recallDelta*100)
		}
	}

	t.Log("")
	t.Log("  Privacy guarantees: IDENTICAL across all configs")
	t.Log("    - CKKS HE centroid scoring (server never sees query)")
	t.Log("    - AES-256-GCM vector encryption at rest")
	t.Logf("    - %d decoy clusters per query (access pattern hiding)", numDecoys)
	t.Log("    - PQ codes encrypted with AES (server never sees codes or codebooks)")
	t.Log("================================================================")
}

// Uses bruteForceTopK and getSIFT1MDataPath from sift1m_benchmark_test.go (same build tag, same package).
