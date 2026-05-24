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

// TestSIFT1MAccuracy_NC256 mirrors TestSIFT1MAccuracy but with NumClusters=256.
// Compares against the headline NC=128 numbers (probe-8: 99.8% R@10 / 464ms on
// AWS m6i.2xlarge as of May 2026). Hypothesis: smaller per-cluster blobs reduce
// fetch + local-scoring time, possibly winning latency at equal recall.
//
// TopClusters values are scaled 2x relative to NC=128 to keep coverage % equal.
//
//	NC=128 probe-8 (6.25%)  ↔  NC=256 probe-16 (6.25%)
//	NC=128 probe-16 (12.5%) ↔  NC=256 probe-32 (12.5%)
func TestSIFT1MAccuracy_NC256(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M NC=256 benchmark in short mode")
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
		t.Fatalf("Failed to load SIFT1M: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	numClusters := 256
	numQueries := 50
	topK := 10
	numDecoys := 8

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}

	t.Log("Computing brute-force ground truth...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(dataset.Queries[:numQueries], dataset.Vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Coverage % scaled to mirror NC=128 configs.
	configs := []struct {
		name        string
		topClusters int
		probeThresh float64
	}{
		{"strict-8", 8, 1.0},   // 3.1% — mirrors NC=128 strict-4
		{"strict-16", 16, 1.0}, // 6.2% — mirrors NC=128 strict-8
		{"strict-32", 32, 1.0}, // 12.5% — mirrors NC=128 strict-16
		{"probe-16", 16, 0.95}, // 6.2%+ — mirrors NC=128 probe-8 (HEADLINE comparison)
		{"probe-32", 32, 0.95}, // 12.5%+ — mirrors NC=128 probe-16
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("       SIFT1M ACCURACY BENCHMARK — NumClusters=256")
	t.Log("================================================================")
	t.Logf("Vectors:     %d", len(dataset.Vectors))
	t.Logf("Dimension:   %d", dataset.Dimension)
	t.Logf("Clusters:    %d (~%d vectors each, vs ~%d at NC=128)",
		numClusters, len(dataset.Vectors)/numClusters, len(dataset.Vectors)/128)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d", numDecoys)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	type result struct {
		name      string
		topK      int
		probe     float64
		buildTime time.Duration
		avgQuery  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []result

	for _, cfg := range configs {
		t.Logf("--- Config: %s (TopClusters=%d, ProbeThreshold=%.2f) ---",
			cfg.name, cfg.topClusters, cfg.probeThresh)

		db, err := opaque.NewDB(opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
			PaddingMode:    opaque.PaddingBucketed,
			TargetEpsilon:  2.5,
		})
		if err != nil {
			t.Fatalf("NewDB failed: %v", err)
		}

		if err := db.AddBatch(ctx, dataset.IDs, dataset.Vectors); err != nil {
			t.Fatalf("AddBatch failed: %v", err)
		}

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build failed: %v", err)
		}
		buildTime := time.Since(buildStart)

		db.Search(ctx, dataset.Queries[0], topK)

		var totalLatency time.Duration
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, dataset.Queries[q], topK)
			totalLatency += time.Since(start)

			if err != nil {
				t.Fatalf("Search %d failed: %v", q, err)
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

		avgLatency := totalLatency / time.Duration(numQueries)
		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, result{
			name:      cfg.name,
			topK:      cfg.topClusters,
			probe:     cfg.probeThresh,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Recall@1: %.1f%%, Recall@10: %.1f%%, Avg Query: %v",
			r1*100, r10*100, avgLatency)

		db.Close()
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("                  RESULTS — NumClusters=256")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-12s  %5s  %6s  %9s  %9s  %10s", "Config", "Probe", "Multi", "Recall@1", "Recall@10", "Avg Query")
	t.Logf("  %-12s  %5s  %6s  %9s  %9s  %10s", "------", "-----", "-----", "--------", "---------", "---------")
	for _, r := range results {
		multiProbe := "no"
		if r.probe < 1.0 {
			multiProbe = "yes"
		}
		t.Logf("  %-12s  %4d%%  %5s   %7.1f%%   %7.1f%%   %9v",
			r.name,
			r.topK*100/numClusters,
			multiProbe,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery)
	}
	t.Log("")
	t.Logf("  Each cluster: ~%d vectors, Decoys: %d per query", len(dataset.Vectors)/numClusters, numDecoys)
	t.Log("================================================================")
}

// TestPQ_SIFT1M_NC256 mirrors TestPQ_SIFT1M but with NumClusters=256.
// Same TopClusters scaling rule (2x of NC=128 configs to match coverage %).
func TestPQ_SIFT1M_NC256(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M PQ NC=256 benchmark in short mode")
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

	numClusters := 256
	numQueries := 50
	topK := 10
	numDecoys := 8

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	t.Log("Computing brute-force cosine ground truth (1M vectors)...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("    SIFT 1M (128-dim) — PQ vs Standard, NumClusters=256")
	t.Log("================================================================")
	t.Logf("Vectors:     %d (real SIFT 128-dim image descriptors)", N)
	t.Logf("Clusters:    %d (~%d vectors each, vs ~%d at NC=128)",
		numClusters, N/numClusters, N/128)
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

	// TopClusters scaled 2x of NC=128 PQ configs.
	configs := []struct {
		name        string
		pqM         int
		topClusters int
		probeThresh float64
		epsilon     float64
	}{
		// HEADLINE comparison: mirrors NC=128 PQ-M8-probe8-eps25.
		{"PQ-M8-probe16-eps25", 8, 16, 0.95, 2.5},
		{"PQ-M8-probe32-eps25", 8, 32, 0.95, 2.5},
		{"PQ-M8-probe16-eps271", 8, 16, 0.95, 2.71},
		{"PQ-M8-probe16-eps20", 8, 16, 0.95, 2.0},
		{"standard-probe16-eps25", 0, 16, 0.95, 2.5},
	}

	for _, cfg := range configs {
		t.Logf("--- %s (TopClusters=%d, PQ=%d, probe=%.2f, eps=%.2f) ---",
			cfg.name, cfg.topClusters, cfg.pqM, cfg.probeThresh, cfg.epsilon)

		dbCfg := opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
			PaddingMode:    opaque.PaddingBucketed,
			TargetEpsilon:  cfg.epsilon,
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

	t.Log("")
	t.Log("================================================================")
	t.Log("              RESULTS — NumClusters=256")
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

	t.Log("")
	t.Log("  Compare to NC=128 PQ-M8-probe8-eps25 headline (~409ms / 98.4%% R@10)")
	t.Log("================================================================")
}
