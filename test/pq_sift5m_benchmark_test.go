//go:build sift10m

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

// TestPQ_SIFT5M benchmarks PQ on 5M vectors with the best configs only.
// Skips standard baseline (too slow at 5M) and low-probe configs (poor recall).
// Focuses on PQ-M8 with probe16 and probe32 — the sweet spot for recall+latency.
//
// Run: go test -tags sift10m -v -run TestPQ_SIFT5M ./test/ -timeout 300m
func TestPQ_SIFT5M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT5M benchmark in short mode")
	}

	dataPath := getSIFT10MDataPath()
	if dataPath == "" {
		t.Skip("SIFT10M dataset not found")
	}

	t.Log("Loading SIFT10M dataset...")
	loadStart := time.Now()
	dataset, err := embeddings.SIFT10M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	N := 5000000
	if N > len(dataset.Vectors) {
		t.Fatalf("Need 5M vectors, only have %d", len(dataset.Vectors))
	}

	ids := dataset.IDs[:N]
	vectors := dataset.Vectors[:N]
	numQueries := 20
	topK := 10
	queries := dataset.Queries[:numQueries]
	numClusters := 256

	t.Log("Computing brute-force cosine ground truth (5M)...")
	gtStart := time.Now()
	gt := sift10mBruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth in %v", time.Since(gtStart))

	t.Log("")
	t.Log("================================================================")
	t.Log("     SIFT 5M — Best PQ Configs Only")
	t.Log("================================================================")
	t.Logf("Vectors: %dM, Dim: %d, Clusters: %d", N/1000000, dataset.Dimension, numClusters)
	t.Logf("Queries: %d, TopK: %d, Decoys: 8, CPUs: %d", numQueries, topK, runtime.NumCPU())
	t.Log("")

	ctx := context.Background()

	configs := []struct {
		name        string
		pqM         int
		topClusters int
		probeThresh float64
	}{
		{"PQ-M8-probe16", 8, 16, 0.95},
		{"PQ-M8-probe32", 8, 32, 0.95},
	}

	type result struct {
		name      string
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []result

	for _, cfg := range configs {
		t.Logf("--- %s ---", cfg.name)

		db, err := opaque.NewDB(opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      8,
			ProbeThreshold: cfg.probeThresh,
			PQSubspaces:    cfg.pqM,
		})
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

		db.Search(ctx, queries[0], topK) // warm-up

		latencies := make([]time.Duration, numQueries)
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			latencies[q] = time.Since(start)
			if err != nil {
				t.Fatalf("Search: %v", err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			if resultIDs[fmt.Sprintf("sift10m_%d", gt[q][0])] {
				recall1Sum++
			}
			hits := 0
			for i := 0; i < topK; i++ {
				if resultIDs[fmt.Sprintf("sift10m_%d", gt[q][i])] {
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

		results = append(results, result{
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
		runtime.GC()
	}

	t.Log("")
	t.Log("================================================================")
	t.Logf("  %-20s  %9s  %9s  %10s  %10s", "Config", "Recall@1", "Recall@10", "Avg Query", "P50")
	t.Logf("  %-20s  %9s  %9s  %10s  %10s", "------", "--------", "---------", "---------", "---")
	for _, r := range results {
		t.Logf("  %-20s  %7.1f%%   %7.1f%%   %9v  %9v",
			r.name, r.recall1*100, r.recall10*100,
			r.avgQuery.Round(time.Millisecond), r.p50Query.Round(time.Millisecond))
	}
	t.Log("================================================================")
}

// Uses getSIFT10MDataPath from pq_sift10m_benchmark_test.go (same build tag, same package).
// Uses sift10mBruteForceTopK from pq_sift10m_benchmark_test.go.
