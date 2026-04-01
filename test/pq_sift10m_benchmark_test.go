//go:build sift10m

package test

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque"
	"github.com/Prasad-178/opaque/pkg/embeddings"
)

func getSIFT10MDataPath() string {
	candidates := []string{
		"../data/bigann",
		"../../data/bigann",
		"../../../data/bigann",
	}
	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(filepath.Join(absPath, "sift10m_base.bvecs")); err == nil {
			return absPath
		}
	}
	return ""
}

// TestPQ_SIFT10M benchmarks PQ-accelerated search at multi-million scale.
// Tests 2M and 5M subsets of SIFT10M (10M exceeds 16GB RAM as float64).
// No other HE-based private search system has published results at this scale.
//
// Run: go test -tags sift10m -v -run TestPQ_SIFT10M ./test/ -timeout 120m
func TestPQ_SIFT10M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT10M benchmark in short mode")
	}

	dataPath := getSIFT10MDataPath()
	if dataPath == "" {
		t.Skip("SIFT10M dataset not found; run scripts/download_sift10m.sh first")
	}

	t.Log("Loading SIFT10M dataset (this may take a minute)...")
	loadStart := time.Now()
	dataset, err := embeddings.SIFT10M(dataPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	numQueries := 20
	topK := 10
	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	// Test at multiple scales. 10M float64 = ~10GB, too much for 16GB RAM.
	// 5M = ~5GB, 2M = ~2GB — both fit comfortably.
	scales := []int{2000000, 5000000}

	// Check if we have enough vectors
	maxScale := scales[len(scales)-1]
	if maxScale > len(dataset.Vectors) {
		t.Logf("WARNING: dataset has %d vectors, capping at available", len(dataset.Vectors))
		// Adjust scales
		var adjusted []int
		for _, s := range scales {
			if s <= len(dataset.Vectors) {
				adjusted = append(adjusted, s)
			}
		}
		scales = adjusted
	}

	ctx := context.Background()

	t.Log("")
	t.Log("================================================================")
	t.Log("     SIFT Multi-Million Scale — PQ Benchmark")
	t.Log("================================================================")
	t.Logf("Scales: %v", scales)
	t.Logf("Queries: %d, TopK: %d", numQueries, topK)
	t.Logf("CPUs: %d, RAM: 16GB", runtime.NumCPU())
	t.Log("")

	type benchResult struct {
		scale     int
		name      string
		clusters  int
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		recall1   float64
		recall10  float64
	}
	var allResults []benchResult

	for _, N := range scales {
		t.Logf("===== %dM vectors =====", N/1000000)

		ids := dataset.IDs[:N]
		vectors := dataset.Vectors[:N]

		// Compute brute-force ground truth at this scale.
		t.Logf("Computing brute-force cosine ground truth (%dM)...", N/1000000)
		gtStart := time.Now()
		gt := sift10mBruteForceTopK(queries, vectors, dataset.Dimension, topK)
		t.Logf("Ground truth in %v", time.Since(gtStart))

		// Choose cluster count based on scale.
		numClusters := 256
		if N <= 2000000 {
			numClusters = 128
		}

		configs := []struct {
			name        string
			pqM         int
			topClusters int
			probeThresh float64
		}{
			{"standard-strict8", 0, 8, 1.0},
			{"PQ-M8-strict8", 8, 8, 1.0},
			{"PQ-M8-strict16", 8, 16, 1.0},
			{"PQ-M8-probe16", 8, 16, 0.95},
			{"PQ-M8-probe32", 8, 32, 0.95},
		}

		for _, cfg := range configs {
			t.Logf("  --- %s (clusters=%d, top=%d, PQ=%d) ---",
				cfg.name, numClusters, cfg.topClusters, cfg.pqM)

			dbCfg := opaque.Config{
				Dimension:      dataset.Dimension,
				NumClusters:    numClusters,
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
				results, err := db.Search(ctx, queries[q], topK)
				latencies[q] = time.Since(start)
				if err != nil {
					t.Fatalf("Search: %v", err)
				}

				resultIDs := make(map[string]bool)
				for _, r := range results {
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

			allResults = append(allResults, benchResult{
				scale:     N,
				name:      cfg.name,
				clusters:  numClusters,
				buildTime: buildTime,
				avgQuery:  avgLatency,
				p50Query:  p50,
				recall1:   r1,
				recall10:  r10,
			})

			t.Logf("    Build: %v, Recall@1: %.1f%%, Recall@10: %.1f%%, Avg: %v, P50: %v",
				buildTime.Round(time.Millisecond), r1*100, r10*100,
				avgLatency.Round(time.Millisecond), p50.Round(time.Millisecond))

			db.Close()
			runtime.GC() // Free memory between configs
		}

		runtime.GC() // Free memory between scales
		t.Log("")
	}

	// Summary.
	t.Log("================================================================")
	t.Log("                    FULL RESULTS")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-6s  %-20s  %4s  %9s  %9s  %10s  %10s",
		"Scale", "Config", "K", "Recall@1", "Recall@10", "Avg Query", "P50")
	t.Logf("  %-6s  %-20s  %4s  %9s  %9s  %10s  %10s",
		"-----", "------", "-", "--------", "---------", "---------", "---")
	for _, r := range allResults {
		t.Logf("  %-6s  %-20s  %4d  %7.1f%%   %7.1f%%   %9v  %9v",
			fmt.Sprintf("%dM", r.scale/1000000),
			r.name,
			r.clusters,
			r.recall1*100,
			r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond))
	}

	t.Log("")
	t.Log("  Privacy: IDENTICAL across all configs (CKKS HE + AES-256-GCM + 8 decoys)")
	t.Log("  Ground truth: brute-force cosine similarity at each scale")
	t.Log("  NOTE: No other HE-based system has published benchmarks at multi-million scale")
	t.Log("================================================================")
}

func sift10mBruteForceTopK(queries, vectors [][]float64, dim, topK int) [][]int {
	type scored struct {
		idx   int
		score float64
	}
	result := make([][]int, len(queries))
	for q := range queries {
		scores := make([]scored, len(vectors))
		for i := range vectors {
			var dot, normA, normB float64
			for d := 0; d < dim; d++ {
				dot += queries[q][d] * vectors[i][d]
				normA += queries[q][d] * queries[q][d]
				normB += vectors[i][d] * vectors[i][d]
			}
			scores[i] = scored{idx: i, score: dot / (math.Sqrt(normA) * math.Sqrt(normB))}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})
		result[q] = make([]int, topK)
		for i := 0; i < topK && i < len(scores); i++ {
			result[q][i] = scores[i].idx
		}
	}
	return result
}
