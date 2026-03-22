//go:build gist

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

func getGISTDataPath() string {
	candidates := []string{
		"../data/gist",
		"../../data/gist",
		"../../../data/gist",
	}
	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(filepath.Join(absPath, "gist_base.fvecs")); err == nil {
			return absPath
		}
	}
	return ""
}

// TestGIST100K_PCA_Benchmark benchmarks PCA dimensionality reduction on GIST 100K
// (960-dim real image descriptors). Compares three configurations:
//
//   - Baseline: full 960-dim (8 centroids/pack, 4 HE ops for 32 clusters)
//   - PCA 960→256: (32 centroids/pack, 1 HE op for 32 clusters)
//   - PCA 960→128: (64 centroids/pack, 1 HE op for 32 clusters)
//
// PCA is applied client-side before CKKS encryption, so it has zero privacy impact.
// The server never sees original or reduced vectors.
func TestGIST100K_PCA_Benchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GIST PCA benchmark in short mode")
	}

	dataPath := getGISTDataPath()
	if dataPath == "" {
		t.Skip("GIST1M dataset not found; run scripts/download_gist1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading GIST1M dataset...")
	loadStart := time.Now()
	dataset, err := embeddings.GIST1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load GIST1M: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	// Use first 100K vectors for tractable runtime
	n := 100000
	if n > len(dataset.Vectors) {
		n = len(dataset.Vectors)
	}
	vectors := dataset.Vectors[:n]
	ids := dataset.IDs[:n]

	numQueries := 50
	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]

	numClusters := 32
	topK := 10
	numDecoys := 8

	// Compute brute-force ground truth on the ORIGINAL 960-dim vectors.
	// This is the true ground truth — PCA recall is measured against this.
	t.Log("Computing brute-force ground truth (960-dim)...")
	gtStart := time.Now()
	groundTruth := gistBruteForceTopK(queries, vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Test configs: baseline (no PCA), PCA→256, PCA→128
	type pcaConfig struct {
		name        string
		pcaDim      int // 0 = no PCA
		topClusters int
		probeThresh float64
	}

	configs := []pcaConfig{
		// Baseline: full 960-dim
		{"960d-probe-8", 0, 8, 0.95},
		{"960d-probe-16", 0, 16, 0.95},

		// PCA 960→256
		{"pca256-probe-8", 256, 8, 0.95},
		{"pca256-probe-16", 256, 16, 0.95},

		// PCA 960→128
		{"pca128-probe-8", 128, 8, 0.95},
		{"pca128-probe-16", 128, 16, 0.95},
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("      GIST 100K PCA BENCHMARK (960-dim Real Embeddings)")
	t.Log("================================================================")
	t.Logf("Vectors:     %d (%d-dim)", n, dataset.Dimension)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, n/numClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
	t.Logf("Decoys:      %d", numDecoys)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	type benchResult struct {
		name      string
		pcaDim    int
		buildTime time.Duration
		avgQuery  time.Duration
		p50Query  time.Duration
		recall1   float64
		recall10  float64
	}
	var results []benchResult

	for _, cfg := range configs {
		pcaLabel := "none"
		if cfg.pcaDim > 0 {
			pcaLabel = fmt.Sprintf("960→%d", cfg.pcaDim)
		}
		t.Logf("--- %s (PCA: %s, TopClusters=%d, Probe=%.2f) ---",
			cfg.name, pcaLabel, cfg.topClusters, cfg.probeThresh)

		dbCfg := opaque.Config{
			Dimension:      dataset.Dimension,
			NumClusters:    numClusters,
			TopClusters:    cfg.topClusters,
			NumDecoys:      numDecoys,
			ProbeThreshold: cfg.probeThresh,
		}
		if cfg.pcaDim > 0 {
			dbCfg.PCADimension = cfg.pcaDim
		}

		db, err := opaque.NewDB(dbCfg)
		if err != nil {
			t.Fatalf("NewDB failed for %s: %v", cfg.name, err)
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch failed for %s: %v", cfg.name, err)
		}

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build failed for %s: %v", cfg.name, err)
		}
		buildTime := time.Since(buildStart)
		t.Logf("  Build: %v", buildTime)

		// Warm up
		db.Search(ctx, queries[0], topK)

		latencies := make([]time.Duration, numQueries)
		var recall1Sum, recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, queries[q], topK)
			latencies[q] = time.Since(start)

			if err != nil {
				t.Fatalf("Search %d failed for %s: %v", q, cfg.name, err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			// Recall@1 — against original 960-dim ground truth
			gtID := fmt.Sprintf("gist_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			// Recall@10
			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("gist_%d", groundTruth[q][i])
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
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		avgLatency := totalLatency / time.Duration(numQueries)
		p50Latency := latencies[numQueries/2]
		r1 := recall1Sum / float64(numQueries)
		r10 := recall10Sum / float64(numQueries)

		results = append(results, benchResult{
			name:      cfg.name,
			pcaDim:    cfg.pcaDim,
			buildTime: buildTime,
			avgQuery:  avgLatency,
			p50Query:  p50Latency,
			recall1:   r1,
			recall10:  r10,
		})

		t.Logf("  Recall@1: %.1f%%, Recall@10: %.1f%%, Avg: %v, P50: %v",
			r1*100, r10*100, avgLatency.Round(time.Millisecond), p50Latency.Round(time.Millisecond))

		db.Close()
	}

	// Summary table
	t.Log("")
	t.Log("================================================================")
	t.Log("                         RESULTS")
	t.Log("================================================================")
	t.Log("")
	t.Logf("  %-18s  %6s  %8s  %9s  %9s  %10s  %10s",
		"Config", "PCA", "Build", "Recall@1", "Recall@10", "Avg Query", "P50 Query")
	t.Logf("  %-18s  %6s  %8s  %9s  %9s  %10s  %10s",
		"------", "---", "-----", "--------", "---------", "---------", "---------")
	for _, r := range results {
		pcaLabel := "none"
		if r.pcaDim > 0 {
			pcaLabel = fmt.Sprintf("→%d", r.pcaDim)
		}
		t.Logf("  %-18s  %6s  %7v   %7.1f%%   %7.1f%%   %9v   %9v",
			r.name, pcaLabel,
			r.buildTime.Round(time.Millisecond),
			r.recall1*100, r.recall10*100,
			r.avgQuery.Round(time.Millisecond),
			r.p50Query.Round(time.Millisecond))
	}

	t.Log("")
	t.Log("  PCA is applied client-side before encryption — zero privacy impact.")
	t.Logf("  Ground truth: brute-force cosine on original %d-dim vectors.", dataset.Dimension)
	t.Logf("  Each cluster: ~%d vectors, Decoys: %d per query", n/numClusters, numDecoys)
	t.Log("================================================================")
}

// gistBruteForceTopK computes exact cosine similarity top-K for ground truth.
func gistBruteForceTopK(queries, vectors [][]float64, dim, topK int) [][]int {
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
			denom := math.Sqrt(normA) * math.Sqrt(normB)
			if denom > 0 {
				scores[i] = scored{idx: i, score: dot / denom}
			}
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
