//go:build sift1m

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

	"github.com/opaque/opaque/go"
	"github.com/opaque/opaque/go/pkg/embeddings"
)

func getSIFT1MDataPath() string {
	candidates := []string{
		"../data/sift",
		"../../data/sift",
		"../../../data/sift",
	}
	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(filepath.Join(absPath, "sift_base.fvecs")); err == nil {
			return absPath
		}
	}
	return ""
}

// TestSIFT1MAccuracy benchmarks privacy-preserving search on 1M real vectors
// across multiple configurations to show the recall/latency/privacy tradeoff.
//
// Each config varies TopClusters (how many clusters we probe out of 128 total).
// More probing = better recall but slower queries and weaker access pattern privacy.
func TestSIFT1MAccuracy(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M benchmark in short mode")
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

	numClusters := 128
	numQueries := 50
	topK := 10
	numDecoys := 8 // production-realistic decoy count

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}

	// Compute brute-force ground truth for proper recall measurement
	t.Log("Computing brute-force ground truth...")
	gtStart := time.Now()
	groundTruth := bruteForceTopK(dataset.Queries[:numQueries], dataset.Vectors, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Test configs: varying TopClusters from aggressive (4) to conservative (32)
	// With 128 clusters and 1M vectors, each cluster has ~7,800 vectors.
	configs := []struct {
		name        string
		topClusters int
		probeThresh float64 // 1.0 = strict top-K only, 0.95 = multi-probe
	}{
		{"strict-4", 4, 1.0},   // 3.1% of data, max privacy
		{"strict-8", 8, 1.0},   // 6.2% of data
		{"strict-16", 16, 1.0}, // 12.5% of data
		{"probe-8", 8, 0.95},   // 6.2%+ with multi-probe expansion
		{"probe-16", 16, 0.95}, // 12.5%+ with multi-probe expansion
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("         SIFT1M ACCURACY BENCHMARK (1M Vectors)")
	t.Log("================================================================")
	t.Logf("Vectors:     %d", len(dataset.Vectors))
	t.Logf("Dimension:   %d", dataset.Dimension)
	t.Logf("Clusters:    %d (~%d vectors each)", numClusters, len(dataset.Vectors)/numClusters)
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

		// Warm up
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

			// Recall@1
			gtID := fmt.Sprintf("sift_%d", groundTruth[q][0])
			if resultIDs[gtID] {
				recall1Sum++
			}

			// Recall@10
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

	// Summary table
	t.Log("")
	t.Log("================================================================")
	t.Log("                         RESULTS")
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

// TestSIFT1MScaling tests how the system scales with increasing dataset size
// using a fixed production-realistic config. Ground truth is computed via
// brute force within each subset.
func TestSIFT1MScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT1M scaling benchmark in short mode")
	}

	dataPath := getSIFT1MDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	t.Log("Loading SIFT1M dataset...")
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT1M: %v", err)
	}

	// Production-realistic config
	numClusters := 128
	topClusters := 8  // 6.2% probe — realistic
	numDecoys := 8    // production decoy count
	numQueries := 20
	topK := 10
	sizes := []int{100000, 500000, 1000000}

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("         SIFT1M SCALING BENCHMARK")
	t.Log("================================================================")
	t.Logf("Config: %d clusters, top %d probed (%.1f%%), %d decoys, probe_thresh=0.95",
		numClusters, topClusters, float64(topClusters)/float64(numClusters)*100, numDecoys)
	t.Logf("Queries: %d, TopK: %d", numQueries, topK)
	t.Log("")

	type scalingResult struct {
		size      int
		buildTime time.Duration
		avgQuery  time.Duration
		recall10  float64
	}
	var results []scalingResult

	for _, size := range sizes {
		if size > len(dataset.Vectors) {
			continue
		}

		t.Logf("--- %dK vectors ---", size/1000)

		ids := dataset.IDs[:size]
		vectors := dataset.Vectors[:size]

		// Compute brute-force ground truth within this subset
		t.Log("  Computing brute-force ground truth...")
		localGT := bruteForceTopK(dataset.Queries[:numQueries], vectors, dataset.Dimension, topK)

		db, err := opaque.NewDB(opaque.Config{
			Dimension:   dataset.Dimension,
			NumClusters: numClusters,
			TopClusters: topClusters,
			NumDecoys:   numDecoys,
		})
		if err != nil {
			t.Fatalf("NewDB failed: %v", err)
		}

		if err := db.AddBatch(ctx, ids, vectors); err != nil {
			t.Fatalf("AddBatch failed: %v", err)
		}

		buildStart := time.Now()
		if err := db.Build(ctx); err != nil {
			t.Fatalf("Build failed for size %d: %v", size, err)
		}
		buildTime := time.Since(buildStart)

		// Warm up
		db.Search(ctx, dataset.Queries[0], topK)

		var totalLatency time.Duration
		var recall10Sum float64

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			searchResults, err := db.Search(ctx, dataset.Queries[q], topK)
			totalLatency += time.Since(start)

			if err != nil {
				t.Fatalf("Search failed: %v", err)
			}

			resultIDs := make(map[string]bool)
			for _, r := range searchResults {
				resultIDs[r.ID] = true
			}

			hits := 0
			for i := 0; i < topK; i++ {
				gtID := fmt.Sprintf("sift_%d", localGT[q][i])
				if resultIDs[gtID] {
					hits++
				}
			}
			recall10Sum += float64(hits) / float64(topK)
		}

		avgQuery := totalLatency / time.Duration(numQueries)
		recall10 := recall10Sum / float64(numQueries)

		results = append(results, scalingResult{
			size:      size,
			buildTime: buildTime,
			avgQuery:  avgQuery,
			recall10:  recall10,
		})

		t.Logf("  Build: %v, Query: %v, Recall@10: %.1f%%",
			buildTime, avgQuery, recall10*100)

		db.Close()
	}

	// Summary
	t.Log("")
	t.Log("================================================================")
	t.Log("                    SCALING SUMMARY")
	t.Log("================================================================")
	t.Log("")
	t.Log("  Vectors    Build Time    Avg Query    Recall@10")
	t.Log("  -------    ----------    ---------    ---------")
	for _, r := range results {
		t.Logf("  %7dK   %10v    %9v    %8.1f%%",
			r.size/1000, r.buildTime, r.avgQuery, r.recall10*100)
	}
	t.Log("")

	if len(results) >= 2 {
		first := results[0]
		last := results[len(results)-1]
		sizeRatio := float64(last.size) / float64(first.size)
		latencyRatio := float64(last.avgQuery) / float64(first.avgQuery)
		t.Logf("  Size ratio:    %.1fx (%dK -> %dK)", sizeRatio, first.size/1000, last.size/1000)
		t.Logf("  Latency ratio: %.1fx", latencyRatio)
	}
}

// bruteForceTopK computes exact top-K nearest neighbors by cosine similarity.
func bruteForceTopK(queries, vectors [][]float64, dim, topK int) [][]int {
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
