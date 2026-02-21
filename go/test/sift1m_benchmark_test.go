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

// TestSIFT1MAccuracy benchmarks privacy-preserving search on the full SIFT1M dataset.
// This validates that the system handles 1 million real-world vectors with acceptable
// recall and latency.
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
	loadTime := time.Since(loadStart)

	stats := dataset.Stats()
	t.Logf("Loaded %d vectors (%d-dim) in %v", stats.NumVectors, stats.Dimension, loadTime)
	t.Logf("Queries: %d, Ground truth depth: %d", stats.NumQueries, stats.GroundTruthDepth)

	numClusters := 128
	topClusters := 64
	numQueries := 50
	topK := 10

	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("         SIFT1M ACCURACY BENCHMARK (1M Vectors)")
	t.Log("================================================================")
	t.Logf("Vectors:     %d", stats.NumVectors)
	t.Logf("Dimension:   %d", stats.Dimension)
	t.Logf("Clusters:    %d", numClusters)
	t.Logf("TopClusters: %d (%.0f%%)", topClusters, float64(topClusters)/float64(numClusters)*100)
	t.Logf("Queries:     %d", numQueries)
	t.Logf("TopK:        %d", topK)
	t.Logf("CPUs:        %d", runtime.NumCPU())
	t.Log("")

	// Build index
	t.Log("Building index...")
	db, err := opaque.NewDB(opaque.Config{
		Dimension:   stats.Dimension,
		NumClusters: numClusters,
		TopClusters: topClusters,
		NumDecoys:   0, // Disable decoys for pure accuracy measurement
	})
	if err != nil {
		t.Fatalf("NewDB failed: %v", err)
	}
	defer db.Close()

	addStart := time.Now()
	if err := db.AddBatch(ctx, dataset.IDs, dataset.Vectors); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}
	addTime := time.Since(addStart)
	t.Logf("AddBatch:    %v (%.0f vectors/sec)", addTime, float64(stats.NumVectors)/addTime.Seconds())

	buildStart := time.Now()
	if err := db.Build(ctx); err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	buildTime := time.Since(buildStart)
	t.Logf("Build:       %v", buildTime)
	t.Log("")

	// Run queries
	t.Log("Running queries...")
	var totalLatency time.Duration
	var recall1Sum, recall10Sum, recall100Sum float64

	for q := 0; q < numQueries; q++ {
		start := time.Now()
		results, err := db.Search(ctx, dataset.Queries[q], topK)
		elapsed := time.Since(start)
		totalLatency += elapsed

		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		// Compute recall against ground truth
		if len(dataset.GroundTruth) > q {
			gt := dataset.GroundTruth[q]
			resultIDs := make(map[string]bool)
			for _, r := range results {
				resultIDs[r.ID] = true
			}

			// Recall@1: is the top ground truth result in our results?
			if len(gt) > 0 {
				gtID := fmt.Sprintf("sift_%d", gt[0])
				if resultIDs[gtID] {
					recall1Sum++
				}
			}

			// Recall@10: how many of top-10 ground truth are in our results?
			gtTop10 := topK
			if len(gt) < gtTop10 {
				gtTop10 = len(gt)
			}
			hits10 := 0
			for i := 0; i < gtTop10; i++ {
				gtID := fmt.Sprintf("sift_%d", gt[i])
				if resultIDs[gtID] {
					hits10++
				}
			}
			recall10Sum += float64(hits10) / float64(gtTop10)

			// Recall@100: how many of top-100 ground truth overlap with our top-10?
			gtTop100 := 100
			if len(gt) < gtTop100 {
				gtTop100 = len(gt)
			}
			hits100 := 0
			for i := 0; i < gtTop100; i++ {
				gtID := fmt.Sprintf("sift_%d", gt[i])
				if resultIDs[gtID] {
					hits100++
				}
			}
			recall100Sum += float64(hits100) / float64(topK)
		}
	}

	avgLatency := totalLatency / time.Duration(numQueries)
	recall1 := recall1Sum / float64(numQueries)
	recall10 := recall10Sum / float64(numQueries)
	recall100 := recall100Sum / float64(numQueries)

	t.Log("")
	t.Log("================================================================")
	t.Log("                         RESULTS")
	t.Log("================================================================")
	t.Logf("  Build Time:        %v", buildTime)
	t.Logf("  Avg Query Latency: %v", avgLatency)
	t.Logf("  Throughput:        %.2f queries/sec", float64(numQueries)/totalLatency.Seconds())
	t.Log("")
	t.Logf("  Recall@1:          %.1f%%", recall1*100)
	t.Logf("  Recall@10:         %.1f%%", recall10*100)
	t.Logf("  Recall@100:        %.1f%%", recall100*100)
	t.Log("================================================================")

	// Sanity checks
	if recall10 < 0.3 {
		t.Errorf("Recall@10 too low: %.1f%% (expected >= 30%%)", recall10*100)
	}
}

// TestSIFT1MScaling tests how the system scales with increasing dataset size.
// Ground truth is computed via brute force within each subset so recall is
// meaningful at every scale.
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

	numClusters := 128
	topClusters := 64
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
	t.Logf("Clusters:    %d, TopClusters: %d", numClusters, topClusters)
	t.Logf("Queries:     %d, TopK: %d", numQueries, topK)
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
			t.Logf("Skipping size %d (dataset has %d vectors)", size, len(dataset.Vectors))
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
			NumDecoys:   0,
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

		// Run queries
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

	// Print summary table
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

	// Verify scaling
	if len(results) >= 2 {
		first := results[0]
		last := results[len(results)-1]
		sizeRatio := float64(last.size) / float64(first.size)
		latencyRatio := float64(last.avgQuery) / float64(first.avgQuery)

		t.Logf("  Size ratio:    %.1fx (%dK -> %dK)", sizeRatio, first.size/1000, last.size/1000)
		t.Logf("  Latency ratio: %.1fx", latencyRatio)
	}

	// All subsets should achieve good recall with proper ground truth
	for _, r := range results {
		if r.recall10 < 0.5 {
			t.Errorf("Recall@10 too low at %dK: %.1f%% (expected >= 50%%)", r.size/1000, r.recall10*100)
		}
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
