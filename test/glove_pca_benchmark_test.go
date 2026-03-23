//go:build glove

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
	"github.com/Prasad-178/opaque/pkg/pca"
)

func getGloVeDataPath() string {
	candidates := []string{
		"../data/glove/glove.6B.300d.txt",
		"../../data/glove/glove.6B.300d.txt",
		"../../../data/glove/glove.6B.300d.txt",
	}
	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(absPath); err == nil {
			return absPath
		}
	}
	return ""
}

// TestGloVe_PCA_Benchmark benchmarks PCA on GloVe 6B 300-dim word vectors.
//
// GloVe embeddings have concentrated variance — the top principal components
// capture most of the information, making PCA highly effective. This contrasts
// with GIST image descriptors where variance is spread uniformly across all
// 960 dimensions and PCA destroys recall.
//
// Run with:
//
//	go test -tags glove -v -run TestGloVe_PCA_Benchmark ./test/ -timeout 30m
func TestGloVe_PCA_Benchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GloVe PCA benchmark in short mode")
	}

	dataPath := getGloVeDataPath()
	if dataPath == "" {
		t.Skip("GloVe 6B 300d not found; download from https://nlp.stanford.edu/projects/glove/")
	}

	ctx := context.Background()

	t.Log("Loading GloVe 6B 300d...")
	loadStart := time.Now()
	dataset, err := embeddings.GloVe(dataPath)
	if err != nil {
		t.Fatalf("Failed to load GloVe: %v", err)
	}
	t.Logf("Loaded %d vectors (%d-dim) in %v", len(dataset.Vectors), dataset.Dimension, time.Since(loadStart))

	// Use first 100K vectors for tractable runtime
	n := 100000
	if n > len(dataset.Vectors) {
		n = len(dataset.Vectors)
	}
	vectors := dataset.Vectors[:n]
	ids := dataset.IDs[:n]

	// Use a separate set of vectors as queries (next 50 after the index set)
	numQueries := 50
	queryStart := n
	if queryStart+numQueries > len(dataset.Vectors) {
		// Fall back to using vectors from within the index
		queryStart = n - numQueries
	}
	queries := dataset.Vectors[queryStart : queryStart+numQueries]

	numClusters := 32
	topK := 10
	numDecoys := 8

	// Compute brute-force ground truth on the ORIGINAL 300-dim vectors.
	t.Log("Computing brute-force ground truth (300-dim)...")
	gtStart := time.Now()
	groundTruth := gloveBruteForceTopK(queries, vectors, ids, dataset.Dimension, topK)
	t.Logf("Ground truth computed in %v", time.Since(gtStart))

	// Measure PCA variance explained — expect concentrated variance for GloVe.
	t.Log("Measuring PCA variance explained...")
	pcaSample := vectors
	for _, targetDim := range []int{32, 64, 96, 128, 192, 256} {
		model, err := pca.Fit(pcaSample, targetDim)
		if err != nil {
			t.Logf("  PCA %d→%d: fit failed: %v", dataset.Dimension, targetDim, err)
			continue
		}
		t.Logf("  PCA %d→%d: variance explained = %.2f%%",
			dataset.Dimension, targetDim, model.TotalVarianceExplained()*100)
	}
	t.Log("")

	type pcaConfig struct {
		name        string
		pcaDim      int // 0 = no PCA
		topClusters int
		probeThresh float64
	}

	configs := []pcaConfig{
		// Baseline: full 300-dim
		{"300d-probe-8", 0, 8, 0.95},
		{"300d-probe-16", 0, 16, 0.95},

		// PCA 300→128 (should be efficient — 64 centroids/pack, 1 HE op)
		{"pca128-probe-8", 128, 8, 0.95},
		{"pca128-probe-16", 128, 16, 0.95},

		// PCA 300→96 (96 centroids/pack, 1 HE op)
		{"pca96-probe-8", 96, 8, 0.95},
		{"pca96-probe-16", 96, 16, 0.95},

		// PCA 300→64 (128 centroids/pack, 1 HE op)
		{"pca64-probe-8", 64, 8, 0.95},
		{"pca64-probe-16", 64, 16, 0.95},
	}

	t.Log("")
	t.Log("================================================================")
	t.Log("     GloVe 100K PCA BENCHMARK (300-dim Word Embeddings)")
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
			pcaLabel = fmt.Sprintf("300→%d", cfg.pcaDim)
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

			// Recall@1 — against original 300-dim ground truth
			if resultIDs[groundTruth[q][0]] {
				recall1Sum++
			}

			// Recall@10
			hits := 0
			for i := 0; i < topK; i++ {
				if resultIDs[groundTruth[q][i]] {
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
	t.Log("")
	t.Log("  GloVe word vectors have CONCENTRATED variance — PCA should retain")
	t.Log("  high recall even at aggressive reduction ratios, unlike GIST where")
	t.Log("  variance is uniform across all 960 dimensions.")
	t.Log("================================================================")
}

// gloveBruteForceTopK computes exact cosine similarity top-K for ground truth.
// Returns IDs (word strings from the dataset) for direct comparison with search results.
func gloveBruteForceTopK(queries [][]float64, vectors [][]float64, ids []string, dim, topK int) [][]string {
	type scored struct {
		id    string
		score float64
	}

	result := make([][]string, len(queries))
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
				scores[i] = scored{id: ids[i], score: dot / denom}
			}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})
		result[q] = make([]string, topK)
		for i := 0; i < topK && i < len(scores); i++ {
			result[q][i] = scores[i].id
		}
	}
	return result
}
