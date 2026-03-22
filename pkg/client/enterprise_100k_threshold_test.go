//go:build integration

package client

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/pkg/auth"
	"github.com/Prasad-178/opaque/pkg/blob"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/crypto/threshold"
	"github.com/Prasad-178/opaque/pkg/embeddings"
	"github.com/Prasad-178/opaque/pkg/enterprise"
	"github.com/Prasad-178/opaque/pkg/hierarchical"
)

func getThresholdSIFTDataPath() string {
	candidates := []string{
		"../../data/sift",
		"../data/sift",
		"../../../data/sift",
	}
	for _, p := range candidates {
		abs, _ := filepath.Abs(p)
		if _, err := os.Stat(filepath.Join(abs, "sift_base.fvecs")); err == nil {
			return abs
		}
	}
	return ""
}

// TestEnterprise100KDirectVsThreshold benchmarks direct vs threshold CKKS on
// the first 100K vectors of SIFT1M with real query vectors and brute-force
// ground truth. Tests multiple probe configs to show the recall/latency tradeoff.
func TestEnterprise100KDirectVsThreshold(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100K threshold comparison in short mode")
	}

	dataPath := getThresholdSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT1M dataset not found; run scripts/download_sift1m.sh first")
	}

	ctx := context.Background()

	// Load SIFT1M and take first 100K
	fmt.Println("============================================================")
	fmt.Println("SIFT 100K: DIRECT vs THRESHOLD CKKS")
	fmt.Println("============================================================")

	fmt.Println("\n[1] Loading SIFT1M dataset...")
	dataset, err := embeddings.SIFT1M(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT1M: %v", err)
	}

	numVectors := 100000
	if len(dataset.Vectors) < numVectors {
		numVectors = len(dataset.Vectors)
	}
	vectors := dataset.Vectors[:numVectors]
	ids := dataset.IDs[:numVectors]
	dimension := dataset.Dimension

	numQueries := 50
	if numQueries > len(dataset.Queries) {
		numQueries = len(dataset.Queries)
	}
	queries := dataset.Queries[:numQueries]
	topK := 10

	fmt.Printf("  Vectors: %d (%d-dim), Queries: %d\n", numVectors, dimension, numQueries)

	// Compute brute-force ground truth (cosine similarity)
	fmt.Println("\n[2] Computing brute-force ground truth...")
	startGT := time.Now()
	groundTruth := make([][]string, numQueries)
	for q := 0; q < numQueries; q++ {
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			scores[i] = scored{ids[i], cosineSimThreshold(queries[q], vectors[i])}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		groundTruth[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruth[q][i] = scores[i].id
		}
	}
	fmt.Printf("  Ground truth computed in %v\n", time.Since(startGT))

	// Probe configs: sweep from conservative to aggressive
	numSuperBuckets := 64
	probeConfigs := []struct {
		name            string
		topSuperBuckets int
	}{
		{"probe-4", 4},   // 6.2%
		{"probe-8", 8},   // 12.5%
		{"probe-16", 16}, // 25%
		{"probe-32", 32}, // 50%
		{"probe-48", 48}, // 75%
	}

	// Simulated network latency for threshold mode
	simulatedRTT := 2 * time.Millisecond

	// Results storage
	type benchResult struct {
		name            string
		topSuperBuckets int
		directLatency   time.Duration
		threshLatency   time.Duration
		directRecall    float64
		threshRecall    float64
		directScored    int
		threshScored    int
	}
	var allResults []benchResult

	for _, pc := range probeConfigs {
		fmt.Printf("\n============================================================\n")
		fmt.Printf("CONFIG: %s (TopSuperBuckets=%d/%d = %.1f%% probe)\n",
			pc.name, pc.topSuperBuckets, numSuperBuckets,
			float64(pc.topSuperBuckets)/float64(numSuperBuckets)*100)
		fmt.Println("============================================================")

		// Build index
		fmt.Println("\n  Building index...")
		startBuild := time.Now()
		enterpriseCfg, _ := enterprise.NewConfig("bench-threshold", dimension, numSuperBuckets)
		store := blob.NewMemoryStore()
		cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
		cfg.TopSuperBuckets = pc.topSuperBuckets
		cfg.SubBucketsPerSuper = 4
		cfg.NumDecoys = 8

		builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
		builder.Build(ctx, ids, vectors, store)
		enterpriseCfg = builder.GetEnterpriseConfig()
		fmt.Printf("  Index built in %v\n", time.Since(startBuild))

		// Auth
		enterpriseStore := enterprise.NewMemoryStore()
		enterpriseStore.Put(ctx, enterpriseCfg)
		authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
		authService.RegisterUser(ctx, "user", "bench-threshold", []byte("pass"), []string{auth.ScopeSearch})
		creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

		// --- Direct mode ---
		fmt.Println("\n  DIRECT MODE:")
		directProvider, err := crypto.NewDirectHEProvider(4)
		if err != nil {
			t.Fatalf("NewDirectHEProvider: %v", err)
		}
		directClient, err := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, directProvider)
		if err != nil {
			t.Fatalf("NewClient (direct): %v", err)
		}
		dRes := runSIFTBenchmark(t, directClient, queries, groundTruth, topK)
		fmt.Printf("    Avg: %v, Recall@%d: %.1f%%, Scored: %d\n",
			dRes.avgLatency.Round(time.Millisecond), topK, dRes.avgRecall*100, dRes.avgScored)

		// --- Threshold mode ---
		fmt.Printf("\n  THRESHOLD MODE (3-of-5, %v RTT):\n", simulatedRTT)
		committee, err := threshold.NewCommittee(5, 3)
		if err != nil {
			t.Fatalf("NewCommittee: %v", err)
		}
		committee.SimulatedRTT = simulatedRTT
		threshProvider, err := crypto.NewThresholdHEProvider(committee, 4)
		if err != nil {
			t.Fatalf("NewThresholdHEProvider: %v", err)
		}
		threshClient, err := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, threshProvider)
		if err != nil {
			t.Fatalf("NewClient (threshold): %v", err)
		}
		tRes := runSIFTBenchmark(t, threshClient, queries, groundTruth, topK)
		fmt.Printf("    Avg: %v, Recall@%d: %.1f%%, Scored: %d\n",
			tRes.avgLatency.Round(time.Millisecond), topK, tRes.avgRecall*100, tRes.avgScored)

		directProvider.Close()
		threshProvider.Close()

		allResults = append(allResults, benchResult{
			name:            pc.name,
			topSuperBuckets: pc.topSuperBuckets,
			directLatency:   dRes.avgLatency,
			threshLatency:   tRes.avgLatency,
			directRecall:    dRes.avgRecall,
			threshRecall:    tRes.avgRecall,
			directScored:    dRes.avgScored,
			threshScored:    tRes.avgScored,
		})
	}

	// --- Summary table ---
	fmt.Println("\n============================================================")
	fmt.Printf("SUMMARY: SIFT 100K, %d-dim, 50 queries, 3-of-5 committee, %v RTT\n", dimension, simulatedRTT)
	fmt.Println("============================================================")
	fmt.Println()
	fmt.Println("┌────────────┬────────┬──────────────────────┬──────────────────────┬──────────┐")
	fmt.Println("│ Config     │ Probe% │ DIRECT (lat / rec)   │ THRESHOLD (lat / rec)│ Overhead │")
	fmt.Println("├────────────┼────────┼──────────────────────┼──────────────────────┼──────────┤")
	for _, r := range allResults {
		probe := float64(r.topSuperBuckets) / float64(numSuperBuckets) * 100
		overhead := float64(r.threshLatency) / float64(r.directLatency)
		fmt.Printf("│ %-10s │ %4.1f%% │ %7v / %5.1f%%      │ %7v / %5.1f%%      │  %.2fx   │\n",
			r.name, probe,
			r.directLatency.Round(time.Millisecond), r.directRecall*100,
			r.threshLatency.Round(time.Millisecond), r.threshRecall*100,
			overhead)
	}
	fmt.Println("└────────────┴────────┴──────────────────────┴──────────────────────┴──────────┘")
	fmt.Println()
	fmt.Println("PRIVACY:  Direct = single-key (partial), Threshold = 3-of-5 (full key distribution)")
	fmt.Println("NETWORK:  Threshold adds simulated 2ms datacenter RTT (2 round trips, all nodes parallel)")
}

type siftBenchResult struct {
	avgLatency time.Duration
	avgRecall  float64
	avgScored  int
}

func runSIFTBenchmark(t *testing.T, client *EnterpriseHierarchicalClient, queries [][]float64, groundTruth [][]string, topK int) siftBenchResult {
	t.Helper()
	ctx := context.Background()
	numQueries := len(queries)

	var totalLatency time.Duration
	var totalRecall float64
	var totalScored int

	for q := 0; q < numQueries; q++ {
		start := time.Now()
		result, err := client.SearchBatch(ctx, queries[q], topK)
		latency := time.Since(start)
		if err != nil {
			t.Fatalf("SearchBatch query %d failed: %v", q, err)
		}

		totalLatency += latency
		totalScored += result.Stats.VectorsScored

		// Compute recall against ground truth
		resultIDs := make(map[string]bool)
		for _, r := range result.Results {
			resultIDs[r.ID] = true
		}
		hits := 0
		for _, gtID := range groundTruth[q] {
			if resultIDs[gtID] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(topK)
	}

	return siftBenchResult{
		avgLatency: totalLatency / time.Duration(numQueries),
		avgRecall:  totalRecall / float64(numQueries),
		avgScored:  totalScored / numQueries,
	}
}

func cosineSimThreshold(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
