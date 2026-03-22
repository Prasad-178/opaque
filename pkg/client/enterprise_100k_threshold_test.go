//go:build integration

package client

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/pkg/auth"
	"github.com/Prasad-178/opaque/pkg/blob"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/crypto/threshold"
	"github.com/Prasad-178/opaque/pkg/enterprise"
	"github.com/Prasad-178/opaque/pkg/hierarchical"
)

// TestEnterprise100KDirectVsThreshold runs 100K vector search in both direct and
// threshold CKKS mode, reporting end-to-end query times and recall@10 against
// brute-force ground truth.
func TestEnterprise100KDirectVsThreshold(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100K threshold comparison in short mode")
	}
	ctx := context.Background()

	dimension := 128
	numVectors := 100000
	numSuperBuckets := 64
	numQueries := 20
	topK := 10

	fmt.Println("============================================================")
	fmt.Println("100K VECTORS: DIRECT vs THRESHOLD CKKS")
	fmt.Println("============================================================")

	// Generate test data
	fmt.Println("\n[1] Generating 100K vectors...")
	startGen := time.Now()
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}
	fmt.Printf("  Generated in %v\n", time.Since(startGen))

	// Generate independent random query vectors (NOT from the dataset)
	fmt.Printf("\n[2] Generating %d random query vectors...\n", numQueries)
	queries := make([][]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			queries[i][j] = rng.NormFloat64()
		}
	}

	// Compute brute-force ground truth for each query
	fmt.Println("\n[3] Computing brute-force ground truth...")
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

	// Build index (shared between both modes)
	fmt.Println("\n[4] Building hierarchical index...")
	startBuild := time.Now()
	enterpriseCfg, _ := enterprise.NewConfig("bench-threshold", dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = 8
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()
	fmt.Printf("  Index built in %v\n", time.Since(startBuild))

	// Setup auth
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "bench-threshold", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	// --- Direct mode ---
	fmt.Println("\n[5] DIRECT MODE (single-key)")
	directProvider, err := crypto.NewDirectHEProvider(4)
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	directClient, err := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, directProvider)
	if err != nil {
		t.Fatalf("NewClient (direct): %v", err)
	}

	directResults := runSearchBenchmark(t, directClient, queries, groundTruth, topK)

	// --- Threshold mode (3-of-5) with simulated network latency ---
	// In a real deployment, each committee node runs on a separate machine.
	// We simulate 2ms RTT (same data center) — each node needs 2 RTTs
	// (receive ct + send share back), but all nodes operate in parallel.
	simulatedRTT := 2 * time.Millisecond

	fmt.Printf("\n[6] THRESHOLD MODE (3-of-5 committee, simulated %v RTT)\n", simulatedRTT)
	startCommittee := time.Now()
	committee, err := threshold.NewCommittee(5, 3)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}
	committee.SimulatedRTT = simulatedRTT
	committeeSetup := time.Since(startCommittee)
	fmt.Printf("  Committee setup: %v\n", committeeSetup)

	threshProvider, err := crypto.NewThresholdHEProvider(committee, 4)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider: %v", err)
	}
	threshClient, err := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, threshProvider)
	if err != nil {
		t.Fatalf("NewClient (threshold): %v", err)
	}

	threshResults := runSearchBenchmark(t, threshClient, queries, groundTruth, topK)

	// Cleanup
	directProvider.Close()
	threshProvider.Close()

	// --- Comparison table ---
	fmt.Println("\n============================================================")
	fmt.Printf("COMPARISON SUMMARY (100K vectors, 128-dim, %d queries, %v simulated RTT)\n", numQueries, simulatedRTT)
	fmt.Println("============================================================")
	fmt.Println()
	fmt.Println("┌────────────────────┬──────────────┬──────────────┐")
	fmt.Println("│                    │    DIRECT    │  THRESHOLD   │")
	fmt.Println("├────────────────────┼──────────────┼──────────────┤")
	fmt.Printf("│ Avg Query Time     │ %10v │ %10v │\n", directResults.avgTotal.Round(time.Millisecond), threshResults.avgTotal.Round(time.Millisecond))
	fmt.Printf("│  HE Encrypt        │ %10v │ %10v │\n", directResults.avgHEEncrypt.Round(time.Millisecond), threshResults.avgHEEncrypt.Round(time.Millisecond))
	fmt.Printf("│  HE Dot+Decrypt    │ %10v │ %10v │\n", directResults.avgHEDot.Round(time.Millisecond), threshResults.avgHEDot.Round(time.Millisecond))
	fmt.Printf("│  AES + Scoring     │ %10v │ %10v │\n", directResults.avgOther.Round(time.Millisecond), threshResults.avgOther.Round(time.Millisecond))
	fmt.Printf("│ Recall@%-2d (vs BF)  │     %5.1f%% │     %5.1f%% │\n", topK, directResults.avgRecall*100, threshResults.avgRecall*100)
	fmt.Printf("│ Vectors Scored     │ %10d │ %10d │\n", directResults.avgVectorsScored, threshResults.avgVectorsScored)
	fmt.Println("├────────────────────┼──────────────┼──────────────┤")
	fmt.Println("│ PRIVACY            │   PARTIAL    │     FULL     │")
	fmt.Println("│ Key Ownership      │  single-key  │  3-of-5 t/N  │")
	fmt.Println("│ Network Model      │      n/a     │  2ms RTT DC  │")
	fmt.Println("└────────────────────┴──────────────┴──────────────┘")
	fmt.Println()
	fmt.Printf("Threshold overhead vs direct: %.2fx total, %.2fx HE\n",
		float64(threshResults.avgTotal)/float64(directResults.avgTotal),
		float64(threshResults.avgHEDot)/float64(directResults.avgHEDot))
	fmt.Printf("Recall difference: %.1f%% (direct) vs %.1f%% (threshold)\n",
		directResults.avgRecall*100, threshResults.avgRecall*100)
}

type searchBenchResults struct {
	avgTotal         time.Duration
	avgHEEncrypt     time.Duration
	avgHEDot         time.Duration
	avgHEDecrypt     time.Duration
	avgOther         time.Duration
	avgVectorsScored int
	avgRecall        float64
}

func runSearchBenchmark(t *testing.T, client *EnterpriseHierarchicalClient, queries [][]float64, groundTruth [][]string, topK int) searchBenchResults {
	t.Helper()
	ctx := context.Background()
	numQueries := len(queries)

	var totalTime, heEncrypt, heDot, heDecrypt, otherTime time.Duration
	var totalVectorsScored int
	var totalRecall float64

	for q := 0; q < numQueries; q++ {
		start := time.Now()
		result, err := client.SearchBatch(ctx, queries[q], topK)
		searchTime := time.Since(start)
		if err != nil {
			t.Fatalf("SearchBatch failed: %v", err)
		}

		totalTime += searchTime
		heEncrypt += result.Timing.HEEncryptQuery
		heDot += result.Timing.HECentroidScores
		heDecrypt += result.Timing.HEDecryptScores
		otherTime += result.Timing.BucketSelection + result.Timing.BucketFetch + result.Timing.AESDecrypt + result.Timing.LocalScoring
		totalVectorsScored += result.Stats.VectorsScored

		// Compute recall against brute-force ground truth
		gtSet := make(map[string]bool)
		for _, id := range groundTruth[q] {
			gtSet[id] = true
		}
		matches := 0
		for _, r := range result.Results {
			if gtSet[r.ID] {
				matches++
			}
		}
		recall := float64(matches) / float64(topK)
		totalRecall += recall

		fmt.Printf("  Query %d: %v (HE: enc=%v dot=%v dec=%v, scored=%d, recall=%.0f%%)\n",
			q+1, searchTime.Round(time.Millisecond),
			result.Timing.HEEncryptQuery.Round(time.Millisecond),
			result.Timing.HECentroidScores.Round(time.Millisecond),
			result.Timing.HEDecryptScores.Round(time.Millisecond),
			result.Stats.VectorsScored,
			recall*100)
	}

	n := time.Duration(numQueries)
	return searchBenchResults{
		avgTotal:         totalTime / n,
		avgHEEncrypt:     heEncrypt / n,
		avgHEDot:         heDot / n,
		avgHEDecrypt:     heDecrypt / n,
		avgOther:         otherTime / n,
		avgVectorsScored: totalVectorsScored / numQueries,
		avgRecall:        totalRecall / float64(numQueries),
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
