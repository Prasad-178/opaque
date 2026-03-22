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
// threshold CKKS mode, reporting end-to-end query times and privacy overhead.
func TestEnterprise100KDirectVsThreshold(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100K threshold comparison in short mode")
	}
	ctx := context.Background()

	dimension := 128
	numVectors := 100000
	numSuperBuckets := 64
	numQueries := 5
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

	// Build index (shared between both modes)
	fmt.Println("\n[2] Building hierarchical index...")
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

	// Select query vectors
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = rng.Intn(numVectors)
	}

	// Brute force baseline
	fmt.Println("\n[3] Brute force baseline...")
	var bfTotal time.Duration
	for q := 0; q < numQueries; q++ {
		query := vectors[queryIndices[q]]
		start := time.Now()
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			scores[i] = scored{ids[i], cosineSimThreshold(query, vectors[i])}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		bfTotal += time.Since(start)
	}
	avgBF := bfTotal / time.Duration(numQueries)
	fmt.Printf("  Brute force avg: %v\n", avgBF)

	// --- Direct mode ---
	fmt.Println("\n[4] DIRECT MODE (single-key)")
	directProvider, err := crypto.NewDirectHEProvider(4)
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	directClient, err := NewEnterpriseHierarchicalClientWithProvider(cfg, creds, store, directProvider)
	if err != nil {
		t.Fatalf("NewClient (direct): %v", err)
	}

	directResults := runSearchBenchmark(t, directClient, vectors, queryIndices, topK, ids)

	// --- Threshold mode (3-of-5) with simulated network latency ---
	// In a real deployment, each committee node runs on a separate machine.
	// We simulate 2ms RTT (same data center) — each node needs 2 RTTs
	// (receive ct + send share back), but all nodes operate in parallel.
	simulatedRTT := 2 * time.Millisecond

	fmt.Printf("\n[5] THRESHOLD MODE (3-of-5 committee, simulated %v RTT)\n", simulatedRTT)
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

	threshResults := runSearchBenchmark(t, threshClient, vectors, queryIndices, topK, ids)

	// Cleanup
	directProvider.Close()
	threshProvider.Close()

	// --- Comparison table ---
	fmt.Println("\n============================================================")
	fmt.Printf("COMPARISON SUMMARY (100K vectors, 128-dim, %v simulated RTT)\n", simulatedRTT)
	fmt.Println("============================================================")
	fmt.Println()
	fmt.Println("┌──────────────────┬──────────────┬──────────────┬──────────────┐")
	fmt.Println("│                  │ BRUTE FORCE  │    DIRECT    │  THRESHOLD   │")
	fmt.Println("├──────────────────┼──────────────┼──────────────┼──────────────┤")
	fmt.Printf("│ Avg Query Time   │ %10v │ %10v │ %10v │\n", avgBF.Round(time.Millisecond), directResults.avgTotal.Round(time.Millisecond), threshResults.avgTotal.Round(time.Millisecond))
	fmt.Printf("│  HE Encrypt      │          n/a │ %10v │ %10v │\n", directResults.avgHEEncrypt.Round(time.Millisecond), threshResults.avgHEEncrypt.Round(time.Millisecond))
	fmt.Printf("│  HE Dot Products │          n/a │ %10v │ %10v │\n", directResults.avgHEDot.Round(time.Millisecond), threshResults.avgHEDot.Round(time.Millisecond))
	fmt.Printf("│  HE Decrypt      │          n/a │ %10v │ %10v │\n", directResults.avgHEDecrypt.Round(time.Millisecond), threshResults.avgHEDecrypt.Round(time.Millisecond))
	fmt.Printf("│  AES + Scoring   │          n/a │ %10v │ %10v │\n", directResults.avgOther.Round(time.Millisecond), threshResults.avgOther.Round(time.Millisecond))
	fmt.Printf("│ Recall@%d        │          n/a │       %d/%d │       %d/%d │\n", topK, directResults.recall, numQueries, threshResults.recall, numQueries)
	fmt.Printf("│ Vectors Scored   │ %10d │ %10d │ %10d │\n", numVectors, directResults.avgVectorsScored, threshResults.avgVectorsScored)
	fmt.Println("├──────────────────┼──────────────┼──────────────┼──────────────┤")
	fmt.Println("│ PRIVACY          │     NONE     │   PARTIAL    │     FULL     │")
	fmt.Println("│ Key Ownership    │      n/a     │  single-key  │  3-of-5 t/N  │")
	fmt.Println("│ Network Model    │      n/a     │      n/a     │  2ms RTT DC  │")
	fmt.Println("└──────────────────┴──────────────┴──────────────┴──────────────┘")
	fmt.Println()
	fmt.Printf("Threshold overhead vs direct: %.2fx total, %.2fx HE\n",
		float64(threshResults.avgTotal)/float64(directResults.avgTotal),
		float64(threshResults.avgHEDot)/float64(directResults.avgHEDot))
}

type searchBenchResults struct {
	avgTotal         time.Duration
	avgHEEncrypt     time.Duration
	avgHEDot         time.Duration
	avgHEDecrypt     time.Duration
	avgOther         time.Duration
	avgVectorsScored int
	recall           int
}

func runSearchBenchmark(t *testing.T, client *EnterpriseHierarchicalClient, vectors [][]float64, queryIndices []int, topK int, ids []string) searchBenchResults {
	t.Helper()
	ctx := context.Background()
	numQueries := len(queryIndices)

	var totalTime, heEncrypt, heDot, heDecrypt, otherTime time.Duration
	var totalVectorsScored, recall int

	for q := 0; q < numQueries; q++ {
		query := vectors[queryIndices[q]]

		start := time.Now()
		result, err := client.SearchBatch(ctx, query, topK)
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

		targetID := ids[queryIndices[q]]
		for _, r := range result.Results {
			if r.ID == targetID {
				recall++
				break
			}
		}

		fmt.Printf("  Query %d: %v (HE: enc=%v dot=%v dec=%v, scored=%d)\n",
			q+1, searchTime.Round(time.Millisecond),
			result.Timing.HEEncryptQuery.Round(time.Millisecond),
			result.Timing.HECentroidScores.Round(time.Millisecond),
			result.Timing.HEDecryptScores.Round(time.Millisecond),
			result.Stats.VectorsScored)
	}

	n := time.Duration(numQueries)
	return searchBenchResults{
		avgTotal:         totalTime / n,
		avgHEEncrypt:     heEncrypt / n,
		avgHEDot:         heDot / n,
		avgHEDecrypt:     heDecrypt / n,
		avgOther:         otherTime / n,
		avgVectorsScored: totalVectorsScored / numQueries,
		recall:           recall,
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
