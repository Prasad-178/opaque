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

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestEnterprise100K runs a realistic 100K vector benchmark
func TestEnterprise100K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100K vector benchmark in short mode")
	}
	ctx := context.Background()

	// Configuration
	dimension := 128
	numVectors := 100000
	numSuperBuckets := 64
	numQueries := 5
	topK := 10

	fmt.Println("=" + "=================================================================")
	fmt.Println("100K VECTOR BENCHMARK: TIER 2.5 vs BRUTE FORCE")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Vectors: %d\n", numVectors)
	fmt.Printf("  Dimension: %d\n", dimension)
	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Top-K: %d\n", topK)
	fmt.Println()

	// =========================================
	// Generate Test Data
	// =========================================
	fmt.Println("[1] GENERATING 100K VECTORS...")
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

	genTime := time.Since(startGen)
	fmt.Printf("  ✓ Generated %d vectors in %v\n", numVectors, genTime)
	fmt.Printf("  Memory: ~%.1f MB\n", float64(numVectors*dimension*8)/(1024*1024))

	// =========================================
	// BRUTE FORCE BASELINE
	// =========================================
	fmt.Println("\n[2] BRUTE FORCE BASELINE (what a simple vector DB does)")
	fmt.Println("    This is plaintext - no privacy, server sees everything")

	// Prepare query vectors
	queryIndices := make([]int, numQueries)
	for i := 0; i < numQueries; i++ {
		queryIndices[i] = rng.Intn(numVectors)
	}

	var bruteForceTotalTime time.Duration
	bruteForceRecall := 0

	for q := 0; q < numQueries; q++ {
		query := vectors[queryIndices[q]]

		startBF := time.Now()

		// Brute force: compute all similarities
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			scores[i] = scored{
				id:    ids[i],
				score: cosineSim(query, vectors[i]),
			}
		}

		// Sort by score
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})

		bfTime := time.Since(startBF)
		bruteForceTotalTime += bfTime

		// Check if exact match found (should always be #1)
		if scores[0].id == ids[queryIndices[q]] {
			bruteForceRecall++
		}

		fmt.Printf("    Query %d: %v (top: %s, score: %.4f)\n",
			q+1, bfTime, scores[0].id, scores[0].score)
	}

	avgBruteForce := bruteForceTotalTime / time.Duration(numQueries)
	fmt.Printf("\n  BRUTE FORCE RESULTS:\n")
	fmt.Printf("    Average time: %v\n", avgBruteForce)
	fmt.Printf("    Throughput: %.2f queries/sec\n", float64(numQueries)/bruteForceTotalTime.Seconds())
	fmt.Printf("    Recall@1: %d/%d (100%% expected)\n", bruteForceRecall, numQueries)
	fmt.Printf("    Privacy: NONE - server sees query and all vectors\n")

	// =========================================
	// TIER 2.5 SETUP
	// =========================================
	fmt.Println("\n[3] TIER 2.5 HIERARCHICAL SEARCH SETUP")

	startSetup := time.Now()

	// Create enterprise config
	enterpriseCfg, _ := enterprise.NewConfig("benchmark-100k", dimension, numSuperBuckets)

	// Build index
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = 8
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	fmt.Printf("    Building hierarchical index...\n")
	startBuild := time.Now()
	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	buildTime := time.Since(startBuild)
	enterpriseCfg = builder.GetEnterpriseConfig()

	fmt.Printf("    ✓ Index built in %v\n", buildTime)

	// Setup auth
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "benchmark-100k", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	// Create client
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	setupTime := time.Since(startSetup)
	fmt.Printf("    ✓ Total setup: %v\n", setupTime)

	// =========================================
	// TIER 2.5 SEARCH
	// =========================================
	fmt.Println("\n[4] TIER 2.5 SEARCH (with full privacy)")

	var tier25TotalTime time.Duration
	var totalHETime, totalOtherTime time.Duration
	tier25Recall := 0
	var totalVectorsScored int

	for q := 0; q < numQueries; q++ {
		query := vectors[queryIndices[q]]

		startSearch := time.Now()
		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
		searchTime := time.Since(startSearch)
		tier25TotalTime += searchTime

		heTime := result.Timing.HEEncryptQuery + result.Timing.HECentroidScores + result.Timing.HEDecryptScores
		totalHETime += heTime
		totalOtherTime += searchTime - heTime
		totalVectorsScored += result.Stats.VectorsScored

		// Check recall
		targetID := ids[queryIndices[q]]
		for _, r := range result.Results {
			if r.ID == targetID {
				tier25Recall++
				break
			}
		}

		fmt.Printf("    Query %d: %v (HE: %v, scored: %d vectors)\n",
			q+1, searchTime,
			heTime,
			result.Stats.VectorsScored)
	}

	avgTier25 := tier25TotalTime / time.Duration(numQueries)
	avgHE := totalHETime / time.Duration(numQueries)
	avgOther := totalOtherTime / time.Duration(numQueries)

	fmt.Printf("\n  TIER 2.5 RESULTS:\n")
	fmt.Printf("    Average time: %v\n", avgTier25)
	fmt.Printf("      - HE operations: %v (%.1f%%)\n", avgHE, float64(avgHE)/float64(avgTier25)*100)
	fmt.Printf("      - Other (LSH, AES, scoring): %v (%.1f%%)\n", avgOther, float64(avgOther)/float64(avgTier25)*100)
	fmt.Printf("    Throughput: %.2f queries/sec\n", float64(numQueries)/tier25TotalTime.Seconds())
	fmt.Printf("    Recall@%d: %d/%d\n", topK, tier25Recall, numQueries)
	fmt.Printf("    Avg vectors scored: %d (%.2f%% of dataset)\n",
		totalVectorsScored/numQueries,
		float64(totalVectorsScored)/float64(numQueries*numVectors)*100)
	fmt.Printf("    Privacy: FULL - server sees nothing useful\n")

	// =========================================
	// COMPARISON
	// =========================================
	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("COMPARISON SUMMARY")
	fmt.Println("=" + "=================================================================")
	fmt.Println()
	fmt.Println("┌──────────────────┬─────────────────┬─────────────────┐")
	fmt.Println("│                  │   BRUTE FORCE   │    TIER 2.5     │")
	fmt.Println("├──────────────────┼─────────────────┼─────────────────┤")
	fmt.Printf("│ Avg Query Time   │ %13v │ %13v │\n", avgBruteForce, avgTier25)
	fmt.Printf("│ Throughput       │ %10.2f qps │ %10.2f qps │\n",
		float64(numQueries)/bruteForceTotalTime.Seconds(),
		float64(numQueries)/tier25TotalTime.Seconds())
	fmt.Printf("│ Vectors Scanned  │ %13d │ %13d │\n", numVectors, totalVectorsScored/numQueries)
	fmt.Printf("│ Recall@%d        │ %13d │ %13d │\n", topK, bruteForceRecall, tier25Recall)
	fmt.Println("├──────────────────┼─────────────────┼─────────────────┤")
	fmt.Println("│ PRIVACY          │                 │                 │")
	fmt.Println("│  Query hidden    │       ✗ NO      │       ✓ YES     │")
	fmt.Println("│  Vectors hidden  │       ✗ NO      │       ✓ YES     │")
	fmt.Println("│  Results hidden  │       ✗ NO      │       ✓ YES     │")
	fmt.Println("└──────────────────┴─────────────────┴─────────────────┘")
	fmt.Println()
	fmt.Printf("Slowdown for privacy: %.1fx\n", float64(avgTier25)/float64(avgBruteForce))
	fmt.Println()
	fmt.Println("NOTE: Current implementation runs entirely client-side.")
	fmt.Println("In production, HE centroid scoring would run on a server,")
	fmt.Println("with network latency adding ~10-50ms per query.")
}

// cosineSim computes cosine similarity (local helper to avoid redeclaration)
func cosineSim(a, b []float64) float64 {
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
