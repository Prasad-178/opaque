// Hierarchical Private Search Demo
//
// This demonstrates the three-level privacy-preserving vector search:
//
//	Level 1: HE on super-bucket centroids (64 operations)
//	Level 2: Decoy-based sub-bucket fetch
//	Level 3: Local AES decrypt + scoring
//
// Run: go run ./examples/hierarchical/main.go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  Hierarchical Private Search Demo")
	fmt.Println("  Three-Level Privacy-Preserving Vector Search")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Configuration
	numVectors := 100000
	dimension := 128
	numQueries := 5
	topK := 10

	fmt.Println("Configuration:")
	fmt.Printf("  Vectors: %d\n", numVectors)
	fmt.Printf("  Dimension: %d\n", dimension)
	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Top-K: %d\n", topK)
	fmt.Println()

	// Generate vectors
	fmt.Print("Generating vectors... ")
	start := time.Now()
	ids, vectors := generateVectors(rng, numVectors, dimension)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Create encryption key
	fmt.Print("Generating AES-256 key... ")
	key, err := encrypt.GenerateKey()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Println("done")

	// Get config and print details
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension
	fmt.Println()
	fmt.Println("Hierarchical Index Configuration:")
	fmt.Printf("  Super-buckets: %d (for HE scoring)\n", cfg.NumSuperBuckets)
	fmt.Printf("  Sub-buckets per super: %d\n", cfg.NumSubBuckets)
	fmt.Printf("  Total sub-buckets: %d\n", cfg.NumSuperBuckets*cfg.NumSubBuckets)
	fmt.Printf("  Top super-buckets to select: %d\n", cfg.TopSuperBuckets)
	fmt.Printf("  Sub-buckets per selected super: %d\n", cfg.SubBucketsPerSuper)
	fmt.Printf("  Decoy sub-buckets: %d\n", cfg.NumDecoys)
	fmt.Println()

	// Build index
	fmt.Print("Building hierarchical index... ")
	start = time.Now()

	builder, err := hierarchical.NewBuilder(cfg, key)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	buildTime := time.Since(start)
	fmt.Printf("done (%v)\n", buildTime)

	// Print index stats
	stats := idx.GetStats()
	fmt.Println()
	fmt.Println("Index Statistics:")
	fmt.Printf("  Total vectors: %d\n", stats.TotalVectors)
	fmt.Printf("  Super-buckets used: %d/%d\n", stats.NumSuperBuckets, cfg.NumSuperBuckets)
	fmt.Printf("  Sub-buckets used: %d/%d\n", stats.NumSubBuckets, cfg.NumSuperBuckets*cfg.NumSubBuckets)
	fmt.Printf("  Avg vectors per sub-bucket: %.1f\n", stats.AvgVectorsPerSub)
	fmt.Printf("  Empty sub-buckets: %d\n", stats.EmptySubBuckets)
	fmt.Println()

	// Create client
	fmt.Print("Creating hierarchical client... ")
	start = time.Now()
	hClient, err := client.NewHierarchicalClient(idx)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("done (%v)\n", time.Since(start))
	fmt.Println()

	// Run searches
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("Running Searches")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	var totalTime time.Duration
	var totalHETime time.Duration
	var totalFetchTime time.Duration
	var totalScoreTime time.Duration
	correctResults := 0

	for q := 0; q < numQueries; q++ {
		// Create query similar to a random vector
		targetIdx := rng.Intn(numVectors)
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		// Add small noise
		for i := range query {
			query[i] += (rng.Float64() - 0.5) * 0.02
		}

		fmt.Printf("Query %d (target: doc-%d):\n", q+1, targetIdx)

		result, err := hClient.Search(ctx, query, topK)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}

		totalTime += result.Timing.Total
		totalHETime += result.Timing.HECentroidScores
		totalFetchTime += result.Timing.BucketFetch
		totalScoreTime += result.Timing.LocalScoring

		// Print timing breakdown
		fmt.Printf("  Level 1 (HE centroid scoring): %v\n", result.Timing.HECentroidScores)
		fmt.Printf("  Level 2 (Sub-bucket fetch):    %v\n", result.Timing.BucketFetch)
		fmt.Printf("  Level 3 (Local scoring):       %v\n", result.Timing.LocalScoring)
		fmt.Printf("  Total:                         %v\n", result.Timing.Total)

		// Print stats
		fmt.Printf("  HE operations: %d\n", result.Stats.HEOperations)
		fmt.Printf("  Real buckets: %d, Decoy buckets: %d\n",
			result.Stats.RealSubBuckets, result.Stats.DecoySubBuckets)
		fmt.Printf("  Vectors scored: %d\n", result.Stats.VectorsScored)

		// Check if target found
		targetID := fmt.Sprintf("doc-%d", targetIdx)
		found := false
		for i, r := range result.Results {
			if r.ID == targetID {
				found = true
				correctResults++
				fmt.Printf("  Target found at position %d (score: %.4f)\n", i+1, r.Score)
				break
			}
		}
		if !found {
			fmt.Printf("  Target not in top %d\n", topK)
		}
		fmt.Println()
	}

	// Print summary
	avgTime := totalTime / time.Duration(numQueries)
	avgHETime := totalHETime / time.Duration(numQueries)
	avgFetchTime := totalFetchTime / time.Duration(numQueries)
	avgScoreTime := totalScoreTime / time.Duration(numQueries)
	recall := float64(correctResults) / float64(numQueries) * 100

	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Summary")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	fmt.Println("Average Timing Breakdown:")
	fmt.Printf("  Level 1 (HE): %v (%.1f%%)\n", avgHETime, float64(avgHETime)/float64(avgTime)*100)
	fmt.Printf("  Level 2 (Fetch): %v (%.1f%%)\n", avgFetchTime, float64(avgFetchTime)/float64(avgTime)*100)
	fmt.Printf("  Level 3 (Score): %v (%.1f%%)\n", avgScoreTime, float64(avgScoreTime)/float64(avgTime)*100)
	fmt.Printf("  Total: %v\n", avgTime)
	fmt.Println()

	fmt.Printf("Recall@%d: %.1f%% (%d/%d)\n", topK, recall, correctResults, numQueries)
	fmt.Println()

	// Comparison with naive Tier 1
	naiveHEOps := numVectors
	actualHEOps := cfg.NumSuperBuckets
	speedup := float64(naiveHEOps) / float64(actualHEOps)

	fmt.Println("Privacy-Performance Tradeoff:")
	fmt.Printf("  HE operations: %d (vs %d for naive Tier 1)\n", actualHEOps, naiveHEOps)
	fmt.Printf("  Speedup: %.0fx fewer HE operations\n", speedup)
	fmt.Println()

	// Estimate naive Tier 1 time
	// ~33ms per HE operation
	naiveTime := time.Duration(naiveHEOps) * 33 * time.Millisecond
	fmt.Println("Estimated Performance:")
	fmt.Printf("  Hierarchical (this demo): %v\n", avgTime)
	fmt.Printf("  Naive Tier 1 (all HE):    ~%v\n", naiveTime)
	fmt.Printf("  Time savings:             ~%.0fx faster\n", float64(naiveTime)/float64(avgTime))
	fmt.Println()

	fmt.Println("Privacy Guarantees:")
	fmt.Println("  - Query vector: hidden from server (HE encryption)")
	fmt.Println("  - Super-bucket selection: hidden from server (client-side decrypt)")
	fmt.Println("  - Sub-bucket interest: hidden from server (decoy buckets)")
	fmt.Println("  - Vector values: hidden from storage (AES-256-GCM)")
	fmt.Println("  - Final scores: hidden from everyone (local computation)")
	fmt.Println()
}

func generateVectors(rng *rand.Rand, count, dimension int) ([]string, [][]float64) {
	ids := make([]string, count)
	vectors := make([][]float64, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("doc-%d", i)
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1 // [-1, 1]
		}
		vectors[i] = vec
	}

	return ids, vectors
}
