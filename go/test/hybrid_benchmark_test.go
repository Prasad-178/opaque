package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/encrypt"
)

// TestHybridBenchmark compares Tier 1, Tier 2, and Hybrid (Tier 2.5)
func TestHybridBenchmark(t *testing.T) {
	fmt.Println("============================================================")
	fmt.Println("TIER 1 vs TIER 2 vs HYBRID (TIER 2.5) COMPARISON")
	fmt.Println("============================================================")
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Configuration
	numVectors := 10000 // Smaller for faster testing
	dimension := 128
	numQueries := 5
	topK := 10

	fmt.Printf("Configuration:\n")
	fmt.Printf("  Vectors: %d\n", numVectors)
	fmt.Printf("  Dimension: %d\n", dimension)
	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Top-K: %d\n", topK)
	fmt.Println()

	// Generate vectors
	fmt.Print("Generating vectors... ")
	vectors := make([][]float64, numVectors)
	ids := make([]string, numVectors)
	for i := 0; i < numVectors; i++ {
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
		}
		vectors[i] = vec
		ids[i] = fmt.Sprintf("doc-%d", i)
	}
	fmt.Println("done")

	// Generate queries
	queries := make([][]float64, numQueries)
	targetIndices := make([]int, numQueries)
	for i := 0; i < numQueries; i++ {
		targetIdx := rng.Intn(numVectors)
		targetIndices[i] = targetIdx
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		for j := range query {
			query[j] += rng.Float64()*0.1 - 0.05
		}
		queries[i] = query
	}
	fmt.Println()

	// Benchmark Tier 2 (baseline - fastest)
	fmt.Println("============================================================")
	fmt.Println("TIER 2: Data-Private (AES only, no HE)")
	fmt.Println("============================================================")
	tier2Results := benchmarkTier2Only(ctx, vectors, ids, queries, targetIndices, dimension, topK)

	// Benchmark Hybrid - Fast mode (no HE)
	fmt.Println("============================================================")
	fmt.Println("HYBRID (Fast Mode): AES + Plaintext scoring")
	fmt.Println("============================================================")
	hybridFastResults := benchmarkHybridFast(ctx, vectors, ids, queries, targetIndices, dimension, topK)

	// Benchmark Hybrid - Private mode (with HE)
	fmt.Println("============================================================")
	fmt.Println("HYBRID (Private Mode): AES + HE scoring on top 20 candidates")
	fmt.Println("============================================================")
	hybridPrivateResults := benchmarkHybridPrivate(ctx, vectors, ids, queries, targetIndices, dimension, topK)

	// Print comparison
	printHybridComparison(tier2Results, hybridFastResults, hybridPrivateResults)
}

type hybridBenchResults struct {
	name           string
	avgQueryTime   time.Duration
	avgLSHTime     time.Duration
	avgFetchTime   time.Duration
	avgDecryptTime time.Duration
	avgCoarseTime  time.Duration
	avgHETime      time.Duration
	recall         float64
	heOperations   int
}

func benchmarkTier2Only(ctx context.Context, vectors [][]float64, ids []string, queries [][]float64, targets []int, dimension, topK int) hybridBenchResults {
	results := hybridBenchResults{name: "Tier 2 (AES only)"}

	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension: dimension,
		LSHBits:   12,
		LSHSeed:   42,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)

	// Insert
	fmt.Print("Inserting vectors... ")
	start := time.Now()
	tier2Client.InsertBatch(ctx, ids, vectors, nil)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Query
	fmt.Printf("Running %d queries...\n", len(queries))
	var totalTime time.Duration
	correctResults := 0

	for i, query := range queries {
		queryStart := time.Now()
		searchResults, _ := tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
			TopK:          topK,
			NumBuckets:    5,
			UseMultiProbe: true,
		})
		queryTime := time.Since(queryStart)
		totalTime += queryTime

		targetID := fmt.Sprintf("doc-%d", targets[i])
		for _, r := range searchResults {
			if r.ID == targetID {
				correctResults++
				break
			}
		}
		fmt.Printf("  Query %d: %v\n", i+1, queryTime)
	}

	results.avgQueryTime = totalTime / time.Duration(len(queries))
	results.recall = float64(correctResults) / float64(len(queries)) * 100
	fmt.Printf("Average: %v, Recall: %.1f%%\n\n", results.avgQueryTime, results.recall)

	return results
}

func benchmarkHybridFast(ctx context.Context, vectors [][]float64, ids []string, queries [][]float64, targets []int, dimension, topK int) hybridBenchResults {
	results := hybridBenchResults{name: "Hybrid (Fast)"}

	key, _ := encrypt.GenerateKey()
	store := blob.NewMemoryStore()

	cfg := client.HybridConfig{
		Dimension:            dimension,
		LSHBits:              12,
		LSHSeed:              42,
		CoarseCandidates:     200,
		FineCandidates:       20,
		UseHEForFinalScoring: false, // Fast mode - no HE
	}

	hybridClient, err := client.NewHybridClient(cfg, key, store)
	if err != nil {
		fmt.Printf("Error creating hybrid client: %v\n", err)
		return results
	}

	// Insert
	fmt.Print("Inserting vectors... ")
	start := time.Now()
	hybridClient.InsertBatch(ctx, ids, vectors, nil)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Query
	fmt.Printf("Running %d queries...\n", len(queries))
	var totalTime, totalLSH, totalFetch, totalDecrypt, totalCoarse time.Duration
	correctResults := 0

	for i, query := range queries {
		searchResult, _ := hybridClient.SearchFast(ctx, query, topK)
		totalTime += searchResult.TotalTime
		totalLSH += searchResult.LSHTime
		totalFetch += searchResult.FetchTime
		totalDecrypt += searchResult.DecryptTime
		totalCoarse += searchResult.CoarseTime

		targetID := fmt.Sprintf("doc-%d", targets[i])
		for _, r := range searchResult.Results {
			if r.ID == targetID {
				correctResults++
				break
			}
		}
		fmt.Printf("  Query %d: %v (LSH: %v, Fetch: %v, Decrypt: %v, Score: %v)\n",
			i+1, searchResult.TotalTime, searchResult.LSHTime, searchResult.FetchTime,
			searchResult.DecryptTime, searchResult.CoarseTime)
	}

	n := time.Duration(len(queries))
	results.avgQueryTime = totalTime / n
	results.avgLSHTime = totalLSH / n
	results.avgFetchTime = totalFetch / n
	results.avgDecryptTime = totalDecrypt / n
	results.avgCoarseTime = totalCoarse / n
	results.recall = float64(correctResults) / float64(len(queries)) * 100
	fmt.Printf("Average: %v, Recall: %.1f%%\n\n", results.avgQueryTime, results.recall)

	return results
}

func benchmarkHybridPrivate(ctx context.Context, vectors [][]float64, ids []string, queries [][]float64, targets []int, dimension, topK int) hybridBenchResults {
	results := hybridBenchResults{name: "Hybrid (Private)"}

	key, _ := encrypt.GenerateKey()
	store := blob.NewMemoryStore()

	cfg := client.HybridConfig{
		Dimension:            dimension,
		LSHBits:              12,
		LSHSeed:              42,
		CoarseCandidates:     200,
		FineCandidates:       20,
		UseHEForFinalScoring: true, // Private mode - use HE for final scoring
	}

	hybridClient, err := client.NewHybridClient(cfg, key, store)
	if err != nil {
		fmt.Printf("Error creating hybrid client: %v\n", err)
		return results
	}

	// Insert
	fmt.Print("Inserting vectors... ")
	start := time.Now()
	hybridClient.InsertBatch(ctx, ids, vectors, nil)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Query
	fmt.Printf("Running %d queries (with HE scoring on top 20 candidates)...\n", len(queries))
	var totalTime, totalLSH, totalFetch, totalDecrypt, totalCoarse, totalHE time.Duration
	var totalHEOps int
	correctResults := 0

	for i, query := range queries {
		searchResult, _ := hybridClient.SearchPrivate(ctx, query, topK)
		totalTime += searchResult.TotalTime
		totalLSH += searchResult.LSHTime
		totalFetch += searchResult.FetchTime
		totalDecrypt += searchResult.DecryptTime
		totalCoarse += searchResult.CoarseTime
		totalHE += searchResult.HEScoreTime
		totalHEOps += searchResult.HEOperations

		targetID := fmt.Sprintf("doc-%d", targets[i])
		for _, r := range searchResult.Results {
			if r.ID == targetID {
				correctResults++
				break
			}
		}
		fmt.Printf("  Query %d: %v (LSH: %v, Fetch: %v, Decrypt: %v, Coarse: %v, HE[%d]: %v)\n",
			i+1, searchResult.TotalTime, searchResult.LSHTime, searchResult.FetchTime,
			searchResult.DecryptTime, searchResult.CoarseTime, searchResult.HEOperations, searchResult.HEScoreTime)
	}

	n := time.Duration(len(queries))
	results.avgQueryTime = totalTime / n
	results.avgLSHTime = totalLSH / n
	results.avgFetchTime = totalFetch / n
	results.avgDecryptTime = totalDecrypt / n
	results.avgCoarseTime = totalCoarse / n
	results.avgHETime = totalHE / n
	results.heOperations = totalHEOps / len(queries)
	results.recall = float64(correctResults) / float64(len(queries)) * 100
	fmt.Printf("Average: %v, Recall: %.1f%%, HE ops/query: %d\n\n", results.avgQueryTime, results.recall, results.heOperations)

	return results
}

func printHybridComparison(tier2, hybridFast, hybridPrivate hybridBenchResults) {
	fmt.Println("============================================================")
	fmt.Println("COMPARISON SUMMARY")
	fmt.Println("============================================================")
	fmt.Println()

	fmt.Println("┌─────────────────────┬───────────────┬───────────────┬───────────────┐")
	fmt.Println("│ Metric              │ Tier 2 (AES)  │ Hybrid Fast   │ Hybrid Private│")
	fmt.Println("├─────────────────────┼───────────────┼───────────────┼───────────────┤")
	fmt.Printf("│ Avg Query Time      │ %13v │ %13v │ %13v │\n",
		tier2.avgQueryTime.Round(time.Microsecond),
		hybridFast.avgQueryTime.Round(time.Microsecond),
		hybridPrivate.avgQueryTime.Round(time.Millisecond))
	fmt.Printf("│ Recall@10           │ %12.1f%% │ %12.1f%% │ %12.1f%% │\n",
		tier2.recall, hybridFast.recall, hybridPrivate.recall)
	fmt.Printf("│ HE Operations       │ %13d │ %13d │ %13d │\n",
		0, 0, hybridPrivate.heOperations)
	fmt.Println("└─────────────────────┴───────────────┴───────────────┴───────────────┘")
	fmt.Println()

	// Timing breakdown for hybrid private
	fmt.Println("Hybrid Private Timing Breakdown:")
	fmt.Printf("  LSH:          %v\n", hybridPrivate.avgLSHTime)
	fmt.Printf("  Fetch:        %v\n", hybridPrivate.avgFetchTime)
	fmt.Printf("  AES Decrypt:  %v\n", hybridPrivate.avgDecryptTime)
	fmt.Printf("  Coarse Score: %v\n", hybridPrivate.avgCoarseTime)
	fmt.Printf("  HE Score:     %v (on %d candidates)\n", hybridPrivate.avgHETime, hybridPrivate.heOperations)
	fmt.Println()

	// Speed comparison
	speedupVsTier2 := float64(hybridPrivate.avgQueryTime) / float64(tier2.avgQueryTime)
	fmt.Printf("Speed Comparison:\n")
	fmt.Printf("  Hybrid Private is %.1fx slower than Tier 2 (pure AES)\n", speedupVsTier2)
	fmt.Printf("  But Hybrid Private does HE scoring for query privacy!\n")
	fmt.Println()

	// Privacy comparison
	fmt.Println("PRIVACY COMPARISON")
	fmt.Println("==================")
	fmt.Println()
	fmt.Println("                    │ Storage Sees  │ Compute Sees  │ Query Private │ Data Private")
	fmt.Println("────────────────────┼───────────────┼───────────────┼───────────────┼──────────────")
	fmt.Println("Tier 2 (AES)        │ Encrypted     │ N/A (local)   │ Yes (local)   │ Yes")
	fmt.Println("Hybrid Fast         │ Encrypted     │ N/A (local)   │ Yes (local)   │ Yes")
	fmt.Println("Hybrid Private      │ Encrypted     │ Vectors only* │ Yes (HE)      │ Yes")
	fmt.Println()
	fmt.Println("* In Hybrid Private mode, a compute server could see decrypted vectors")
	fmt.Println("  but NOT the query or final scores (HE encrypted).")
	fmt.Println()

	fmt.Println("KEY INSIGHT:")
	fmt.Println("============")
	fmt.Println("Hybrid Private does HE operations on only 20 candidates instead of 10,000+")
	fmt.Printf("This makes it ~%.0fx faster than pure Tier 1 while maintaining query privacy!\n",
		float64(10000)/float64(hybridPrivate.heOperations))
}
