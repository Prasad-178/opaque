//go:build integration

package test

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// TestTier1VsTier2Comparison runs a comprehensive comparison of both tiers
func TestTier1VsTier2Comparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping tier comparison in short mode")
	}

	fmt.Println("============================================================")
	fmt.Println("TIER 1 vs TIER 2 COMPREHENSIVE COMPARISON")
	fmt.Println("============================================================")
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducibility

	// Configuration
	numVectors := 100000
	dimension := 128
	numQueries := 10
	topK := 10

	fmt.Printf("Configuration:\n")
	fmt.Printf("  Vectors: %d\n", numVectors)
	fmt.Printf("  Dimension: %d\n", dimension)
	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Top-K: %d\n", topK)
	fmt.Printf("  CPU Cores: %d\n", runtime.NumCPU())
	fmt.Println()

	// Generate vectors
	fmt.Print("Generating vectors... ")
	start := time.Now()
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
	fmt.Printf("done (%v)\n\n", time.Since(start))

	// Generate queries (similar to random vectors with noise)
	queries := make([][]float64, numQueries)
	targetIndices := make([]int, numQueries)
	for i := 0; i < numQueries; i++ {
		targetIdx := rng.Intn(numVectors)
		targetIndices[i] = targetIdx
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		// Add noise
		for j := range query {
			query[j] += rng.Float64()*0.1 - 0.05
		}
		queries[i] = query
	}

	// Run Tier 1 benchmark
	tier1Results := benchmarkTier1(ctx, vectors, ids, queries, targetIndices, dimension, topK)

	// Run Tier 2 benchmark
	tier2Results := benchmarkTier2(ctx, vectors, ids, queries, targetIndices, dimension, topK)

	// Print comparison
	printComparison(tier1Results, tier2Results, numVectors)
}

type benchmarkResults struct {
	name              string
	indexBuildTime    time.Duration
	avgQueryTime      time.Duration
	minQueryTime      time.Duration
	maxQueryTime      time.Duration
	recall            float64
	memoryUsed        uint64
	privacyFeatures   []string
	serverSees        []string
	serverDoesNotSee  []string
}

func benchmarkTier1(ctx context.Context, vectors [][]float64, ids []string, queries [][]float64, targets []int, dimension, topK int) benchmarkResults {
	fmt.Println("============================================================")
	fmt.Println("TIER 1: Query-Private Search (Homomorphic Encryption)")
	fmt.Println("============================================================")
	fmt.Println()

	results := benchmarkResults{
		name: "Tier 1 (Query-Private)",
		privacyFeatures: []string{
			"Query encrypted with BFV homomorphic encryption",
			"Similarity computed on encrypted data",
			"Results decrypted client-side",
		},
		serverSees: []string{
			"Plaintext vectors (server owns the data)",
			"LSH bucket of query (approximate region)",
			"Encrypted query ciphertext",
			"Encrypted similarity scores",
		},
		serverDoesNotSee: []string{
			"Actual query vector values",
			"Decrypted similarity scores",
			"Which results client selected",
		},
	}

	// Memory before
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Initialize crypto engine
	fmt.Print("1. Initializing HE engine... ")
	start := time.Now()
	engine, err := crypto.NewClientEngine()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return results
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Build LSH index
	fmt.Print("2. Building LSH index... ")
	start = time.Now()
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   64,
		Seed:      42,
	})
	err = lshIndex.Add(ids, vectors)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return results
	}
	results.indexBuildTime = time.Since(start)
	fmt.Printf("done (%v)\n", results.indexBuildTime)

	// Memory after index
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	results.memoryUsed = memAfter.Alloc - memBefore.Alloc

	// Run queries
	fmt.Printf("3. Running %d queries...\n", len(queries))

	queryTimes := make([]time.Duration, len(queries))
	correctResults := 0

	for i, query := range queries {
		queryStart := time.Now()

		// Normalize query
		normalizedQuery := normalizeVec(query)

		// LSH search for candidates
		candidates, err := lshIndex.SearchVector(normalizedQuery, 200)
		if err != nil {
			continue
		}

		// Encrypt query
		encQuery, err := engine.EncryptVector(normalizedQuery)
		if err != nil {
			continue
		}

		// Score top candidates with HE (limit to 20 for reasonable time)
		numToScore := min(20, len(candidates))
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, 0, numToScore)

		for j := 0; j < numToScore; j++ {
			idx := 0
			fmt.Sscanf(candidates[j].ID, "doc-%d", &idx)
			vec := vectors[idx]

			normalizedVec := normalizeVec(vec)
			encScore, err := engine.HomomorphicDotProduct(encQuery, normalizedVec)
			if err != nil {
				continue
			}

			score, err := engine.DecryptScalar(encScore)
			if err != nil {
				continue
			}

			scores = append(scores, scored{id: candidates[j].ID, score: score})
		}

		queryTimes[i] = time.Since(queryStart)

		// Check if target was found in top-K
		targetID := fmt.Sprintf("doc-%d", targets[i])
		for j := 0; j < min(topK, len(scores)); j++ {
			if scores[j].id == targetID {
				correctResults++
				break
			}
		}

		fmt.Printf("   Query %d: %v (candidates: %d, scored: %d)\n",
			i+1, queryTimes[i], len(candidates), len(scores))
	}

	// Calculate statistics
	var totalTime time.Duration
	results.minQueryTime = queryTimes[0]
	results.maxQueryTime = queryTimes[0]
	for _, t := range queryTimes {
		totalTime += t
		if t < results.minQueryTime {
			results.minQueryTime = t
		}
		if t > results.maxQueryTime {
			results.maxQueryTime = t
		}
	}
	results.avgQueryTime = totalTime / time.Duration(len(queries))
	results.recall = float64(correctResults) / float64(len(queries)) * 100

	fmt.Println()
	fmt.Printf("   Average query time: %v\n", results.avgQueryTime)
	fmt.Printf("   Min/Max: %v / %v\n", results.minQueryTime, results.maxQueryTime)
	fmt.Printf("   Recall@%d: %.1f%%\n", topK, results.recall)
	fmt.Println()

	return results
}

func benchmarkTier2(ctx context.Context, vectors [][]float64, ids []string, queries [][]float64, targets []int, dimension, topK int) benchmarkResults {
	fmt.Println("============================================================")
	fmt.Println("TIER 2: Data-Private Search (AES-256-GCM)")
	fmt.Println("============================================================")
	fmt.Println()

	results := benchmarkResults{
		name: "Tier 2 (Data-Private)",
		privacyFeatures: []string{
			"Vectors encrypted with AES-256-GCM before storage",
			"All computation happens client-side",
			"Optional timing obfuscation",
			"Optional decoy bucket fetches",
			"Optional dummy queries for noise",
		},
		serverSees: []string{
			"Encrypted blobs (opaque ciphertext)",
			"LSH bucket identifiers",
			"Access patterns (which buckets fetched)",
			"Blob count per bucket",
		},
		serverDoesNotSee: []string{
			"Actual vector values",
			"Query vectors",
			"Similarity scores",
			"Which results client selected",
			"Decryption keys",
		},
	}

	// Memory before
	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	// Create encryption key
	fmt.Print("1. Creating AES-256-GCM key... ")
	key, err := encrypt.GenerateKey()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return results
	}
	encryptor, _ := encrypt.NewAESGCM(key)
	fmt.Printf("done (fingerprint: %s)\n", encryptor.KeyFingerprint())

	// Create storage and client
	fmt.Print("2. Creating Tier 2 client... ")
	store := blob.NewMemoryStore()
	cfg := client.Tier2Config{
		Dimension: dimension,
		LSHBits:   12, // More bits for 100K vectors
		LSHSeed:   42,
	}
	tier2Client, _ := client.NewTier2Client(cfg, encryptor, store)
	fmt.Println("done")

	// Insert vectors
	fmt.Print("3. Inserting vectors (encrypted)... ")
	start := time.Now()

	// Insert in batches for better performance
	batchSize := 10000
	for i := 0; i < len(vectors); i += batchSize {
		end := min(i+batchSize, len(vectors))
		err = tier2Client.InsertBatch(ctx, ids[i:end], vectors[i:end], nil)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return results
		}
		fmt.Printf("\r3. Inserting vectors (encrypted)... %d/%d", end, len(vectors))
	}
	results.indexBuildTime = time.Since(start)
	fmt.Printf("\r3. Inserting vectors (encrypted)... done (%v)\n", results.indexBuildTime)

	stats, _ := tier2Client.GetStats(ctx)
	fmt.Printf("   %d blobs in %d buckets (%.1f blobs/bucket)\n",
		stats.TotalBlobs, stats.TotalBuckets, stats.AvgBlobsPerBucket)

	// Memory after
	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	results.memoryUsed = memAfter.Alloc - memBefore.Alloc

	// Run queries (basic search without privacy overhead)
	fmt.Printf("4. Running %d queries (basic search)...\n", len(queries))

	queryTimes := make([]time.Duration, len(queries))
	correctResults := 0

	for i, query := range queries {
		queryStart := time.Now()

		searchResults, err := tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
			TopK:          topK,
			NumBuckets:    5,
			UseMultiProbe: true,
		})
		if err != nil {
			continue
		}

		queryTimes[i] = time.Since(queryStart)

		// Check if target was found
		targetID := fmt.Sprintf("doc-%d", targets[i])
		for _, r := range searchResults {
			if r.ID == targetID {
				correctResults++
				break
			}
		}

		fmt.Printf("   Query %d: %v (results: %d)\n", i+1, queryTimes[i], len(searchResults))
	}

	// Calculate statistics
	var totalTime time.Duration
	results.minQueryTime = queryTimes[0]
	results.maxQueryTime = queryTimes[0]
	for _, t := range queryTimes {
		totalTime += t
		if t < results.minQueryTime {
			results.minQueryTime = t
		}
		if t > results.maxQueryTime {
			results.maxQueryTime = t
		}
	}
	results.avgQueryTime = totalTime / time.Duration(len(queries))
	results.recall = float64(correctResults) / float64(len(queries)) * 100

	fmt.Println()
	fmt.Printf("   Average query time: %v\n", results.avgQueryTime)
	fmt.Printf("   Min/Max: %v / %v\n", results.minQueryTime, results.maxQueryTime)
	fmt.Printf("   Recall@%d: %.1f%%\n", topK, results.recall)

	// Also test privacy-enhanced search
	fmt.Printf("\n5. Running %d queries (privacy-enhanced)...\n", len(queries))
	tier2Client.SetPrivacyConfig(client.DefaultPrivacyConfig())

	var privacyTotalTime time.Duration
	for i, query := range queries {
		queryStart := time.Now()
		_, _ = tier2Client.SearchWithPrivacy(ctx, query, topK)
		privacyTime := time.Since(queryStart)
		privacyTotalTime += privacyTime
		fmt.Printf("   Query %d: %v (with timing obfuscation)\n", i+1, privacyTime)
	}
	avgPrivacyTime := privacyTotalTime / time.Duration(len(queries))
	fmt.Printf("   Average privacy-enhanced time: %v\n", avgPrivacyTime)
	fmt.Println()

	return results
}

func printComparison(tier1, tier2 benchmarkResults, numVectors int) {
	fmt.Println("============================================================")
	fmt.Println("COMPARISON SUMMARY")
	fmt.Println("============================================================")
	fmt.Println()

	fmt.Println("┌────────────────────────┬─────────────────────┬─────────────────────┐")
	fmt.Println("│ Metric                 │ Tier 1 (Query-Priv) │ Tier 2 (Data-Priv)  │")
	fmt.Println("├────────────────────────┼─────────────────────┼─────────────────────┤")
	fmt.Printf("│ Index Build Time       │ %19s │ %19s │\n",
		tier1.indexBuildTime.Round(time.Millisecond),
		tier2.indexBuildTime.Round(time.Millisecond))
	fmt.Printf("│ Avg Query Time         │ %19s │ %19s │\n",
		tier1.avgQueryTime.Round(time.Millisecond),
		tier2.avgQueryTime.Round(time.Microsecond))
	fmt.Printf("│ Min Query Time         │ %19s │ %19s │\n",
		tier1.minQueryTime.Round(time.Millisecond),
		tier2.minQueryTime.Round(time.Microsecond))
	fmt.Printf("│ Max Query Time         │ %19s │ %19s │\n",
		tier1.maxQueryTime.Round(time.Millisecond),
		tier2.maxQueryTime.Round(time.Microsecond))
	fmt.Printf("│ Recall@10              │ %18.1f%% │ %18.1f%% │\n",
		tier1.recall, tier2.recall)
	fmt.Printf("│ Memory Used            │ %15.1f MB │ %15.1f MB │\n",
		float64(tier1.memoryUsed)/1024/1024,
		float64(tier2.memoryUsed)/1024/1024)
	fmt.Println("└────────────────────────┴─────────────────────┴─────────────────────┘")
	fmt.Println()

	// Speed comparison
	speedup := float64(tier1.avgQueryTime) / float64(tier2.avgQueryTime)
	fmt.Printf("Speed Comparison: Tier 2 is %.1fx faster than Tier 1\n", speedup)
	fmt.Printf("  (Tier 1: %v vs Tier 2: %v per query)\n", tier1.avgQueryTime, tier2.avgQueryTime)
	fmt.Println()

	// QPS comparison
	tier1QPS := 1.0 / tier1.avgQueryTime.Seconds()
	tier2QPS := 1.0 / tier2.avgQueryTime.Seconds()
	fmt.Printf("Throughput (QPS):\n")
	fmt.Printf("  Tier 1: %.2f queries/sec\n", tier1QPS)
	fmt.Printf("  Tier 2: %.2f queries/sec\n", tier2QPS)
	fmt.Println()

	// Privacy comparison
	fmt.Println("PRIVACY COMPARISON")
	fmt.Println("==================")
	fmt.Println()

	fmt.Println("Tier 1 - What server SEES:")
	for _, s := range tier1.serverSees {
		fmt.Printf("  ✓ %s\n", s)
	}
	fmt.Println()
	fmt.Println("Tier 1 - What server DOES NOT see:")
	for _, s := range tier1.serverDoesNotSee {
		fmt.Printf("  ✗ %s\n", s)
	}
	fmt.Println()

	fmt.Println("Tier 2 - What server SEES:")
	for _, s := range tier2.serverSees {
		fmt.Printf("  ✓ %s\n", s)
	}
	fmt.Println()
	fmt.Println("Tier 2 - What server DOES NOT see:")
	for _, s := range tier2.serverDoesNotSee {
		fmt.Printf("  ✗ %s\n", s)
	}
	fmt.Println()

	// Use case recommendations
	fmt.Println("USE CASE RECOMMENDATIONS")
	fmt.Println("========================")
	fmt.Println()
	fmt.Println("Choose Tier 1 when:")
	fmt.Println("  - You trust the server with the vector data")
	fmt.Println("  - You need to hide what users are searching for")
	fmt.Println("  - Server owns/manages the vector database")
	fmt.Println("  - Example: Enterprise search where queries are sensitive")
	fmt.Println()
	fmt.Println("Choose Tier 2 when:")
	fmt.Println("  - You don't trust the storage backend")
	fmt.Println("  - Users control their own encryption keys")
	fmt.Println("  - Data must be encrypted at rest")
	fmt.Println("  - Example: Blockchain storage, zero-trust cloud, personal vaults")
	fmt.Println()
}

func normalizeVec(v []float64) []float64 {
	result := make([]float64, len(v))
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = sqrt(norm)
	if norm > 0 {
		for i, val := range v {
			result[i] = val / norm
		}
	}
	return result
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 20; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
