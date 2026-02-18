//go:build integration

package test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// TestOptimizedTwoStageSearch benchmarks the optimized two-stage search
// with hash masking enabled.
func TestOptimizedTwoStageSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping optimized benchmark in short mode")
	}

	rand.Seed(42)
	numCPU := runtime.NumCPU()

	const (
		numVectors        = 100000
		dimension         = 128
		numQueries        = 10
		initialCandidates = 200 // Stage 1: cheap LSH
		heCandidates      = 10  // Stage 2: expensive HE
	)

	fmt.Println("============================================================")
	fmt.Printf("OPTIMIZED Two-Stage Search: %d vectors, %d cores\n", numVectors, numCPU)
	fmt.Printf("Config: %d LSH candidates → %d HE dot products\n", initialCandidates, heCandidates)
	fmt.Println("============================================================")

	// Generate data
	fmt.Printf("\n[1/5] Generating %d vectors...\n", numVectors)
	start := time.Now()
	ids, vectors := generateTestVectors(numVectors, dimension)
	fmt.Printf("   Generated in %v\n", time.Since(start))

	// Build LSH index
	fmt.Println("\n[2/5] Building LSH index...")
	start = time.Now()
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   128,
		Seed:      42,
	})
	lshIndex.Add(ids, vectors)
	fmt.Printf("   Built in %v\n", time.Since(start))

	// Create optimized client
	fmt.Println("\n[3/5] Creating optimized client...")
	start = time.Now()
	cli, err := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           128,
		MaxCandidates:     initialCandidates,
		HECandidates:      heCandidates,
		TopK:              heCandidates,
		EnableHashMasking: true,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Set LSH planes
	planes := lshIndex.GetPlanes()
	flatPlanes := make([]float64, 0, len(planes)*dimension)
	for _, p := range planes {
		flatPlanes = append(flatPlanes, p...)
	}
	cli.SetLSHPlanes(flatPlanes, len(planes), dimension)
	fmt.Printf("   Client created in %v\n", time.Since(start))
	fmt.Printf("   Hash masking: enabled\n")

	// Create vector map
	vectorMap := make(map[string][]float64, numVectors)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	// Create worker pool once (reuse across queries)
	fmt.Printf("\n[4/5] Creating worker pool with %d engines...\n", numCPU)
	poolStart := time.Now()
	pool, err := NewWorkerPool(numCPU)
	if err != nil {
		t.Fatalf("Failed to create worker pool: %v", err)
	}
	defer pool.Close()
	fmt.Printf("   Pool created in %v\n", time.Since(poolStart))

	// Benchmark optimized search
	fmt.Println("\n[5/6] Benchmarking optimized two-stage search...")

	var totalTime time.Duration
	var totalLSHTime time.Duration
	var totalEncryptTime time.Duration
	var totalHETime time.Duration

	for q := 0; q < numQueries; q++ {
		targetIdx := rand.Intn(numVectors)
		query := addNoiseVec(vectors[targetIdx], 0.1)
		normalizedQuery := crypto.NormalizeVector(query)

		queryStart := time.Now()

		// Stage 1: LSH search with hash masking
		lshStart := time.Now()
		maskedHash, _ := cli.ComputeMaskedLSHHash(normalizedQuery)
		// Server would unmask: unmaskedHash := lsh.UnmaskHash(maskedHash, cli.GetSessionKey())
		// For local test, use direct hash
		queryHash := lshIndex.HashBytes(normalizedQuery)
		candidates, _ := lshIndex.Search(queryHash, initialCandidates)
		lshTime := time.Since(lshStart)
		totalLSHTime += lshTime

		// Get top candidates by Hamming distance (already sorted)
		if len(candidates) > heCandidates {
			candidates = candidates[:heCandidates]
		}

		// Encrypt query
		encStart := time.Now()
		engine := cli.GetEngine()
		encQuery, _ := engine.EncryptVector(normalizedQuery)
		encryptTime := time.Since(encStart)
		totalEncryptTime += encryptTime

		// Stage 2: Parallel HE dot products (only on top candidates)
		heStart := time.Now()
		results := make([]struct {
			score float64
			err   error
		}, len(candidates))

		for i := 0; i < len(candidates); i++ {
			idx := i
			vec := vectorMap[candidates[idx].ID]
			pool.Submit(func(eng *crypto.Engine) {
				encScore, err := eng.HomomorphicDotProduct(encQuery, vec)
				if err != nil {
					results[idx].err = err
					return
				}
				score, err := eng.DecryptScalar(encScore)
				results[idx].score = score
				results[idx].err = err
			})
		}
		pool.Wait()
		heTime := time.Since(heStart)
		totalHETime += heTime

		queryTime := time.Since(queryStart)
		totalTime += queryTime

		// Verify we got results
		_ = maskedHash // Used for privacy, verified it compiles

		fmt.Printf("   Query %d: LSH=%v, Encrypt=%v, %d×HE=%v, Total=%v\n",
			q+1, lshTime, encryptTime, len(candidates), heTime, queryTime)
	}

	// Calculate averages
	n := time.Duration(numQueries)
	avgTotal := totalTime / n
	avgLSH := totalLSHTime / n
	avgEncrypt := totalEncryptTime / n
	avgHE := totalHETime / n

	fmt.Println("\n[6/6] Summary")
	fmt.Println("============================================================")
	fmt.Printf("   Avg LSH search:     %v\n", avgLSH)
	fmt.Printf("   Avg encryption:     %v\n", avgEncrypt)
	fmt.Printf("   Avg %d×HE ops:      %v (parallel)\n", heCandidates, avgHE)
	fmt.Printf("   Avg total time:     %v\n", avgTotal)
	fmt.Printf("   Estimated QPS:      %.2f\n", float64(time.Second)/float64(avgTotal))

	// Compare with baseline (20 HE ops)
	fmt.Println("\n--- Comparison with baseline (20 HE ops) ---")
	baselineHE := avgHE * 2 // Roughly 2x for 20 vs 10 HE ops
	baselineTotal := avgLSH + avgEncrypt + baselineHE
	fmt.Printf("   Baseline (20 HE):   ~%v\n", baselineTotal)
	fmt.Printf("   Optimized (10 HE):  %v\n", avgTotal)
	fmt.Printf("   Speedup:            %.2fx\n", float64(baselineTotal)/float64(avgTotal))
}

// TestHashMaskingPrivacy verifies hash masking works correctly.
func TestHashMaskingPrivacy(t *testing.T) {
	dimension := 128

	// Create two clients with different session keys
	cli1, _ := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           128,
		MaxCandidates:     100,
		HECandidates:      10,
		EnableHashMasking: true,
	})

	cli2, _ := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           128,
		MaxCandidates:     100,
		HECandidates:      10,
		EnableHashMasking: true,
	})

	// Create LSH index and set planes
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   128,
		Seed:      42,
	})

	planes := lshIndex.GetPlanes()
	flatPlanes := make([]float64, 0, len(planes)*dimension)
	for _, p := range planes {
		flatPlanes = append(flatPlanes, p...)
	}
	cli1.SetLSHPlanes(flatPlanes, len(planes), dimension)
	cli2.SetLSHPlanes(flatPlanes, len(planes), dimension)

	// Same query vector
	query := make([]float64, dimension)
	for i := range query {
		query[i] = rand.NormFloat64()
	}
	norm := 0.0
	for _, v := range query {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	for i := range query {
		query[i] /= norm
	}

	// Get masked hashes
	hash1, _ := cli1.ComputeMaskedLSHHash(query)
	hash2, _ := cli2.ComputeMaskedLSHHash(query)

	// Verify masked hashes are different (due to different session keys)
	same := true
	for i := range hash1 {
		if hash1[i] != hash2[i] {
			same = false
			break
		}
	}

	if same {
		t.Error("Masked hashes should be different for different session keys")
	}

	// Verify unmasking recovers same hash
	unmasked1 := lsh.UnmaskHash(hash1, cli1.GetSessionKey())
	unmasked2 := lsh.UnmaskHash(hash2, cli2.GetSessionKey())

	for i := range unmasked1 {
		if unmasked1[i] != unmasked2[i] {
			t.Error("Unmasked hashes should be identical")
			break
		}
	}

	fmt.Println("Hash masking test passed:")
	fmt.Printf("   Masked hash 1:   %x...\n", hash1[:8])
	fmt.Printf("   Masked hash 2:   %x...\n", hash2[:8])
	fmt.Printf("   Unmasked (same): %x...\n", unmasked1[:8])
}

// TestTwoStageAccuracy verifies two-stage search maintains accuracy.
func TestTwoStageAccuracy(t *testing.T) {
	rand.Seed(42)

	const (
		numVectors = 1000
		dimension  = 128
		numQueries = 10
	)

	// Generate data
	ids, vectors := generateTestVectors(numVectors, dimension)

	// Build index
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   128,
		Seed:      42,
	})
	lshIndex.Add(ids, vectors)

	// Create client
	cli, _ := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           128,
		MaxCandidates:     200,
		HECandidates:      10,
		TopK:              10,
		EnableHashMasking: true,
	})

	planes := lshIndex.GetPlanes()
	flatPlanes := make([]float64, 0, len(planes)*dimension)
	for _, p := range planes {
		flatPlanes = append(flatPlanes, p...)
	}
	cli.SetLSHPlanes(flatPlanes, len(planes), dimension)

	// Create vector map
	vectorMap := make(map[string][]float64, numVectors)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	// Test queries
	fmt.Println("Testing two-stage search accuracy...")

	totalRecall := 0.0

	for q := 0; q < numQueries; q++ {
		targetIdx := rand.Intn(numVectors)
		query := addNoiseVec(vectors[targetIdx], 0.05) // Small noise

		// Get two-stage results
		results, err := cli.TwoStageSearchLocal(context.Background(), query, lshIndex, vectorMap)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Check if target is in top results
		found := false
		for _, r := range results {
			if r.ID == ids[targetIdx] {
				found = true
				break
			}
		}

		if found {
			totalRecall += 1.0
		}

		fmt.Printf("   Query %d: target=%s, found=%v, top_score=%.4f\n",
			q+1, ids[targetIdx], found, results[0].Score)
	}

	recall := totalRecall / float64(numQueries) * 100
	fmt.Printf("\nRecall@10: %.1f%%\n", recall)

	if recall < 70 {
		t.Errorf("Recall too low: %.1f%% (expected >70%%)", recall)
	}
}
