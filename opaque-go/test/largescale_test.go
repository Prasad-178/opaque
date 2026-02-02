package test

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/opaque/opaque-go/pkg/client"
	"github.com/opaque/opaque-go/pkg/crypto"
	"github.com/opaque/opaque-go/pkg/lsh"
)

func TestLargeScale100K(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large scale test in short mode")
	}

	rand.Seed(42)

	const (
		numVectors    = 100000
		dimension     = 128
		numQueries    = 5
		topK          = 20
		maxCandidates = 100
	)

	fmt.Println("=" + "===========================================================")
	fmt.Printf("Large Scale Benchmark: %d vectors, %d dimensions\n", numVectors, dimension)
	fmt.Println("============================================================")

	// Track memory
	var memStart, memAfterIndex runtime.MemStats
	runtime.ReadMemStats(&memStart)

	// 1. Generate vectors
	fmt.Printf("\n[1/6] Generating %d vectors...\n", numVectors)
	start := time.Now()
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)

	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vec := make([]float64, dimension)
		var norm float64
		for j := 0; j < dimension; j++ {
			vec[j] = rand.NormFloat64()
			norm += vec[j] * vec[j]
		}
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}
		vectors[i] = vec
	}
	fmt.Printf("   Generated in %v\n", time.Since(start))

	// 2. Build LSH index
	fmt.Println("\n[2/6] Building LSH index...")
	start = time.Now()
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   128, // More bits for larger dataset
		Seed:      42,
	})
	if err := lshIndex.Add(ids, vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}
	indexTime := time.Since(start)
	fmt.Printf("   Built index in %v\n", indexTime)
	fmt.Printf("   Vectors: %d, Buckets: %d\n", lshIndex.Count(), lshIndex.BucketCount())

	runtime.ReadMemStats(&memAfterIndex)
	memUsed := (memAfterIndex.Alloc - memStart.Alloc) / 1024 / 1024
	fmt.Printf("   Memory used: ~%d MB\n", memUsed)

	// 3. Initialize client
	fmt.Println("\n[3/6] Initializing crypto client...")
	start = time.Now()
	cli, err := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           128,
		MaxCandidates:     200, // Stage 1: Get more candidates
		HECandidates:      topK,
		TopK:              topK,
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
	fmt.Printf("   Client initialized in %v\n", time.Since(start))

	// 4. Benchmark LSH search
	fmt.Println("\n[4/6] Benchmarking LSH search...")
	var totalLSHTime time.Duration
	for i := 0; i < numQueries; i++ {
		query := vectors[rand.Intn(numVectors)]
		queryHash := lshIndex.HashBytes(query)

		start = time.Now()
		candidates, err := lshIndex.Search(queryHash, maxCandidates)
		lshTime := time.Since(start)
		totalLSHTime += lshTime

		if err != nil {
			t.Fatalf("LSH search failed: %v", err)
		}
		fmt.Printf("   Query %d: Found %d candidates in %v\n", i+1, len(candidates), lshTime)
	}
	avgLSHTime := totalLSHTime / time.Duration(numQueries)
	fmt.Printf("   Average LSH search time: %v\n", avgLSHTime)

	// 5. Benchmark full search pipeline (with encryption)
	fmt.Println("\n[5/6] Benchmarking full search pipeline...")

	// Create vector map for scoring
	vectorMap := make(map[string][]float64, numVectors)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	var results []struct {
		queryIdx       int
		lshTime        time.Duration
		encryptTime    time.Duration
		dotProductTime time.Duration
		totalTime      time.Duration
		numCandidates  int
	}

	engine := cli.GetEngine()

	for i := 0; i < numQueries; i++ {
		targetIdx := rand.Intn(numVectors)
		query := addNoiseToVector(vectors[targetIdx], 0.1)
		normalizedQuery := crypto.NormalizeVector(query)

		result := struct {
			queryIdx       int
			lshTime        time.Duration
			encryptTime    time.Duration
			dotProductTime time.Duration
			totalTime      time.Duration
			numCandidates  int
		}{queryIdx: targetIdx}

		totalStart := time.Now()

		// LSH
		lshStart := time.Now()
		queryHash := lshIndex.HashBytes(normalizedQuery)
		candidates, _ := lshIndex.Search(queryHash, maxCandidates)
		result.lshTime = time.Since(lshStart)
		result.numCandidates = len(candidates)

		// Encryption
		encStart := time.Now()
		encQuery, err := engine.EncryptVector(normalizedQuery)
		if err != nil {
			t.Fatalf("Encryption failed: %v", err)
		}
		result.encryptTime = time.Since(encStart)

		// Homomorphic dot products (parallel)
		dotStart := time.Now()
		numToScore := topK
		if numToScore > len(candidates) {
			numToScore = len(candidates)
		}

		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())

		for j := 0; j < numToScore; j++ {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()

				vec := vectorMap[candidates[idx].ID]
				_, _ = engine.HomomorphicDotProduct(encQuery, vec)
			}(j)
		}
		wg.Wait()
		result.dotProductTime = time.Since(dotStart)

		result.totalTime = time.Since(totalStart)
		results = append(results, result)

		fmt.Printf("   Query %d (similar to doc_%d):\n", i+1, targetIdx)
		fmt.Printf("      LSH:         %v (%d candidates)\n", result.lshTime, result.numCandidates)
		fmt.Printf("      Encryption:  %v\n", result.encryptTime)
		fmt.Printf("      %d× Dot Prod: %v (parallel)\n", numToScore, result.dotProductTime)
		fmt.Printf("      Total:       %v\n", result.totalTime)
	}

	// 6. Summary
	fmt.Println("\n[6/6] Summary")
	fmt.Println("============================================================")

	var totalLSH, totalEnc, totalDot, totalTotal time.Duration
	for _, r := range results {
		totalLSH += r.lshTime
		totalEnc += r.encryptTime
		totalDot += r.dotProductTime
		totalTotal += r.totalTime
	}
	n := time.Duration(len(results))

	fmt.Printf("\nAverage times per query:\n")
	fmt.Printf("   LSH Search:      %v\n", totalLSH/n)
	fmt.Printf("   Encryption:      %v\n", totalEnc/n)
	fmt.Printf("   %d× Dot Products: %v (parallel)\n", topK, totalDot/n)
	fmt.Printf("   Total:           %v\n", totalTotal/n)
	fmt.Printf("\nEstimated QPS:      %.2f\n", float64(time.Second)/(float64(totalTotal/n)))
	fmt.Printf("Dataset size:       %d vectors\n", numVectors)
	fmt.Printf("Memory usage:       ~%d MB\n", memUsed)
}

func addNoiseToVector(vec []float64, scale float64) []float64 {
	noisy := make([]float64, len(vec))
	var norm float64
	for i, v := range vec {
		noisy[i] = v + rand.NormFloat64()*scale
		norm += noisy[i] * noisy[i]
	}
	norm = math.Sqrt(norm)
	for i := range noisy {
		noisy[i] /= norm
	}
	return noisy
}

// Benchmark with different dataset sizes
func TestScalabilityBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping scalability test in short mode")
	}

	rand.Seed(42)
	sizes := []int{1000, 10000, 100000}
	dimension := 128

	fmt.Println("============================================================")
	fmt.Println("Scalability Benchmark")
	fmt.Println("============================================================")

	for _, size := range sizes {
		fmt.Printf("\n--- Dataset size: %d ---\n", size)

		// Generate vectors
		ids := make([]string, size)
		vectors := make([][]float64, size)
		for i := 0; i < size; i++ {
			ids[i] = fmt.Sprintf("doc_%d", i)
			vec := make([]float64, dimension)
			var norm float64
			for j := 0; j < dimension; j++ {
				vec[j] = rand.NormFloat64()
				norm += vec[j] * vec[j]
			}
			norm = math.Sqrt(norm)
			for j := range vec {
				vec[j] /= norm
			}
			vectors[i] = vec
		}

		// Build index
		start := time.Now()
		lshIndex := lsh.NewIndex(lsh.Config{
			Dimension: dimension,
			NumBits:   128,
			Seed:      42,
		})
		lshIndex.Add(ids, vectors)
		indexTime := time.Since(start)

		// Search
		query := vectors[0]
		queryHash := lshIndex.HashBytes(query)

		start = time.Now()
		numSearches := 100
		for i := 0; i < numSearches; i++ {
			lshIndex.Search(queryHash, 100)
		}
		searchTime := time.Since(start) / time.Duration(numSearches)

		fmt.Printf("   Index build: %v\n", indexTime)
		fmt.Printf("   LSH search:  %v (avg of %d)\n", searchTime, numSearches)
		fmt.Printf("   Buckets:     %d\n", lshIndex.BucketCount())
	}
}
