// Example: Privacy-preserving vector search demo
package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/lsh"
)

const (
	dimension     = 32  // Vector dimension (smaller for faster demo)
	numVectors    = 100 // Number of vectors in database
	numQueries    = 5   // Number of search queries
	topK          = 5   // Top results to return
	maxCandidates = 20  // LSH candidates
)

func main() {
	fmt.Println("=== Opaque: Privacy-Preserving Vector Search Demo ===")
	fmt.Println()

	// Seed random for reproducibility
	rand.Seed(42)

	// 1. Create synthetic database
	fmt.Println("1. Creating synthetic vector database...")
	ids, vectors := createSyntheticData(numVectors, dimension)
	fmt.Printf("   Created %d vectors of dimension %d\n", numVectors, dimension)

	// 2. Initialize LSH index
	fmt.Println("\n2. Building LSH index...")
	start := time.Now()
	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   64,
		Seed:      42,
	})
	if err := lshIndex.Add(ids, vectors); err != nil {
		panic(err)
	}
	fmt.Printf("   Built index in %v\n", time.Since(start))
	fmt.Printf("   Buckets: %d\n", lshIndex.BucketCount())

	// 3. Initialize client
	fmt.Println("\n3. Initializing client with homomorphic encryption...")
	start = time.Now()
	cli, err := client.NewClient(client.Config{
		Dimension:         dimension,
		LSHBits:           64,
		MaxCandidates:     maxCandidates,
		HECandidates:      topK,
		TopK:              topK,
		EnableHashMasking: true,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("   Client initialized in %v\n", time.Since(start))

	// 4. Get public key
	pubKey, err := cli.GetPublicKey()
	if err != nil {
		panic(err)
	}
	fmt.Printf("   Public key size: %d bytes\n", len(pubKey))

	// 5. Set LSH planes from index
	planes := lshIndex.GetPlanes()
	flatPlanes := make([]float64, 0, len(planes)*dimension)
	for _, p := range planes {
		flatPlanes = append(flatPlanes, p...)
	}
	if err := cli.SetLSHPlanes(flatPlanes, len(planes), dimension); err != nil {
		panic(err)
	}

	// 6. Test homomorphic encryption
	fmt.Println("\n4. Testing homomorphic encryption accuracy...")
	testHomomorphicAccuracy(cli, vectors)

	// 7. Perform searches
	fmt.Println("\n5. Performing privacy-preserving searches...")
	vectorsMap := make(map[string][]float64)
	for i, id := range ids {
		vectorsMap[id] = vectors[i]
	}

	for i := 0; i < numQueries; i++ {
		// Create a query similar to a random vector
		targetIdx := rand.Intn(numVectors)
		query := addNoise(vectors[targetIdx], 0.1)

		fmt.Printf("\n   Query %d (similar to doc_%d):\n", i+1, targetIdx)

		start := time.Now()
		results, err := cli.SearchLocal(context.Background(), query, lshIndex, vectorsMap, topK)
		if err != nil {
			fmt.Printf("   Error: %v\n", err)
			continue
		}
		elapsed := time.Since(start)

		fmt.Printf("   Found %d results in %v:\n", len(results), elapsed)
		for j, r := range results {
			fmt.Printf("     %d. %s (score: %.4f)\n", j+1, r.ID, r.Score)
		}
	}

	// 8. Benchmark
	fmt.Println("\n6. Benchmarking...")
	benchmark(cli, vectors, lshIndex, vectorsMap)

	fmt.Println("\n=== Demo Complete ===")
}

// createSyntheticData creates random normalized vectors
func createSyntheticData(n, dim int) ([]string, [][]float64) {
	ids := make([]string, n)
	vectors := make([][]float64, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)

		// Generate random vector
		vec := make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			vec[j] = rand.NormFloat64()
			norm += vec[j] * vec[j]
		}

		// Normalize
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}

		vectors[i] = vec
	}

	return ids, vectors
}

// addNoise adds Gaussian noise to a vector
func addNoise(vec []float64, scale float64) []float64 {
	noisy := make([]float64, len(vec))
	var norm float64
	for i, v := range vec {
		noisy[i] = v + rand.NormFloat64()*scale
		norm += noisy[i] * noisy[i]
	}

	// Renormalize
	norm = math.Sqrt(norm)
	for i := range noisy {
		noisy[i] /= norm
	}

	return noisy
}

// testHomomorphicAccuracy tests the accuracy of homomorphic operations
func testHomomorphicAccuracy(cli *client.Client, vectors [][]float64) {
	// Test with a few vector pairs
	numTests := 5
	var totalError float64

	for i := 0; i < numTests; i++ {
		query := vectors[rand.Intn(len(vectors))]
		vector := vectors[rand.Intn(len(vectors))]

		decrypted, plaintext, err := cli.EncryptAndComputeLocal(query, vector)
		if err != nil {
			fmt.Printf("   Test %d: Error - %v\n", i+1, err)
			continue
		}

		// Note: The homomorphic result will differ from plaintext due to:
		// 1. Fixed-point encoding
		// 2. Offset handling in BFV
		// 3. The rotation-based sum may not be complete without evaluation keys
		// For now, we just verify the encryption/decryption works
		error := math.Abs(decrypted - plaintext)
		totalError += error

		fmt.Printf("   Test %d: plaintext=%.6f, decrypted=%.6f, diff=%.6f\n",
			i+1, plaintext, decrypted, error)
	}

	avgError := totalError / float64(numTests)
	fmt.Printf("   Average error: %.6f\n", avgError)
}

// benchmark runs performance benchmarks
func benchmark(cli *client.Client, vectors [][]float64, lshIndex *lsh.Index, vectorsMap map[string][]float64) {
	numBenchQueries := 100
	query := vectors[0]

	// Benchmark LSH
	start := time.Now()
	for i := 0; i < numBenchQueries; i++ {
		_, _ = cli.ComputeLSHHash(query)
	}
	lshTime := time.Since(start) / time.Duration(numBenchQueries)
	fmt.Printf("   LSH hash:        %v per query\n", lshTime)

	// Benchmark encryption
	start = time.Now()
	numEncrypt := 10
	for i := 0; i < numEncrypt; i++ {
		_, _ = cli.EncryptQuery(query)
	}
	encTime := time.Since(start) / time.Duration(numEncrypt)
	fmt.Printf("   Encryption:      %v per query\n", encTime)

	// Benchmark full search
	start = time.Now()
	numSearches := 10
	for i := 0; i < numSearches; i++ {
		_, _ = cli.SearchLocal(context.Background(), query, lshIndex, vectorsMap, topK)
	}
	searchTime := time.Since(start) / time.Duration(numSearches)
	fmt.Printf("   Full search:     %v per query\n", searchTime)

	fmt.Printf("   Estimated QPS:   %.1f\n", float64(time.Second)/float64(searchTime))
}
