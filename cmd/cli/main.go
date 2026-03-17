// Command cli provides a command-line interface for testing Opaque locally.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	opaque "github.com/Prasad-178/opaque"
)

var (
	dimension  = flag.Int("dim", 32, "Vector dimension")
	numVectors = flag.Int("vectors", 100, "Number of vectors to index")
	numQueries = flag.Int("queries", 5, "Number of queries to run")
	topK       = flag.Int("k", 5, "Number of results per query")
	benchmark  = flag.Bool("bench", false, "Run benchmarks")
)

func main() {
	flag.Parse()

	fmt.Println("=== Opaque CLI - Privacy-Preserving Vector Search ===")
	fmt.Println()

	rand.Seed(42)

	// Create opaque DB
	db, err := opaque.NewDB(opaque.Config{
		Dimension:   *dimension,
		NumClusters: 16,
	})
	if err != nil {
		log.Fatalf("Failed to create DB: %v", err)
	}
	defer db.Close()

	// Generate test data
	fmt.Printf("Generating %d vectors of dimension %d...\n", *numVectors, *dimension)
	ids, vectors := generateVectors(*numVectors, *dimension)

	// Add to DB
	ctx := context.Background()
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("Added %d vectors\n", *numVectors)

	// Build the index
	fmt.Println("Building index...")
	start := time.Now()
	if err := db.Build(ctx); err != nil {
		log.Fatalf("Failed to build index: %v", err)
	}
	fmt.Printf("Index built in %v\n\n", time.Since(start))

	// Run queries
	fmt.Printf("Running %d queries (top-%d results each)...\n\n", *numQueries, *topK)

	for i := 0; i < *numQueries; i++ {
		// Create query similar to a random vector
		targetIdx := rand.Intn(*numVectors)
		query := addNoise(vectors[targetIdx], 0.1)

		fmt.Printf("Query %d (similar to %s):\n", i+1, ids[targetIdx])

		start := time.Now()
		results, err := db.Search(ctx, query, *topK)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Printf("  Error: %v\n\n", err)
			continue
		}

		for j, r := range results {
			match := ""
			if r.ID == ids[targetIdx] {
				match = " ← exact match!"
			}
			fmt.Printf("  %d. %s (score: %.4f)%s\n", j+1, r.ID, r.Score, match)
		}
		fmt.Printf("  Time: %v\n\n", elapsed)
	}

	// Run benchmarks
	if *benchmark {
		runBenchmarks(db, vectors)
	}

	fmt.Println("=== Done ===")
}

func generateVectors(n, dim int) ([]string, [][]float64) {
	ids := make([]string, n)
	vectors := make([][]float64, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vec := make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			vec[j] = rand.NormFloat64()
			norm += vec[j] * vec[j]
		}
		norm = math.Sqrt(norm)
		for j := range vec {
			vec[j] /= norm
		}
		vectors[i] = vec
	}

	return ids, vectors
}

func addNoise(vec []float64, scale float64) []float64 {
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

func runBenchmarks(db *opaque.DB, vectors [][]float64) {
	fmt.Println("=== Benchmarks ===")
	fmt.Println()

	ctx := context.Background()
	query := vectors[0]

	// Full search
	n := 20
	start := time.Now()
	for i := 0; i < n; i++ {
		db.Search(ctx, query, 5)
	}
	searchTime := time.Since(start) / time.Duration(n)
	fmt.Printf("Full search:    %v/op\n", searchTime)
	fmt.Printf("Estimated QPS:  %.1f\n", float64(time.Second)/float64(searchTime))
	fmt.Println()
}
