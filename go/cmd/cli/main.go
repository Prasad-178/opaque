// Command cli provides a command-line interface for testing Opaque locally.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/opaque/opaque/go/internal/service"
	"github.com/opaque/opaque/go/internal/store"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/lsh"
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

	// Create vector store and service
	vectorStore := store.NewMemoryStore()
	cfg := service.Config{
		LSHNumBits:          64,
		LSHDimension:        *dimension,
		LSHSeed:             42,
		MaxSessionTTL:       time.Hour,
		MaxConcurrentScores: 8,
	}

	svc, err := service.NewSearchService(cfg, vectorStore)
	if err != nil {
		log.Fatalf("Failed to create service: %v", err)
	}

	// Generate test data
	fmt.Printf("Generating %d vectors of dimension %d...\n", *numVectors, *dimension)
	ids, vectors := generateVectors(*numVectors, *dimension)

	// Add to service
	if err := svc.AddVectors(context.Background(), ids, vectors, nil); err != nil {
		log.Fatalf("Failed to add vectors: %v", err)
	}
	fmt.Printf("Indexed %d vectors\n\n", *numVectors)

	// Create client
	fmt.Println("Initializing client...")
	cli, err := client.NewClient(client.Config{
		Dimension:         *dimension,
		LSHBits:           64,
		MaxCandidates:     100,
		HECandidates:      *topK,
		TopK:              *topK,
		EnableHashMasking: true,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	// Get and set LSH planes
	planes := svc.GetLSHIndex().GetPlanes()
	flatPlanes := make([]float64, 0, len(planes)**dimension)
	for _, p := range planes {
		flatPlanes = append(flatPlanes, p...)
	}
	cli.SetLSHPlanes(flatPlanes, len(planes), *dimension)

	// Create a map for local lookup
	vectorMap := make(map[string][]float64)
	for i, id := range ids {
		vectorMap[id] = vectors[i]
	}

	fmt.Println("Client initialized")
	fmt.Println()

	// Run queries
	fmt.Printf("Running %d queries (top-%d results each)...\n\n", *numQueries, *topK)

	for i := 0; i < *numQueries; i++ {
		// Create query similar to a random vector
		targetIdx := rand.Intn(*numVectors)
		query := addNoise(vectors[targetIdx], 0.1)

		fmt.Printf("Query %d (similar to %s):\n", i+1, ids[targetIdx])

		start := time.Now()
		results := searchWithPlaintext(cli, svc.GetLSHIndex(), query, vectorMap, *topK)
		elapsed := time.Since(start)

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
		runBenchmarks(cli, svc.GetLSHIndex(), vectors, vectorMap)
	}

	fmt.Println("=== Done ===")
}

// searchWithPlaintext performs search using LSH for candidates and plaintext scoring
func searchWithPlaintext(cli *client.Client, idx *lsh.Index, query []float64, vectors map[string][]float64, k int) []client.Result {
	// Normalize query
	normalizedQuery := crypto.NormalizeVector(query)

	// Get LSH hash
	queryHash := idx.HashBytes(normalizedQuery)

	// Get candidates
	candidates, err := idx.Search(queryHash, 50)
	if err != nil {
		log.Printf("LSH search failed: %v", err)
		return nil
	}

	// Score candidates
	results := make([]client.Result, 0, len(candidates))
	for _, cand := range candidates {
		vec, ok := vectors[cand.ID]
		if !ok {
			continue
		}
		normalizedVec := crypto.NormalizeVector(vec)
		score := dotProduct(normalizedQuery, normalizedVec)
		results = append(results, client.Result{ID: cand.ID, Score: score})
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > k {
		results = results[:k]
	}

	return results
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

func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func runBenchmarks(cli *client.Client, idx *lsh.Index, vectors [][]float64, vectorMap map[string][]float64) {
	fmt.Println("=== Benchmarks ===")
	fmt.Println()

	query := vectors[0]
	n := 100

	// LSH hash
	start := time.Now()
	for i := 0; i < n; i++ {
		idx.HashBytes(query)
	}
	fmt.Printf("LSH hash:       %v/op\n", time.Since(start)/time.Duration(n))

	// Encryption
	start = time.Now()
	encN := 10
	for i := 0; i < encN; i++ {
		cli.EncryptQuery(query)
	}
	fmt.Printf("Encryption:     %v/op\n", time.Since(start)/time.Duration(encN))

	// Full search
	start = time.Now()
	searchN := 20
	for i := 0; i < searchN; i++ {
		searchWithPlaintext(cli, idx, query, vectorMap, 5)
	}
	searchTime := time.Since(start) / time.Duration(searchN)
	fmt.Printf("Full search:    %v/op\n", searchTime)
	fmt.Printf("Estimated QPS:  %.1f\n", float64(time.Second)/float64(searchTime))
	fmt.Println()
}
