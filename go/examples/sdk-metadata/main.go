// Example sdk-metadata demonstrates adding metadata to vectors
// and using filtered search to narrow results by metadata fields.
//
// Run: go run ./examples/sdk-metadata/
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/opaque/opaque/go"
)

func main() {
	ctx := context.Background()

	db, err := opaque.NewDB(opaque.Config{
		Dimension:   128,
		NumClusters: 16,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Add vectors with metadata.
	rng := rand.New(rand.NewSource(42))
	categories := []string{"tech", "science", "art", "history"}

	const numVectors = 400
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	metadatas := make([]opaque.Metadata, numVectors)

	for i := range ids {
		ids[i] = fmt.Sprintf("doc-%d", i)
		vectors[i] = randomUnitVector(rng, 128)
		metadatas[i] = opaque.Metadata{
			"category": categories[i%len(categories)],
			"year":     2020 + i%5,
			"featured": i%10 == 0,
		}
	}

	if err := db.AddBatchWithMetadata(ctx, ids, vectors, metadatas); err != nil {
		log.Fatal(err)
	}
	if err := db.Build(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Built index with %d vectors\n\n", numVectors)

	query := vectors[0] // Search for something similar to doc-0

	// Unfiltered search.
	all, _ := db.Search(ctx, query, 5)
	fmt.Println("Unfiltered results:")
	printResults(ctx, db, all)

	// Filter by category.
	techOnly, _ := db.SearchWithFilter(ctx, query, 5, opaque.Filter{
		Where: map[string]any{"category": "tech"},
	})
	fmt.Println("Category = 'tech':")
	printResults(ctx, db, techOnly)

	// Filter by year.
	recent, _ := db.SearchWithFilter(ctx, query, 5, opaque.Filter{
		Where: map[string]any{"year": 2024},
	})
	fmt.Println("Year = 2024:")
	printResults(ctx, db, recent)

	// Filter by multiple conditions (AND).
	combined, _ := db.SearchWithFilter(ctx, query, 5, opaque.Filter{
		Where: map[string]any{"category": "science", "featured": true},
	})
	fmt.Println("Category = 'science' AND featured = true:")
	printResults(ctx, db, combined)
}

func printResults(ctx context.Context, db *opaque.DB, results []opaque.Result) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
	}
	for i, r := range results {
		meta, _ := db.GetMetadata(ctx, r.ID)
		fmt.Printf("  %d. %s (%.4f) — %v\n", i+1, r.ID, r.Score, meta)
	}
	fmt.Println()
}

func randomUnitVector(rng *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	var norm float64
	for j := range v {
		v[j] = rng.Float64()*2 - 1
		norm += v[j] * v[j]
	}
	norm = math.Sqrt(norm)
	for j := range v {
		v[j] /= norm
	}
	return v
}
