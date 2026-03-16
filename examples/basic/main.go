// Example sdk-basic demonstrates the simplest Opaque workflow:
// create a DB, add vectors, build the index, and search.
//
// Run: go run ./examples/sdk-basic/
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/Prasad-178/opaque"
)

func main() {
	ctx := context.Background()

	// 1. Create a new database.
	db, err := opaque.NewDB(opaque.Config{
		Dimension:   128,
		NumClusters: 16,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// 2. Generate some vectors and add them.
	const numVectors = 500
	ids, vectors := generateVectors(numVectors, 128)

	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Added %d vectors\n", numVectors)

	// 3. Build the index (k-means clustering + HE engine init).
	if err := db.Build(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Index built")

	// 4. Search for the 5 nearest neighbors to vectors[0].
	results, err := db.Search(ctx, vectors[0], 5)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nSearch results for %q:\n", ids[0])
	for i, r := range results {
		fmt.Printf("  %d. %s (score: %.4f)\n", i+1, r.ID, r.Score)
	}

	// 5. Check some info APIs.
	fmt.Printf("\nDB stats: %d vectors, %d clusters\n",
		db.Size(), db.Stats(ctx).ClusterStats.NumClusters)
}

func generateVectors(n, dim int) ([]string, [][]float64) {
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, n)
	vecs := make([][]float64, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("doc-%d", i)
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
		vecs[i] = v
	}
	return ids, vecs
}
