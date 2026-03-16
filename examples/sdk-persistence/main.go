// Example sdk-persistence demonstrates saving a built index to disk
// and loading it back in a new process.
//
// Run: go run ./examples/sdk-persistence/
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/Prasad-178/opaque"
)

func main() {
	ctx := context.Background()
	tmpDir, err := os.MkdirTemp("", "opaque-persist-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	savePath := filepath.Join(tmpDir, "my-index")

	ids, vectors := generateVectors(300, 128)

	// Phase 1: Build and save.
	fmt.Println("=== Phase 1: Build & Save ===")
	{
		db, err := opaque.NewDB(opaque.Config{
			Dimension:   128,
			NumClusters: 16,
		})
		if err != nil {
			log.Fatal(err)
		}

		db.AddBatch(ctx, ids, vectors)
		if err := db.Build(ctx); err != nil {
			log.Fatal(err)
		}

		results, _ := db.Search(ctx, vectors[0], 3)
		fmt.Printf("Before save — top result: %s (%.4f)\n", results[0].ID, results[0].Score)

		if err := db.Save(savePath); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Saved to %s\n", savePath)
		db.Close()
	}

	// Phase 2: Load and search (simulates a new process).
	fmt.Println("\n=== Phase 2: Load & Search ===")
	{
		db, err := opaque.Load(savePath)
		if err != nil {
			log.Fatal(err)
		}
		defer db.Close()

		fmt.Printf("Loaded: %d vectors, ready=%v\n", db.Size(), db.IsReady())

		results, _ := db.Search(ctx, vectors[0], 3)
		fmt.Printf("After load — top result: %s (%.4f)\n", results[0].ID, results[0].Score)

		// You can add more vectors and rebuild after loading.
		newIDs, newVecs := generateVectorsWithSeed(50, 128, 99)
		for i := range newIDs {
			db.Add(ctx, newIDs[i], newVecs[i])
		}
		if err := db.Rebuild(ctx); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("After rebuild: %d vectors\n", db.Size())
	}
}

func generateVectors(n, dim int) ([]string, [][]float64) {
	return generateVectorsWithSeed(n, dim, 42)
}

func generateVectorsWithSeed(n, dim int, seed int64) ([]string, [][]float64) {
	rng := rand.New(rand.NewSource(seed))
	ids := make([]string, n)
	vecs := make([][]float64, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("doc-%d-%d", seed, i)
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
