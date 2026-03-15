// Example sdk-file-storage demonstrates using file-backed storage
// instead of in-memory storage. This keeps encrypted blobs on disk,
// reducing memory usage for large datasets.
//
// Run: go run ./examples/sdk-file-storage/
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/opaque/opaque/go"
)

func main() {
	ctx := context.Background()

	tmpDir, err := os.MkdirTemp("", "opaque-file-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	storagePath := filepath.Join(tmpDir, "blobs")
	savePath := filepath.Join(tmpDir, "saved-index")

	// Use File storage backend — blobs are written to disk instead of RAM.
	db, err := opaque.NewDB(opaque.Config{
		Dimension:   128,
		NumClusters: 16,
		Storage:     opaque.File,
		StoragePath: storagePath,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Add vectors.
	ids, vectors := generateVectors(500, 128)
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		log.Fatal(err)
	}

	// Build the index — blobs are written to storagePath.
	if err := db.Build(ctx); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Built with file storage at %s\n", storagePath)
	fmt.Printf("Vectors: %d\n", db.Size())

	// Search works the same way.
	results, err := db.Search(ctx, vectors[0], 5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nSearch results:\n")
	for i, r := range results {
		fmt.Printf("  %d. %s (%.4f)\n", i+1, r.ID, r.Score)
	}

	// Save to a different location — blobs are copied.
	if err := db.Save(savePath); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nSaved to %s\n", savePath)
	db.Close()

	// Load from saved location.
	loaded, err := opaque.Load(savePath)
	if err != nil {
		log.Fatal(err)
	}
	defer loaded.Close()

	loadedResults, err := loaded.Search(ctx, vectors[0], 5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded and searched: top=%s (%.4f)\n",
		loadedResults[0].ID, loadedResults[0].Score)

	// List directory contents to show what's saved.
	entries, _ := os.ReadDir(savePath)
	fmt.Printf("\nSaved files:\n")
	for _, e := range entries {
		info, _ := e.Info()
		if e.IsDir() {
			fmt.Printf("  %s/\n", e.Name())
		} else {
			fmt.Printf("  %s (%d bytes)\n", e.Name(), info.Size())
		}
	}
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
