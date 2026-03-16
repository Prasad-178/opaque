// Example sdk-large-scale demonstrates tuning Opaque for larger datasets.
// It indexes 10,000 vectors and shows how NumClusters, TopClusters, and
// WorkerPoolSize affect performance.
//
// Run: go run ./examples/sdk-large-scale/
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/Prasad-178/opaque"
)

func main() {
	ctx := context.Background()

	const (
		numVectors = 10_000
		dim        = 128
		topK       = 10
	)

	ids, vectors := generateVectors(numVectors, dim)

	// For larger datasets, increase NumClusters for faster search
	// and WorkerPoolSize for parallel decryption.
	db, err := opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: 64, // More clusters = faster search, fewer vectors per cluster
		TopClusters: 8,  // Probe 8 of 64 clusters (~12.5% of data)
		NumDecoys:   4,  // Privacy: fetch 4 extra random clusters as cover traffic
		WorkerPoolSize: 4, // Parallel AES decryption
	})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	// Batch add is much faster than individual adds.
	start := time.Now()
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Added %d vectors in %v\n", numVectors, time.Since(start))

	// Build: k-means clustering + HE engine initialization.
	start = time.Now()
	if err := db.Build(ctx); err != nil {
		log.Fatal(err)
	}
	buildTime := time.Since(start)
	fmt.Printf("Built index in %v\n", buildTime)

	// Print cluster statistics.
	stats := db.Stats(ctx)
	cs := stats.ClusterStats
	fmt.Printf("\nCluster stats:\n")
	fmt.Printf("  Clusters: %d\n", cs.NumClusters)
	fmt.Printf("  Min/Avg/Max size: %d / %.0f / %d\n", cs.MinSize, cs.AvgSize, cs.MaxSize)
	fmt.Printf("  Empty clusters: %d\n", cs.EmptyClusters)
	fmt.Printf("  K-means iterations: %d\n", cs.Iterations)

	// Benchmark search latency.
	fmt.Printf("\nSearching (topK=%d)...\n", topK)
	const numQueries = 20
	var totalTime time.Duration
	for i := range numQueries {
		start = time.Now()
		results, err := db.Search(ctx, vectors[i*50], topK)
		elapsed := time.Since(start)
		totalTime += elapsed

		if err != nil {
			log.Fatal(err)
		}
		if i == 0 {
			fmt.Printf("  Query 1: top=%s (%.4f), %v\n", results[0].ID, results[0].Score, elapsed)
		}
	}

	avgLatency := totalTime / numQueries
	qps := float64(numQueries) / totalTime.Seconds()
	fmt.Printf("\n  %d queries: avg=%v, QPS=%.1f\n", numQueries, avgLatency, qps)

	// Tuning tips.
	fmt.Println("\nTuning tips:")
	fmt.Println("  - More NumClusters → faster search (fewer vectors per cluster)")
	fmt.Println("  - More TopClusters → better recall (more clusters probed)")
	fmt.Println("  - More NumDecoys → stronger privacy (more cover traffic)")
	fmt.Println("  - More WorkerPoolSize → faster decryption (parallel AES)")
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
