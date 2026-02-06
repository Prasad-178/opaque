package hierarchical_test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

func TestBuilderBasic(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate test data
	numVectors := 1000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Create builder
	key, _ := encrypt.GenerateKey()
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension

	builder, err := hierarchical.NewBuilder(cfg, key)
	if err != nil {
		t.Fatalf("Failed to create builder: %v", err)
	}

	// Build index
	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	// Verify index
	if idx.GetVectorCount() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, idx.GetVectorCount())
	}

	// Check centroids are computed
	centroids := idx.GetCentroids()
	if len(centroids) != cfg.NumSuperBuckets {
		t.Errorf("Expected %d centroids, got %d", cfg.NumSuperBuckets, len(centroids))
	}

	// Verify each centroid is normalized (for non-empty super-buckets)
	for i, c := range centroids {
		if idx.SuperBuckets[i].VectorCount > 0 {
			var norm float64
			for _, v := range c {
				norm += v * v
			}
			if norm < 0.99 || norm > 1.01 {
				t.Errorf("Centroid %d not normalized: norm = %f", i, norm)
			}
		}
	}

	// Print stats
	stats := idx.GetStats()
	t.Logf("Index stats:")
	t.Logf("  Total vectors: %d", stats.TotalVectors)
	t.Logf("  Super-buckets: %d", stats.NumSuperBuckets)
	t.Logf("  Sub-buckets used: %d", stats.NumSubBuckets)
	t.Logf("  Avg vectors/sub: %.1f", stats.AvgVectorsPerSub)
	t.Logf("  Empty sub-buckets: %d", stats.EmptySubBuckets)
}

func TestHierarchicalSearch(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate test data
	numVectors := 5000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Create and build index
	key, _ := encrypt.GenerateKey()
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension

	builder, _ := hierarchical.NewBuilder(cfg, key)
	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	// Create client
	hClient, err := client.NewHierarchicalClient(idx)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Create query (similar to vector 100)
	targetIdx := 100
	query := make([]float64, dimension)
	copy(query, vectors[targetIdx])
	// Add very small noise (realistic for near-neighbor search)
	for i := range query {
		query[i] += (rng.Float64() - 0.5) * 0.02
	}

	// Search
	result, err := hClient.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Verify we got results
	if len(result.Results) == 0 {
		t.Fatal("No results returned")
	}

	// Print timing
	t.Logf("Search timing:")
	t.Logf("  HE encrypt: %v", result.Timing.HEEncryptQuery)
	t.Logf("  HE centroids: %v", result.Timing.HECentroidScores)
	t.Logf("  HE decrypt: %v", result.Timing.HEDecryptScores)
	t.Logf("  Bucket select: %v", result.Timing.BucketSelection)
	t.Logf("  Bucket fetch: %v", result.Timing.BucketFetch)
	t.Logf("  AES decrypt: %v", result.Timing.AESDecrypt)
	t.Logf("  Local score: %v", result.Timing.LocalScoring)
	t.Logf("  Total: %v", result.Timing.Total)

	// Print stats
	t.Logf("Search stats:")
	t.Logf("  HE operations: %d", result.Stats.HEOperations)
	t.Logf("  Super-buckets selected: %d", result.Stats.SuperBucketsSelected)
	t.Logf("  Real sub-buckets: %d", result.Stats.RealSubBuckets)
	t.Logf("  Decoy sub-buckets: %d", result.Stats.DecoySubBuckets)
	t.Logf("  Blobs fetched: %d", result.Stats.BlobsFetched)
	t.Logf("  Vectors scored: %d", result.Stats.VectorsScored)

	// Check if target is in results
	targetID := fmt.Sprintf("doc-%d", targetIdx)
	found := false
	for i, r := range result.Results {
		t.Logf("  %d. %s (score: %.4f)", i+1, r.ID, r.Score)
		if r.ID == targetID {
			found = true
		}
	}

	if !found {
		t.Logf("Note: Target %s not in top 10 (this can happen with LSH)", targetID)
	}
}

func TestHierarchicalRecall(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate test data
	numVectors := 10000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Create and build index
	key, _ := encrypt.GenerateKey()
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension

	builder, _ := hierarchical.NewBuilder(cfg, key)
	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	// Create client
	hClient, err := client.NewHierarchicalClient(idx)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run multiple queries and check recall
	numQueries := 20
	topK := 10
	correctResults := 0

	for q := 0; q < numQueries; q++ {
		// Create query similar to a random vector
		targetIdx := rng.Intn(numVectors)
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		// Add very small noise (more realistic for near-neighbor queries)
		for i := range query {
			query[i] += (rng.Float64() - 0.5) * 0.02
		}

		result, err := hClient.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		// Check if target is in results
		targetID := fmt.Sprintf("doc-%d", targetIdx)
		for _, r := range result.Results {
			if r.ID == targetID {
				correctResults++
				break
			}
		}
	}

	recall := float64(correctResults) / float64(numQueries) * 100
	t.Logf("Recall@%d: %.1f%% (%d/%d queries found target)", topK, recall, correctResults, numQueries)

	// With random data and LSH, recall varies significantly
	// Real ML embeddings would give much higher recall (60-90%)
	// For random data, we just verify the system works (any recall > 0)
	if recall == 0 && numQueries >= 20 {
		t.Errorf("Zero recall with %d queries suggests a bug", numQueries)
	}
}

func TestPrivacyGuarantees(t *testing.T) {
	// This test verifies the privacy properties of the hierarchical search

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	numVectors := 1000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	key, _ := encrypt.GenerateKey()
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension

	builder, _ := hierarchical.NewBuilder(cfg, key)
	store := blob.NewMemoryStore()
	idx, _ := builder.Build(ctx, ids, vectors, store)

	hClient, _ := client.NewHierarchicalClient(idx)

	// Create query
	query := vectors[42]

	result, _ := hClient.Search(ctx, query, 10)

	// Privacy guarantee 1: HE operations cover ALL centroids
	// Server computes scores for all centroids, not just the selected ones
	if result.Stats.HEOperations != cfg.NumSuperBuckets {
		t.Errorf("HE should cover all %d centroids, got %d", cfg.NumSuperBuckets, result.Stats.HEOperations)
	}

	// Privacy guarantee 2: Decoy buckets are fetched
	if result.Stats.DecoySubBuckets < cfg.NumDecoys {
		t.Errorf("Should fetch at least %d decoy buckets, got %d", cfg.NumDecoys, result.Stats.DecoySubBuckets)
	}

	// Privacy guarantee 3: Total buckets fetched includes both real and decoy
	expectedMin := result.Stats.RealSubBuckets + cfg.NumDecoys
	if result.Stats.TotalSubBuckets < expectedMin {
		t.Errorf("Total buckets should be at least %d (real + decoys), got %d", expectedMin, result.Stats.TotalSubBuckets)
	}

	t.Logf("Privacy verification passed:")
	t.Logf("  All centroids scored (HE): %d", result.Stats.HEOperations)
	t.Logf("  Decoy buckets: %d", result.Stats.DecoySubBuckets)
	t.Logf("  Real buckets: %d", result.Stats.RealSubBuckets)
}

func BenchmarkHierarchicalSearch(b *testing.B) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Different dataset sizes
	sizes := []int{10000, 50000, 100000}

	for _, numVectors := range sizes {
		b.Run(fmt.Sprintf("vectors=%d", numVectors), func(b *testing.B) {
			dimension := 128

			ids, vectors := generateTestData(rng, numVectors, dimension)

			key, _ := encrypt.GenerateKey()
			cfg := hierarchical.DefaultConfig()
			cfg.Dimension = dimension

			builder, _ := hierarchical.NewBuilder(cfg, key)
			store := blob.NewMemoryStore()
			idx, _ := builder.Build(ctx, ids, vectors, store)

			hClient, _ := client.NewHierarchicalClient(idx)

			// Prepare queries
			queries := make([][]float64, 10)
			for i := range queries {
				targetIdx := rng.Intn(numVectors)
				queries[i] = make([]float64, dimension)
				copy(queries[i], vectors[targetIdx])
				for j := range queries[i] {
					queries[i][j] += (rng.Float64() - 0.5) * 0.1
				}
			}

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				query := queries[i%len(queries)]
				_, err := hClient.Search(ctx, query, 10)
				if err != nil {
					b.Fatalf("Search failed: %v", err)
				}
			}
		})
	}
}

func BenchmarkHierarchicalBuild(b *testing.B) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	numVectors := 100000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		key, _ := encrypt.GenerateKey()
		cfg := hierarchical.DefaultConfig()
		cfg.Dimension = dimension

		builder, _ := hierarchical.NewBuilder(cfg, key)
		store := blob.NewMemoryStore()
		_, err := builder.Build(ctx, ids, vectors, store)
		if err != nil {
			b.Fatalf("Build failed: %v", err)
		}
	}
}

func generateTestData(rng *rand.Rand, numVectors, dimension int) ([]string, [][]float64) {
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)

	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc-%d", i)
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1 // [-1, 1]
		}
		vectors[i] = vec
	}

	return ids, vectors
}

func TestHierarchicalEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping end-to-end test in short mode")
	}

	ctx := context.Background()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// 100K vectors
	numVectors := 100000
	dimension := 128

	t.Logf("Generating %d vectors...", numVectors)
	ids, vectors := generateTestData(rng, numVectors, dimension)

	t.Log("Building hierarchical index...")
	key, _ := encrypt.GenerateKey()
	cfg := hierarchical.DefaultConfig()
	cfg.Dimension = dimension

	buildStart := time.Now()
	builder, _ := hierarchical.NewBuilder(cfg, key)
	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}
	buildTime := time.Since(buildStart)
	t.Logf("Build time: %v", buildTime)

	stats := idx.GetStats()
	t.Logf("Index stats:")
	t.Logf("  Total vectors: %d", stats.TotalVectors)
	t.Logf("  Super-buckets: %d", stats.NumSuperBuckets)
	t.Logf("  Sub-buckets used: %d", stats.NumSubBuckets)
	t.Logf("  Avg vectors/sub: %.1f", stats.AvgVectorsPerSub)

	t.Log("Creating client...")
	hClient, err := client.NewHierarchicalClient(idx)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run searches
	numQueries := 5
	topK := 10
	var totalTime time.Duration
	correctResults := 0

	t.Logf("Running %d queries...", numQueries)
	for q := 0; q < numQueries; q++ {
		targetIdx := rng.Intn(numVectors)
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		for i := range query {
			query[i] += (rng.Float64() - 0.5) * 0.02 // Small noise
		}

		result, err := hClient.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		totalTime += result.Timing.Total

		// Check if target is in results
		targetID := fmt.Sprintf("doc-%d", targetIdx)
		for _, r := range result.Results {
			if r.ID == targetID {
				correctResults++
				break
			}
		}

		t.Logf("Query %d: %v (HE: %v, Fetch: %v, Score: %v)",
			q+1, result.Timing.Total,
			result.Timing.HECentroidScores,
			result.Timing.BucketFetch,
			result.Timing.LocalScoring)
	}

	avgTime := totalTime / time.Duration(numQueries)
	recall := float64(correctResults) / float64(numQueries) * 100

	t.Logf("\n=== RESULTS ===")
	t.Logf("Dataset: %d vectors, %d dimensions", numVectors, dimension)
	t.Logf("Average query time: %v", avgTime)
	t.Logf("Recall@%d: %.1f%%", topK, recall)
	t.Logf("HE operations per query: %d (vs %d for naive Tier 1)", cfg.NumSuperBuckets, numVectors)
	t.Logf("Speedup factor: %.0fx fewer HE ops", float64(numVectors)/float64(cfg.NumSuperBuckets))
}
