package client

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/embeddings"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// getSIFTDataPath returns the path to the SIFT dataset
func getSIFTDataPath() string {
	// Try relative path from go directory
	candidates := []string{
		"../data/siftsmall",
		"../../data/siftsmall",
		"../../../data/siftsmall",
	}

	for _, path := range candidates {
		absPath, _ := filepath.Abs(path)
		if _, err := os.Stat(filepath.Join(absPath, "siftsmall_base.fvecs")); err == nil {
			return absPath
		}
	}

	return ""
}

// TestSIFTAccuracyWithGroundTruth tests accuracy using real SIFT embeddings with ground truth
func TestSIFTAccuracyWithGroundTruth(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found. Download from http://corpus-texmex.irisa.fr/")
	}

	ctx := context.Background()

	// Load SIFT10K dataset
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT10K: %v", err)
	}

	stats := dataset.Stats()
	fmt.Println("=" + "=================================================================")
	fmt.Println("SIFT10K ACCURACY TEST WITH REAL EMBEDDINGS")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nDataset: %s\n", stats.Name)
	fmt.Printf("Vectors: %d (dimension: %d)\n", stats.NumVectors, stats.Dimension)
	fmt.Printf("Queries: %d\n", stats.NumQueries)
	fmt.Printf("Ground Truth Depth: %d\n", stats.GroundTruthDepth)

	// Test configurations - reduced for faster execution
	configs := []struct {
		numSuperBuckets int
		topSuperBuckets int
		numSubBuckets   int
	}{
		{32, 8, 64},
		{32, 16, 64},
		{32, 32, 64}, // Fetch all super-buckets
	}

	topK := 10
	numQueries := min(20, len(dataset.Queries)) // Reduced for faster testing

	fmt.Println("\n┌────────────┬───────────┬──────────┬────────────┬────────────┬────────────┬───────────┐")
	fmt.Println("│ SuperBkts  │ TopSelect │ Recall@1 │  Recall@10 │   NDCG@10  │  Vectors   │ % Dataset │")
	fmt.Println("│            │           │ (GT)     │   (GT)     │            │  Scanned   │  Scanned  │")
	fmt.Println("├────────────┼───────────┼──────────┼────────────┼────────────┼────────────┼───────────┤")

	for _, cfg := range configs {
		recall1, recall10, ndcg, avgScanned := runSIFTAccuracyTest(
			ctx, t, dataset, cfg.numSuperBuckets, cfg.topSuperBuckets, cfg.numSubBuckets, numQueries, topK,
		)

		pctScanned := float64(avgScanned) / float64(stats.NumVectors) * 100

		fmt.Printf("│     %2d     │    %2d     │  %5.1f%%  │   %5.1f%%   │   %6.4f   │   %6d   │   %5.2f%%  │\n",
			cfg.numSuperBuckets, cfg.topSuperBuckets,
			recall1*100, recall10*100, ndcg, avgScanned, pctScanned)
	}

	fmt.Println("└────────────┴───────────┴──────────┴────────────┴────────────┴────────────┴───────────┘")

	fmt.Println("\nMetrics explained:")
	fmt.Println("  Recall@1 (GT):  % of queries where top-1 result is in ground truth top-1")
	fmt.Println("  Recall@10 (GT): % of ground truth top-10 found in our top-10")
	fmt.Println("  NDCG@10:        Normalized Discounted Cumulative Gain (ranking quality)")
	fmt.Println("  Vectors Scanned: Average number of vectors decrypted and scored per query")
}

func runSIFTAccuracyTest(
	ctx context.Context,
	t *testing.T,
	dataset *embeddings.Dataset,
	numSuperBuckets, topSuperBuckets, numSubBuckets, numQueries, topK int,
) (recall1, recall10, ndcg float64, avgScanned int) {

	// Setup Tier 2.5 with enterprise configuration
	enterpriseCfg, err := enterprise.NewConfig("sift-test", dataset.Dimension, numSuperBuckets)
	if err != nil {
		t.Fatalf("Failed to create enterprise config: %v", err)
	}
	enterpriseCfg.NumSubBuckets = numSubBuckets

	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	// Build index with AES encryption
	builder, err := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	if err != nil {
		t.Fatalf("Failed to create builder: %v", err)
	}

	_, err = builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	enterpriseCfg = builder.GetEnterpriseConfig()

	// Setup auth and client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "sift-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Run queries using SIFT query vectors
	var totalRecall1, totalRecall10, totalNDCG float64
	var totalScanned int

	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]
		groundTruth := dataset.GroundTruth[q] // True nearest neighbor indices

		// Tier 2.5 search
		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Convert result IDs to indices
		resultIndices := make([]int, len(result.Results))
		for i, r := range result.Results {
			// Parse index from ID like "sift_123"
			var idx int
			fmt.Sscanf(r.ID, "sift_%d", &idx)
			resultIndices[i] = idx
		}

		// Calculate metrics using ground truth
		r1, r10, n := calculateSIFTMetrics(groundTruth, resultIndices, topK)
		totalRecall1 += r1
		totalRecall10 += r10
		totalNDCG += n
		totalScanned += result.Stats.VectorsScored
	}

	recall1 = totalRecall1 / float64(numQueries)
	recall10 = totalRecall10 / float64(numQueries)
	ndcg = totalNDCG / float64(numQueries)
	avgScanned = totalScanned / numQueries

	return
}

func calculateSIFTMetrics(groundTruth, predicted []int, topK int) (recall1, recall10, ndcg float64) {
	if len(predicted) == 0 {
		return 0, 0, 0
	}

	// Recall@1: Is the top result correct?
	if len(groundTruth) > 0 && len(predicted) > 0 && groundTruth[0] == predicted[0] {
		recall1 = 1.0
	}

	// Recall@K: What fraction of ground truth top-K is in predicted top-K?
	gtSet := make(map[int]bool)
	for i := 0; i < len(groundTruth) && i < topK; i++ {
		gtSet[groundTruth[i]] = true
	}

	found := 0
	for i := 0; i < len(predicted) && i < topK; i++ {
		if gtSet[predicted[i]] {
			found++
		}
	}
	if len(gtSet) > 0 {
		recall10 = float64(found) / float64(min(len(gtSet), topK))
	}

	// NDCG@K: Normalized Discounted Cumulative Gain
	gtRanks := make(map[int]int)
	for i, idx := range groundTruth {
		gtRanks[idx] = i + 1
	}

	var dcg, idcg float64
	for i := 0; i < len(predicted) && i < topK; i++ {
		if rank, ok := gtRanks[predicted[i]]; ok {
			// Relevance based on ground truth rank (higher rank = more relevant)
			rel := float64(topK - rank + 1)
			if rel > 0 {
				dcg += rel / math.Log2(float64(i+2))
			}
		}
	}

	// Ideal DCG (if we had perfect ranking)
	for i := 0; i < topK; i++ {
		rel := float64(topK - i)
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg > 0 {
		ndcg = dcg / idcg
	}

	return
}

// TestSIFTVsBruteForce compares Tier 2.5 results against brute force search
func TestSIFTVsBruteForce(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found. Download from http://corpus-texmex.irisa.fr/")
	}

	ctx := context.Background()

	// Load SIFT10K dataset
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT10K: %v", err)
	}

	stats := dataset.Stats()
	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SIFT10K: TIER 2.5 vs BRUTE FORCE COMPARISON")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nDataset: %s (%d vectors, %d queries)\n", stats.Name, stats.NumVectors, stats.NumQueries)

	// Setup Tier 2.5
	numSuperBuckets := 64
	topSuperBuckets := 16
	topK := 10
	numQueries := min(50, len(dataset.Queries))

	enterpriseCfg, _ := enterprise.NewConfig("sift-bf-test", dataset.Dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "sift-bf-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	fmt.Printf("\nConfig: %d super-buckets, top %d selected, %d queries\n",
		numSuperBuckets, topSuperBuckets, numQueries)

	fmt.Println("\n┌─────────┬────────────────────────────┬────────────────────────────┬────────────┐")
	fmt.Println("│ Query   │       Brute Force          │         Tier 2.5           │   Match    │")
	fmt.Println("│         │   Top-1 ID    (Score)      │   Top-1 ID    (Score)      │            │")
	fmt.Println("├─────────┼────────────────────────────┼────────────────────────────┼────────────┤")

	var totalMatch float64
	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]

		// Brute force search
		bfResults := siftBruteForceSearch(query, dataset.IDs, dataset.Vectors, topK)

		// Tier 2.5 search
		result, _ := client.Search(ctx, query, topK)

		tier25Top := "N/A"
		tier25Score := 0.0
		if len(result.Results) > 0 {
			tier25Top = result.Results[0].ID
			tier25Score = result.Results[0].Score
		}

		bfTop := bfResults[0].id
		bfScore := bfResults[0].score

		match := "  ❌"
		if bfTop == tier25Top {
			match = "  ✓"
			totalMatch += 1.0
		}

		if q < 15 { // Show first 15 queries
			fmt.Printf("│   %2d    │ %12s (%8.5f)  │ %12s (%8.5f)  │    %s     │\n",
				q, bfTop, bfScore, tier25Top, tier25Score, match)
		}
	}

	fmt.Println("├─────────┼────────────────────────────┼────────────────────────────┼────────────┤")
	fmt.Printf("│  Total  │         Brute Force        │         Tier 2.5           │  %.1f%%    │\n",
		totalMatch/float64(numQueries)*100)
	fmt.Println("└─────────┴────────────────────────────┴────────────────────────────┴────────────┘")
}

func siftBruteForceSearch(query []float64, ids []string, vectors [][]float64, topK int) []searchResult {
	results := make([]searchResult, len(vectors))
	for i := range vectors {
		results[i] = searchResult{
			id:    ids[i],
			score: siftCosineSim(query, vectors[i]),
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

func siftCosineSim(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// TestSIFTRandomVsReal compares accuracy between random vectors and SIFT embeddings
func TestSIFTRandomVsReal(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found. Download from http://corpus-texmex.irisa.fr/")
	}

	ctx := context.Background()

	// Load SIFT10K dataset
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT10K: %v", err)
	}

	stats := dataset.Stats()
	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("RANDOM vs REAL EMBEDDINGS ACCURACY COMPARISON")
	fmt.Println("=" + "=================================================================")

	numVectors := stats.NumVectors
	dimension := stats.Dimension
	numSuperBuckets := 64
	topSuperBuckets := 8
	topK := 10
	numQueries := 50

	// Generate random vectors for comparison
	randomDataset := embeddings.Generate(numVectors, dimension, 42)

	fmt.Printf("\nBoth datasets: %d vectors, %d dimensions\n", numVectors, dimension)
	fmt.Printf("Config: %d super-buckets, top %d selected\n", numSuperBuckets, topSuperBuckets)

	fmt.Println("\n┌───────────────┬──────────┬────────────┬────────────┐")
	fmt.Println("│    Dataset    │ Recall@1 │  Recall@10 │   NDCG@10  │")
	fmt.Println("│               │ (vs BF)  │  (vs BF)   │            │")
	fmt.Println("├───────────────┼──────────┼────────────┼────────────┤")

	// Test with random vectors
	randomRecall1, randomRecall10, randomNDCG := runComparisonTest(
		ctx, t, randomDataset, numSuperBuckets, topSuperBuckets, numQueries, topK)

	fmt.Printf("│ Random        │  %5.1f%%  │   %5.1f%%   │   %6.4f   │\n",
		randomRecall1*100, randomRecall10*100, randomNDCG)

	// Test with SIFT vectors
	siftRecall1, siftRecall10, siftNDCG := runComparisonTest(
		ctx, t, dataset, numSuperBuckets, topSuperBuckets, numQueries, topK)

	fmt.Printf("│ SIFT (real)   │  %5.1f%%  │   %5.1f%%   │   %6.4f   │\n",
		siftRecall1*100, siftRecall10*100, siftNDCG)

	fmt.Println("└───────────────┴──────────┴────────────┴────────────┘")

	improvement := (siftRecall10 - randomRecall10) / randomRecall10 * 100
	fmt.Printf("\nReal embeddings improvement: %+.1f%% in Recall@10\n", improvement)

	if siftRecall10 > randomRecall10 {
		fmt.Println("✓ SIFT embeddings have better clustering structure than random vectors")
	} else {
		fmt.Println("! Unexpectedly, random vectors performed similar or better")
	}
}

func runComparisonTest(
	ctx context.Context,
	t *testing.T,
	dataset *embeddings.Dataset,
	numSuperBuckets, topSuperBuckets, numQueries, topK int,
) (recall1, recall10, ndcg float64) {

	// Setup Tier 2.5
	enterpriseCfg, _ := enterprise.NewConfig("compare-test", dataset.Dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "compare-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	var totalRecall1, totalRecall10, totalNDCG float64

	for q := 0; q < numQueries; q++ {
		// Use random query vector with noise from actual vectors
		baseIdx := q % len(dataset.Vectors)
		query := make([]float64, dataset.Dimension)
		for j := 0; j < dataset.Dimension; j++ {
			query[j] = dataset.Vectors[baseIdx][j] + 0.1*(float64(q)/float64(numQueries)-0.5)
		}

		// Brute force ground truth
		bfResults := siftBruteForceSearch(query, dataset.IDs, dataset.Vectors, topK)

		// Tier 2.5 search
		result, _ := client.Search(ctx, query, topK)
		tier25Results := make([]searchResult, len(result.Results))
		for i, r := range result.Results {
			tier25Results[i] = searchResult{id: r.ID, score: r.Score}
		}

		r1, r10, n, _, _ := calculateMetrics(bfResults, tier25Results, topK)
		totalRecall1 += r1
		totalRecall10 += r10
		totalNDCG += n
	}

	recall1 = totalRecall1 / float64(numQueries)
	recall10 = totalRecall10 / float64(numQueries)
	ndcg = totalNDCG / float64(numQueries)

	return
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestSIFTCentroidQualityWithLSH tests centroid quality using actual LSH (fast, no HE)
// This isolates whether poor accuracy is due to centroid quality vs HE operations
func TestSIFTCentroidQualityWithLSH(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()

	// Load SIFT10K dataset
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT10K: %v", err)
	}

	stats := dataset.Stats()
	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SIFT10K CENTROID QUALITY TEST WITH LSH (NO HE - FAST)")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nDataset: %s (%d vectors, %d queries)\n", stats.Name, stats.NumVectors, stats.NumQueries)

	// Test different super-bucket configurations
	superBucketOptions := []int{16, 32, 64}
	topKOptions := []int{2, 4, 8, 16, 32}

	fmt.Println("\nThis test uses real LSH clustering and centroid scoring WITHOUT HE.")
	fmt.Println("Measures pure centroid quality for bucket selection.")

	for _, numBuckets := range superBucketOptions {
		fmt.Printf("\n=== %d Super-Buckets (via LSH) ===\n", numBuckets)

		// Build actual hierarchical index to get real LSH bucket assignments
		enterpriseCfg, _ := enterprise.NewConfig("centroid-test", stats.Dimension, numBuckets)
		store := blob.NewMemoryStore()
		cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
		cfg.SubBucketsPerSuper = 4
		cfg.NumDecoys = 8

		builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
		// Build index returns the Index which has VectorLocations with actual LSH assignments
		idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
		enterpriseCfg = builder.GetEnterpriseConfig()

		// Get centroids from the built index
		centroids := enterpriseCfg.Centroids

		// Count vectors per bucket using ACTUAL LSH assignments (not nearest centroid)
		bucketCounts := make([]int, numBuckets)
		vectorToBucket := make(map[int]int)

		// Use the actual LSH bucket assignments from the built index
		for i, id := range dataset.IDs {
			loc := idx.VectorLocations[id]
			vectorToBucket[i] = loc.SuperID
			bucketCounts[loc.SuperID]++
		}

		// Show bucket distribution
		minCount, maxCount := 10000, 0
		for _, c := range bucketCounts {
			if c < minCount {
				minCount = c
			}
			if c > maxCount {
				maxCount = c
			}
		}
		avgCount := stats.NumVectors / numBuckets
		fmt.Printf("Bucket sizes: min=%d, max=%d, avg=%d\n", minCount, maxCount, avgCount)

		fmt.Println("┌───────────┬──────────┬────────────┬────────────┬───────────┐")
		fmt.Println("│ TopSelect │ Recall@1 │  Recall@10 │  Avg Rank  │ % Dataset │")
		fmt.Println("├───────────┼──────────┼────────────┼────────────┼───────────┤")

		numQueries := min(100, len(dataset.Queries))

		for _, topK := range topKOptions {
			if topK > numBuckets {
				continue
			}

			var totalRecall1, totalRecall10, totalRank float64
			var totalVectorsScanned int

			for q := 0; q < numQueries; q++ {
				query := dataset.Queries[q]
				groundTruth := dataset.GroundTruth[q]

				// Score all centroids (NO HE - just plaintext dot product)
				scores := make([]struct {
					bucketID int
					score    float64
				}, numBuckets)

				for i, centroid := range centroids {
					scores[i].bucketID = i
					scores[i].score = siftCosineSim(query, centroid)
				}

				// Sort by score
				sort.Slice(scores, func(i, j int) bool {
					return scores[i].score > scores[j].score
				})

				// Get vectors from top-K buckets
				selectedBuckets := make(map[int]bool)
				vectorsInSelected := 0
				for i := 0; i < topK; i++ {
					selectedBuckets[scores[i].bucketID] = true
					vectorsInSelected += bucketCounts[scores[i].bucketID]
				}

				// Count how many ground truth vectors are in selected buckets
				recall1 := 0.0
				recall10 := 0.0

				// Check if top-1 ground truth is in selected buckets
				if selectedBuckets[vectorToBucket[groundTruth[0]]] {
					recall1 = 1.0
				}

				// Check top-10 ground truth
				gtTop10 := min(10, len(groundTruth))
				foundInTop10 := 0
				for i := 0; i < gtTop10; i++ {
					if selectedBuckets[vectorToBucket[groundTruth[i]]] {
						foundInTop10++
					}
				}
				recall10 = float64(foundInTop10) / float64(gtTop10)

				// Calculate rank of bucket containing top-1 ground truth
				targetBucket := vectorToBucket[groundTruth[0]]
				for rank, s := range scores {
					if s.bucketID == targetBucket {
						totalRank += float64(rank + 1)
						break
					}
				}

				totalVectorsScanned += vectorsInSelected
				totalRecall1 += recall1
				totalRecall10 += recall10
			}

			avgScanned := totalVectorsScanned / numQueries
			pctScanned := float64(avgScanned) / float64(stats.NumVectors) * 100

			fmt.Printf("│    %2d     │  %5.1f%%  │   %5.1f%%   │   %5.1f    │   %5.2f%%  │\n",
				topK,
				totalRecall1/float64(numQueries)*100,
				totalRecall10/float64(numQueries)*100,
				totalRank/float64(numQueries),
				pctScanned)
		}

		fmt.Println("└───────────┴──────────┴────────────┴────────────┴───────────┘")
	}

	fmt.Println("\nConclusion: If recall is low here, the problem is centroid quality (not HE).")
	fmt.Println("Consider: K-means clustering, more buckets, or different centroid computation.")
}
