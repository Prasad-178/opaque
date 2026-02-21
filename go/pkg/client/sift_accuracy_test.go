//go:build integration

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
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/embeddings"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
	"github.com/opaque/opaque/go/pkg/lsh"
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
	if testing.Short() {
		t.Skip("skipping SIFT accuracy with ground truth test in short mode")
	}
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
	if testing.Short() {
		t.Skip("skipping SIFT vs brute force test in short mode")
	}
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
	if testing.Short() {
		t.Skip("skipping SIFT random vs real embeddings test in short mode")
	}
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

// TestSIFTHEvPlaintext compares HE centroid ranking vs plaintext centroid ranking
func TestSIFTHEvPlaintext(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT HE vs plaintext centroid ranking test in short mode")
	}
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
	fmt.Println("SIFT10K: HE vs PLAINTEXT CENTROID RANKING COMPARISON")
	fmt.Println("=" + "=================================================================")

	numSuperBuckets := 32
	numQueries := 10

	// Build index to get centroids
	enterpriseCfg, _ := enterprise.NewConfig("he-test", stats.Dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()
	centroids := enterpriseCfg.Centroids

	fmt.Printf("\nConfig: %d super-buckets, %d centroids, %d queries\n",
		numSuperBuckets, len(centroids), numQueries)

	// Create HE engine
	heEngine, _ := crypto.NewClientEngine()

	fmt.Println("\n┌─────────┬──────────────────────────┬──────────────────────────┬─────────┐")
	fmt.Println("│ Query   │    Plaintext Top-4       │      HE Top-4            │ Match   │")
	fmt.Println("├─────────┼──────────────────────────┼──────────────────────────┼─────────┤")

	totalMatch := 0
	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]
		normalizedQuery := crypto.NormalizeVector(query)

		// Plaintext centroid scoring
		plainScores := make([]float64, len(centroids))
		for i, c := range centroids {
			plainScores[i] = siftCosineSim(normalizedQuery, c)
		}

		// HE centroid scoring
		encQuery, _ := heEngine.EncryptVector(normalizedQuery)
		heScores := make([]float64, len(centroids))
		for i, c := range centroids {
			encResult, _ := heEngine.HomomorphicDotProduct(encQuery, c)
			heScores[i], _ = heEngine.DecryptScalar(encResult)
		}

		// Get top-4 for each
		plainTop := getTopKIndices(plainScores, 4)
		heTop := getTopKIndices(heScores, 4)

		// Check match
		matchCount := 0
		for _, pt := range plainTop {
			for _, ht := range heTop {
				if pt == ht {
					matchCount++
				}
			}
		}

		match := "  ✓"
		if matchCount < 4 {
			match = fmt.Sprintf("%d/4", matchCount)
		} else {
			totalMatch++
		}

		fmt.Printf("│   %2d    │ %v │ %v │   %s   │\n",
			q, plainTop, heTop, match)
	}

	fmt.Println("└─────────┴──────────────────────────┴──────────────────────────┴─────────┘")
	fmt.Printf("\nPerfect top-4 match: %d/%d (%.1f%%)\n",
		totalMatch, numQueries, float64(totalMatch)/float64(numQueries)*100)
}

func getTopKIndices(scores []float64, k int) []int {
	type idxScore struct {
		idx   int
		score float64
	}
	items := make([]idxScore, len(scores))
	for i, s := range scores {
		items[i] = idxScore{i, s}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	result := make([]int, k)
	for i := 0; i < k && i < len(items); i++ {
		result[i] = items[i].idx
	}
	return result
}

// TestSIFTSubBucketConsistency verifies builder and client use same LSH
func TestSIFTSubBucketConsistency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT sub-bucket consistency test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()

	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT10K: %v", err)
	}

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SUB-BUCKET ASSIGNMENT CONSISTENCY CHECK")
	fmt.Println("=" + "=================================================================")

	numSuperBuckets := 32
	numSubBuckets := 64

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("sub-test", dataset.Dimension, numSuperBuckets)
	enterpriseCfg.NumSubBuckets = numSubBuckets
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Get credentials (same as client would)
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "sub-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	fmt.Printf("\nConfig: %d super-buckets, %d sub-buckets\n", numSuperBuckets, numSubBuckets)
	fmt.Printf("LSH hyperplanes from auth: %d planes, %d dimensions\n",
		len(creds.LSHHyperplanes), len(creds.LSHHyperplanes[0]))
	fmt.Printf("Enterprise NumSubBuckets: %d\n", enterpriseCfg.NumSubBuckets)
	fmt.Printf("Enterprise SubLSHBits: %d\n", enterpriseCfg.GetSubLSHBits())

	// Create client LSH hasher
	clientHasher := lsh.NewEnterpriseHasher(creds.LSHHyperplanes)

	// Check a sample of vectors - does client's sub-bucket match builder's?
	numCheck := 100
	matches := 0

	fmt.Println("\n┌────────┬────────────────┬────────────────┬─────────┐")
	fmt.Println("│ Vector │ Builder SubID  │ Client SubID   │  Match  │")
	fmt.Println("├────────┼────────────────┼────────────────┼─────────┤")

	for i := 0; i < numCheck; i++ {
		id := dataset.IDs[i]
		vec := dataset.Vectors[i]
		normalizedVec := crypto.NormalizeVector(vec)

		// Builder's sub-bucket (from the built index)
		loc := idx.VectorLocations[id]
		builderSubID := loc.SubID

		// Client's sub-bucket (computed with LSH hasher from credentials)
		clientSubID := clientHasher.HashToIndex(normalizedVec, numSubBuckets)

		match := "  ✓"
		if builderSubID != clientSubID {
			match = "  ❌"
		} else {
			matches++
		}

		if i < 15 { // Show first 15
			fmt.Printf("│  %4d  │      %3d       │      %3d       │   %s   │\n",
				i, builderSubID, clientSubID, match)
		}
	}

	fmt.Println("└────────┴────────────────┴────────────────┴─────────┘")
	fmt.Printf("\nConsistency: %d/%d (%.1f%%)\n", matches, numCheck, float64(matches)/float64(numCheck)*100)

	if matches < numCheck*9/10 { // Less than 90% match
		t.Errorf("Sub-bucket assignment inconsistent: only %d/%d match", matches, numCheck)
	}
}

// TestSIFTSubBucketCoverage tests if sub-bucket fetching covers the ground truth
func TestSIFTSubBucketCoverage(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT sub-bucket coverage test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SUB-BUCKET COVERAGE ANALYSIS")
	fmt.Println("=" + "=================================================================")

	numSuperBuckets := 32
	numSubBuckets := 64
	numQueries := 50

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("coverage-test", dataset.Dimension, numSuperBuckets)
	enterpriseCfg.NumSubBuckets = numSubBuckets
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "coverage-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	clientHasher := lsh.NewEnterpriseHasher(creds.LSHHyperplanes)

	fmt.Printf("\nConfig: %d super-buckets, %d sub-buckets\n", numSuperBuckets, numSubBuckets)

	// For each SubBucketsPerSuper setting, check coverage
	subBucketOptions := []int{1, 2, 4, 8, 16, 32, 64}

	fmt.Println("\n┌──────────────────┬───────────────┬───────────────┬───────────────┐")
	fmt.Println("│ SubBktsPerSuper  │ Same SubBkt   │ Query Finds   │ % Coverage    │")
	fmt.Println("│                  │ as GT Top-1   │ GT Sub-Bkt    │               │")
	fmt.Println("├──────────────────┼───────────────┼───────────────┼───────────────┤")

	for _, subBktsPerSuper := range subBucketOptions {
		sameSubBucket := 0
		queryCovered := 0

		for q := 0; q < numQueries; q++ {
			query := dataset.Queries[q]
			normalizedQuery := crypto.NormalizeVector(query)
			groundTruthIdx := dataset.GroundTruth[q][0]
			gtID := fmt.Sprintf("sift_%d", groundTruthIdx)

			// Get ground truth vector's sub-bucket
			gtLoc := idx.VectorLocations[gtID]
			gtSubID := gtLoc.SubID

			// Get query's computed sub-bucket
			querySubID := clientHasher.HashToIndex(normalizedQuery, numSubBuckets)

			// Check if same sub-bucket
			if gtSubID == querySubID {
				sameSubBucket++
			}

			// Check if query's neighbors include GT sub-bucket
			// This simulates what the client does
			covered := false
			for i := 0; i < subBktsPerSuper; i++ {
				neighborID := (querySubID + i) % numSubBuckets
				if neighborID == gtSubID {
					covered = true
					break
				}
			}
			if covered {
				queryCovered++
			}
		}

		pctSame := float64(sameSubBucket) / float64(numQueries) * 100
		pctCovered := float64(queryCovered) / float64(numQueries) * 100

		fmt.Printf("│       %2d         │    %5.1f%%     │    %5.1f%%     │    %5.1f%%     │\n",
			subBktsPerSuper, pctSame, pctCovered, pctCovered)
	}

	fmt.Println("└──────────────────┴───────────────┴───────────────┴───────────────┘")
	fmt.Println("\nNote: 'Query Finds GT Sub-Bkt' shows % of queries where the sub-bucket")
	fmt.Println("range fetched by the client would include the ground truth vector's sub-bucket.")
}

// TestSIFTSubBucketDistance shows the distribution of sub-bucket ID differences
func TestSIFTSubBucketDistance(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT sub-bucket distance distribution test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SUB-BUCKET ID DISTANCE DISTRIBUTION")
	fmt.Println("=" + "=================================================================")

	numSubBuckets := 64
	enterpriseCfg, _ := enterprise.NewConfig("dist-test", dataset.Dimension, 32)
	enterpriseCfg.NumSubBuckets = numSubBuckets
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "dist-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	clientHasher := lsh.NewEnterpriseHasher(creds.LSHHyperplanes)

	// For each query, compute distance to GT sub-bucket
	distances := make(map[int]int)
	numQueries := 100

	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]
		normalizedQuery := crypto.NormalizeVector(query)
		gtIdx := dataset.GroundTruth[q][0]
		gtID := fmt.Sprintf("sift_%d", gtIdx)

		gtLoc := idx.VectorLocations[gtID]
		gtSubID := gtLoc.SubID
		querySubID := clientHasher.HashToIndex(normalizedQuery, numSubBuckets)

		// Circular distance
		dist := querySubID - gtSubID
		if dist < 0 {
			dist = -dist
		}
		if dist > numSubBuckets/2 {
			dist = numSubBuckets - dist
		}
		distances[dist]++
	}

	fmt.Println("\nDistance from query sub-bucket to GT sub-bucket:")
	fmt.Println("(If neighbors were useful, we'd see high counts at small distances)")
	fmt.Println("\n┌──────────────┬──────────────┬──────────────┐")
	fmt.Println("│ Distance     │   Count      │  Cumulative  │")
	fmt.Println("├──────────────┼──────────────┼──────────────┤")

	cumulative := 0
	for d := 0; d <= 32; d++ {
		cumulative += distances[d]
		if distances[d] > 0 || d < 10 {
			fmt.Printf("│     %2d       │     %3d      │    %5.1f%%    │\n",
				d, distances[d], float64(cumulative)/float64(numQueries)*100)
		}
	}
	fmt.Println("└──────────────┴──────────────┴──────────────┘")

	fmt.Println("\nConclusion: If counts are spread evenly across distances,")
	fmt.Println("then fetching 'neighbor' sub-buckets by ID increment is useless.")
	fmt.Println("Sub-bucket ID has no correlation with vector similarity.")
}

// TestSIFTSubBucketsPerSuperImpact tests how SubBucketsPerSuper affects accuracy
func TestSIFTSubBucketsPerSuperImpact(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT sub-buckets per super impact test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("IMPACT OF SubBucketsPerSuper ON ACCURACY")
	fmt.Println("=" + "=================================================================")

	numSuperBuckets := 32
	numSubBuckets := 64
	topSuperBuckets := 16
	topK := 10
	numQueries := 30

	fmt.Printf("\nFixed: %d super-buckets, %d sub-buckets, top %d selected\n",
		numSuperBuckets, numSubBuckets, topSuperBuckets)

	subBktOptions := []int{1, 4, 8, 16, 32, 64} // Number of sub-buckets to fetch per super

	fmt.Println("\n┌────────────────────┬──────────┬────────────┬────────────┬───────────┐")
	fmt.Println("│ SubBktsPerSuper    │ Recall@1 │  Recall@10 │  Vectors   │ % Dataset │")
	fmt.Println("├────────────────────┼──────────┼────────────┼────────────┼───────────┤")

	for _, subBktsPerSuper := range subBktOptions {
		// Build index
		enterpriseCfg, _ := enterprise.NewConfig("impact-test", dataset.Dimension, numSuperBuckets)
		enterpriseCfg.NumSubBuckets = numSubBuckets
		store := blob.NewMemoryStore()
		cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
		cfg.TopSuperBuckets = topSuperBuckets
		cfg.SubBucketsPerSuper = subBktsPerSuper
		cfg.NumDecoys = 8

		builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
		builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
		enterpriseCfg = builder.GetEnterpriseConfig()

		enterpriseStore := enterprise.NewMemoryStore()
		enterpriseStore.Put(ctx, enterpriseCfg)
		authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
		authService.RegisterUser(ctx, "user", "impact-test", []byte("pass"), []string{auth.ScopeSearch})
		creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
		client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

		var totalRecall1, totalRecall10 float64
		var totalScanned int

		for q := 0; q < numQueries; q++ {
			query := dataset.Queries[q]
			result, _ := client.Search(ctx, query, topK)

			gtSet := make(map[int]bool)
			for i := 0; i < topK && i < len(dataset.GroundTruth[q]); i++ {
				gtSet[dataset.GroundTruth[q][i]] = true
			}

			if len(result.Results) > 0 {
				var idx int
				fmt.Sscanf(result.Results[0].ID, "sift_%d", &idx)
				if idx == dataset.GroundTruth[q][0] {
					totalRecall1++
				}
			}

			found := 0
			for _, r := range result.Results {
				var idx int
				fmt.Sscanf(r.ID, "sift_%d", &idx)
				if gtSet[idx] {
					found++
				}
			}
			totalRecall10 += float64(found) / float64(min(topK, len(gtSet)))
			totalScanned += result.Stats.VectorsScored
		}

		avgScanned := totalScanned / numQueries
		pctScanned := float64(avgScanned) / float64(len(dataset.Vectors)) * 100

		fmt.Printf("│        %2d          │  %5.1f%%  │   %5.1f%%   │   %6d   │   %5.2f%%  │\n",
			subBktsPerSuper,
			totalRecall1/float64(numQueries)*100,
			totalRecall10/float64(numQueries)*100,
			avgScanned,
			pctScanned)
	}

	fmt.Println("└────────────────────┴──────────┴────────────┴────────────┴───────────┘")
	fmt.Println("\nConclusion: Higher SubBucketsPerSuper = better recall but more vectors scanned.")
}

// TestSIFTCentroidQualityWithLSH tests centroid quality using actual LSH (fast, no HE)
// This isolates whether poor accuracy is due to centroid quality vs HE operations
func TestSIFTCentroidQualityWithLSH(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT centroid quality with LSH test in short mode")
	}
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
