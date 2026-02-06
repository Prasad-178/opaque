package client

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestAccuracyComparison compares accuracy between brute force and Tier 2.5
func TestAccuracyComparison(t *testing.T) {
	ctx := context.Background()

	// Test configurations
	configs := []struct {
		numVectors      int
		dimension       int
		numSuperBuckets int
		topSuperBuckets int
	}{
		{1000, 64, 16, 4},
		{5000, 64, 32, 8},
		{10000, 128, 32, 8},
		{50000, 128, 64, 16},
	}

	numQueries := 20
	topK := 10

	fmt.Println("=" + "=================================================================")
	fmt.Println("ACCURACY COMPARISON: BRUTE FORCE vs TIER 2.5")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nQueries per config: %d, Top-K: %d\n", numQueries, topK)

	fmt.Println("\n┌────────────┬─────┬────────┬────────────┬────────────┬────────────┬────────────┐")
	fmt.Println("│  Vectors   │ Dim │ Recall │  Recall@10 │    NDCG    │  Avg Score │ Score Diff │")
	fmt.Println("│            │     │  @1    │            │    @10     │   (Tier2.5)│            │")
	fmt.Println("├────────────┼─────┼────────┼────────────┼────────────┼────────────┼────────────┤")

	for _, cfg := range configs {
		recall1, recall10, ndcg, avgScore, scoreDiff := runAccuracyTest(
			ctx, t, cfg.numVectors, cfg.dimension, cfg.numSuperBuckets,
			cfg.topSuperBuckets, numQueries, topK,
		)

		fmt.Printf("│ %10d │ %3d │ %5.1f%% │    %5.1f%%  │   %6.4f   │   %6.4f   │   %6.4f   │\n",
			cfg.numVectors, cfg.dimension, recall1*100, recall10*100, ndcg, avgScore, scoreDiff)
	}

	fmt.Println("└────────────┴─────┴────────┴────────────┴────────────┴────────────┴────────────┘")

	fmt.Println("\nMetrics explained:")
	fmt.Println("  Recall@1:   % of queries where top-1 result matches brute force")
	fmt.Println("  Recall@10:  % of brute force top-10 found in Tier 2.5 top-10")
	fmt.Println("  NDCG@10:    Normalized Discounted Cumulative Gain (ranking quality)")
	fmt.Println("  Avg Score:  Average similarity score of Tier 2.5 results")
	fmt.Println("  Score Diff: Avg difference between brute force and Tier 2.5 top scores")
}

func runAccuracyTest(
	ctx context.Context,
	t *testing.T,
	numVectors, dimension, numSuperBuckets, topSuperBuckets, numQueries, topK int,
) (recall1, recall10, ndcg, avgScore, scoreDiff float64) {

	// Generate vectors
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}

	// Setup Tier 2.5
	enterpriseCfg, _ := enterprise.NewConfig("accuracy-test", dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "accuracy-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	// Run queries
	var totalRecall1, totalRecall10, totalNDCG, totalAvgScore, totalScoreDiff float64

	for q := 0; q < numQueries; q++ {
		// Generate random query (not from dataset to test generalization)
		query := make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			query[j] = rng.NormFloat64()
		}

		// Brute force ground truth
		bfResults := bruteForceSearch(query, ids, vectors, topK)

		// Tier 2.5 search
		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Convert to comparable format
		tier25Results := make([]searchResult, len(result.Results))
		for i, r := range result.Results {
			tier25Results[i] = searchResult{id: r.ID, score: r.Score}
		}

		// Calculate metrics
		r1, r10, n, avg, diff := calculateMetrics(bfResults, tier25Results, topK)
		totalRecall1 += r1
		totalRecall10 += r10
		totalNDCG += n
		totalAvgScore += avg
		totalScoreDiff += diff
	}

	recall1 = totalRecall1 / float64(numQueries)
	recall10 = totalRecall10 / float64(numQueries)
	ndcg = totalNDCG / float64(numQueries)
	avgScore = totalAvgScore / float64(numQueries)
	scoreDiff = totalScoreDiff / float64(numQueries)

	return
}

type searchResult struct {
	id    string
	score float64
}

func bruteForceSearch(query []float64, ids []string, vectors [][]float64, topK int) []searchResult {
	results := make([]searchResult, len(vectors))
	for i := range vectors {
		results[i] = searchResult{
			id:    ids[i],
			score: cosineSimBF(query, vectors[i]),
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

func cosineSimBF(a, b []float64) float64 {
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

func calculateMetrics(groundTruth, predicted []searchResult, topK int) (recall1, recall10, ndcg, avgScore, scoreDiff float64) {
	if len(predicted) == 0 {
		return 0, 0, 0, 0, 0
	}

	// Recall@1: Is the top result correct?
	if len(groundTruth) > 0 && len(predicted) > 0 && groundTruth[0].id == predicted[0].id {
		recall1 = 1.0
	}

	// Recall@K: What fraction of ground truth top-K is in predicted top-K?
	gtSet := make(map[string]bool)
	for i := 0; i < len(groundTruth) && i < topK; i++ {
		gtSet[groundTruth[i].id] = true
	}

	found := 0
	for i := 0; i < len(predicted) && i < topK; i++ {
		if gtSet[predicted[i].id] {
			found++
		}
	}
	if len(gtSet) > 0 {
		recall10 = float64(found) / float64(len(gtSet))
	}

	// NDCG@K: Normalized Discounted Cumulative Gain
	// Measures ranking quality - higher if relevant items appear earlier
	gtRanks := make(map[string]int)
	for i, r := range groundTruth {
		gtRanks[r.id] = i + 1
	}

	var dcg, idcg float64
	for i := 0; i < len(predicted) && i < topK; i++ {
		if rank, ok := gtRanks[predicted[i].id]; ok {
			// Relevance based on ground truth rank (higher rank = more relevant)
			rel := float64(topK - rank + 1)
			dcg += rel / math.Log2(float64(i+2))
		}
	}

	// Ideal DCG (if we had perfect ranking)
	for i := 0; i < topK && i < len(groundTruth); i++ {
		rel := float64(topK - i)
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg > 0 {
		ndcg = dcg / idcg
	}

	// Average score of predicted results
	for _, r := range predicted {
		avgScore += r.score
	}
	if len(predicted) > 0 {
		avgScore /= float64(len(predicted))
	}

	// Score difference between top results
	if len(groundTruth) > 0 && len(predicted) > 0 {
		scoreDiff = groundTruth[0].score - predicted[0].score
	}

	return
}

// TestAccuracyWithSimilarQueries tests accuracy when query is similar to indexed vectors
func TestAccuracyWithSimilarQueries(t *testing.T) {
	ctx := context.Background()

	numVectors := 10000
	dimension := 128
	numSuperBuckets := 32
	topSuperBuckets := 8
	numQueries := 50
	topK := 10

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("ACCURACY TEST: QUERIES SIMILAR TO INDEXED VECTORS")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nVectors: %d, Dim: %d, Queries: %d\n", numVectors, dimension, numQueries)

	// Generate vectors
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}

	// Setup Tier 2.5
	enterpriseCfg, _ := enterprise.NewConfig("similar-test", dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "similar-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	// Test with different noise levels
	noiseLevels := []float64{0.0, 0.05, 0.1, 0.2, 0.5}

	fmt.Println("\n┌───────────┬──────────┬────────────┬────────────┬────────────┐")
	fmt.Println("│   Noise   │ Recall@1 │  Recall@10 │   NDCG@10  │ Avg Top    │")
	fmt.Println("│   Level   │          │            │            │ Score      │")
	fmt.Println("├───────────┼──────────┼────────────┼────────────┼────────────┤")

	for _, noise := range noiseLevels {
		var totalRecall1, totalRecall10, totalNDCG, totalTopScore float64

		for q := 0; q < numQueries; q++ {
			// Pick a random vector and add noise
			baseIdx := rng.Intn(numVectors)
			query := make([]float64, dimension)
			for j := 0; j < dimension; j++ {
				query[j] = vectors[baseIdx][j] + noise*rng.NormFloat64()
			}

			// Brute force
			bfResults := bruteForceSearch(query, ids, vectors, topK)

			// Tier 2.5
			result, _ := client.Search(ctx, query, topK)
			tier25Results := make([]searchResult, len(result.Results))
			for i, r := range result.Results {
				tier25Results[i] = searchResult{id: r.ID, score: r.Score}
			}

			r1, r10, n, _, _ := calculateMetrics(bfResults, tier25Results, topK)
			totalRecall1 += r1
			totalRecall10 += r10
			totalNDCG += n

			if len(tier25Results) > 0 {
				totalTopScore += tier25Results[0].score
			}
		}

		fmt.Printf("│   %.2f    │  %5.1f%%  │   %5.1f%%   │   %6.4f   │   %6.4f   │\n",
			noise,
			totalRecall1/float64(numQueries)*100,
			totalRecall10/float64(numQueries)*100,
			totalNDCG/float64(numQueries),
			totalTopScore/float64(numQueries))
	}

	fmt.Println("└───────────┴──────────┴────────────┴────────────┴────────────┘")

	fmt.Println("\nNoise level 0.0 = exact vector from dataset")
	fmt.Println("Noise level 0.5 = significant perturbation")
}

// TestAccuracyVsTopSuperBuckets shows accuracy/performance tradeoff
func TestAccuracyVsTopSuperBuckets(t *testing.T) {
	ctx := context.Background()

	numVectors := 10000
	dimension := 128
	numSuperBuckets := 64
	numQueries := 30
	topK := 10

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("ACCURACY vs TOP-SUPER-BUCKETS TRADEOFF")
	fmt.Println("=" + "=================================================================")
	fmt.Printf("\nVectors: %d, Total Super-Buckets: %d\n", numVectors, numSuperBuckets)

	// Generate vectors
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}

	// Build index once
	enterpriseCfg, _ := enterprise.NewConfig("tradeoff-test", dimension, numSuperBuckets)
	store := blob.NewMemoryStore()
	baseCfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	baseCfg.SubBucketsPerSuper = 4
	baseCfg.NumDecoys = 8

	builder, _ := hierarchical.NewEnterpriseBuilder(baseCfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "tradeoff-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	topSuperBucketOptions := []int{2, 4, 8, 16, 32, 64}

	fmt.Println("\n┌─────────────┬──────────┬────────────┬────────────┬────────────┬───────────┐")
	fmt.Println("│ Top Buckets │ Recall@1 │  Recall@10 │   NDCG@10  │ Vectors    │ % Dataset │")
	fmt.Println("│  Selected   │          │            │            │ Scanned    │  Scanned  │")
	fmt.Println("├─────────────┼──────────┼────────────┼────────────┼────────────┼───────────┤")

	for _, topSuper := range topSuperBucketOptions {
		cfg := baseCfg
		cfg.TopSuperBuckets = topSuper

		client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

		var totalRecall1, totalRecall10, totalNDCG float64
		var totalScanned int

		for q := 0; q < numQueries; q++ {
			// Use vectors from dataset with small noise
			baseIdx := rng.Intn(numVectors)
			query := make([]float64, dimension)
			for j := 0; j < dimension; j++ {
				query[j] = vectors[baseIdx][j] + 0.1*rng.NormFloat64()
			}

			bfResults := bruteForceSearch(query, ids, vectors, topK)

			result, _ := client.Search(ctx, query, topK)
			tier25Results := make([]searchResult, len(result.Results))
			for i, r := range result.Results {
				tier25Results[i] = searchResult{id: r.ID, score: r.Score}
			}

			r1, r10, n, _, _ := calculateMetrics(bfResults, tier25Results, topK)
			totalRecall1 += r1
			totalRecall10 += r10
			totalNDCG += n
			totalScanned += result.Stats.VectorsScored
		}

		avgScanned := totalScanned / numQueries
		pctScanned := float64(avgScanned) / float64(numVectors) * 100

		fmt.Printf("│     %2d      │  %5.1f%%  │   %5.1f%%   │   %6.4f   │   %6d   │   %5.2f%%  │\n",
			topSuper,
			totalRecall1/float64(numQueries)*100,
			totalRecall10/float64(numQueries)*100,
			totalNDCG/float64(numQueries),
			avgScanned,
			pctScanned)
	}

	fmt.Println("└─────────────┴──────────┴────────────┴────────────┴────────────┴───────────┘")

	fmt.Println("\nObservation: More super-buckets = better accuracy but more vectors to decrypt")
}
