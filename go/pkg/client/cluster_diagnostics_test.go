package client

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestClusterSelectionDiagnostics performs comprehensive analysis to identify
// WHERE recall is being lost: cluster selection vs post-selection scoring.
//
// Key insight: If cluster coverage >> actual recall, the problem is local scoring.
// If cluster coverage ≈ actual recall, the problem is cluster selection.
func TestClusterSelectionDiagnostics(t *testing.T) {
	// Configuration
	// Note: HE operations are slow (~3s per query due to serialized ops)
	// Using smaller numbers for diagnostic purposes
	numVectors := 10000  // Use 10K for faster diagnostics
	dimension := 128
	numClusters := 64
	topSelect := 16
	numQueries := 10     // Reduced to 10 for speed (HE is ~3s per query)
	topK := 10

	t.Logf("Configuration: %d vectors, %d clusters, top-%d selection, %d queries",
		numVectors, numClusters, topSelect, numQueries)

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate synthetic data
	t.Log("Generating synthetic data...")
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
		// Normalize
		var norm float64
		for _, v := range vectors[i] {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range vectors[i] {
			vectors[i][j] /= norm
		}
	}

	// Generate queries
	queries := make([][]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			queries[i][j] = rng.NormFloat64()
		}
		var norm float64
		for _, v := range queries[i] {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range queries[i] {
			queries[i][j] /= norm
		}
	}

	// Build index
	t.Log("Building index with K-means clustering...")
	enterpriseCfg, _ := enterprise.NewConfig("diag-test", dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	index, _ := builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Create vector-to-cluster mapping from index
	vectorToCluster := make(map[string]int)
	for vecID, loc := range index.VectorLocations {
		vectorToCluster[vecID] = loc.SuperID
	}

	// Setup client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "diag-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Compute ground truth for all queries (brute force)
	t.Log("Computing ground truth (brute force)...")
	groundTruths := make([][]string, numQueries)
	for q := 0; q < numQueries; q++ {
		type scored struct {
			id    string
			score float64
		}
		allScores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			score := dotProductDiag(queries[q], vectors[i])
			allScores[i] = scored{id: ids[i], score: score}
		}
		sort.Slice(allScores, func(i, j int) bool {
			return allScores[i].score > allScores[j].score
		})
		groundTruths[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruths[q][i] = allScores[i].id
		}
	}

	// Compute plaintext cluster scores for comparison with HE
	t.Log("Computing plaintext cluster scores...")
	centroids := enterpriseCfg.Centroids
	plaintextClusterScores := make([][]float64, numQueries)
	plaintextTopK := make([][]int, numQueries)
	for q := 0; q < numQueries; q++ {
		scores := make([]float64, numClusters)
		for c := 0; c < numClusters; c++ {
			scores[c] = dotProductDiag(queries[q], centroids[c])
		}
		plaintextClusterScores[q] = scores
		plaintextTopK[q] = selectTopKIndicesLocal(scores, topSelect)
	}

	// Run searches and collect diagnostics
	t.Log("Running searches with HE and collecting diagnostics...")

	var (
		totalRecall              float64
		totalClusterCoverage     float64
		totalHEPlaintextMatch    int
		totalHEPlaintextDiverge1 int
		totalHEPlaintextDiverge2 int
		totalHEPlaintextDiverge3 int
		gtClusterInTopK          int
	)

	for q := 0; q < numQueries; q++ {
		result, err := client.Search(ctx, queries[q], topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Get selected clusters from HE search
		heSelected := result.Stats.SelectedClusters
		heSelectedSet := make(map[int]bool)
		for _, c := range heSelected {
			heSelectedSet[c] = true
		}

		// 1. Compute actual recall
		resultSet := make(map[string]bool)
		for _, r := range result.Results {
			resultSet[r.ID] = true
		}
		hits := 0
		for _, gt := range groundTruths[q] {
			if resultSet[gt] {
				hits++
			}
		}
		recall := float64(hits) / float64(topK)
		totalRecall += recall

		// 2. Compute cluster coverage (theoretical max recall)
		// How many of the GT top-10 vectors are in selected clusters?
		gtInSelected := 0
		for _, gtID := range groundTruths[q] {
			if cluster, ok := vectorToCluster[gtID]; ok {
				if heSelectedSet[cluster] {
					gtInSelected++
				}
			}
		}
		clusterCoverage := float64(gtInSelected) / float64(topK)
		totalClusterCoverage += clusterCoverage

		// 3. Check if #1 GT vector's cluster is selected
		gt1Cluster := vectorToCluster[groundTruths[q][0]]
		if heSelectedSet[gt1Cluster] {
			gtClusterInTopK++
		}

		// 4. Compare HE vs plaintext cluster selection
		plaintextSet := make(map[int]bool)
		for _, c := range plaintextTopK[q] {
			plaintextSet[c] = true
		}

		matchCount := 0
		for _, c := range heSelected {
			if plaintextSet[c] {
				matchCount++
			}
		}
		divergence := topSelect - matchCount

		if divergence == 0 {
			totalHEPlaintextMatch++
		} else if divergence <= 2 {
			totalHEPlaintextDiverge1++
		} else if divergence <= 4 {
			totalHEPlaintextDiverge2++
		} else {
			totalHEPlaintextDiverge3++
		}
	}

	// Print results
	avgRecall := totalRecall / float64(numQueries)
	avgCoverage := totalClusterCoverage / float64(numQueries)

	t.Log("")
	t.Log("==================================================================")
	t.Log("            CLUSTER SELECTION DIAGNOSTICS REPORT")
	t.Log("==================================================================")
	t.Log("")
	t.Logf("Dataset:        %d vectors, %d dimensions", numVectors, dimension)
	t.Logf("Clusters:       %d total, selecting top %d (%.1f%% coverage)",
		numClusters, topSelect, float64(topSelect)/float64(numClusters)*100)
	t.Logf("Queries:        %d", numQueries)
	t.Log("")
	t.Log("------------------------------------------------------------------")
	t.Log("                     ACCURACY ANALYSIS")
	t.Log("------------------------------------------------------------------")
	t.Logf("  Actual Recall@%d:           %.1f%%", topK, avgRecall*100)
	t.Logf("  Cluster Coverage:           %.1f%% (theoretical max recall)", avgCoverage*100)
	t.Logf("  GT#1 cluster in top-K:      %.1f%% of queries", float64(gtClusterInTopK)/float64(numQueries)*100)
	t.Log("")
	t.Log("------------------------------------------------------------------")
	t.Log("                     DIAGNOSIS")
	t.Log("------------------------------------------------------------------")

	gap := avgCoverage - avgRecall
	if gap > 0.15 {
		t.Log("  PROBLEM: LOCAL SCORING")
		t.Logf("  Coverage %.1f%% >> Recall %.1f%% (gap: %.1f%%)", avgCoverage*100, avgRecall*100, gap*100)
		t.Log("  The correct clusters ARE being selected, but vectors aren't")
		t.Log("  being found. Check:")
		t.Log("    - AES decryption correctness")
		t.Log("    - Vector normalization")
		t.Log("    - Score computation")
	} else if avgCoverage < 0.80 {
		t.Log("  PROBLEM: CLUSTER SELECTION")
		t.Logf("  Coverage %.1f%% is too low (need >80%% for good recall)", avgCoverage*100)
		t.Log("  The correct clusters are NOT being selected. Consider:")
		t.Log("    - Increasing TopSuperBuckets (selecting more clusters)")
		t.Log("    - Improving K-means quality (more iterations)")
		t.Log("    - Multi-probe search (nearby clusters)")
	} else {
		t.Log("  STATUS: OK")
		t.Logf("  Coverage %.1f%% and Recall %.1f%% are aligned", avgCoverage*100, avgRecall*100)
		t.Log("  The search is working as expected for this cluster count.")
	}

	t.Log("")
	t.Log("------------------------------------------------------------------")
	t.Log("                 HE vs PLAINTEXT COMPARISON")
	t.Log("------------------------------------------------------------------")
	t.Logf("  Perfect match (0 divergence):    %d/%d (%.1f%%)",
		totalHEPlaintextMatch, numQueries, float64(totalHEPlaintextMatch)/float64(numQueries)*100)
	t.Logf("  Minor divergence (1-2 clusters): %d/%d (%.1f%%)",
		totalHEPlaintextDiverge1, numQueries, float64(totalHEPlaintextDiverge1)/float64(numQueries)*100)
	t.Logf("  Medium divergence (3-4 clusters): %d/%d (%.1f%%)",
		totalHEPlaintextDiverge2, numQueries, float64(totalHEPlaintextDiverge2)/float64(numQueries)*100)
	t.Logf("  Major divergence (5+ clusters):  %d/%d (%.1f%%)",
		totalHEPlaintextDiverge3, numQueries, float64(totalHEPlaintextDiverge3)/float64(numQueries)*100)

	if float64(totalHEPlaintextMatch)/float64(numQueries) < 0.80 {
		t.Log("")
		t.Log("  WARNING: HE and plaintext select different clusters!")
		t.Log("  This suggests HE precision issues. Consider:")
		t.Log("    - Increasing CKKS scale")
		t.Log("    - Better vector normalization")
	}

	t.Log("")
	t.Log("==================================================================")
}

// TestClusterDistribution analyzes how vectors are distributed across clusters.
func TestClusterDistribution(t *testing.T) {
	numVectors := 10000
	dimension := 128
	numClusters := 64

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate synthetic data
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
		var norm float64
		for _, v := range vectors[i] {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range vectors[i] {
			vectors[i][j] /= norm
		}
	}

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("dist-test", dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	index, _ := builder.Build(ctx, ids, vectors, store)

	// Count vectors per cluster
	clusterCounts := make([]int, numClusters)
	for _, loc := range index.VectorLocations {
		clusterCounts[loc.SuperID]++
	}

	// Statistics
	var min, max, sum int
	min = clusterCounts[0]
	for _, count := range clusterCounts {
		if count < min {
			min = count
		}
		if count > max {
			max = count
		}
		sum += count
	}
	avg := float64(sum) / float64(numClusters)

	// Variance
	var variance float64
	for _, count := range clusterCounts {
		diff := float64(count) - avg
		variance += diff * diff
	}
	variance /= float64(numClusters)
	stddev := math.Sqrt(variance)

	t.Log("")
	t.Log("==================================================================")
	t.Log("               CLUSTER DISTRIBUTION ANALYSIS")
	t.Log("==================================================================")
	t.Logf("Total vectors:    %d", numVectors)
	t.Logf("Total clusters:   %d", numClusters)
	t.Log("")
	t.Log("Vectors per cluster:")
	t.Logf("  Min:     %d", min)
	t.Logf("  Max:     %d", max)
	t.Logf("  Average: %.1f", avg)
	t.Logf("  StdDev:  %.1f", stddev)
	t.Logf("  CV:      %.2f (coefficient of variation)", stddev/avg)
	t.Log("")

	// Sort for histogram
	sorted := make([]int, numClusters)
	copy(sorted, clusterCounts)
	sort.Ints(sorted)

	t.Log("Distribution (sorted):")
	for i := 0; i < numClusters; i += 8 {
		end := i + 8
		if end > numClusters {
			end = numClusters
		}
		t.Logf("  Clusters %2d-%2d: %v", i, end-1, sorted[i:end])
	}
	t.Log("==================================================================")
}

// TestTopSelectImpact measures how TopSuperBuckets affects coverage and recall.
func TestTopSelectImpact(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping impact test in short mode")
	}

	numVectors := 10000
	dimension := 128
	numClusters := 64
	numQueries := 5     // Keep low due to slow HE (~3s per query)
	topK := 10

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate data
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec_%d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
		var norm float64
		for _, v := range vectors[i] {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range vectors[i] {
			vectors[i][j] /= norm
		}
	}

	queries := make([][]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			queries[i][j] = rng.NormFloat64()
		}
		var norm float64
		for _, v := range queries[i] {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := range queries[i] {
			queries[i][j] /= norm
		}
	}

	// Build index once
	enterpriseCfg, _ := enterprise.NewConfig("impact-test", dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	index, _ := builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	vectorToCluster := make(map[string]int)
	for vecID, loc := range index.VectorLocations {
		vectorToCluster[vecID] = loc.SuperID
	}

	// Compute ground truth
	groundTruths := make([][]string, numQueries)
	for q := 0; q < numQueries; q++ {
		type scored struct {
			id    string
			score float64
		}
		allScores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			score := dotProductDiag(queries[q], vectors[i])
			allScores[i] = scored{id: ids[i], score: score}
		}
		sort.Slice(allScores, func(i, j int) bool {
			return allScores[i].score > allScores[j].score
		})
		groundTruths[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruths[q][i] = allScores[i].id
		}
	}

	// Test different TopSelect values
	topSelectValues := []int{8, 16, 32, 48}

	t.Log("")
	t.Log("==================================================================")
	t.Log("            TOP SELECT IMPACT ANALYSIS")
	t.Log("==================================================================")
	t.Logf("Dataset:   %d vectors, %d clusters", numVectors, numClusters)
	t.Logf("Queries:   %d", numQueries)
	t.Log("")
	t.Log("TopSelect | Coverage | Recall@10 | Vectors/Query | Time")
	t.Log("----------|----------|-----------|---------------|--------")

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "impact-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	for _, topSel := range topSelectValues {
		cfg.TopSuperBuckets = topSel
		client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

		var totalRecall, totalCoverage float64
		var totalVectors int
		var totalTime time.Duration

		for q := 0; q < numQueries; q++ {
			start := time.Now()
			result, _ := client.Search(ctx, queries[q], topK)
			totalTime += time.Since(start)

			// Recall
			resultSet := make(map[string]bool)
			for _, r := range result.Results {
				resultSet[r.ID] = true
			}
			hits := 0
			for _, gt := range groundTruths[q] {
				if resultSet[gt] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(topK)

			// Coverage
			heSelected := make(map[int]bool)
			for _, c := range result.Stats.SelectedClusters {
				heSelected[c] = true
			}
			gtInSel := 0
			for _, gtID := range groundTruths[q] {
				if cluster, ok := vectorToCluster[gtID]; ok {
					if heSelected[cluster] {
						gtInSel++
					}
				}
			}
			totalCoverage += float64(gtInSel) / float64(topK)
			totalVectors += result.Stats.VectorsScored
		}

		avgRecall := totalRecall / float64(numQueries) * 100
		avgCoverage := totalCoverage / float64(numQueries) * 100
		avgVectors := float64(totalVectors) / float64(numQueries)
		avgTime := totalTime / time.Duration(numQueries)

		t.Logf("    %2d    |  %5.1f%%  |   %5.1f%%  |     %6.0f    | %v",
			topSel, avgCoverage, avgRecall, avgVectors, avgTime)
	}

	t.Log("==================================================================")
}

// Helper functions

func dotProductDiag(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func selectTopKIndicesLocal(scores []float64, k int) []int {
	type indexed struct {
		idx   int
		score float64
	}
	items := make([]indexed, len(scores))
	for i, s := range scores {
		items[i] = indexed{idx: i, score: s}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})
	if k > len(items) {
		k = len(items)
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = items[i].idx
	}
	return result
}
