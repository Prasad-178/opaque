//go:build integration

package client

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestComparisonBenchmark compares our privacy-preserving search against
// standard vector search approaches on 100K vectors.
//
// Systems compared:
// 1. Brute Force (baseline) - 100% recall, no privacy
// 2. IVF (Inverted File Index) - Similar to FAISS IVF, no privacy
// 3. Our System (HE + AES) - Privacy-preserving with HE encryption
//
// This provides context for our performance/accuracy trade-offs.
func TestComparisonBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100K vector comparison benchmark in short mode")
	}

	// Configuration
	numVectors := 100000
	dimension := 128
	numQueries := 20
	topK := 10

	t.Log("")
	t.Log("╔══════════════════════════════════════════════════════════════════╗")
	t.Log("║     VECTOR SEARCH COMPARISON BENCHMARK (100K Vectors)            ║")
	t.Log("╚══════════════════════════════════════════════════════════════════╝")
	t.Log("")
	t.Logf("Dataset:    %d vectors, %d dimensions", numVectors, dimension)
	t.Logf("Queries:    %d queries, top-%d results", numQueries, topK)
	t.Logf("Hardware:   %d CPU cores", runtime.NumCPU())
	t.Log("")

	// Generate data
	t.Log("Generating synthetic data...")
	rng := rand.New(rand.NewSource(42))
	ids, vectors := generateNormalizedVectors(rng, numVectors, dimension)
	queries := generateNormalizedQueries(rng, numQueries, dimension)

	// Compute ground truth (brute force)
	t.Log("Computing ground truth...")
	groundTruth := computeGroundTruth(vectors, queries, topK)

	t.Log("")
	t.Log("Running benchmarks...")
	t.Log("")

	// ========================================
	// 1. BRUTE FORCE (Baseline)
	// ========================================
	t.Log("┌──────────────────────────────────────────────────────────────────┐")
	t.Log("│ 1. BRUTE FORCE (Baseline - No Index)                             │")
	t.Log("└──────────────────────────────────────────────────────────────────┘")

	bruteResults := benchmarkBruteForce(vectors, queries, topK)
	bruteRecall := computeRecall(bruteResults.results, groundTruth, topK)

	t.Logf("   Setup Time:      N/A (no index)")
	t.Logf("   Query Latency:   %v (avg)", bruteResults.avgLatency)
	t.Logf("   Recall@%d:       %.1f%%", topK, bruteRecall*100)
	t.Logf("   Privacy:         NONE (plaintext vectors, plaintext query)")
	t.Logf("   Vectors Scanned: %d (100%%)", numVectors)
	t.Log("")

	// ========================================
	// 2. IVF (Inverted File Index) - Similar to FAISS
	// ========================================
	t.Log("┌──────────────────────────────────────────────────────────────────┐")
	t.Log("│ 2. IVF (Inverted File Index) - Similar to FAISS IVFFlat         │")
	t.Log("└──────────────────────────────────────────────────────────────────┘")

	ivfClusters := 64
	ivfProbe := 32 // Same as our TopSelect for fair comparison

	ivfResults := benchmarkIVF(ids, vectors, queries, topK, ivfClusters, ivfProbe)
	ivfRecall := computeRecall(ivfResults.results, groundTruth, topK)

	t.Logf("   Clusters:        %d (probing %d = %.0f%%)", ivfClusters, ivfProbe, float64(ivfProbe)/float64(ivfClusters)*100)
	t.Logf("   Setup Time:      %v (K-means clustering)", ivfResults.setupTime)
	t.Logf("   Query Latency:   %v (avg)", ivfResults.avgLatency)
	t.Logf("   Recall@%d:       %.1f%%", topK, ivfRecall*100)
	t.Logf("   Privacy:         NONE (plaintext vectors, plaintext query)")
	t.Logf("   Vectors Scanned: %d (%.1f%%)", ivfResults.vectorsScanned, float64(ivfResults.vectorsScanned)/float64(numVectors)*100)
	t.Log("")

	// ========================================
	// 3. IVF with Product Quantization (PQ) - Similar to FAISS IVFPQ
	// ========================================
	t.Log("┌──────────────────────────────────────────────────────────────────┐")
	t.Log("│ 3. IVF-PQ (with Product Quantization) - Similar to FAISS IVFPQ  │")
	t.Log("└──────────────────────────────────────────────────────────────────┘")

	pqResults := benchmarkIVFPQ(ids, vectors, queries, topK, ivfClusters, ivfProbe, 8) // 8 subquantizers
	pqRecall := computeRecall(pqResults.results, groundTruth, topK)

	t.Logf("   Clusters:        %d (probing %d)", ivfClusters, ivfProbe)
	t.Logf("   PQ Subspaces:    8 (16-dim each)")
	t.Logf("   Setup Time:      %v", pqResults.setupTime)
	t.Logf("   Query Latency:   %v (avg)", pqResults.avgLatency)
	t.Logf("   Recall@%d:       %.1f%%", topK, pqRecall*100)
	t.Logf("   Privacy:         NONE (plaintext vectors, plaintext query)")
	t.Logf("   Memory:          ~%.1f%% of original (compressed)", 100.0/float64(dimension)*8*8)
	t.Log("")

	// ========================================
	// 4. OUR SYSTEM (HE + AES Privacy-Preserving)
	// ========================================
	t.Log("┌──────────────────────────────────────────────────────────────────┐")
	t.Log("│ 4. OPAQUE (HE + AES Privacy-Preserving Search)                   │")
	t.Log("└──────────────────────────────────────────────────────────────────┘")

	ctx := context.Background()
	opaqueResults := benchmarkOpaque(ctx, t, ids, vectors, queries, topK, ivfClusters, ivfProbe)
	opaqueRecall := computeRecall(opaqueResults.results, groundTruth, topK)

	t.Logf("   Clusters:        %d (selecting %d = %.0f%%)", ivfClusters, ivfProbe, float64(ivfProbe)/float64(ivfClusters)*100)
	t.Logf("   Setup Time:      %v (K-means + HE keys + AES encrypt)", opaqueResults.setupTime)
	t.Logf("   Query Latency:   %v (avg)", opaqueResults.avgLatency)
	t.Logf("   Recall@%d:       %.1f%%", topK, opaqueRecall*100)
	t.Logf("   Privacy:         FULL")
	t.Logf("     - Query:       Encrypted (CKKS homomorphic encryption)")
	t.Logf("     - Vectors:     Encrypted (AES-256-GCM)")
	t.Logf("     - Selection:   Hidden (client-side after HE decrypt)")
	t.Logf("   Vectors Scanned: %d (%.1f%%)", opaqueResults.vectorsScanned, float64(opaqueResults.vectorsScanned)/float64(numVectors)*100)
	t.Log("")

	// ========================================
	// COMPARISON SUMMARY
	// ========================================
	t.Log("╔══════════════════════════════════════════════════════════════════╗")
	t.Log("║                      COMPARISON SUMMARY                          ║")
	t.Log("╚══════════════════════════════════════════════════════════════════╝")
	t.Log("")
	t.Log("┌────────────────┬────────────┬────────────┬────────────┬──────────┐")
	t.Log("│ System         │ Latency    │ Recall@10  │ Privacy    │ Overhead │")
	t.Log("├────────────────┼────────────┼────────────┼────────────┼──────────┤")
	t.Logf("│ Brute Force    │ %10v │    %5.1f%% │ None       │ 1.0x     │",
		bruteResults.avgLatency, bruteRecall*100)
	t.Logf("│ IVF (FAISS)    │ %10v │    %5.1f%% │ None       │ %.1fx     │",
		ivfResults.avgLatency, ivfRecall*100, float64(ivfResults.avgLatency)/float64(bruteResults.avgLatency))
	t.Logf("│ IVF-PQ         │ %10v │    %5.1f%% │ None       │ %.1fx     │",
		pqResults.avgLatency, pqRecall*100, float64(pqResults.avgLatency)/float64(bruteResults.avgLatency))
	t.Logf("│ OPAQUE (Ours)  │ %10v │    %5.1f%% │ FULL       │ %.0fx     │",
		opaqueResults.avgLatency, opaqueRecall*100, float64(opaqueResults.avgLatency)/float64(bruteResults.avgLatency))
	t.Log("└────────────────┴────────────┴────────────┴────────────┴──────────┘")
	t.Log("")

	// Privacy cost analysis
	privacyOverhead := float64(opaqueResults.avgLatency) / float64(ivfResults.avgLatency)
	t.Log("PRIVACY COST ANALYSIS:")
	t.Logf("  - Privacy overhead vs IVF: %.0fx slower", privacyOverhead)
	t.Logf("  - But provides: Query privacy, Vector privacy, Access pattern hiding")
	t.Logf("  - HE operations: %d per query (dominates latency)", ivfClusters)
	t.Log("")

	// Recall analysis
	t.Log("RECALL ANALYSIS:")
	t.Logf("  - Brute force baseline: %.1f%% (expected 100%%)", bruteRecall*100)
	t.Logf("  - IVF vs Brute force:   %.1f%% recall with %.1fx speedup",
		ivfRecall*100, float64(bruteResults.avgLatency)/float64(ivfResults.avgLatency))
	t.Logf("  - OPAQUE vs IVF:        %.1f%% vs %.1f%% (similar recall)",
		opaqueRecall*100, ivfRecall*100)
	t.Log("")
}

// Benchmark result types
type benchResult struct {
	avgLatency     time.Duration
	results        [][][]string // [query][result_idx] = vector_id
	setupTime      time.Duration
	vectorsScanned int
}

// generateNormalizedVectors creates normalized random vectors
func generateNormalizedVectors(rng *rand.Rand, n, dim int) ([]string, [][]float64) {
	ids := make([]string, n)
	vectors := make([][]float64, n)
	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("vec_%06d", i)
		vectors[i] = make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			vectors[i][j] = rng.NormFloat64()
			norm += vectors[i][j] * vectors[i][j]
		}
		norm = math.Sqrt(norm)
		for j := range vectors[i] {
			vectors[i][j] /= norm
		}
	}
	return ids, vectors
}

// generateNormalizedQueries creates normalized random queries
func generateNormalizedQueries(rng *rand.Rand, n, dim int) [][]float64 {
	queries := make([][]float64, n)
	for i := 0; i < n; i++ {
		queries[i] = make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			queries[i][j] = rng.NormFloat64()
			norm += queries[i][j] * queries[i][j]
		}
		norm = math.Sqrt(norm)
		for j := range queries[i] {
			queries[i][j] /= norm
		}
	}
	return queries
}

// computeGroundTruth computes brute-force top-K for each query
func computeGroundTruth(vectors, queries [][]float64, topK int) [][][]string {
	groundTruth := make([][][]string, len(queries))

	for q := range queries {
		type scored struct {
			idx   int
			score float64
		}
		scores := make([]scored, len(vectors))
		for i := range vectors {
			scores[i] = scored{idx: i, score: cosineSimCompare(queries[q], vectors[i])}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})

		groundTruth[q] = make([][]string, 1)
		groundTruth[q][0] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruth[q][0][i] = fmt.Sprintf("vec_%06d", scores[i].idx)
		}
	}
	return groundTruth
}

// computeRecall computes average recall@K
func computeRecall(results, groundTruth [][][]string, topK int) float64 {
	var totalRecall float64
	for q := range results {
		if len(results[q]) == 0 || len(groundTruth[q]) == 0 {
			continue
		}
		gtSet := make(map[string]bool)
		for _, id := range groundTruth[q][0] {
			gtSet[id] = true
		}
		hits := 0
		for _, id := range results[q][0] {
			if gtSet[id] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(topK)
	}
	return totalRecall / float64(len(results))
}

// cosineSimCompare computes cosine similarity (assumes normalized vectors)
func cosineSimCompare(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// benchmarkBruteForce runs brute force search
func benchmarkBruteForce(vectors, queries [][]float64, topK int) benchResult {
	results := make([][][]string, len(queries))
	var totalLatency time.Duration

	for q := range queries {
		start := time.Now()

		type scored struct {
			idx   int
			score float64
		}
		scores := make([]scored, len(vectors))
		for i := range vectors {
			scores[i] = scored{idx: i, score: cosineSimCompare(queries[q], vectors[i])}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})

		results[q] = make([][]string, 1)
		results[q][0] = make([]string, topK)
		for i := 0; i < topK; i++ {
			results[q][0][i] = fmt.Sprintf("vec_%06d", scores[i].idx)
		}

		totalLatency += time.Since(start)
	}

	return benchResult{
		avgLatency: totalLatency / time.Duration(len(queries)),
		results:    results,
	}
}

// benchmarkIVF runs IVF (Inverted File Index) search - similar to FAISS IVFFlat
func benchmarkIVF(ids []string, vectors, queries [][]float64, topK, nClusters, nProbe int) benchResult {
	// Setup: K-means clustering
	setupStart := time.Now()

	// Simple K-means (same as our system)
	centroids, assignments := simpleKMeans(vectors, nClusters, 10)

	// Build inverted lists
	invertedLists := make([][]int, nClusters)
	for i := range invertedLists {
		invertedLists[i] = make([]int, 0)
	}
	for vecIdx, clusterIdx := range assignments {
		invertedLists[clusterIdx] = append(invertedLists[clusterIdx], vecIdx)
	}

	setupTime := time.Since(setupStart)

	// Query
	results := make([][][]string, len(queries))
	var totalLatency time.Duration
	var totalVectorsScanned int

	for q := range queries {
		start := time.Now()

		// Find top nProbe clusters
		type scored struct {
			idx   int
			score float64
		}
		clusterScores := make([]scored, nClusters)
		for c := range centroids {
			clusterScores[c] = scored{idx: c, score: cosineSimCompare(queries[q], centroids[c])}
		}
		sort.Slice(clusterScores, func(i, j int) bool {
			return clusterScores[i].score > clusterScores[j].score
		})

		// Search in top clusters
		var candidates []scored
		for p := 0; p < nProbe; p++ {
			clusterIdx := clusterScores[p].idx
			for _, vecIdx := range invertedLists[clusterIdx] {
				score := cosineSimCompare(queries[q], vectors[vecIdx])
				candidates = append(candidates, scored{idx: vecIdx, score: score})
			}
		}
		totalVectorsScanned += len(candidates)

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].score > candidates[j].score
		})

		results[q] = make([][]string, 1)
		results[q][0] = make([]string, topK)
		for i := 0; i < topK && i < len(candidates); i++ {
			results[q][0][i] = fmt.Sprintf("vec_%06d", candidates[i].idx)
		}

		totalLatency += time.Since(start)
	}

	return benchResult{
		avgLatency:     totalLatency / time.Duration(len(queries)),
		results:        results,
		setupTime:      setupTime,
		vectorsScanned: totalVectorsScanned / len(queries),
	}
}

// benchmarkIVFPQ runs IVF with Product Quantization - similar to FAISS IVFPQ
func benchmarkIVFPQ(ids []string, vectors, queries [][]float64, topK, nClusters, nProbe, nSubquantizers int) benchResult {
	dim := len(vectors[0])
	subDim := dim / nSubquantizers

	// Setup: K-means + PQ codebook training
	setupStart := time.Now()

	// K-means for IVF
	centroids, assignments := simpleKMeans(vectors, nClusters, 10)

	// Build inverted lists
	invertedLists := make([][]int, nClusters)
	for i := range invertedLists {
		invertedLists[i] = make([]int, 0)
	}
	for vecIdx, clusterIdx := range assignments {
		invertedLists[clusterIdx] = append(invertedLists[clusterIdx], vecIdx)
	}

	// Train PQ codebooks (simplified - just use random samples)
	// In real FAISS, this would be more sophisticated
	nCodes := 256 // 8-bit codes
	codebooks := make([][][]float64, nSubquantizers)
	for s := 0; s < nSubquantizers; s++ {
		// Extract subvectors and cluster them
		subvectors := make([][]float64, len(vectors))
		for i := range vectors {
			subvectors[i] = vectors[i][s*subDim : (s+1)*subDim]
		}
		codebooks[s], _ = simpleKMeans(subvectors, nCodes, 5)
	}

	// Encode all vectors
	codes := make([][]int, len(vectors))
	for i := range vectors {
		codes[i] = make([]int, nSubquantizers)
		for s := 0; s < nSubquantizers; s++ {
			subvec := vectors[i][s*subDim : (s+1)*subDim]
			codes[i][s] = findNearestCode(subvec, codebooks[s])
		}
	}

	setupTime := time.Since(setupStart)

	// Query using asymmetric distance computation (ADC)
	results := make([][][]string, len(queries))
	var totalLatency time.Duration
	var totalVectorsScanned int

	for q := range queries {
		start := time.Now()

		// Find top nProbe clusters
		type scored struct {
			idx   int
			score float64
		}
		clusterScores := make([]scored, nClusters)
		for c := range centroids {
			clusterScores[c] = scored{idx: c, score: cosineSimCompare(queries[q], centroids[c])}
		}
		sort.Slice(clusterScores, func(i, j int) bool {
			return clusterScores[i].score > clusterScores[j].score
		})

		// Precompute query-to-codebook distances (ADC tables)
		adcTables := make([][]float64, nSubquantizers)
		for s := 0; s < nSubquantizers; s++ {
			querySubvec := queries[q][s*subDim : (s+1)*subDim]
			adcTables[s] = make([]float64, len(codebooks[s]))
			for c := range codebooks[s] {
				adcTables[s][c] = cosineSimCompare(querySubvec, codebooks[s][c])
			}
		}

		// Search in top clusters using ADC
		var candidates []scored
		for p := 0; p < nProbe; p++ {
			clusterIdx := clusterScores[p].idx
			for _, vecIdx := range invertedLists[clusterIdx] {
				// Approximate distance using ADC
				var approxScore float64
				for s := 0; s < nSubquantizers; s++ {
					approxScore += adcTables[s][codes[vecIdx][s]]
				}
				candidates = append(candidates, scored{idx: vecIdx, score: approxScore})
			}
		}
		totalVectorsScanned += len(candidates)

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].score > candidates[j].score
		})

		results[q] = make([][]string, 1)
		results[q][0] = make([]string, topK)
		for i := 0; i < topK && i < len(candidates); i++ {
			results[q][0][i] = fmt.Sprintf("vec_%06d", candidates[i].idx)
		}

		totalLatency += time.Since(start)
	}

	return benchResult{
		avgLatency:     totalLatency / time.Duration(len(queries)),
		results:        results,
		setupTime:      setupTime,
		vectorsScanned: totalVectorsScanned / len(queries),
	}
}

// benchmarkOpaque runs our privacy-preserving search
func benchmarkOpaque(ctx context.Context, t *testing.T, ids []string, vectors, queries [][]float64, topK, nClusters, nProbe int) benchResult {
	dimension := len(vectors[0])

	// Setup
	setupStart := time.Now()

	enterpriseCfg, _ := enterprise.NewConfig("compare-test", dimension, nClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = nProbe
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "compare-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	setupTime := time.Since(setupStart)

	// Warm up
	client.Search(ctx, queries[0], topK)

	// Query
	results := make([][][]string, len(queries))
	var totalLatency time.Duration
	var totalVectorsScanned int

	for q := range queries {
		start := time.Now()
		result, _ := client.Search(ctx, queries[q], topK)
		totalLatency += time.Since(start)

		results[q] = make([][]string, 1)
		results[q][0] = make([]string, len(result.Results))
		for i, r := range result.Results {
			results[q][0][i] = r.ID
		}
		totalVectorsScanned += result.Stats.VectorsScored
	}

	return benchResult{
		avgLatency:     totalLatency / time.Duration(len(queries)),
		results:        results,
		setupTime:      setupTime,
		vectorsScanned: totalVectorsScanned / len(queries),
	}
}

// simpleKMeans performs basic K-means clustering
func simpleKMeans(vectors [][]float64, k, maxIter int) ([][]float64, []int) {
	dim := len(vectors[0])
	n := len(vectors)

	// Initialize centroids randomly
	rng := rand.New(rand.NewSource(42))
	centroids := make([][]float64, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, dim)
		copy(centroids[i], vectors[rng.Intn(n)])
	}

	assignments := make([]int, n)

	for iter := 0; iter < maxIter; iter++ {
		// Assign vectors to nearest centroid (parallel)
		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU()
		chunkSize := (n + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}
			if start >= n {
				break
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					bestCluster := 0
					bestScore := math.Inf(-1)
					for c := range centroids {
						score := cosineSimCompare(vectors[i], centroids[c])
						if score > bestScore {
							bestScore = score
							bestCluster = c
						}
					}
					assignments[i] = bestCluster
				}
			}(start, end)
		}
		wg.Wait()

		// Update centroids
		counts := make([]int, k)
		sums := make([][]float64, k)
		for i := 0; i < k; i++ {
			sums[i] = make([]float64, dim)
		}

		for i, cluster := range assignments {
			counts[cluster]++
			for d := 0; d < dim; d++ {
				sums[cluster][d] += vectors[i][d]
			}
		}

		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				var norm float64
				for d := 0; d < dim; d++ {
					centroids[c][d] = sums[c][d] / float64(counts[c])
					norm += centroids[c][d] * centroids[c][d]
				}
				norm = math.Sqrt(norm)
				if norm > 0 {
					for d := 0; d < dim; d++ {
						centroids[c][d] /= norm
					}
				}
			}
		}
	}

	return centroids, assignments
}

// findNearestCode finds the nearest code in a codebook
func findNearestCode(vec []float64, codebook [][]float64) int {
	bestIdx := 0
	bestScore := math.Inf(-1)
	for i, code := range codebook {
		score := cosineSimCompare(vec, code)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	return bestIdx
}
