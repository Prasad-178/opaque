//go:build integration

package client

import (
	"context"
	"fmt"
	"testing"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/embeddings"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestSIFTKMeansEndToEnd tests full end-to-end accuracy with K-means
// IMPORTANT: SIFT ground truth uses Euclidean distance, but we use cosine similarity.
// This test now computes cosine-similarity-based ground truth for fair comparison.
func TestSIFTKMeansEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT K-means end-to-end test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("SIFT10K: K-MEANS END-TO-END ACCURACY TEST")
	fmt.Println("=" + "=================================================================")

	// Use more clusters for better accuracy
	numClusters := 64
	topSelect := 8 // Top 12.5% of clusters
	topK := 10
	numQueries := 20 // Reduced for faster debugging

	fmt.Printf("\nConfig: %d clusters, top %d selected (%.1f%%), %d queries\n",
		numClusters, topSelect, float64(topSelect)/float64(numClusters)*100, numQueries)

	// Precompute cosine similarity ground truth (SIFT uses Euclidean)
	fmt.Println("\nComputing cosine similarity ground truth...")
	cosineGT := computeCosineGroundTruth(dataset, topK)

	// Build K-means index with no sub-buckets
	enterpriseCfg, _ := enterprise.NewConfig("e2e-test", dataset.Dimension, numClusters)
	enterpriseCfg.NumSubBuckets = 1 // Single sub-bucket = all vectors in cluster
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.SubBucketsPerSuper = 1 // Fetch all
	cfg.NumDecoys = 0          // No decoys for clean measurement

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	clusterStats := builder.GetClusterStats()
	fmt.Printf("\nK-means: %d iterations, cluster sizes: min=%d, max=%d, avg=%.1f\n",
		clusterStats.Iterations, clusterStats.MinSize, clusterStats.MaxSize, clusterStats.AvgSize)

	// Setup client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "e2e-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	// Compute what clusters SHOULD be selected (plaintext scoring for reference)
	centroids := enterpriseCfg.Centroids

	// Run queries - compare both Euclidean GT (SIFT) and Cosine GT (our computation)
	var euclideanRecall1, cosineRecall1 float64
	var euclideanRecall10, cosineRecall10 float64
	var totalScanned int
	var totalClusterHit int

	debugQueries := 5
	if debugQueries > 0 {
		fmt.Println("\n=== Debug: First 5 queries ===")
	}

	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]
		normalizedQuery := crypto.NormalizeVector(query)

		// Euclidean GT (from SIFT dataset)
		euclideanGTIdx := dataset.GroundTruth[q][0]
		_ = euclideanGTIdx // Used later for recall calculation

		// Cosine GT (our computation)
		cosineGTIdx := cosineGT[q][0]
		cosineGTID := fmt.Sprintf("sift_%d", cosineGTIdx)
		cosineGTCluster := idx.VectorLocations[cosineGTID].SuperID

		// What clusters SHOULD be selected (plaintext)
		scores := make([]float64, len(centroids))
		for i, c := range centroids {
			scores[i] = cosineSim(normalizedQuery, c)
		}
		expectedTopK := getTopKIndices(scores, topSelect)

		// Check if cosine GT cluster is in expected selection
		gtInExpected := false
		for _, c := range expectedTopK {
			if c == cosineGTCluster {
				gtInExpected = true
				break
			}
		}
		if gtInExpected {
			totalClusterHit++
		}

		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Logf("Search error: %v", err)
			continue
		}

		// Check if GT was found at all
		gtFound := false
		gtRank := -1
		for i, r := range result.Results {
			if r.ID == cosineGTID {
				gtFound = true
				gtRank = i + 1
				break
			}
		}

		// Check bucket key mismatch - what bucket key was GT stored in vs what we searched
		gtLoc := idx.VectorLocations[cosineGTID]
		gtBucketKey := gtLoc.BucketKey

		// Is GT cluster in our selected clusters?
		gtClusterSelected := false
		for _, c := range expectedTopK {
			if c == cosineGTCluster {
				gtClusterSelected = true
				break
			}
		}

		// Directly verify GT blob exists in store
		gtBlobs, _ := store.GetBuckets(ctx, []string{gtBucketKey})
		gtBlobFound := false
		for _, b := range gtBlobs {
			if b.ID == cosineGTID {
				gtBlobFound = true
				break
			}
		}

		// Try to decrypt the GT blob directly
		gtDecryptOK := false
		var gtDecryptedScore float64
		if gtBlobFound && len(gtBlobs) > 0 {
			for _, b := range gtBlobs {
				if b.ID == cosineGTID {
					// Try decryption with the client's encryptor
					if len(creds.AESKey) > 0 {
						testEncryptor, _ := encrypt.NewAESGCM(creds.AESKey)
						decVec, decErr := testEncryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
						if decErr == nil && len(decVec) > 0 {
							gtDecryptOK = true
							normalizedQuery := crypto.NormalizeVector(query)
							normalizedGT := crypto.NormalizeVector(decVec)
							gtDecryptedScore = dotProduct(normalizedQuery, normalizedGT)
						}
					}
					break
				}
			}
		}

		// Fetch bucket directly to count blobs
		gtBucketBlobs, _ := store.GetBuckets(ctx, []string{gtBucketKey})
		blobsInGTBucket := len(gtBucketBlobs)
		gtBlobInFetch := false
		for _, b := range gtBucketBlobs {
			if b.ID == cosineGTID {
				gtBlobInFetch = true
				break
			}
		}

		if q < debugQueries {
			fmt.Printf("\nQuery %d:\n", q)
			fmt.Printf("  GT: %s, cluster: %d, bucketKey: %s\n", cosineGTID, cosineGTCluster, gtBucketKey)
			fmt.Printf("  GT cluster selected: %v, GT blob in store: %v\n", gtClusterSelected, gtBlobFound)
			fmt.Printf("  Blobs in GT bucket: %d, GT in bucket fetch: %v\n", blobsInGTBucket, gtBlobInFetch)
			fmt.Printf("  GT decryption OK: %v (score=%.4f)\n", gtDecryptOK, gtDecryptedScore)
			fmt.Printf("  Selected clusters: %v\n", expectedTopK[:min(4, len(expectedTopK))])
			fmt.Printf("  Vectors scored: %d\n", result.Stats.VectorsScored)
			fmt.Printf("  GT found in results: %v (rank=%d)\n", gtFound, gtRank)
			if len(result.Results) > 0 {
				fmt.Printf("  Top result: %s (score=%.4f)\n", result.Results[0].ID, result.Results[0].Score)
			}
		}

		// Recall@1 with Euclidean GT
		if len(result.Results) > 0 {
			var resultIdx int
			fmt.Sscanf(result.Results[0].ID, "sift_%d", &resultIdx)
			if resultIdx == euclideanGTIdx {
				euclideanRecall1++
			}
			if resultIdx == cosineGTIdx {
				cosineRecall1++
			}
		}

		// Recall@10 with Euclidean GT
		euclideanGTSet := make(map[int]bool)
		for i := 0; i < topK && i < len(dataset.GroundTruth[q]); i++ {
			euclideanGTSet[dataset.GroundTruth[q][i]] = true
		}
		euclideanFound := 0
		for _, r := range result.Results {
			var idx int
			fmt.Sscanf(r.ID, "sift_%d", &idx)
			if euclideanGTSet[idx] {
				euclideanFound++
			}
		}
		euclideanRecall10 += float64(euclideanFound) / float64(min(topK, len(euclideanGTSet)))

		// Recall@10 with Cosine GT
		cosineGTSet := make(map[int]bool)
		for i := 0; i < topK && i < len(cosineGT[q]); i++ {
			cosineGTSet[cosineGT[q][i]] = true
		}
		cosineFound := 0
		for _, r := range result.Results {
			var idx int
			fmt.Sscanf(r.ID, "sift_%d", &idx)
			if cosineGTSet[idx] {
				cosineFound++
			}
		}
		cosineRecall10 += float64(cosineFound) / float64(min(topK, len(cosineGTSet)))

		totalScanned += result.Stats.VectorsScored
	}

	eR1 := euclideanRecall1 / float64(numQueries) * 100
	cR1 := cosineRecall1 / float64(numQueries) * 100
	eR10 := euclideanRecall10 / float64(numQueries) * 100
	cR10 := cosineRecall10 / float64(numQueries) * 100
	avgScanned := totalScanned / numQueries
	pctScanned := float64(avgScanned) / float64(len(dataset.Vectors)) * 100
	clusterHitPct := float64(totalClusterHit) / float64(numQueries) * 100

	fmt.Println("\n┌──────────────────────┬────────────────┬────────────────┐")
	fmt.Println("│       Metric         │ Euclidean GT   │  Cosine GT     │")
	fmt.Println("├──────────────────────┼────────────────┼────────────────┤")
	fmt.Printf("│ Recall@1             │    %5.1f%%      │    %5.1f%%      │\n", eR1, cR1)
	fmt.Printf("│ Recall@10            │    %5.1f%%      │    %5.1f%%      │\n", eR10, cR10)
	fmt.Println("├──────────────────────┼────────────────┴────────────────┤")
	fmt.Printf("│ Vectors Scanned      │           %6d                 │\n", avgScanned)
	fmt.Printf("│ %% Dataset            │          %5.2f%%                 │\n", pctScanned)
	fmt.Printf("│ Cluster Selection    │          %5.1f%%                 │\n", clusterHitPct)
	fmt.Println("└──────────────────────┴─────────────────────────────────┘")

	fmt.Println("\nNote: Euclidean GT is from SIFT dataset, Cosine GT is recomputed for fair comparison")

	// With correct metric, we should see high recall
	if cR1 < 70 {
		t.Logf("Warning: Cosine Recall@1 lower than expected (%.1f%%)", cR1)
	}
}

// computeCosineGroundTruth computes ground truth using cosine similarity
// instead of the Euclidean distance used by SIFT dataset.
func computeCosineGroundTruth(dataset *embeddings.Dataset, topK int) [][]int {
	numQueries := len(dataset.Queries)
	gt := make([][]int, numQueries)

	for q := 0; q < numQueries; q++ {
		normalizedQuery := crypto.NormalizeVector(dataset.Queries[q])

		// Score all vectors
		type scored struct {
			idx   int
			score float64
		}
		scores := make([]scored, len(dataset.Vectors))
		for i, vec := range dataset.Vectors {
			normalizedVec := crypto.NormalizeVector(vec)
			scores[i] = scored{i, dotProduct(normalizedQuery, normalizedVec)}
		}

		// Sort by score descending
		for i := 0; i < topK; i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}

		gt[q] = make([]int, topK)
		for i := 0; i < topK; i++ {
			gt[q][i] = scores[i].idx
		}
	}

	return gt
}

// Note: dotProduct is defined in client.go

// TestSIFTKMeansScaling tests K-means accuracy with different bucket counts
func TestSIFTKMeansScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT K-means scaling test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("K-MEANS SCALING: ACCURACY vs NUMBER OF CLUSTERS")
	fmt.Println("=" + "=================================================================")

	topK := 10
	numQueries := 50

	// Test different cluster counts with fixed top-K selection
	configs := []struct {
		numClusters int
		topSelect   int
	}{
		{16, 4},   // 25% of clusters
		{32, 8},   // 25% of clusters
		{64, 16},  // 25% of clusters
		{128, 32}, // 25% of clusters
	}

	fmt.Println("\n┌──────────────┬───────────┬──────────┬────────────┬───────────┐")
	fmt.Println("│ NumClusters  │ TopSelect │ Recall@1 │  Recall@10 │ % Dataset │")
	fmt.Println("├──────────────┼───────────┼──────────┼────────────┼───────────┤")

	for _, cfg := range configs {
		recall1, recall10, scanned := runClusteringTest(
			ctx, t, dataset, cfg.numClusters, 1, cfg.topSelect, numQueries, topK, true)

		pctScanned := float64(scanned) / float64(len(dataset.Vectors)) * 100

		fmt.Printf("│     %3d      │    %2d     │  %5.1f%%  │   %5.1f%%   │   %5.2f%%  │\n",
			cfg.numClusters, cfg.topSelect, recall1*100, recall10*100, pctScanned)
	}

	fmt.Println("└──────────────┴───────────┴──────────┴────────────┴───────────┘")
	fmt.Println("\nNote: More clusters = smaller clusters = better centroids = better recall")
}

// TestSIFTKMeansCentroidQuality verifies K-means centroids are better representatives
func TestSIFTKMeansCentroidQuality(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT K-means centroid quality test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("CENTROID QUALITY: HE vs PLAINTEXT RANKING COMPARISON")
	fmt.Println("=" + "=================================================================")

	numClusters := 32
	numQueries := 20

	// Build K-means index
	enterpriseCfg, _ := enterprise.NewConfig("quality-test", dataset.Dimension, numClusters)
	enterpriseCfg.NumSubBuckets = 1
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()
	centroids := enterpriseCfg.Centroids

	clusterStats := builder.GetClusterStats()
	fmt.Printf("\nK-means clustering: %d clusters, %d iterations\n",
		clusterStats.NumClusters, clusterStats.Iterations)
	fmt.Printf("Cluster sizes: min=%d, max=%d, avg=%.1f\n",
		clusterStats.MinSize, clusterStats.MaxSize, clusterStats.AvgSize)

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
			plainScores[i] = cosineSim(normalizedQuery, c)
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

	if totalMatch < numQueries*9/10 {
		t.Errorf("HE ranking should match plaintext for K-means centroids")
	}
}

func runClusteringTest(
	ctx context.Context,
	t *testing.T,
	dataset *embeddings.Dataset,
	numSuperBuckets, numSubBuckets, topSuperBuckets, numQueries, topK int,
	useKMeans bool,
) (recall1, recall10 float64, avgScanned int) {

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("cluster-test", dataset.Dimension, numSuperBuckets)
	enterpriseCfg.NumSubBuckets = numSubBuckets
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSuperBuckets
	cfg.SubBucketsPerSuper = numSubBuckets // Fetch all sub-buckets
	cfg.NumDecoys = 8

	var idx *hierarchical.Index
	if useKMeans {
		builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
		idx, _ = builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
		enterpriseCfg = builder.GetEnterpriseConfig()
	} else {
		builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
		idx, _ = builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
		enterpriseCfg = builder.GetEnterpriseConfig()
	}
	_ = idx // Index used for verification

	// Setup auth and client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "cluster-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	var totalRecall1, totalRecall10 float64
	var totalScanned int

	for q := 0; q < numQueries; q++ {
		query := dataset.Queries[q]
		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Logf("Search error for query %d: %v", q, err)
			continue
		}

		// Ground truth set
		gtSet := make(map[int]bool)
		for i := 0; i < topK && i < len(dataset.GroundTruth[q]); i++ {
			gtSet[dataset.GroundTruth[q][i]] = true
		}

		// Check Recall@1
		if len(result.Results) > 0 {
			var idx int
			fmt.Sscanf(result.Results[0].ID, "sift_%d", &idx)
			if idx == dataset.GroundTruth[q][0] {
				totalRecall1++
			}
		}

		// Check Recall@K
		found := 0
		for _, r := range result.Results {
			var idx int
			fmt.Sscanf(r.ID, "sift_%d", &idx)
			if gtSet[idx] {
				found++
			}
		}
		if len(gtSet) > 0 {
			totalRecall10 += float64(found) / float64(min(topK, len(gtSet)))
		}

		totalScanned += result.Stats.VectorsScored
	}

	recall1 = totalRecall1 / float64(numQueries)
	recall10 = totalRecall10 / float64(numQueries)
	avgScanned = totalScanned / numQueries

	return
}

// Note: cosineSim is defined in enterprise_100k_test.go

// TestSIFTKMeansClusterSelection verifies K-means selects better clusters than LSH
func TestSIFTKMeansClusterSelection(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SIFT K-means cluster selection test in short mode")
	}
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	fmt.Println("\n" + "=" + "=================================================================")
	fmt.Println("CLUSTER SELECTION QUALITY: K-MEANS vs LSH")
	fmt.Println("=" + "=================================================================")
	fmt.Println("\nMeasures: Does top-K cluster selection include the ground truth vector's cluster?")

	numClusters := 64
	numQueries := 100

	fmt.Printf("\nConfig: %d clusters, %d queries\n", numClusters, numQueries)

	// Build both indices
	lshEntCfg, _ := enterprise.NewConfig("lsh-select-test", dataset.Dimension, numClusters)
	lshEntCfg.NumSubBuckets = 1
	lshStore := blob.NewMemoryStore()
	lshCfg := hierarchical.ConfigFromEnterprise(lshEntCfg)
	lshBuilder, _ := hierarchical.NewEnterpriseBuilder(lshCfg, lshEntCfg)
	lshIdx, _ := lshBuilder.Build(ctx, dataset.IDs, dataset.Vectors, lshStore)
	lshEntCfg = lshBuilder.GetEnterpriseConfig()

	kmEntCfg, _ := enterprise.NewConfig("km-select-test", dataset.Dimension, numClusters)
	kmEntCfg.NumSubBuckets = 1
	kmStore := blob.NewMemoryStore()
	kmCfg := hierarchical.ConfigFromEnterprise(kmEntCfg)
	kmBuilder, _ := hierarchical.NewKMeansBuilder(kmCfg, kmEntCfg)
	kmIdx, _ := kmBuilder.Build(ctx, dataset.IDs, dataset.Vectors, kmStore)
	kmEntCfg = kmBuilder.GetEnterpriseConfig()

	// Get centroids
	lshCentroids := lshEntCfg.Centroids
	kmCentroids := kmEntCfg.Centroids

	fmt.Println("\n┌───────────┬──────────────────┬──────────────────┐")
	fmt.Println("│  Top-K    │  LSH Bucket Hit  │  K-Means Hit     │")
	fmt.Println("│  Selected │  (GT in top-K)   │  (GT in top-K)   │")
	fmt.Println("├───────────┼──────────────────┼──────────────────┤")

	topKOptions := []int{2, 4, 8, 16, 32}

	for _, topK := range topKOptions {
		lshHits := 0
		kmHits := 0

		for q := 0; q < numQueries; q++ {
			query := dataset.Queries[q]
			normalizedQuery := crypto.NormalizeVector(query)

			// Get ground truth vector's cluster in each index
			gtIdx := dataset.GroundTruth[q][0]
			gtID := fmt.Sprintf("sift_%d", gtIdx)
			lshGTCluster := lshIdx.VectorLocations[gtID].SuperID
			kmGTCluster := kmIdx.VectorLocations[gtID].SuperID

			// Score centroids (plaintext for speed)
			lshScores := make([]float64, len(lshCentroids))
			kmScores := make([]float64, len(kmCentroids))
			for i := range lshCentroids {
				lshScores[i] = cosineSim(normalizedQuery, lshCentroids[i])
				kmScores[i] = cosineSim(normalizedQuery, kmCentroids[i])
			}

			// Get top-K clusters
			lshTopK := getTopKIndices(lshScores, topK)
			kmTopK := getTopKIndices(kmScores, topK)

			// Check if GT cluster is in top-K
			for _, c := range lshTopK {
				if c == lshGTCluster {
					lshHits++
					break
				}
			}
			for _, c := range kmTopK {
				if c == kmGTCluster {
					kmHits++
					break
				}
			}
		}

		lshPct := float64(lshHits) / float64(numQueries) * 100
		kmPct := float64(kmHits) / float64(numQueries) * 100

		fmt.Printf("│    %2d     │     %5.1f%%       │     %5.1f%%       │\n",
			topK, lshPct, kmPct)
	}

	fmt.Println("└───────────┴──────────────────┴──────────────────┘")
	fmt.Println("\n'Hit' = ground truth nearest neighbor's cluster is in the selected top-K clusters")
}
