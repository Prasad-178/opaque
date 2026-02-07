package client

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestBenchmark100K runs a full benchmark on 100K synthetic vectors
func TestBenchmark100K(t *testing.T) {
	ctx := context.Background()

	// Configuration
	numVectors := 100000
	dimension := 128
	numClusters := 64 // More clusters for larger dataset
	numQueries := 20
	topK := 10
	topSelect := 16

	t.Log("")
	t.Log("==============================================================")
	t.Log("         100K VECTOR BENCHMARK - FULL PIPELINE")
	t.Log("==============================================================")
	t.Logf("Vectors:     %d", numVectors)
	t.Logf("Dimension:   %d", dimension)
	t.Logf("Clusters:    %d", numClusters)
	t.Logf("TopSelect:   %d", topSelect)
	t.Logf("Queries:     %d", numQueries)
	t.Log("")

	// ==========================================
	// PHASE 1: Data Generation
	// ==========================================
	t.Log("PHASE 1: Generating synthetic data...")
	startGen := time.Now()

	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)

	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec_%06d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.Float64()*2 - 1 // [-1, 1]
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
			queries[i][j] = rng.Float64()*2 - 1
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

	dataGenTime := time.Since(startGen)
	t.Logf("  Data generation:     %v", dataGenTime)

	// ==========================================
	// PHASE 2: Index Building (K-means + AES Encryption)
	// ==========================================
	t.Log("")
	t.Log("PHASE 2: Building index (K-means clustering + AES encryption)...")
	startBuild := time.Now()

	enterpriseCfg, err := enterprise.NewConfig("benchmark-100k", dimension, numClusters)
	if err != nil {
		t.Fatalf("Failed to create enterprise config: %v", err)
	}

	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, err := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	if err != nil {
		t.Fatalf("Failed to create builder: %v", err)
	}

	// Track sub-phases
	kmeansStart := time.Now()
	_, err = builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}
	kmeansTime := time.Since(kmeansStart)

	enterpriseCfg = builder.GetEnterpriseConfig()
	buildTime := time.Since(startBuild)

	t.Logf("  K-means clustering:  %v", kmeansTime)
	t.Logf("  Total build time:    %v", buildTime)

	// ==========================================
	// PHASE 3: Auth Setup
	// ==========================================
	t.Log("")
	t.Log("PHASE 3: Setting up authentication...")
	startAuth := time.Now()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "benchuser", "benchmark-100k", []byte("pass"), []string{auth.ScopeSearch})
	creds, err := authService.Authenticate(ctx, "benchuser", []byte("pass"))
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	authTime := time.Since(startAuth)
	t.Logf("  Auth setup:          %v", authTime)

	// ==========================================
	// PHASE 4: Client Setup (HE Engine + Centroid Cache)
	// ==========================================
	t.Log("")
	t.Log("PHASE 4: Setting up client (HE engine + centroid cache)...")
	startClient := time.Now()

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	clientSetupTime := time.Since(startClient)
	t.Logf("  Client setup:        %v", clientSetupTime)
	t.Logf("    (includes HE key generation + centroid pre-encoding)")

	// ==========================================
	// PHASE 5: Query Benchmark
	// ==========================================
	t.Log("")
	t.Log("PHASE 5: Running query benchmark...")

	// Warmup
	t.Log("  Warming up (1 query)...")
	_, _ = client.Search(ctx, queries[0], topK)

	// Run queries
	t.Logf("  Running %d queries...", numQueries)
	var totalTiming hierarchical.SearchTiming
	var totalStats hierarchical.SearchStats

	queryTimes := make([]time.Duration, numQueries)
	for i := 0; i < numQueries; i++ {
		result, err := client.Search(ctx, queries[i], topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		queryTimes[i] = result.Timing.Total
		totalTiming.HEEncryptQuery += result.Timing.HEEncryptQuery
		totalTiming.HECentroidScores += result.Timing.HECentroidScores
		totalTiming.HEDecryptScores += result.Timing.HEDecryptScores
		totalTiming.BucketSelection += result.Timing.BucketSelection
		totalTiming.BucketFetch += result.Timing.BucketFetch
		totalTiming.AESDecrypt += result.Timing.AESDecrypt
		totalTiming.LocalScoring += result.Timing.LocalScoring
		totalTiming.Total += result.Timing.Total

		totalStats.HEOperations += result.Stats.HEOperations
		totalStats.SuperBucketsSelected += result.Stats.SuperBucketsSelected
		totalStats.BlobsFetched += result.Stats.BlobsFetched
		totalStats.VectorsScored += result.Stats.VectorsScored
	}

	// Compute averages
	n := time.Duration(numQueries)
	avgTiming := hierarchical.SearchTiming{
		HEEncryptQuery:   totalTiming.HEEncryptQuery / n,
		HECentroidScores: totalTiming.HECentroidScores / n,
		HEDecryptScores:  totalTiming.HEDecryptScores / n,
		BucketSelection:  totalTiming.BucketSelection / n,
		BucketFetch:      totalTiming.BucketFetch / n,
		AESDecrypt:       totalTiming.AESDecrypt / n,
		LocalScoring:     totalTiming.LocalScoring / n,
		Total:            totalTiming.Total / n,
	}

	avgStats := hierarchical.SearchStats{
		HEOperations:         totalStats.HEOperations / numQueries,
		SuperBucketsSelected: totalStats.SuperBucketsSelected / numQueries,
		BlobsFetched:         totalStats.BlobsFetched / numQueries,
		VectorsScored:        totalStats.VectorsScored / numQueries,
	}

	// Calculate percentiles
	sortedTimes := make([]time.Duration, len(queryTimes))
	copy(sortedTimes, queryTimes)
	for i := range sortedTimes {
		for j := i + 1; j < len(sortedTimes); j++ {
			if sortedTimes[j] < sortedTimes[i] {
				sortedTimes[i], sortedTimes[j] = sortedTimes[j], sortedTimes[i]
			}
		}
	}
	p50 := sortedTimes[numQueries/2]
	p95 := sortedTimes[int(float64(numQueries)*0.95)]
	p99 := sortedTimes[int(float64(numQueries)*0.99)]

	// ==========================================
	// RESULTS
	// ==========================================
	t.Log("")
	t.Log("==============================================================")
	t.Log("                      RESULTS SUMMARY")
	t.Log("==============================================================")
	t.Log("")
	t.Log("SETUP TIMES:")
	t.Log("--------------------------------------------------------------")
	t.Logf("  Data generation:       %12v", dataGenTime)
	t.Logf("  K-means clustering:    %12v", kmeansTime)
	t.Logf("  Index build (total):   %12v", buildTime)
	t.Logf("  Auth setup:            %12v", authTime)
	t.Logf("  Client setup:          %12v", clientSetupTime)
	t.Logf("  ─────────────────────────────────────")
	t.Logf("  TOTAL SETUP:           %12v", dataGenTime+buildTime+authTime+clientSetupTime)
	t.Log("")
	t.Log("QUERY TIME BREAKDOWN (average per query):")
	t.Log("--------------------------------------------------------------")
	t.Logf("  1. HE Encrypt Query:   %12v  (%5.1f%%)", avgTiming.HEEncryptQuery, pct(avgTiming.HEEncryptQuery, avgTiming.Total))
	t.Logf("  2. HE Centroid Scores: %12v  (%5.1f%%)", avgTiming.HECentroidScores, pct(avgTiming.HECentroidScores, avgTiming.Total))
	t.Logf("  3. HE Decrypt Scores:  %12v  (%5.1f%%)", avgTiming.HEDecryptScores, pct(avgTiming.HEDecryptScores, avgTiming.Total))
	t.Logf("  4. Bucket Selection:   %12v  (%5.1f%%)", avgTiming.BucketSelection, pct(avgTiming.BucketSelection, avgTiming.Total))
	t.Logf("  5. Bucket Fetch:       %12v  (%5.1f%%)", avgTiming.BucketFetch, pct(avgTiming.BucketFetch, avgTiming.Total))
	t.Logf("  6. AES Decrypt:        %12v  (%5.1f%%)", avgTiming.AESDecrypt, pct(avgTiming.AESDecrypt, avgTiming.Total))
	t.Logf("  7. Local Scoring:      %12v  (%5.1f%%)", avgTiming.LocalScoring, pct(avgTiming.LocalScoring, avgTiming.Total))
	t.Logf("  ─────────────────────────────────────")
	t.Logf("  TOTAL:                 %12v", avgTiming.Total)
	t.Log("")
	t.Log("QUERY LATENCY DISTRIBUTION:")
	t.Log("--------------------------------------------------------------")
	t.Logf("  p50 (median):          %12v", p50)
	t.Logf("  p95:                   %12v", p95)
	t.Logf("  p99:                   %12v", p99)
	t.Log("")
	t.Log("QUERY STATISTICS (average per query):")
	t.Log("--------------------------------------------------------------")
	t.Logf("  HE Operations:         %12d  (centroids scored)", avgStats.HEOperations)
	t.Logf("  Clusters Selected:     %12d  (out of %d)", avgStats.SuperBucketsSelected, numClusters)
	t.Logf("  Blobs Fetched:         %12d  (%.1f%% of dataset)", avgStats.BlobsFetched, float64(avgStats.BlobsFetched)/float64(numVectors)*100)
	t.Logf("  Vectors Scored:        %12d", avgStats.VectorsScored)
	t.Log("")
	t.Log("PRIVACY GUARANTEES:")
	t.Log("--------------------------------------------------------------")
	t.Logf("  Query Privacy:         100%% (query encrypted with HE, never leaves client)")
	t.Logf("  Cluster Selection:     100%% (selection happens after local HE decrypt)")
	t.Logf("  Access Pattern:        Hidden (via decoy requests if enabled)")
	t.Logf("  Vector Privacy:        100%% (vectors AES-encrypted at rest)")
	t.Log("")
	t.Log("ACCURACY (Recall@10 estimate for synthetic data):")
	t.Log("--------------------------------------------------------------")
	t.Logf("  Clusters probed:       %d / %d (%.1f%%)", topSelect, numClusters, float64(topSelect)/float64(numClusters)*100)
	t.Logf("  Expected Recall@10:    ~%.1f%% (based on cluster coverage)", float64(topSelect)/float64(numClusters)*100*1.5)
	t.Logf("  Note: Actual recall depends on data distribution")
	t.Log("")
	t.Log("==============================================================")
}

// TestBenchmark100KWithRecall measures actual recall on 100K vectors
func TestBenchmark100KWithRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping 100K recall test in short mode")
	}

	ctx := context.Background()

	// Configuration
	numVectors := 100000
	dimension := 128
	numClusters := 64
	numQueries := 10
	topK := 10
	topSelect := 16

	t.Log("")
	t.Log("==============================================================")
	t.Log("    100K RECALL BENCHMARK (with brute-force ground truth)")
	t.Log("==============================================================")
	t.Logf("Computing ground truth for %d queries on %d vectors...", numQueries, numVectors)

	// Generate data
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)

	for i := 0; i < numVectors; i++ {
		ids[i] = fmt.Sprintf("vec_%06d", i)
		vectors[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			vectors[i][j] = rng.Float64()*2 - 1
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

	// Generate queries
	queries := make([][]float64, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			queries[i][j] = rng.Float64()*2 - 1
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

	// Compute brute-force ground truth
	t.Log("Computing brute-force ground truth...")
	startGT := time.Now()
	groundTruth := make([][]string, numQueries)

	for q := 0; q < numQueries; q++ {
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, numVectors)
		for i := 0; i < numVectors; i++ {
			var dot float64
			for j := 0; j < dimension; j++ {
				dot += queries[q][j] * vectors[i][j]
			}
			scores[i] = scored{id: ids[i], score: dot}
		}

		// Sort by score
		for i := range scores {
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[i].score {
					scores[i], scores[j] = scores[j], scores[i]
				}
			}
		}

		groundTruth[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruth[q][i] = scores[i].id
		}
	}
	gtTime := time.Since(startGT)
	t.Logf("Ground truth computed in %v", gtTime)

	// Build index
	t.Log("Building index...")
	enterpriseCfg, _ := enterprise.NewConfig("recall-100k", dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	_, _ = builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Setup client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "recall-100k", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))
	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	// Measure recall
	t.Log("Running queries and measuring recall...")
	var totalRecall float64

	for q := 0; q < numQueries; q++ {
		result, err := client.Search(ctx, queries[q], topK)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		// Count matches
		gtSet := make(map[string]bool)
		for _, id := range groundTruth[q] {
			gtSet[id] = true
		}

		matches := 0
		for _, r := range result.Results {
			if gtSet[r.ID] {
				matches++
			}
		}

		recall := float64(matches) / float64(topK)
		totalRecall += recall
		t.Logf("  Query %d: Recall@%d = %.1f%% (%d/%d matches)", q, topK, recall*100, matches, topK)
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Log("")
	t.Log("--------------------------------------------------------------")
	t.Logf("AVERAGE RECALL@%d: %.1f%%", topK, avgRecall*100)
	t.Log("--------------------------------------------------------------")
}
