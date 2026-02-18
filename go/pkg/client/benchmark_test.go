//go:build integration

package client

import (
	"context"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/embeddings"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestSearchTimingBreakdown measures time spent in each phase
func TestSearchTimingBreakdown(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		t.Fatalf("Failed to load SIFT dataset: %v", err)
	}

	numClusters := 32
	topSelect := 16

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("timing-test", dataset.Dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Setup client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "timing-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Warm up
	client.Search(ctx, dataset.Queries[0], 10)

	// Run multiple queries and collect timing
	numQueries := 10
	var totalTiming hierarchical.SearchTiming

	for i := 0; i < numQueries; i++ {
		result, err := client.Search(ctx, dataset.Queries[i], 10)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		totalTiming.HEEncryptQuery += result.Timing.HEEncryptQuery
		totalTiming.HECentroidScores += result.Timing.HECentroidScores
		totalTiming.HEDecryptScores += result.Timing.HEDecryptScores
		totalTiming.BucketSelection += result.Timing.BucketSelection
		totalTiming.BucketFetch += result.Timing.BucketFetch
		totalTiming.AESDecrypt += result.Timing.AESDecrypt
		totalTiming.LocalScoring += result.Timing.LocalScoring
		totalTiming.Total += result.Timing.Total
	}

	// Compute averages
	avgTiming := hierarchical.SearchTiming{
		HEEncryptQuery:   totalTiming.HEEncryptQuery / time.Duration(numQueries),
		HECentroidScores: totalTiming.HECentroidScores / time.Duration(numQueries),
		HEDecryptScores:  totalTiming.HEDecryptScores / time.Duration(numQueries),
		BucketSelection:  totalTiming.BucketSelection / time.Duration(numQueries),
		BucketFetch:      totalTiming.BucketFetch / time.Duration(numQueries),
		AESDecrypt:       totalTiming.AESDecrypt / time.Duration(numQueries),
		LocalScoring:     totalTiming.LocalScoring / time.Duration(numQueries),
		Total:            totalTiming.Total / time.Duration(numQueries),
	}

	t.Log("")
	t.Log("==============================================================")
	t.Log("       SEARCH TIMING BREAKDOWN (average of 10 queries)")
	t.Log("==============================================================")
	t.Logf("  HE Encrypt Query:      %12v  (%5.1f%%)", avgTiming.HEEncryptQuery, pct(avgTiming.HEEncryptQuery, avgTiming.Total))
	t.Logf("  HE Centroid Scores:    %12v  (%5.1f%%)", avgTiming.HECentroidScores, pct(avgTiming.HECentroidScores, avgTiming.Total))
	t.Logf("  HE Decrypt Scores:     %12v  (%5.1f%%)", avgTiming.HEDecryptScores, pct(avgTiming.HEDecryptScores, avgTiming.Total))
	t.Logf("  Bucket Selection:      %12v  (%5.1f%%)", avgTiming.BucketSelection, pct(avgTiming.BucketSelection, avgTiming.Total))
	t.Logf("  Bucket Fetch:          %12v  (%5.1f%%)", avgTiming.BucketFetch, pct(avgTiming.BucketFetch, avgTiming.Total))
	t.Logf("  AES Decrypt:           %12v  (%5.1f%%)", avgTiming.AESDecrypt, pct(avgTiming.AESDecrypt, avgTiming.Total))
	t.Logf("  Local Scoring:         %12v  (%5.1f%%)", avgTiming.LocalScoring, pct(avgTiming.LocalScoring, avgTiming.Total))
	t.Log("--------------------------------------------------------------")
	t.Logf("  TOTAL:                 %12v", avgTiming.Total)
	t.Log("==============================================================")

	// Performance targets
	t.Log("")
	if avgTiming.Total < 2*time.Second {
		t.Log("✓ Total time under 2s target")
	} else {
		t.Logf("✗ Total time %v exceeds 2s target", avgTiming.Total)
	}
}

func pct(part, total time.Duration) float64 {
	if total == 0 {
		return 0
	}
	return float64(part) / float64(total) * 100
}

// BenchmarkSearch provides a proper Go benchmark
func BenchmarkSearch(b *testing.B) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		b.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, err := embeddings.SIFT10K(dataPath)
	if err != nil {
		b.Fatalf("Failed to load SIFT dataset: %v", err)
	}

	numClusters := 32
	topSelect := 16

	enterpriseCfg, _ := enterprise.NewConfig("bench", dataset.Dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "bench", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		b.Fatalf("Failed to create client: %v", err)
	}

	// Warm up
	client.Search(ctx, dataset.Queries[0], 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := dataset.Queries[i%len(dataset.Queries)]
		_, err := client.Search(ctx, query, 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}

// TestWorkerCountImpact tests different worker counts
func TestWorkerCountImpact(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	t.Log("")
	t.Log("Note: Worker count is currently hardcoded to 4 in enterprise_hierarchical.go:152")
	t.Log("To test different worker counts, modify that value and re-run.")
	t.Log("")

	// Just run the timing breakdown to show current performance
	t.Run("Timing", TestSearchTimingBreakdown)
}
