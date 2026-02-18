//go:build integration

package client

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestEnterpriseBenchmarkComplete runs a comprehensive benchmark showing all components
func TestEnterpriseBenchmarkComplete(t *testing.T) {
	ctx := context.Background()

	// Configuration
	dimensions := []int{64, 128, 256}
	vectorCounts := []int{1000, 5000, 10000}

	fmt.Println("=" + "=================================================================")
	fmt.Println("TIER 2.5 ENTERPRISE HIERARCHICAL SEARCH - COMPLETE BENCHMARK")
	fmt.Println("=" + "=================================================================")
	fmt.Println()

	for _, dim := range dimensions {
		for _, numVectors := range vectorCounts {
			runEnterpriseBenchmark(t, ctx, dim, numVectors)
		}
	}
}

func runEnterpriseBenchmark(t *testing.T, ctx context.Context, dimension, numVectors int) {
	numSuperBuckets := 32
	numQueries := 10
	topK := 10

	fmt.Printf("\n--- Dimension: %d, Vectors: %d ---\n", dimension, numVectors)

	// =========================================
	// PHASE 1: Enterprise Setup
	// =========================================
	fmt.Println("\n[1] ENTERPRISE SETUP")

	startSetup := time.Now()
	enterpriseCfg, err := enterprise.NewConfig("benchmark-enterprise", dimension, numSuperBuckets)
	if err != nil {
		t.Fatalf("Failed to create enterprise config: %v", err)
	}
	setupTime := time.Since(startSetup)

	fmt.Printf("  ✓ Created enterprise config: %v\n", setupTime)
	fmt.Printf("    - AES Key: %d bytes (AES-256)\n", len(enterpriseCfg.AESKey))
	fmt.Printf("    - LSH Seed: %d bytes\n", len(enterpriseCfg.LSHSeed))
	fmt.Printf("    - Sub-LSH Bits: %d\n", enterpriseCfg.GetSubLSHBits())
	fmt.Printf("    - NumSubBuckets: %d\n", enterpriseCfg.NumSubBuckets)

	// =========================================
	// PHASE 2: Generate Test Vectors
	// =========================================
	fmt.Println("\n[2] VECTOR GENERATION")

	startGen := time.Now()
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
	genTime := time.Since(startGen)
	fmt.Printf("  ✓ Generated %d vectors: %v\n", numVectors, genTime)

	// =========================================
	// PHASE 3: Build Index (EnterpriseBuilder)
	// =========================================
	fmt.Println("\n[3] INDEX BUILDING (EnterpriseBuilder)")

	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = 8
	cfg.SubBucketsPerSuper = 4
	cfg.NumDecoys = 8

	startBuild := time.Now()
	builder, err := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	if err != nil {
		t.Fatalf("Failed to create builder: %v", err)
	}

	_, err = builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}
	buildTime := time.Since(startBuild)
	enterpriseCfg = builder.GetEnterpriseConfig()

	fmt.Printf("  ✓ Built hierarchical index: %v\n", buildTime)
	fmt.Printf("    - Super-buckets: %d\n", numSuperBuckets)
	fmt.Printf("    - Sub-buckets: %d\n", enterpriseCfg.NumSubBuckets)
	fmt.Printf("    - Vectors per bucket (avg): %.1f\n", float64(numVectors)/float64(numSuperBuckets))
	fmt.Printf("    - AES encryption: included\n")
	fmt.Printf("    - Centroid computation: included\n")

	// =========================================
	// PHASE 4: Authentication
	// =========================================
	fmt.Println("\n[4] AUTHENTICATION (Token-Based)")

	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)

	authCfg := auth.DefaultServiceConfig()
	authService := auth.NewService(authCfg, enterpriseStore)

	startAuth := time.Now()
	authService.RegisterUser(ctx, "benchmark-user", "benchmark-enterprise", []byte("password"), []string{auth.ScopeSearch})
	creds, err := authService.Authenticate(ctx, "benchmark-user", []byte("password"))
	if err != nil {
		t.Fatalf("Authentication failed: %v", err)
	}
	authTime := time.Since(startAuth)

	fmt.Printf("  ✓ User authenticated: %v\n", authTime)
	fmt.Printf("    - Token TTL: %v\n", authCfg.TokenTTL)
	fmt.Printf("    - LSH Hyperplanes distributed: %d planes\n", len(creds.LSHHyperplanes))
	fmt.Printf("    - Centroids distributed: %d\n", len(creds.Centroids))

	// =========================================
	// PHASE 5: Create Enterprise Client
	// =========================================
	fmt.Println("\n[5] CLIENT INITIALIZATION")

	startClient := time.Now()
	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	clientTime := time.Since(startClient)

	fmt.Printf("  ✓ Client initialized: %v\n", clientTime)
	fmt.Printf("    - HE Engine: BFV scheme\n")
	fmt.Printf("    - AES Decryptor: AES-256-GCM\n")
	fmt.Printf("    - LSH Hasher: %d-bit\n", len(creds.LSHHyperplanes))

	// =========================================
	// PHASE 6: Run Search Queries
	// =========================================
	fmt.Println("\n[6] SEARCH QUERIES")
	fmt.Printf("  Running %d queries (top-%d)...\n", numQueries, topK)

	var totalHEEncrypt, totalHEScore, totalHEDecrypt time.Duration
	var totalBucketSelect, totalBucketFetch, totalAESDecrypt, totalLocalScore time.Duration
	var totalSearch time.Duration
	var totalVectorsScored, totalBlobsFetched int

	for q := 0; q < numQueries; q++ {
		queryIdx := rng.Intn(numVectors)
		query := vectors[queryIdx]

		result, err := client.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		totalHEEncrypt += result.Timing.HEEncryptQuery
		totalHEScore += result.Timing.HECentroidScores
		totalHEDecrypt += result.Timing.HEDecryptScores
		totalBucketSelect += result.Timing.BucketSelection
		totalBucketFetch += result.Timing.BucketFetch
		totalAESDecrypt += result.Timing.AESDecrypt
		totalLocalScore += result.Timing.LocalScoring
		totalSearch += result.Timing.Total
		totalVectorsScored += result.Stats.VectorsScored
		totalBlobsFetched += result.Stats.BlobsFetched
	}

	// Print timing breakdown
	fmt.Println("\n  TIMING BREAKDOWN (average per query):")
	fmt.Println("  ┌────────────────────────────────────────────────────────┐")
	fmt.Printf("  │ HE Query Encryption:     %12v                 │\n", totalHEEncrypt/time.Duration(numQueries))
	fmt.Printf("  │ HE Centroid Scoring:     %12v  (server-side)   │\n", totalHEScore/time.Duration(numQueries))
	fmt.Printf("  │ HE Score Decryption:     %12v  (client-side)   │\n", totalHEDecrypt/time.Duration(numQueries))
	fmt.Printf("  │ Bucket Selection:        %12v                 │\n", totalBucketSelect/time.Duration(numQueries))
	fmt.Printf("  │ Bucket Fetch (w/decoys): %12v                 │\n", totalBucketFetch/time.Duration(numQueries))
	fmt.Printf("  │ AES Decryption:          %12v  (client-side)   │\n", totalAESDecrypt/time.Duration(numQueries))
	fmt.Printf("  │ Local Scoring:           %12v  (client-side)   │\n", totalLocalScore/time.Duration(numQueries))
	fmt.Println("  ├────────────────────────────────────────────────────────┤")
	fmt.Printf("  │ TOTAL:                   %12v                 │\n", totalSearch/time.Duration(numQueries))
	fmt.Println("  └────────────────────────────────────────────────────────┘")

	fmt.Println("\n  SEARCH STATISTICS (average):")
	fmt.Printf("    - Vectors scored per query: %d\n", totalVectorsScored/numQueries)
	fmt.Printf("    - Blobs fetched per query: %d\n", totalBlobsFetched/numQueries)
	fmt.Printf("    - Throughput: %.2f queries/sec\n", float64(numQueries)/totalSearch.Seconds())

	// =========================================
	// PHASE 7: Security Summary
	// =========================================
	fmt.Println("\n[7] SECURITY PROPERTIES")
	fmt.Println("  ┌────────────────────────────────────────────────────────┐")
	fmt.Println("  │ What SERVER sees:                                     │")
	fmt.Println("  │   ✓ HE-encrypted query ciphertext                     │")
	fmt.Println("  │   ✓ HE-encrypted centroid scores (computed blindly)   │")
	fmt.Println("  │   ✓ Bucket access pattern (real + decoys mixed)       │")
	fmt.Println("  │   ✓ AES-encrypted vectors (opaque ciphertext)         │")
	fmt.Println("  ├────────────────────────────────────────────────────────┤")
	fmt.Println("  │ What SERVER does NOT see:                             │")
	fmt.Println("  │   ✗ Plaintext query vector                            │")
	fmt.Println("  │   ✗ Plaintext centroid scores                         │")
	fmt.Println("  │   ✗ Which super-buckets were selected                 │")
	fmt.Println("  │   ✗ Which buckets are real vs decoy                   │")
	fmt.Println("  │   ✗ Plaintext vectors                                 │")
	fmt.Println("  │   ✗ Final similarity scores                           │")
	fmt.Println("  │   ✗ Which results were returned to user               │")
	fmt.Println("  └────────────────────────────────────────────────────────┘")
}
