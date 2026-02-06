// Enterprise Hierarchical Private Search Demo
//
// This demonstrates the complete enterprise workflow for Tier 2.5:
//
//  1. Enterprise Setup: Create enterprise config and build index
//  2. User Registration: Register users for the enterprise
//  3. Authentication: User authenticates and receives credentials
//  4. Client Creation: Create enterprise client with credentials
//  5. Private Search: Perform hierarchical private search
//  6. Token Refresh: Refresh token before expiry
//
// Key security properties:
//   - Per-enterprise AES keys (data isolation)
//   - Per-enterprise LSH hyperplanes (query pattern privacy)
//   - HE centroid scoring (query privacy)
//   - Decoy buckets (access pattern privacy)
//   - Token-based authentication (access control)
//
// Run: go run ./examples/enterprise/main.go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  Enterprise Hierarchical Private Search Demo")
	fmt.Println("  Tier 2.5: Per-Enterprise Security with Token Authentication")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Configuration
	numVectors := 50000
	dimension := 128
	numSuperBuckets := 64
	numQueries := 5
	topK := 10

	fmt.Println("Configuration:")
	fmt.Printf("  Vectors: %d\n", numVectors)
	fmt.Printf("  Dimension: %d\n", dimension)
	fmt.Printf("  Super-buckets: %d\n", numSuperBuckets)
	fmt.Printf("  Queries: %d\n", numQueries)
	fmt.Printf("  Top-K: %d\n", topK)
	fmt.Println()

	// =========================================================
	// PHASE 1: Enterprise Admin Setup
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("PHASE 1: Enterprise Admin Setup")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	// Generate vector data
	fmt.Print("Generating vectors... ")
	start := time.Now()
	ids, vectors := generateVectors(rng, numVectors, dimension)
	fmt.Printf("done (%v)\n", time.Since(start))

	// Build enterprise index
	fmt.Print("Building enterprise index... ")
	start = time.Now()

	blobStore := blob.NewMemoryStore()
	idx, enterpriseCfg, err := hierarchical.BuildEnterpriseIndex(
		ctx,
		"acme-corporation",
		dimension,
		numSuperBuckets,
		ids,
		vectors,
		blobStore,
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	buildTime := time.Since(start)
	fmt.Printf("done (%v)\n", buildTime)

	// Print enterprise config
	fmt.Println()
	fmt.Println("Enterprise Configuration:")
	fmt.Printf("  Enterprise ID: %s\n", enterpriseCfg.EnterpriseID)
	fmt.Printf("  AES Key: %d bytes (AES-256)\n", len(enterpriseCfg.AESKey))
	fmt.Printf("  LSH Seed: %d bytes (secret, per-enterprise)\n", len(enterpriseCfg.LSHSeed))
	fmt.Printf("  Centroids: %d computed\n", len(enterpriseCfg.Centroids))
	fmt.Printf("  Created: %v\n", enterpriseCfg.CreatedAt.Format(time.RFC3339))
	fmt.Println()

	// Print index stats
	stats := idx.GetStats()
	fmt.Println("Index Statistics:")
	fmt.Printf("  Total vectors: %d\n", stats.TotalVectors)
	fmt.Printf("  Super-buckets used: %d/%d\n", stats.NumSuperBuckets, numSuperBuckets)
	fmt.Printf("  Sub-buckets used: %d\n", stats.NumSubBuckets)
	fmt.Printf("  Avg vectors per sub-bucket: %.1f\n", stats.AvgVectorsPerSub)
	fmt.Println()

	// Store enterprise config (simulating secure storage)
	configStore := enterprise.NewMemoryStore()
	if err := configStore.Put(ctx, enterpriseCfg); err != nil {
		fmt.Printf("Error storing config: %v\n", err)
		return
	}
	fmt.Println("Enterprise config stored securely")
	fmt.Println()

	// =========================================================
	// PHASE 2: User Registration (Admin action)
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("PHASE 2: User Registration")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	authService := auth.NewService(auth.DefaultServiceConfig(), configStore)

	// Register users
	users := []struct {
		id       string
		password string
		scopes   []string
	}{
		{"alice", "password123", []string{auth.ScopeSearch}},
		{"bob", "secure456", []string{auth.ScopeSearch, auth.ScopeIndex}},
		{"admin", "adminpass", []string{auth.ScopeAdmin}},
	}

	for _, u := range users {
		err := authService.RegisterUser(ctx, u.id, "acme-corporation", []byte(u.password), u.scopes)
		if err != nil {
			fmt.Printf("Error registering %s: %v\n", u.id, err)
			return
		}
		fmt.Printf("  Registered user: %s (scopes: %v)\n", u.id, u.scopes)
	}
	fmt.Println()

	// =========================================================
	// PHASE 3: User Authentication (Client action)
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("PHASE 3: User Authentication")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	fmt.Print("Alice authenticating... ")
	start = time.Now()
	credentials, err := authService.Authenticate(ctx, "alice", []byte("password123"))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Println()
	fmt.Println("Credentials Received:")
	fmt.Printf("  Token: %s...%s\n", credentials.Token[:8], credentials.Token[len(credentials.Token)-8:])
	fmt.Printf("  Expires: %v\n", credentials.TokenExpiry.Format(time.RFC3339))
	fmt.Printf("  AES Key: %d bytes\n", len(credentials.AESKey))
	fmt.Printf("  LSH Hyperplanes: %d planes x %d dims\n",
		len(credentials.LSHHyperplanes), len(credentials.LSHHyperplanes[0]))
	fmt.Printf("  Centroids: %d\n", len(credentials.Centroids))
	fmt.Printf("  Enterprise: %s\n", credentials.EnterpriseID)
	fmt.Println()

	// =========================================================
	// PHASE 4: Create Enterprise Client
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("PHASE 4: Create Enterprise Client")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	clientCfg := hierarchical.Config{
		Dimension:          credentials.Dimension,
		NumSuperBuckets:    credentials.NumSuperBuckets,
		NumSubBuckets:      16,
		TopSuperBuckets:    8,
		SubBucketsPerSuper: 2,
		NumDecoys:          8,
	}

	fmt.Println("Client Configuration:")
	fmt.Printf("  Top super-buckets to select: %d\n", clientCfg.TopSuperBuckets)
	fmt.Printf("  Sub-buckets per super: %d\n", clientCfg.SubBucketsPerSuper)
	fmt.Printf("  Decoy buckets: %d\n", clientCfg.NumDecoys)
	fmt.Println()

	fmt.Print("Creating enterprise client... ")
	start = time.Now()
	enterpriseClient, err := client.NewEnterpriseHierarchicalClient(
		clientCfg,
		credentials,
		blobStore,
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("done (%v)\n", time.Since(start))
	fmt.Println()

	// =========================================================
	// PHASE 5: Perform Private Searches
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("PHASE 5: Perform Private Searches")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	var totalTime time.Duration
	var totalHETime time.Duration
	var totalFetchTime time.Duration
	var totalScoreTime time.Duration
	correctResults := 0

	for q := 0; q < numQueries; q++ {
		// Create query similar to a random vector
		targetIdx := rng.Intn(numVectors)
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		// Add small noise
		for i := range query {
			query[i] += (rng.Float64() - 0.5) * 0.02
		}

		fmt.Printf("Query %d (target: doc-%d):\n", q+1, targetIdx)

		result, err := enterpriseClient.Search(ctx, query, topK)
		if err != nil {
			fmt.Printf("  Error: %v\n", err)
			continue
		}

		totalTime += result.Timing.Total
		totalHETime += result.Timing.HECentroidScores
		totalFetchTime += result.Timing.BucketFetch
		totalScoreTime += result.Timing.LocalScoring

		// Print timing breakdown
		fmt.Printf("  Level 1 (HE centroid scoring): %v\n", result.Timing.HECentroidScores)
		fmt.Printf("  Level 2 (Sub-bucket fetch):    %v\n", result.Timing.BucketFetch)
		fmt.Printf("  Level 3 (Local scoring):       %v\n", result.Timing.LocalScoring)
		fmt.Printf("  Total:                         %v\n", result.Timing.Total)

		// Print stats
		fmt.Printf("  HE operations: %d\n", result.Stats.HEOperations)
		fmt.Printf("  Real buckets: %d, Decoy buckets: %d\n",
			result.Stats.RealSubBuckets, result.Stats.DecoySubBuckets)
		fmt.Printf("  Vectors scored: %d\n", result.Stats.VectorsScored)

		// Check if target found
		targetID := fmt.Sprintf("doc-%d", targetIdx)
		found := false
		for i, r := range result.Results {
			if r.ID == targetID {
				found = true
				correctResults++
				fmt.Printf("  Target found at position %d (score: %.4f)\n", i+1, r.Score)
				break
			}
		}
		if !found {
			fmt.Printf("  Target not in top %d\n", topK)
		}
		fmt.Println()
	}

	// =========================================================
	// RESULTS SUMMARY
	// =========================================================
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Results Summary")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	avgTime := totalTime / time.Duration(numQueries)
	avgHETime := totalHETime / time.Duration(numQueries)
	avgFetchTime := totalFetchTime / time.Duration(numQueries)
	avgScoreTime := totalScoreTime / time.Duration(numQueries)
	recall := float64(correctResults) / float64(numQueries) * 100

	fmt.Println("Average Timing Breakdown:")
	fmt.Printf("  Level 1 (HE): %v (%.1f%%)\n", avgHETime, float64(avgHETime)/float64(avgTime)*100)
	fmt.Printf("  Level 2 (Fetch): %v (%.1f%%)\n", avgFetchTime, float64(avgFetchTime)/float64(avgTime)*100)
	fmt.Printf("  Level 3 (Score): %v (%.1f%%)\n", avgScoreTime, float64(avgScoreTime)/float64(avgTime)*100)
	fmt.Printf("  Total: %v\n", avgTime)
	fmt.Println()

	fmt.Printf("Recall@%d: %.1f%% (%d/%d)\n", topK, recall, correctResults, numQueries)
	fmt.Println()

	// Performance comparison
	naiveHEOps := numVectors
	actualHEOps := numSuperBuckets
	speedup := float64(naiveHEOps) / float64(actualHEOps)

	fmt.Println("Performance Analysis:")
	fmt.Printf("  HE operations: %d (vs %d for naive Tier 1)\n", actualHEOps, naiveHEOps)
	fmt.Printf("  Speedup: %.0fx fewer HE operations\n", speedup)
	fmt.Println()

	// Estimate naive Tier 1 time (~33ms per HE op)
	naiveTime := time.Duration(naiveHEOps) * 33 * time.Millisecond
	fmt.Println("Estimated Performance:")
	fmt.Printf("  Hierarchical (this demo): %v\n", avgTime)
	fmt.Printf("  Naive Tier 1 (all HE):    ~%v\n", naiveTime)
	fmt.Printf("  Time savings:             ~%.0fx faster\n", float64(naiveTime)/float64(avgTime))
	fmt.Println()

	// =========================================================
	// SECURITY SUMMARY
	// =========================================================
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Security Guarantees")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	fmt.Println("Enterprise Isolation:")
	fmt.Println("  - Each enterprise has unique AES-256 key")
	fmt.Println("  - Each enterprise has unique LSH hyperplanes")
	fmt.Println("  - No cross-enterprise data access possible")
	fmt.Println()

	fmt.Println("Query Privacy:")
	fmt.Println("  - Query vector: hidden from server (HE encryption)")
	fmt.Println("  - Super-bucket selection: hidden from server (client-side decrypt)")
	fmt.Println("  - Sub-bucket interest: hidden from server (decoy buckets)")
	fmt.Println()

	fmt.Println("Data Privacy:")
	fmt.Println("  - Vector values: hidden from storage (AES-256-GCM)")
	fmt.Println("  - Final scores: hidden from everyone (local computation)")
	fmt.Println()

	fmt.Println("Access Control:")
	fmt.Printf("  - Token-based authentication (TTL: %v)\n", auth.DefaultServiceConfig().TokenTTL)
	fmt.Println("  - Scope-based authorization (search, index, admin)")
	fmt.Println("  - Token refresh within window")
	fmt.Println("  - Token revocation support")
	fmt.Println()

	// =========================================================
	// TOKEN STATUS
	// =========================================================
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("Token Status")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	if enterpriseClient.IsTokenExpired() {
		fmt.Println("Token Status: EXPIRED (would need refresh)")
	} else {
		fmt.Printf("Token Status: VALID (expires in %v)\n", enterpriseClient.TimeUntilTokenExpiry())
	}
	fmt.Println()
}

func generateVectors(rng *rand.Rand, count, dimension int) ([]string, [][]float64) {
	ids := make([]string, count)
	vectors := make([][]float64, count)

	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("doc-%d", i)
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1 // [-1, 1]
		}
		vectors[i] = vec
	}

	return ids, vectors
}
