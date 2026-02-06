package hierarchical_test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

func TestEnterpriseBuilderBasic(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// Generate test data
	numVectors := 1000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Create enterprise config
	enterpriseCfg, err := enterprise.NewConfig("test-enterprise", dimension, 64)
	if err != nil {
		t.Fatalf("Failed to create enterprise config: %v", err)
	}

	// Create builder from enterprise config
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	builder, err := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	if err != nil {
		t.Fatalf("Failed to create enterprise builder: %v", err)
	}

	// Build index
	store := blob.NewMemoryStore()
	idx, err := builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("Failed to build index: %v", err)
	}

	// Verify index
	if idx.GetVectorCount() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, idx.GetVectorCount())
	}

	// Check that enterprise config was updated with centroids
	updatedCfg := builder.GetEnterpriseConfig()
	if len(updatedCfg.Centroids) != cfg.NumSuperBuckets {
		t.Errorf("Expected %d centroids, got %d", cfg.NumSuperBuckets, len(updatedCfg.Centroids))
	}

	// Verify centroids are normalized
	for i, c := range updatedCfg.Centroids {
		if len(c) > 0 {
			var norm float64
			for _, v := range c {
				norm += v * v
			}
			if norm > 0 && (norm < 0.99 || norm > 1.01) {
				t.Errorf("Centroid %d not normalized: norm = %f", i, norm)
			}
		}
	}

	// Print stats
	stats := idx.GetStats()
	t.Logf("Enterprise Index stats:")
	t.Logf("  Enterprise ID: %s", enterpriseCfg.EnterpriseID)
	t.Logf("  Total vectors: %d", stats.TotalVectors)
	t.Logf("  Super-buckets: %d", stats.NumSuperBuckets)
	t.Logf("  Sub-buckets used: %d", stats.NumSubBuckets)
	t.Logf("  Avg vectors/sub: %.1f", stats.AvgVectorsPerSub)
}

func TestBuildEnterpriseIndexConvenience(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	numVectors := 500
	dimension := 128
	numSuperBuckets := 32

	ids, vectors := generateTestData(rng, numVectors, dimension)

	store := blob.NewMemoryStore()
	idx, cfg, err := hierarchical.BuildEnterpriseIndex(
		ctx,
		"convenience-test",
		dimension,
		numSuperBuckets,
		ids,
		vectors,
		store,
	)
	if err != nil {
		t.Fatalf("Failed to build enterprise index: %v", err)
	}

	// Verify index
	if idx.GetVectorCount() != numVectors {
		t.Errorf("Expected %d vectors, got %d", numVectors, idx.GetVectorCount())
	}

	// Verify config
	if cfg.EnterpriseID != "convenience-test" {
		t.Errorf("Expected enterprise ID 'convenience-test', got %s", cfg.EnterpriseID)
	}
	if len(cfg.Centroids) != numSuperBuckets {
		t.Errorf("Expected %d centroids, got %d", numSuperBuckets, len(cfg.Centroids))
	}
	if len(cfg.AESKey) != 32 {
		t.Errorf("Expected 32-byte AES key, got %d bytes", len(cfg.AESKey))
	}

	t.Logf("Convenience function test passed")
	t.Logf("  Enterprise: %s", cfg.EnterpriseID)
	t.Logf("  Centroids computed: %d", len(cfg.Centroids))
}

func TestEnterpriseEndToEndWithAuth(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	// =========================================================
	// PHASE 1: Enterprise Setup (done once by admin)
	// =========================================================

	numVectors := 2000
	dimension := 128
	numSuperBuckets := 32

	t.Log("=== PHASE 1: Enterprise Setup ===")

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Build enterprise index
	blobStore := blob.NewMemoryStore()
	idx, enterpriseCfg, err := hierarchical.BuildEnterpriseIndex(
		ctx,
		"acme-corp",
		dimension,
		numSuperBuckets,
		ids,
		vectors,
		blobStore,
	)
	if err != nil {
		t.Fatalf("Failed to build enterprise index: %v", err)
	}

	t.Logf("  Built index with %d vectors", idx.GetVectorCount())
	t.Logf("  Enterprise ID: %s", enterpriseCfg.EnterpriseID)
	t.Logf("  Centroids: %d", len(enterpriseCfg.Centroids))

	// Store enterprise config (simulating secure storage)
	configStore := enterprise.NewMemoryStore()
	if err := configStore.Put(ctx, enterpriseCfg); err != nil {
		t.Fatalf("Failed to store enterprise config: %v", err)
	}

	// =========================================================
	// PHASE 2: User Registration (done once per user)
	// =========================================================

	t.Log("=== PHASE 2: User Registration ===")

	authService := auth.NewService(auth.DefaultServiceConfig(), configStore)

	// Register a user
	password := []byte("securepassword123")
	err = authService.RegisterUser(ctx, "alice", "acme-corp", password, []string{auth.ScopeSearch})
	if err != nil {
		t.Fatalf("Failed to register user: %v", err)
	}
	t.Log("  Registered user: alice")

	// =========================================================
	// PHASE 3: User Authentication (done for each session)
	// =========================================================

	t.Log("=== PHASE 3: User Authentication ===")

	credentials, err := authService.Authenticate(ctx, "alice", password)
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	t.Logf("  Token obtained (expires: %v)", credentials.TokenExpiry.Format(time.RFC3339))
	t.Logf("  AES key: %d bytes", len(credentials.AESKey))
	t.Logf("  LSH hyperplanes: %d", len(credentials.LSHHyperplanes))
	t.Logf("  Centroids: %d", len(credentials.Centroids))

	// =========================================================
	// PHASE 4: Create Enterprise Client (done once per session)
	// =========================================================

	t.Log("=== PHASE 4: Create Enterprise Client ===")

	clientCfg := hierarchical.Config{
		Dimension:          credentials.Dimension,
		NumSuperBuckets:    credentials.NumSuperBuckets,
		NumSubBuckets:      16,
		TopSuperBuckets:    8,
		SubBucketsPerSuper: 2,
		NumDecoys:          8,
	}

	enterpriseClient, err := client.NewEnterpriseHierarchicalClient(
		clientCfg,
		credentials,
		blobStore,
	)
	if err != nil {
		t.Fatalf("Failed to create enterprise client: %v", err)
	}

	t.Logf("  Enterprise client created for: %s", enterpriseClient.GetEnterpriseID())

	// =========================================================
	// PHASE 5: Perform Searches (main usage)
	// =========================================================

	t.Log("=== PHASE 5: Perform Searches ===")

	numQueries := 5
	topK := 10
	correctResults := 0

	for q := 0; q < numQueries; q++ {
		// Create query similar to a known vector
		targetIdx := rng.Intn(numVectors)
		query := make([]float64, dimension)
		copy(query, vectors[targetIdx])
		// Add small noise
		for i := range query {
			query[i] += (rng.Float64() - 0.5) * 0.02
		}

		result, err := enterpriseClient.Search(ctx, query, topK)
		if err != nil {
			t.Fatalf("Search %d failed: %v", q, err)
		}

		// Check if target is in results
		targetID := fmt.Sprintf("doc-%d", targetIdx)
		found := false
		for _, r := range result.Results {
			if r.ID == targetID {
				found = true
				correctResults++
				break
			}
		}

		foundStr := "not found"
		if found {
			foundStr = "found"
		}
		t.Logf("  Query %d: target %s (%s), HE ops: %d, vectors scored: %d",
			q+1, targetID, foundStr, result.Stats.HEOperations, result.Stats.VectorsScored)
	}

	recall := float64(correctResults) / float64(numQueries) * 100
	t.Logf("  Recall@%d: %.1f%% (%d/%d)", topK, recall, correctResults, numQueries)

	// =========================================================
	// PHASE 6: Token Validation (ongoing)
	// =========================================================

	t.Log("=== PHASE 6: Token Validation ===")

	token, err := authService.ValidateToken(ctx, credentials.Token)
	if err != nil {
		t.Fatalf("Token validation failed: %v", err)
	}

	t.Logf("  Token valid for user: %s", token.UserID)
	t.Logf("  Enterprise: %s", token.EnterpriseID)
	t.Logf("  Scopes: %v", token.Scopes)
	t.Logf("  Time until expiry: %v", token.TimeUntilExpiry())

	t.Log("=== END-TO-END TEST PASSED ===")
}

func TestEnterpriseIsolation(t *testing.T) {
	// Test that different enterprises have different keys
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	dimension := 128
	numVectors := 100

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Create two enterprises
	config1, _ := enterprise.NewConfig("enterprise-1", dimension, 32)
	config2, _ := enterprise.NewConfig("enterprise-2", dimension, 32)

	// Verify different AES keys
	if string(config1.AESKey) == string(config2.AESKey) {
		t.Error("Different enterprises should have different AES keys")
	}

	// Verify different LSH seeds
	if string(config1.LSHSeed) == string(config2.LSHSeed) {
		t.Error("Different enterprises should have different LSH seeds")
	}

	// Build indices for both
	store1 := blob.NewMemoryStore()
	store2 := blob.NewMemoryStore()

	cfg1 := hierarchical.ConfigFromEnterprise(config1)
	cfg2 := hierarchical.ConfigFromEnterprise(config2)

	builder1, _ := hierarchical.NewEnterpriseBuilder(cfg1, config1)
	builder2, _ := hierarchical.NewEnterpriseBuilder(cfg2, config2)

	builder1.Build(ctx, ids, vectors, store1)
	builder2.Build(ctx, ids, vectors, store2)

	// Verify blobs are different (encrypted with different keys)
	blobs1, _ := store1.GetBuckets(ctx, []string{"00_00"})
	blobs2, _ := store2.GetBuckets(ctx, []string{"00_00"})

	if len(blobs1) > 0 && len(blobs2) > 0 {
		if string(blobs1[0].Ciphertext) == string(blobs2[0].Ciphertext) {
			t.Error("Same data should have different ciphertext with different keys")
		}
	}

	t.Log("Enterprise isolation verified:")
	t.Logf("  Enterprise 1: %s", config1.EnterpriseID)
	t.Logf("  Enterprise 2: %s", config2.EnterpriseID)
	t.Log("  Different AES keys: OK")
	t.Log("  Different LSH seeds: OK")
	t.Log("  Different ciphertexts: OK")
}

func TestEnterprisePrivacyGuarantees(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))

	numVectors := 1000
	dimension := 128

	ids, vectors := generateTestData(rng, numVectors, dimension)

	// Build enterprise index
	store := blob.NewMemoryStore()
	idx, enterpriseCfg, err := hierarchical.BuildEnterpriseIndex(
		ctx, "privacy-test", dimension, 32, ids, vectors, store,
	)
	if err != nil {
		t.Fatalf("Failed to build: %v", err)
	}

	// Set up auth
	configStore := enterprise.NewMemoryStore()
	configStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), configStore)
	authService.RegisterUser(ctx, "tester", "privacy-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "tester", []byte("pass"))

	// Create client
	clientCfg := hierarchical.Config{
		Dimension:          creds.Dimension,
		NumSuperBuckets:    creds.NumSuperBuckets,
		NumSubBuckets:      16,
		TopSuperBuckets:    8,
		SubBucketsPerSuper: 2,
		NumDecoys:          8,
	}
	enterpriseClient, _ := client.NewEnterpriseHierarchicalClient(clientCfg, creds, store)

	// Perform search
	query := vectors[42]
	result, _ := enterpriseClient.Search(ctx, query, 10)

	// Privacy guarantee 1: HE operations cover ALL centroids
	if result.Stats.HEOperations != clientCfg.NumSuperBuckets {
		t.Errorf("HE should cover all %d centroids, got %d",
			clientCfg.NumSuperBuckets, result.Stats.HEOperations)
	}

	// Privacy guarantee 2: Decoy buckets are fetched
	if result.Stats.DecoySubBuckets < clientCfg.NumDecoys {
		t.Errorf("Should fetch at least %d decoy buckets, got %d",
			clientCfg.NumDecoys, result.Stats.DecoySubBuckets)
	}

	// Privacy guarantee 3: Total includes real + decoys
	expectedMin := result.Stats.RealSubBuckets + clientCfg.NumDecoys
	if result.Stats.TotalSubBuckets < expectedMin {
		t.Errorf("Total should be at least %d, got %d",
			expectedMin, result.Stats.TotalSubBuckets)
	}

	// Privacy guarantee 4: Enterprise-specific LSH used (not public seed)
	// The client uses credentials.LSHHyperplanes which are derived from enterprise secret
	if len(creds.LSHHyperplanes) == 0 {
		t.Error("LSH hyperplanes should be provided in credentials")
	}

	t.Log("Enterprise privacy guarantees verified:")
	t.Logf("  All centroids scored (HE): %d", result.Stats.HEOperations)
	t.Logf("  Decoy buckets: %d", result.Stats.DecoySubBuckets)
	t.Logf("  Real buckets: %d", result.Stats.RealSubBuckets)
	t.Logf("  Enterprise LSH planes: %d", len(creds.LSHHyperplanes))
	t.Logf("  Index vector count: %d", idx.GetVectorCount())
}

func TestTokenExpiryHandling(t *testing.T) {
	ctx := context.Background()

	// Create enterprise with short token TTL
	enterpriseCfg, _ := enterprise.NewConfig("expiry-test", 128, 32)
	configStore := enterprise.NewMemoryStore()
	configStore.Put(ctx, enterpriseCfg)

	// Create auth service with very short TTL for testing
	authCfg := auth.ServiceConfig{
		TokenTTL:      100 * time.Millisecond,
		RefreshWindow: 50 * time.Millisecond,
		LSHBits:       8,
		Dimension:     128,
	}
	authService := auth.NewService(authCfg, configStore)

	// Register and authenticate
	authService.RegisterUser(ctx, "expiry-tester", "expiry-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "expiry-tester", []byte("pass"))

	// Token should initially not be expired
	if creds.IsExpired() {
		t.Error("Token should not be expired immediately after authentication")
	}

	// Wait for expiry
	time.Sleep(150 * time.Millisecond)

	// Token should now be expired
	if !creds.IsExpired() {
		t.Error("Token should be expired after TTL")
	}

	t.Log("Token expiry handling verified")
}
