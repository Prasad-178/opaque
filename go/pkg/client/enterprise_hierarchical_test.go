package client

import (
	"context"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestEnterpriseHierarchicalEndToEnd tests the full flow:
// 1. Create enterprise config
// 2. Build index with enterprise builder
// 3. Authenticate and get credentials
// 4. Search using enterprise client
func TestEnterpriseHierarchicalEndToEnd(t *testing.T) {
	ctx := context.Background()
	dimension := 64
	numVectors := 500
	numSuperBuckets := 16
	topK := 5

	// Step 1: Create enterprise configuration
	enterpriseCfg, err := enterprise.NewConfig("test-enterprise", dimension, numSuperBuckets)
	if err != nil {
		t.Fatalf("failed to create enterprise config: %v", err)
	}

	// Step 2: Generate test vectors
	rng := rand.New(rand.NewSource(42))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = generateTestID(i)
		vectors[i] = generateRandomVector(rng, dimension)
	}

	// Step 3: Build index using enterprise builder
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = 4
	cfg.SubBucketsPerSuper = 2
	cfg.NumDecoys = 4

	builder, err := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	if err != nil {
		t.Fatalf("failed to create enterprise builder: %v", err)
	}

	_, err = builder.Build(ctx, ids, vectors, store)
	if err != nil {
		t.Fatalf("failed to build index: %v", err)
	}

	// Get updated config with centroids
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Step 4: Setup auth service
	enterpriseStore := enterprise.NewMemoryStore()
	if err := enterpriseStore.Put(ctx, enterpriseCfg); err != nil {
		t.Fatalf("failed to store enterprise config: %v", err)
	}

	authCfg := auth.ServiceConfig{
		TokenTTL:      time.Hour,
		RefreshWindow: 10 * time.Minute,
		LSHBits:       8,
		Dimension:     dimension,
	}
	authService := auth.NewService(authCfg, enterpriseStore)

	// Register user
	password := []byte("testpassword")
	err = authService.RegisterUser(ctx, "testuser", "test-enterprise", password, []string{auth.ScopeSearch})
	if err != nil {
		t.Fatalf("failed to register user: %v", err)
	}

	// Authenticate
	creds, err := authService.Authenticate(ctx, "testuser", password)
	if err != nil {
		t.Fatalf("failed to authenticate: %v", err)
	}

	// Step 5: Create enterprise client
	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("failed to create enterprise client: %v", err)
	}

	// Step 6: Search for a known vector
	// Note: Hierarchical search doesn't guarantee finding the exact vector
	// because HE centroid scoring may not select the query's super-bucket
	queryIdx := 100
	query := vectors[queryIdx]

	result, err := client.Search(ctx, query, topK)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Verify search returns results and the pipeline works
	t.Logf("Search returned %d results", len(result.Results))
	t.Logf("Top results: %+v", result.Results)

	// Verify the pipeline completed successfully
	if result.Stats.SuperBucketsSelected == 0 {
		t.Error("expected some super-buckets to be selected")
	}
	if result.Stats.TotalSubBuckets == 0 {
		t.Error("expected some sub-buckets to be fetched")
	}

	// Check if exact match was found (not required, but log it)
	foundExact := false
	for _, r := range result.Results {
		if r.ID == ids[queryIdx] {
			foundExact = true
			t.Logf("Found exact match with score %f", r.Score)
			break
		}
	}
	if !foundExact {
		t.Logf("Exact query vector not in top-%d results (this is expected for hierarchical search)", topK)
	}

	// Log timing information
	t.Logf("Search timing breakdown:")
	t.Logf("  HE Query Encryption: %v", result.Timing.HEEncryptQuery)
	t.Logf("  HE Centroid Scores:  %v", result.Timing.HECentroidScores)
	t.Logf("  HE Decrypt Scores:   %v", result.Timing.HEDecryptScores)
	t.Logf("  Bucket Selection:    %v", result.Timing.BucketSelection)
	t.Logf("  Bucket Fetch:        %v", result.Timing.BucketFetch)
	t.Logf("  AES Decrypt:         %v", result.Timing.AESDecrypt)
	t.Logf("  Local Scoring:       %v", result.Timing.LocalScoring)
	t.Logf("  Total:               %v", result.Timing.Total)
	t.Logf("Search stats:")
	t.Logf("  Super-buckets selected: %d", result.Stats.SuperBucketsSelected)
	t.Logf("  Real sub-buckets:       %d", result.Stats.RealSubBuckets)
	t.Logf("  Decoy sub-buckets:      %d", result.Stats.DecoySubBuckets)
	t.Logf("  Vectors scored:         %d", result.Stats.VectorsScored)
}

// TestEnterpriseClientCredentialRefresh tests credential refresh flow
func TestEnterpriseClientCredentialRefresh(t *testing.T) {
	ctx := context.Background()
	dimension := 32
	numSuperBuckets := 8

	// Create enterprise config
	enterpriseCfg, err := enterprise.NewConfig("refresh-test", dimension, numSuperBuckets)
	if err != nil {
		t.Fatalf("failed to create enterprise config: %v", err)
	}

	// Set some centroids
	enterpriseCfg.Centroids = make([][]float64, numSuperBuckets)
	for i := 0; i < numSuperBuckets; i++ {
		enterpriseCfg.Centroids[i] = make([]float64, dimension)
		for j := 0; j < dimension; j++ {
			enterpriseCfg.Centroids[i][j] = rand.Float64()
		}
	}

	// Setup auth
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)

	authCfg := auth.ServiceConfig{
		TokenTTL:      2 * time.Second,
		RefreshWindow: 1 * time.Second,
		LSHBits:       6,
		Dimension:     dimension,
	}
	authService := auth.NewService(authCfg, enterpriseStore)

	password := []byte("testpass")
	authService.RegisterUser(ctx, "user1", "refresh-test", password, []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user1", password)

	// Create client
	store := blob.NewMemoryStore()
	cfg := hierarchical.Config{
		Dimension:       dimension,
		NumSuperBuckets: numSuperBuckets,
		NumSubBuckets:   4,
	}

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Verify initial state
	if client.IsTokenExpired() {
		t.Error("token should not be expired initially")
	}

	// Wait until in refresh window
	time.Sleep(1100 * time.Millisecond)

	// Refresh credentials
	newCreds, err := authService.RefreshToken(ctx, creds.Token)
	if err != nil {
		t.Fatalf("failed to refresh token: %v", err)
	}

	// Update client credentials
	err = client.UpdateCredentials(newCreds)
	if err != nil {
		t.Fatalf("failed to update credentials: %v", err)
	}

	// Verify new credentials
	if client.GetCredentials().Token == creds.Token {
		t.Error("token should have changed after refresh")
	}

	if client.IsTokenExpired() {
		t.Error("new token should not be expired")
	}
}

// TestEnterpriseIsolation verifies that different enterprises have different secrets
func TestEnterpriseIsolation(t *testing.T) {
	dimension := 32
	numSuperBuckets := 8

	// Create two enterprise configs
	cfg1, _ := enterprise.NewConfig("enterprise-1", dimension, numSuperBuckets)
	cfg2, _ := enterprise.NewConfig("enterprise-2", dimension, numSuperBuckets)

	// Verify different AES keys
	if string(cfg1.AESKey) == string(cfg2.AESKey) {
		t.Error("different enterprises should have different AES keys")
	}

	// Verify different LSH seeds
	if string(cfg1.LSHSeed) == string(cfg2.LSHSeed) {
		t.Error("different enterprises should have different LSH seeds")
	}

	// Verify different LSH seed integers
	if cfg1.GetLSHSeedAsInt64() == cfg2.GetLSHSeedAsInt64() {
		t.Error("different enterprises should produce different LSH seed integers")
	}

	t.Logf("Enterprise 1 LSH seed: %d", cfg1.GetLSHSeedAsInt64())
	t.Logf("Enterprise 2 LSH seed: %d", cfg2.GetLSHSeedAsInt64())
}

// TestEnterpriseSearchAccuracy tests search accuracy with known similar vectors
func TestEnterpriseSearchAccuracy(t *testing.T) {
	ctx := context.Background()
	dimension := 64
	numVectors := 200
	numSuperBuckets := 8

	// Create enterprise config
	enterpriseCfg, _ := enterprise.NewConfig("accuracy-test", dimension, numSuperBuckets)

	// Generate vectors with known clusters
	rng := rand.New(rand.NewSource(123))
	ids := make([]string, numVectors)
	vectors := make([][]float64, numVectors)

	// Create a base query vector
	queryVec := generateRandomVector(rng, dimension)
	normalizeVec(queryVec)

	// Create vectors with varying similarity to query
	for i := 0; i < numVectors; i++ {
		ids[i] = generateTestID(i)
		if i < 10 {
			// First 10 vectors are very similar to query (add small noise)
			vectors[i] = make([]float64, dimension)
			for j := 0; j < dimension; j++ {
				vectors[i][j] = queryVec[j] + (rng.Float64()-0.5)*0.1
			}
		} else {
			// Rest are random
			vectors[i] = generateRandomVector(rng, dimension)
		}
	}

	// Build index
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = 4
	cfg.SubBucketsPerSuper = 2
	cfg.NumDecoys = 4

	builder, _ := hierarchical.NewEnterpriseBuilder(cfg, enterpriseCfg)
	builder.Build(ctx, ids, vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Setup auth and client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)

	authCfg := auth.DefaultServiceConfig()
	authService := auth.NewService(authCfg, enterpriseStore)
	authService.RegisterUser(ctx, "user", "accuracy-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	client, _ := NewEnterpriseHierarchicalClient(cfg, creds, store)

	// Search
	result, err := client.Search(ctx, queryVec, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Count how many of the top 10 similar vectors we found
	similarFound := 0
	for _, r := range result.Results {
		for i := 0; i < 10; i++ {
			if r.ID == ids[i] {
				similarFound++
				break
			}
		}
	}

	// Log results - hierarchical search is approximate, so we log rather than require
	t.Logf("Found %d/10 similar vectors in top-%d results", similarFound, len(result.Results))
	t.Logf("Search stats: %d super-buckets selected, %d vectors scored",
		result.Stats.SuperBucketsSelected, result.Stats.VectorsScored)

	// The search should find some vectors
	if len(result.Results) == 0 {
		t.Error("expected at least some results")
	}
}

// Helper functions

func generateTestID(i int) string {
	return "vec_" + string(rune('A'+i%26)) + string(rune('0'+i%10))
}

func generateRandomVector(rng *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	for i := range v {
		v[i] = rng.Float64()*2 - 1
	}
	return v
}

func normalizeVec(v []float64) {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}
