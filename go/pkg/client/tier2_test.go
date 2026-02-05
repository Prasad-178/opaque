package client

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/encrypt"
)

func TestTier2Client_InsertAndSearch(t *testing.T) {
	// Setup
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := Tier2Config{
		Dimension: 8,
		LSHBits:   16,
		LSHSeed:   42,
	}

	client, err := NewTier2Client(cfg, enc, store)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	ctx := context.Background()

	// Insert some vectors
	vectors := [][]float64{
		{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // doc-1: points in x direction
		{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // doc-2: points in y direction
		{0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // doc-3: similar to doc-1
		{0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, // doc-4: similar to doc-2
	}
	ids := []string{"doc-1", "doc-2", "doc-3", "doc-4"}

	err = client.InsertBatch(ctx, ids, vectors, nil)
	if err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	// Verify stats
	stats, _ := client.GetStats(ctx)
	if stats.TotalBlobs != 4 {
		t.Errorf("expected 4 blobs, got %d", stats.TotalBlobs)
	}

	// Search for vector similar to doc-1
	query := []float64{0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	results, err := client.Search(ctx, query, 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Should find some results
	if len(results) == 0 {
		t.Error("expected some results")
	}

	// Results should be sorted by score descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Error("results should be sorted by score descending")
		}
	}
}

func TestTier2Client_SearchWithOptions(t *testing.T) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := Tier2Config{
		Dimension: 4,
		LSHBits:   8,
		LSHSeed:   42,
	}

	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	// Insert vectors
	for i := 0; i < 100; i++ {
		vec := make([]float64, 4)
		for j := range vec {
			vec[j] = rand.Float64()
		}
		client.Insert(ctx, string(rune('a'+i)), vec, nil)
	}

	// Search with options
	query := []float64{0.5, 0.5, 0.5, 0.5}
	results, err := client.SearchWithOptions(ctx, query, SearchOptions{
		TopK:          5,
		NumBuckets:    3,
		UseMultiProbe: true,
	})

	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Should return at most TopK results
	if len(results) > 5 {
		t.Errorf("expected at most 5 results, got %d", len(results))
	}
}

func TestTier2Client_Delete(t *testing.T) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	cfg.Dimension = 4

	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	// Insert
	client.Insert(ctx, "doc-1", []float64{1, 2, 3, 4}, nil)
	client.Insert(ctx, "doc-2", []float64{5, 6, 7, 8}, nil)

	stats, _ := client.GetStats(ctx)
	if stats.TotalBlobs != 2 {
		t.Errorf("expected 2 blobs, got %d", stats.TotalBlobs)
	}

	// Delete
	client.Delete(ctx, "doc-1")

	stats, _ = client.GetStats(ctx)
	if stats.TotalBlobs != 1 {
		t.Errorf("expected 1 blob after delete, got %d", stats.TotalBlobs)
	}
}

func TestTier2Client_WrongKey(t *testing.T) {
	key1, _ := encrypt.GenerateKey()
	key2, _ := encrypt.GenerateKey()

	enc1, _ := encrypt.NewAESGCM(key1)
	enc2, _ := encrypt.NewAESGCM(key2)

	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	cfg.Dimension = 4

	// Client 1 inserts with key1
	client1, _ := NewTier2Client(cfg, enc1, store)
	ctx := context.Background()
	client1.Insert(ctx, "doc-1", []float64{1, 2, 3, 4}, nil)

	// Client 2 tries to search with key2 - should fail to decrypt
	client2, _ := NewTier2Client(cfg, enc2, store)
	results, _ := client2.Search(ctx, []float64{1, 2, 3, 4}, 1)

	// Should get no valid results (decryption failed)
	if len(results) > 0 {
		t.Error("should not get results with wrong key")
	}
}

func TestTier2Client_DimensionMismatch(t *testing.T) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	cfg.Dimension = 4

	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	// Insert with wrong dimension
	err := client.Insert(ctx, "doc-1", []float64{1, 2, 3, 4, 5}, nil)
	if err == nil {
		t.Error("should reject vector with wrong dimension")
	}

	// Search with wrong dimension
	client.Insert(ctx, "doc-1", []float64{1, 2, 3, 4}, nil)
	_, err = client.Search(ctx, []float64{1, 2, 3}, 1)
	if err == nil {
		t.Error("should reject query with wrong dimension")
	}
}

func TestTier2Client_KeyFingerprint(t *testing.T) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	client, _ := NewTier2Client(cfg, enc, store)

	fp := client.GetKeyFingerprint()
	if len(fp) != 16 { // 8 bytes hex encoded
		t.Errorf("fingerprint should be 16 chars, got %d", len(fp))
	}
}

func TestTier2Client_EmptyStore(t *testing.T) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	cfg.Dimension = 4

	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	// Search on empty store should return empty results
	results, err := client.Search(ctx, []float64{1, 2, 3, 4}, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != 0 {
		t.Error("expected empty results for empty store")
	}
}

func TestNormalizeVector(t *testing.T) {
	v := []float64{3.0, 4.0}
	normalized := normalizeVector(v)

	// Should have unit length
	var norm float64
	for _, val := range normalized {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("normalized vector should have unit length, got %f", norm)
	}

	// Direction should be preserved (3/4 = 0.75)
	ratio := normalized[0] / normalized[1]
	if math.Abs(ratio-0.75) > 1e-10 {
		t.Errorf("normalization changed direction: ratio = %f, expected 0.75", ratio)
	}
}

func TestTier2DotProduct(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	// Using the dotProduct from client.go
	result := dotProduct(a, b)
	if result != 32.0 {
		t.Errorf("expected 32, got %f", result)
	}
}

func BenchmarkTier2Client_Insert(b *testing.B) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	vector := make([]float64, 128)
	for i := range vector {
		vector[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.Insert(ctx, string(rune(i)), vector, nil)
	}
}

func BenchmarkTier2Client_Search(b *testing.B) {
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := DefaultTier2Config()
	client, _ := NewTier2Client(cfg, enc, store)
	ctx := context.Background()

	// Pre-populate with vectors
	for i := 0; i < 1000; i++ {
		vector := make([]float64, 128)
		for j := range vector {
			vector[j] = rand.Float64()
		}
		client.Insert(ctx, string(rune(i)), vector, nil)
	}

	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Search(ctx, query, 10)
	}
}
