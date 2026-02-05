package test

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/encrypt"
)

// TestTier2ComprehensiveBenchmark runs a detailed benchmark of Tier 2 features
func TestTier2ComprehensiveBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping comprehensive benchmark in short mode")
	}

	fmt.Println("============================================================")
	fmt.Println("Tier 2: Data-Private Vector Search Benchmark")
	fmt.Println("============================================================")
	fmt.Println()

	ctx := context.Background()
	dimensions := []int{64, 128, 256}
	vectorCounts := []int{1000, 5000, 10000}

	for _, dim := range dimensions {
		for _, count := range vectorCounts {
			fmt.Printf("--- Dimension: %d, Vectors: %d ---\n", dim, count)
			runTier2Benchmark(ctx, t, dim, count)
			fmt.Println()
		}
	}
}

func runTier2Benchmark(ctx context.Context, t *testing.T, dimension, vectorCount int) {
	// Setup
	key, err := encrypt.GenerateKey()
	if err != nil {
		t.Fatalf("failed to generate key: %v", err)
	}

	enc, err := encrypt.NewAESGCM(key)
	if err != nil {
		t.Fatalf("failed to create encryptor: %v", err)
	}

	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension:         dimension,
		LSHBits:           16,
		LSHSeed:           42,
		MaxBucketsToFetch: 5,
	}

	tier2Client, err := client.NewTier2Client(cfg, enc, store)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Generate vectors
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	vectors := make([][]float64, vectorCount)
	ids := make([]string, vectorCount)

	for i := 0; i < vectorCount; i++ {
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
		}
		vectors[i] = vec
		ids[i] = fmt.Sprintf("doc-%d", i)
	}

	// Benchmark insert
	start := time.Now()
	err = tier2Client.InsertBatch(ctx, ids, vectors, nil)
	if err != nil {
		t.Fatalf("insert failed: %v", err)
	}
	insertTime := time.Since(start)
	insertPerVec := float64(insertTime.Microseconds()) / float64(vectorCount) / 1000.0

	fmt.Printf("   Insert: %.2fms total (%.3fms/vec)\n", float64(insertTime.Milliseconds()), insertPerVec)

	// Benchmark basic search
	query := vectors[vectorCount/2] // Use middle vector as query
	searchIterations := 10

	var totalBasicSearch time.Duration
	for i := 0; i < searchIterations; i++ {
		start = time.Now()
		_, err = tier2Client.Search(ctx, query, 10)
		if err != nil {
			t.Fatalf("search failed: %v", err)
		}
		totalBasicSearch += time.Since(start)
	}
	avgBasicSearch := totalBasicSearch / time.Duration(searchIterations)
	fmt.Printf("   Basic search: %.2fms avg\n", float64(avgBasicSearch.Microseconds())/1000.0)

	// Benchmark multi-probe search
	var totalMultiProbe time.Duration
	for i := 0; i < searchIterations; i++ {
		start = time.Now()
		_, err = tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
			TopK:          10,
			NumBuckets:    3,
			UseMultiProbe: true,
		})
		if err != nil {
			t.Fatalf("multi-probe search failed: %v", err)
		}
		totalMultiProbe += time.Since(start)
	}
	avgMultiProbe := totalMultiProbe / time.Duration(searchIterations)
	fmt.Printf("   Multi-probe search (3 buckets): %.2fms avg\n", float64(avgMultiProbe.Microseconds())/1000.0)

	// Benchmark search with decoys
	var totalDecoySearch time.Duration
	for i := 0; i < searchIterations; i++ {
		start = time.Now()
		_, err = tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
			TopK:         10,
			NumBuckets:   3,
			DecoyBuckets: 5,
		})
		if err != nil {
			t.Fatalf("decoy search failed: %v", err)
		}
		totalDecoySearch += time.Since(start)
	}
	avgDecoySearch := totalDecoySearch / time.Duration(searchIterations)
	fmt.Printf("   Search with 5 decoys: %.2fms avg\n", float64(avgDecoySearch.Microseconds())/1000.0)

	// Benchmark privacy-enhanced search
	tier2Client.SetPrivacyConfig(client.DefaultPrivacyConfig())

	var totalPrivacySearch time.Duration
	for i := 0; i < searchIterations; i++ {
		start = time.Now()
		_, err = tier2Client.SearchWithPrivacy(ctx, query, 10)
		if err != nil {
			t.Fatalf("privacy search failed: %v", err)
		}
		totalPrivacySearch += time.Since(start)
	}
	avgPrivacySearch := totalPrivacySearch / time.Duration(searchIterations)
	fmt.Printf("   Privacy-enhanced search: %.2fms avg (includes timing obfuscation)\n", float64(avgPrivacySearch.Microseconds())/1000.0)

	// Storage stats
	stats, _ := tier2Client.GetStats(ctx)
	fmt.Printf("   Storage: %d blobs, %d buckets, %.1f blobs/bucket\n",
		stats.TotalBlobs, stats.TotalBuckets, stats.AvgBlobsPerBucket)
}

// TestTier2EncryptionOverhead measures the encryption/decryption overhead
func TestTier2EncryptionOverhead(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping encryption overhead test in short mode")
	}

	fmt.Println("============================================================")
	fmt.Println("Tier 2: Encryption Overhead Analysis")
	fmt.Println("============================================================")
	fmt.Println()

	dimensions := []int{64, 128, 256, 512, 1024}

	for _, dim := range dimensions {
		key, _ := encrypt.GenerateKey()
		enc, _ := encrypt.NewAESGCM(key)

		// Generate test vector
		vector := make([]float64, dim)
		for i := range vector {
			vector[i] = rand.Float64()*2 - 1
		}

		iterations := 1000

		// Benchmark encryption
		start := time.Now()
		var ciphertext []byte
		for i := 0; i < iterations; i++ {
			ciphertext, _ = enc.EncryptVectorWithID(vector, "test-id")
		}
		encryptTime := time.Since(start)
		avgEncrypt := float64(encryptTime.Microseconds()) / float64(iterations)

		// Benchmark decryption
		start = time.Now()
		for i := 0; i < iterations; i++ {
			_, _ = enc.DecryptVectorWithID(ciphertext, "test-id")
		}
		decryptTime := time.Since(start)
		avgDecrypt := float64(decryptTime.Microseconds()) / float64(iterations)

		fmt.Printf("   Dim %4d: encrypt=%.2fµs, decrypt=%.2fµs, ciphertext=%d bytes\n",
			dim, avgEncrypt, avgDecrypt, len(ciphertext))
	}
	fmt.Println()
}

// TestTier2Throughput measures sustained throughput
func TestTier2Throughput(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput test in short mode")
	}

	fmt.Println("============================================================")
	fmt.Println("Tier 2: Throughput Analysis")
	fmt.Println("============================================================")
	fmt.Println()

	ctx := context.Background()

	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension:         128,
		LSHBits:           16,
		LSHSeed:           42,
		MaxBucketsToFetch: 5,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)

	// Insert vectors
	vectorCount := 10000
	vectors := make([][]float64, vectorCount)
	ids := make([]string, vectorCount)

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < vectorCount; i++ {
		vec := make([]float64, 128)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
		}
		vectors[i] = vec
		ids[i] = fmt.Sprintf("doc-%d", i)
	}

	tier2Client.InsertBatch(ctx, ids, vectors, nil)
	fmt.Printf("   Indexed %d vectors\n", vectorCount)

	// Measure query throughput
	query := vectors[0]
	duration := 5 * time.Second
	queryCount := 0

	start := time.Now()
	for time.Since(start) < duration {
		tier2Client.Search(ctx, query, 10)
		queryCount++
	}
	elapsed := time.Since(start)

	qps := float64(queryCount) / elapsed.Seconds()
	fmt.Printf("   Search QPS: %.1f queries/sec (5 second test)\n", qps)

	// Measure with privacy
	tier2Client.SetPrivacyConfig(client.LowLatencyConfig())
	queryCount = 0

	start = time.Now()
	for time.Since(start) < duration {
		tier2Client.SearchWithPrivacy(ctx, query, 10)
		queryCount++
	}
	elapsed = time.Since(start)

	privacyQps := float64(queryCount) / elapsed.Seconds()
	fmt.Printf("   Privacy search QPS: %.1f queries/sec (low latency config)\n", privacyQps)
	fmt.Println()
}

// TestTier2VsTier1Comparison shows the difference between Tier 1 and Tier 2
func TestTier2VsTier1Comparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping tier comparison in short mode")
	}

	fmt.Println("============================================================")
	fmt.Println("Tier 1 vs Tier 2 Comparison")
	fmt.Println("============================================================")
	fmt.Println()
	fmt.Println("Tier 1 (Query-Private): Server stores plaintext vectors")
	fmt.Println("  - Fast search (server computes similarity)")
	fmt.Println("  - Query is hidden from server")
	fmt.Println("  - Vectors visible to server")
	fmt.Println()
	fmt.Println("Tier 2 (Data-Private): Server stores encrypted blobs")
	fmt.Println("  - Slower search (client decrypts and computes)")
	fmt.Println("  - Vectors hidden from server")
	fmt.Println("  - Server only sees encrypted blobs + LSH buckets")
	fmt.Println()

	// Tier 2 benchmark
	ctx := context.Background()
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension: 128,
		LSHBits:   16,
		LSHSeed:   42,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)

	// Insert vectors
	vectorCount := 1000
	vectors := make([][]float64, vectorCount)
	ids := make([]string, vectorCount)

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < vectorCount; i++ {
		vec := make([]float64, 128)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
		}
		vectors[i] = vec
		ids[i] = fmt.Sprintf("doc-%d", i)
	}

	tier2Client.InsertBatch(ctx, ids, vectors, nil)

	// Time Tier 2 search
	query := vectors[0]
	iterations := 100

	start := time.Now()
	for i := 0; i < iterations; i++ {
		tier2Client.Search(ctx, query, 10)
	}
	tier2Time := time.Since(start) / time.Duration(iterations)

	fmt.Printf("Tier 2 search latency: %.2fms (1000 vectors, 128 dim)\n", float64(tier2Time.Microseconds())/1000.0)
	fmt.Println()
	fmt.Println("Note: Tier 1 uses homomorphic encryption for query privacy,")
	fmt.Println("which has different performance characteristics (see main benchmarks).")
}

// BenchmarkTier2Insert benchmarks insert operations
func BenchmarkTier2Insert(b *testing.B) {
	ctx := context.Background()
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension: 128,
		LSHBits:   16,
		LSHSeed:   42,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)

	vector := make([]float64, 128)
	for i := range vector {
		vector[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tier2Client.Insert(ctx, fmt.Sprintf("doc-%d", i), vector, nil)
	}
}

// BenchmarkTier2Search benchmarks search operations
func BenchmarkTier2Search(b *testing.B) {
	ctx := context.Background()
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension: 128,
		LSHBits:   16,
		LSHSeed:   42,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)

	// Pre-populate
	for i := 0; i < 1000; i++ {
		vector := make([]float64, 128)
		for j := range vector {
			vector[j] = rand.Float64()
		}
		tier2Client.Insert(ctx, fmt.Sprintf("doc-%d", i), vector, nil)
	}

	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tier2Client.Search(ctx, query, 10)
	}
}

// BenchmarkTier2SearchWithPrivacy benchmarks privacy-enhanced search
func BenchmarkTier2SearchWithPrivacy(b *testing.B) {
	ctx := context.Background()
	key, _ := encrypt.GenerateKey()
	enc, _ := encrypt.NewAESGCM(key)
	store := blob.NewMemoryStore()

	cfg := client.Tier2Config{
		Dimension: 128,
		LSHBits:   16,
		LSHSeed:   42,
	}

	tier2Client, _ := client.NewTier2Client(cfg, enc, store)
	tier2Client.SetPrivacyConfig(client.LowLatencyConfig()) // Use low latency for benchmark

	// Pre-populate
	for i := 0; i < 1000; i++ {
		vector := make([]float64, 128)
		for j := range vector {
			vector[j] = rand.Float64()
		}
		tier2Client.Insert(ctx, fmt.Sprintf("doc-%d", i), vector, nil)
	}

	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tier2Client.SearchWithPrivacy(ctx, query, 10)
	}
}
