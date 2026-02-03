package test

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// BenchmarkResult stores benchmark results for comparison
type BenchmarkResult struct {
	Operation    string
	Dimension    int
	NumOps       int
	TotalTimeMs  float64
	AvgTimeMs    float64
	ThroughputPS float64
}

func (r BenchmarkResult) String() string {
	return fmt.Sprintf("%s (dim=%d): %.2fms avg, %.2f ops/sec",
		r.Operation, r.Dimension, r.AvgTimeMs, r.ThroughputPS)
}

// generateVector creates a normalized random vector
func generateVector(dim int) []float64 {
	vec := make([]float64, dim)
	var norm float64
	for i := range vec {
		vec[i] = rand.NormFloat64()
		norm += vec[i] * vec[i]
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

// generateVectors creates n normalized random vectors
func generateVectors(n, dim int) [][]float64 {
	vectors := make([][]float64, n)
	for i := range vectors {
		vectors[i] = generateVector(dim)
	}
	return vectors
}

func TestComprehensiveBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping comprehensive benchmark in short mode")
	}

	rand.Seed(42)
	dimensions := []int{32, 128, 256}
	numDBVectors := 100

	fmt.Println("============================================================")
	fmt.Println("Opaque-Go Comprehensive Benchmark")
	fmt.Println("============================================================")

	var results []BenchmarkResult

	// 1. Key Generation Benchmark
	fmt.Println("\n[1/5] Key Generation Benchmark")
	numKeyGen := 3
	start := time.Now()
	for i := 0; i < numKeyGen; i++ {
		_, err := crypto.NewClientEngine()
		if err != nil {
			t.Fatalf("Failed to create engine: %v", err)
		}
	}
	keyGenTime := time.Since(start)
	keyGenResult := BenchmarkResult{
		Operation:    "key_generation",
		NumOps:       numKeyGen,
		TotalTimeMs:  float64(keyGenTime.Milliseconds()),
		AvgTimeMs:    float64(keyGenTime.Milliseconds()) / float64(numKeyGen),
		ThroughputPS: float64(numKeyGen) / keyGenTime.Seconds(),
	}
	results = append(results, keyGenResult)
	fmt.Printf("  Key generation: %.2fms avg\n", keyGenResult.AvgTimeMs)

	// Create engine for remaining tests
	engine, _ := crypto.NewClientEngine()

	for _, dim := range dimensions {
		fmt.Println()
		fmt.Println("============================================================")
		fmt.Printf("Testing Dimension: %d\n", dim)
		fmt.Println("============================================================")

		// 2. Encryption Benchmark
		fmt.Println("\n[2/5] Encryption Benchmark")
		numEncrypt := 10
		vectors := generateVectors(numEncrypt, dim)

		start = time.Now()
		var ciphertexts [][]byte
		for i := 0; i < numEncrypt; i++ {
			ct, err := engine.EncryptVector(vectors[i])
			if err != nil {
				t.Fatalf("Encryption failed: %v", err)
			}
			serialized, _ := engine.SerializeCiphertext(ct)
			ciphertexts = append(ciphertexts, serialized)
		}
		encTime := time.Since(start)
		encResult := BenchmarkResult{
			Operation:    "encryption",
			Dimension:    dim,
			NumOps:       numEncrypt,
			TotalTimeMs:  float64(encTime.Milliseconds()),
			AvgTimeMs:    float64(encTime.Milliseconds()) / float64(numEncrypt),
			ThroughputPS: float64(numEncrypt) / encTime.Seconds(),
		}
		results = append(results, encResult)
		fmt.Printf("  Encryption: %.2fms avg, ciphertext size: %d bytes\n",
			encResult.AvgTimeMs, len(ciphertexts[0]))

		// 3. Decryption Benchmark
		fmt.Println("\n[3/5] Decryption Benchmark")
		numDecrypt := 10
		ct, _ := engine.EncryptVector(vectors[0])

		start = time.Now()
		for i := 0; i < numDecrypt; i++ {
			_, err := engine.DecryptVector(ct, dim)
			if err != nil {
				t.Fatalf("Decryption failed: %v", err)
			}
		}
		decTime := time.Since(start)
		decResult := BenchmarkResult{
			Operation:    "decryption",
			Dimension:    dim,
			NumOps:       numDecrypt,
			TotalTimeMs:  float64(decTime.Milliseconds()),
			AvgTimeMs:    float64(decTime.Milliseconds()) / float64(numDecrypt),
			ThroughputPS: float64(numDecrypt) / decTime.Seconds(),
		}
		results = append(results, decResult)
		fmt.Printf("  Decryption: %.2fms avg\n", decResult.AvgTimeMs)

		// 4. LSH Benchmark
		fmt.Println("\n[4/5] LSH Benchmark")
		lshIndex := lsh.NewIndex(lsh.Config{
			Dimension: dim,
			NumBits:   64,
			Seed:      42,
		})

		// Add vectors
		ids := make([]string, numDBVectors)
		dbVectors := generateVectors(numDBVectors, dim)
		for i := range ids {
			ids[i] = fmt.Sprintf("doc_%d", i)
		}
		lshIndex.Add(ids, dbVectors)

		numLSH := 1000
		query := generateVector(dim)

		start = time.Now()
		for i := 0; i < numLSH; i++ {
			_ = lshIndex.HashBytes(query)
		}
		lshHashTime := time.Since(start)
		lshHashResult := BenchmarkResult{
			Operation:    "lsh_hash",
			Dimension:    dim,
			NumOps:       numLSH,
			TotalTimeMs:  float64(lshHashTime.Microseconds()) / 1000.0,
			AvgTimeMs:    float64(lshHashTime.Microseconds()) / float64(numLSH) / 1000.0,
			ThroughputPS: float64(numLSH) / lshHashTime.Seconds(),
		}
		results = append(results, lshHashResult)
		fmt.Printf("  LSH hash: %.4fms avg (%.0f ops/sec)\n",
			lshHashResult.AvgTimeMs, lshHashResult.ThroughputPS)

		numLSHSearch := 100
		queryHash := lshIndex.HashBytes(query)
		start = time.Now()
		for i := 0; i < numLSHSearch; i++ {
			_, _ = lshIndex.Search(queryHash, 50)
		}
		lshSearchTime := time.Since(start)
		lshSearchResult := BenchmarkResult{
			Operation:    "lsh_search",
			Dimension:    dim,
			NumOps:       numLSHSearch,
			TotalTimeMs:  float64(lshSearchTime.Microseconds()) / 1000.0,
			AvgTimeMs:    float64(lshSearchTime.Microseconds()) / float64(numLSHSearch) / 1000.0,
			ThroughputPS: float64(numLSHSearch) / lshSearchTime.Seconds(),
		}
		results = append(results, lshSearchResult)
		fmt.Printf("  LSH search (k=50): %.4fms avg (%.0f ops/sec)\n",
			lshSearchResult.AvgTimeMs, lshSearchResult.ThroughputPS)

		// 5. Homomorphic Dot Product (Component-wise, without rotation sum)
		fmt.Println("\n[5/5] Homomorphic Operations Benchmark")
		numDotProducts := 5
		encQuery, _ := engine.EncryptVector(vectors[0])

		start = time.Now()
		for i := 0; i < numDotProducts; i++ {
			_, err := engine.HomomorphicDotProduct(encQuery, dbVectors[i%numDBVectors])
			if err != nil {
				t.Fatalf("Homomorphic dot product failed: %v", err)
			}
		}
		dotTime := time.Since(start)
		dotResult := BenchmarkResult{
			Operation:    "homomorphic_dot_product",
			Dimension:    dim,
			NumOps:       numDotProducts,
			TotalTimeMs:  float64(dotTime.Milliseconds()),
			AvgTimeMs:    float64(dotTime.Milliseconds()) / float64(numDotProducts),
			ThroughputPS: float64(numDotProducts) / dotTime.Seconds(),
		}
		results = append(results, dotResult)
		fmt.Printf("  Homomorphic dot product: %.2fms avg (%.1f ops/sec)\n",
			dotResult.AvgTimeMs, dotResult.ThroughputPS)
	}

	// Summary
	fmt.Println("\n" + "=" + string(make([]byte, 59, 59)))
	fmt.Println("SUMMARY")
	fmt.Println("============================================================")
	fmt.Println("\nOperation                    | Dim  | Avg Time (ms) | Throughput")
	fmt.Println("-----------------------------------------------------------------------")

	for _, r := range results {
		fmt.Printf("%-28s | %4d | %13.4f | %.2f ops/sec\n",
			r.Operation, r.Dimension, r.AvgTimeMs, r.ThroughputPS)
	}
}

// Individual benchmarks for `go test -bench=.`

func BenchmarkKeyGeneration(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = crypto.NewClientEngine()
	}
}

func BenchmarkEncryption32D(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	vec := generateVector(32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.EncryptVector(vec)
	}
}

func BenchmarkEncryption128D(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	vec := generateVector(128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.EncryptVector(vec)
	}
}

func BenchmarkEncryption256D(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	vec := generateVector(256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.EncryptVector(vec)
	}
}

func BenchmarkDecryption128D(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	vec := generateVector(128)
	ct, _ := engine.EncryptVector(vec)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.DecryptVector(ct, 128)
	}
}

func BenchmarkLSHHash128D(b *testing.B) {
	idx := lsh.NewIndex(lsh.Config{Dimension: 128, NumBits: 64, Seed: 42})
	vec := generateVector(128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.HashBytes(vec)
	}
}

func BenchmarkLSHSearch(b *testing.B) {
	idx := lsh.NewIndex(lsh.Config{Dimension: 128, NumBits: 64, Seed: 42})
	vecs := generateVectors(1000, 128)
	ids := make([]string, 1000)
	for i := range ids {
		ids[i] = fmt.Sprintf("doc_%d", i)
	}
	idx.Add(ids, vecs)
	query := generateVector(128)
	hash := idx.HashBytes(query)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(hash, 50)
	}
}

func BenchmarkHomomorphicDotProduct128D(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	query := generateVector(128)
	vector := generateVector(128)
	encQuery, _ := engine.EncryptVector(query)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.HomomorphicDotProduct(encQuery, vector)
	}
}

// Parallel benchmarks to show real throughput with goroutines
// Note: Lattigo's encryptor is NOT thread-safe, so we need separate engines per goroutine
// or use mutex protection. These benchmarks show realistic parallel scenarios.

func BenchmarkLSHSearchParallel(b *testing.B) {
	idx := lsh.NewIndex(lsh.Config{Dimension: 128, NumBits: 64, Seed: 42})
	vecs := generateVectors(1000, 128)
	ids := make([]string, 1000)
	for i := range ids {
		ids[i] = fmt.Sprintf("doc_%d", i)
	}
	idx.Add(ids, vecs)
	queries := generateVectors(100, 128)
	hashes := make([][]byte, 100)
	for i, q := range queries {
		hashes[i] = idx.HashBytes(q)
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			_, _ = idx.Search(hashes[i%100], 50)
			i++
		}
	})
}

// Batch benchmark - simulates real workload of 20 dot products
func BenchmarkBatch20DotProducts(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	query := generateVector(128)
	vectors := generateVectors(20, 128)
	encQuery, _ := engine.EncryptVector(query)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Sequential
		for j := 0; j < 20; j++ {
			_, _ = engine.HomomorphicDotProduct(encQuery, vectors[j])
		}
	}
}

func BenchmarkBatch20DotProductsParallel(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	query := generateVector(128)
	vectors := generateVectors(20, 128)
	encQuery, _ := engine.EncryptVector(query)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Parallel with goroutines
		var wg sync.WaitGroup
		for j := 0; j < 20; j++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				_, _ = engine.HomomorphicDotProduct(encQuery, vectors[idx])
			}(j)
		}
		wg.Wait()
	}
}

func BenchmarkBatch20DecryptionsParallel(b *testing.B) {
	engine, _ := crypto.NewClientEngine()
	query := generateVector(128)
	vector := generateVector(128)
	encQuery, _ := engine.EncryptVector(query)
	encResult, _ := engine.HomomorphicDotProduct(encQuery, vector)
	serialized, _ := engine.SerializeCiphertext(encResult)

	// Create 20 copies
	ciphertexts := make([][]byte, 20)
	for i := range ciphertexts {
		ciphertexts[i] = serialized
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		for j := 0; j < 20; j++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				ct, _ := engine.DeserializeCiphertext(ciphertexts[idx])
				_, _ = engine.DecryptScalar(ct)
			}(j)
		}
		wg.Wait()
	}
}
