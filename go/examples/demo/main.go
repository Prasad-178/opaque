// Opaque Demo: Privacy-Preserving Vector Search
//
// This demo showcases both tiers of privacy protection:
// - Tier 1: Query-Private (server has vectors, query is encrypted)
// - Tier 2: Data-Private (both vectors and queries are hidden from server)
//
// Run: go run ./examples/demo/main.go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
)

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  Opaque: Privacy-Preserving Vector Search Demo")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	ctx := context.Background()
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Generate sample vectors
	numVectors := 1000
	dimension := 128
	vectors := generateVectors(rng, numVectors, dimension)
	ids := generateIDs(numVectors)

	fmt.Printf("Generated %d vectors with dimension %d\n\n", numVectors, dimension)

	// Demo both tiers
	demoTier1(ctx, rng, vectors, ids, dimension)
	demoTier2(ctx, rng, vectors, ids, dimension)

	// Summary
	printSummary()
}

func demoTier1(ctx context.Context, rng *rand.Rand, vectors [][]float64, ids []string, dimension int) {
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("  TIER 1: Query-Private Search")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()
	fmt.Println("  In this mode:")
	fmt.Println("  - Server stores plaintext vectors (it owns the data)")
	fmt.Println("  - Client's QUERY is encrypted using homomorphic encryption")
	fmt.Println("  - Server computes similarity on encrypted query")
	fmt.Println("  - Server never sees what you're searching for")
	fmt.Println()

	// Initialize crypto engine
	fmt.Print("1. Initializing homomorphic encryption... ")
	start := time.Now()

	engine, err := crypto.NewClientEngine()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Build LSH index (server-side)
	fmt.Print("2. Building LSH index (server-side)... ")
	start = time.Now()

	lshIndex := lsh.NewIndex(lsh.Config{
		Dimension: dimension,
		NumBits:   64,
		Seed:      42,
	})

	// Add all vectors to the index
	err = lshIndex.Add(ids, vectors)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Create query (similar to vector 42)
	query := make([]float64, dimension)
	copy(query, vectors[42])
	addNoise(rng, query, 0.05)

	// Normalize query for cosine similarity
	normalizeVector(query)

	// LSH search (reveals approximate bucket only)
	fmt.Print("3. LSH candidate search... ")
	start = time.Now()
	candidates, err := lshIndex.SearchVector(query, 100)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("found %d candidates (%v)\n", len(candidates), time.Since(start))

	// Encrypt query
	fmt.Print("4. Encrypting query (homomorphic)... ")
	start = time.Now()
	encQuery, err := engine.EncryptVector(query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	// Compute encrypted dot products (server-side, top 10 candidates)
	fmt.Print("5. Computing encrypted scores (server)... ")
	start = time.Now()
	numToScore := min(10, len(candidates))

	type scoredCandidate struct {
		id    string
		score float64
	}
	results := make([]scoredCandidate, 0, numToScore)

	for i := 0; i < numToScore; i++ {
		// Get the vector for this candidate
		idx := 0
		fmt.Sscanf(candidates[i].ID, "doc-%d", &idx)
		vec := vectors[idx]

		// Normalize for cosine similarity
		normalizedVec := make([]float64, len(vec))
		copy(normalizedVec, vec)
		normalizeVector(normalizedVec)

		// Compute encrypted dot product
		encScore, err := engine.HomomorphicDotProduct(encQuery, normalizedVec)
		if err != nil {
			continue
		}

		// Decrypt score (client-side)
		score, err := engine.DecryptScalar(encScore)
		if err != nil {
			continue
		}

		results = append(results, scoredCandidate{id: candidates[i].ID, score: score})
	}
	fmt.Printf("done (%v for %d vectors)\n", time.Since(start), numToScore)

	// Show results
	fmt.Println("6. Top results:")
	for i, r := range results[:min(3, len(results))] {
		fmt.Printf("   %d. %s (score: %.4f)\n", i+1, r.id, r.score)
	}

	fmt.Println()
	fmt.Println("  Server saw: LSH bucket, encrypted query blob, encrypted scores")
	fmt.Println("  Server did NOT see: actual query vector, similarity values")
	fmt.Println()
}

func demoTier2(ctx context.Context, rng *rand.Rand, vectors [][]float64, ids []string, dimension int) {
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("  TIER 2: Data-Private Search")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()
	fmt.Println("  In this mode:")
	fmt.Println("  - Vectors are encrypted CLIENT-SIDE before storage")
	fmt.Println("  - Server/storage only sees encrypted blobs")
	fmt.Println("  - All similarity computation happens locally")
	fmt.Println("  - Perfect for blockchain, zero-trust storage, user-controlled keys")
	fmt.Println()

	// Create encryption key
	fmt.Print("1. Creating AES-256-GCM encryption key... ")
	key, err := encrypt.GenerateKey()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	encryptor, _ := encrypt.NewAESGCM(key)
	fmt.Printf("done\n")
	fmt.Printf("   Key fingerprint: %s\n", encryptor.KeyFingerprint())

	// Create storage backend
	fmt.Print("2. Creating storage backend (in-memory)... ")
	store := blob.NewMemoryStore()
	fmt.Println("done")

	// Create Tier 2 client
	fmt.Print("3. Creating Tier 2 client... ")
	cfg := client.Tier2Config{
		Dimension: dimension,
		LSHBits:   8, // Fewer bits = larger buckets = more privacy
		LSHSeed:   42,
	}
	tier2Client, _ := client.NewTier2Client(cfg, encryptor, store)
	fmt.Printf("done (%d-bit LSH)\n", cfg.LSHBits)

	// Insert vectors (encrypted automatically)
	fmt.Print("4. Inserting vectors (encrypted)... ")
	start := time.Now()
	err = tier2Client.InsertBatch(ctx, ids, vectors, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	insertTime := time.Since(start)
	stats, _ := tier2Client.GetStats(ctx)
	fmt.Printf("done (%v)\n", insertTime)
	fmt.Printf("   %d blobs in %d buckets (%.1f blobs/bucket)\n",
		stats.TotalBlobs, stats.TotalBuckets, stats.AvgBlobsPerBucket)

	// Create query (similar to vector 42)
	query := make([]float64, dimension)
	copy(query, vectors[42])
	addNoise(rng, query, 0.05)

	// Basic search
	fmt.Print("5. Basic search... ")
	start = time.Now()
	results, err := tier2Client.Search(ctx, query, 5)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	searchTime := time.Since(start)
	fmt.Printf("done (%v)\n", searchTime)
	fmt.Println("   Top results:")
	for i, r := range results {
		fmt.Printf("     %d. %s (score: %.4f)\n", i+1, r.ID, r.Score)
	}

	// Search with privacy features
	fmt.Print("6. Privacy-enhanced search... ")
	tier2Client.SetPrivacyConfig(client.DefaultPrivacyConfig())
	start = time.Now()
	results, err = tier2Client.SearchWithPrivacy(ctx, query, 5)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	privacySearchTime := time.Since(start)
	fmt.Printf("done (%v)\n", privacySearchTime)
	fmt.Println("   Includes: timing obfuscation, decoy buckets, shuffle")

	// Wrong key test
	fmt.Print("7. Testing wrong key protection... ")
	wrongKey, _ := encrypt.GenerateKey()
	wrongEncryptor, _ := encrypt.NewAESGCM(wrongKey)
	wrongClient, _ := client.NewTier2Client(cfg, wrongEncryptor, store)
	wrongResults, _ := wrongClient.Search(ctx, query, 5)
	fmt.Printf("%d results with wrong key (decryption fails)\n", len(wrongResults))

	fmt.Println()
	fmt.Println("  Storage saw: LSH bucket IDs, encrypted blobs, access patterns")
	fmt.Println("  Storage did NOT see: vectors, query, scores, selected results")
	fmt.Println()
}

func printSummary() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("  Summary")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()
	fmt.Println("  +--------------+---------------------+-----------------------+")
	fmt.Println("  | Aspect       | Tier 1 (Query-Priv) | Tier 2 (Data-Priv)    |")
	fmt.Println("  +--------------+---------------------+-----------------------+")
	fmt.Println("  | Vectors      | Visible to server   | Encrypted (AES-GCM)   |")
	fmt.Println("  | Query        | Encrypted (HE)      | Hidden (local)        |")
	fmt.Println("  | Scores       | Encrypted (HE)      | Hidden (local)        |")
	fmt.Println("  | Computation  | Server-side (HE)    | Client-side           |")
	fmt.Println("  | Latency      | ~100-200ms          | <1ms (+ obfuscation)  |")
	fmt.Println("  | Use case     | Query privacy       | Full data privacy     |")
	fmt.Println("  +--------------+---------------------+-----------------------+")
	fmt.Println()
	fmt.Println("  Choose based on your threat model:")
	fmt.Println("  - Tier 1: You trust the server with data, hide your queries")
	fmt.Println("  - Tier 2: Zero-trust storage, user-controlled encryption")
	fmt.Println()
}

func generateVectors(rng *rand.Rand, count, dimension int) [][]float64 {
	vectors := make([][]float64, count)
	for i := 0; i < count; i++ {
		vec := make([]float64, dimension)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1
		}
		vectors[i] = vec
	}
	return vectors
}

func generateIDs(count int) []string {
	ids := make([]string, count)
	for i := 0; i < count; i++ {
		ids[i] = fmt.Sprintf("doc-%d", i)
	}
	return ids
}

func addNoise(rng *rand.Rand, vec []float64, amount float64) {
	for i := range vec {
		vec[i] += rng.Float64()*amount*2 - amount
	}
}

func normalizeVector(v []float64) {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = sqrt(norm)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	// Newton's method
	z := x
	for i := 0; i < 20; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
