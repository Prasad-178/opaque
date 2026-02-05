// Tier 2 Example: Data-Private Vector Search
//
// This example demonstrates Tier 2 encrypted vector storage where:
// - Vectors are encrypted client-side before storage
// - Storage backend never sees plaintext vectors
// - Search happens by fetching encrypted blobs and decrypting locally
//
// Use cases:
// - Blockchain vector storage
// - Zero-trust cloud storage
// - User-controlled encryption keys
package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/client"
	"github.com/opaque/opaque/go/pkg/encrypt"
)

func main() {
	fmt.Println("=== Tier 2: Data-Private Vector Search Demo ===")
	fmt.Println()

	ctx := context.Background()

	// Step 1: Create encryption key
	// In production, derive from user password or hardware key
	fmt.Println("1. Creating encryption key...")
	key, err := encrypt.GenerateKey()
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}

	encryptor, err := encrypt.NewAESGCM(key)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("   Key fingerprint: %s\n", encryptor.KeyFingerprint())
	fmt.Println()

	// Step 2: Create storage backend
	// Using in-memory for demo; could use FileStore, S3, IPFS, blockchain, etc.
	fmt.Println("2. Creating storage backend (in-memory)...")
	store := blob.NewMemoryStore()
	fmt.Println("   Storage ready.")
	fmt.Println()

	// Step 3: Create Tier 2 client
	fmt.Println("3. Creating Tier 2 client...")
	cfg := client.Tier2Config{
		Dimension: 128,  // Vector dimension
		LSHBits:   8,    // Fewer bits = larger buckets = more privacy (and better recall)
		LSHSeed:   42,   // Must match across all clients sharing this index
	}

	tier2Client, err := client.NewTier2Client(cfg, encryptor, store)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("   Client created with %d-bit LSH.\n", cfg.LSHBits)
	fmt.Println()

	// Step 4: Insert some vectors
	fmt.Println("4. Inserting vectors (encrypted automatically)...")
	numVectors := 1000
	vectors := make([][]float64, numVectors)
	ids := make([]string, numVectors)

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < numVectors; i++ {
		vec := make([]float64, 128)
		for j := range vec {
			vec[j] = rng.Float64()*2 - 1 // Random values in [-1, 1]
		}
		vectors[i] = vec
		ids[i] = fmt.Sprintf("doc-%04d", i)
	}

	start := time.Now()
	err = tier2Client.InsertBatch(ctx, ids, vectors, nil)
	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}
	insertTime := time.Since(start)

	stats, _ := tier2Client.GetStats(ctx)
	fmt.Printf("   Inserted %d vectors in %v\n", numVectors, insertTime)
	fmt.Printf("   Storage stats: %d blobs in %d buckets\n", stats.TotalBlobs, stats.TotalBuckets)
	fmt.Printf("   Avg blobs per bucket: %.1f\n", stats.AvgBlobsPerBucket)
	fmt.Println()

	// Step 5: Search
	fmt.Println("5. Searching (decrypt + compute happens locally)...")

	// Create a query similar to one of our vectors
	query := make([]float64, 128)
	copy(query, vectors[42]) // Copy vector 42
	// Add some noise
	for i := range query {
		query[i] += rng.Float64()*0.1 - 0.05
	}

	start = time.Now()
	results, err := tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
		TopK:          5,
		NumBuckets:    3,    // Search multiple buckets for better recall
		UseMultiProbe: true, // Enable multi-probe LSH
	})
	searchTime := time.Since(start)

	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("   Search completed in %v\n", searchTime)
	fmt.Printf("   Top %d results:\n", len(results))
	for i, r := range results {
		fmt.Printf("     %d. %s (score: %.4f)\n", i+1, r.ID, r.Score)
	}
	fmt.Println()

	// Step 6: Demonstrate privacy features
	fmt.Println("6. Privacy-enhanced search (with decoy buckets)...")

	start = time.Now()
	results, err = tier2Client.SearchWithOptions(ctx, query, client.SearchOptions{
		TopK:         5,
		NumBuckets:   3,
		DecoyBuckets: 5, // Fetch 5 additional random buckets
	})
	privacySearchTime := time.Since(start)

	if err != nil {
		fmt.Printf("   Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("   Search with decoys completed in %v\n", privacySearchTime)
	fmt.Printf("   (Storage can't tell which buckets you're actually interested in)\n")
	fmt.Println()

	// Step 7: Demonstrate wrong key protection
	fmt.Println("7. Testing wrong key protection...")

	// Create another client with a different key
	wrongKey, _ := encrypt.GenerateKey()
	wrongEncryptor, _ := encrypt.NewAESGCM(wrongKey)
	wrongClient, _ := client.NewTier2Client(cfg, wrongEncryptor, store)

	results, err = wrongClient.Search(ctx, query, 5)
	fmt.Printf("   Attempting search with wrong key...\n")
	fmt.Printf("   Results found: %d (decryption fails, no valid results)\n", len(results))
	fmt.Println()

	// Summary
	fmt.Println("=== Summary ===")
	fmt.Println()
	fmt.Println("What the storage backend sees:")
	fmt.Println("  - LSH bucket identifiers (approximate regions)")
	fmt.Println("  - Encrypted blobs (opaque ciphertext)")
	fmt.Println("  - Which buckets were accessed")
	fmt.Println()
	fmt.Println("What the storage backend does NOT see:")
	fmt.Println("  - Actual vector values")
	fmt.Println("  - Query vectors")
	fmt.Println("  - Similarity scores")
	fmt.Println("  - Which results you selected")
	fmt.Println()
	fmt.Printf("Performance: %d vectors, insert %.2fms/vec, search %v\n",
		numVectors,
		float64(insertTime.Microseconds())/float64(numVectors)/1000,
		searchTime)
}
