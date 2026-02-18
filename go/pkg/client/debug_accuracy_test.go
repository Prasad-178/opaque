//go:build integration

package client

import (
	"context"
	"fmt"
	"testing"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/embeddings"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/hierarchical"
)

// TestDebugAccuracy traces through the search to find where GT gets lost
func TestDebugAccuracy(t *testing.T) {
	dataPath := getSIFTDataPath()
	if dataPath == "" {
		t.Skip("SIFT dataset not found")
	}

	ctx := context.Background()
	dataset, _ := embeddings.SIFT10K(dataPath)

	// Use just 1 query for detailed debugging
	numClusters := 32
	topSelect := 8

	// Build index
	enterpriseCfg, _ := enterprise.NewConfig("debug-test", dataset.Dimension, numClusters)
	store := blob.NewMemoryStore()
	cfg := hierarchical.ConfigFromEnterprise(enterpriseCfg)
	cfg.TopSuperBuckets = topSelect
	cfg.NumDecoys = 0

	builder, _ := hierarchical.NewKMeansBuilder(cfg, enterpriseCfg)
	idx, _ := builder.Build(ctx, dataset.IDs, dataset.Vectors, store)
	enterpriseCfg = builder.GetEnterpriseConfig()

	// Get query 0
	query := dataset.Queries[0]
	normalizedQuery := crypto.NormalizeVector(query)

	// Find cosine GT manually
	bestIdx := -1
	bestScore := -1.0
	for i, vec := range dataset.Vectors {
		normVec := crypto.NormalizeVector(vec)
		score := dotProduct(normalizedQuery, normVec)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	gtID := fmt.Sprintf("sift_%d", bestIdx)
	gtCluster := idx.VectorLocations[gtID].SuperID
	gtBucketKey := idx.VectorLocations[gtID].BucketKey
	t.Logf("Ground Truth: %s (cluster %d, bucket %s, score %.4f)", gtID, gtCluster, gtBucketKey, bestScore)

	// Compute plaintext centroid scores
	centroids := enterpriseCfg.Centroids
	plainScores := make([]float64, len(centroids))
	for i, c := range centroids {
		plainScores[i] = cosineSim(normalizedQuery, c)
	}
	plainTopK := getTopKIndices(plainScores, topSelect)
	t.Logf("Plaintext top-%d clusters: %v", topSelect, plainTopK)

	gtInPlainTop := false
	for _, c := range plainTopK {
		if c == gtCluster {
			gtInPlainTop = true
			break
		}
	}
	t.Logf("GT cluster %d in plaintext top-%d: %v", gtCluster, topSelect, gtInPlainTop)

	// Setup client
	enterpriseStore := enterprise.NewMemoryStore()
	enterpriseStore.Put(ctx, enterpriseCfg)
	authService := auth.NewService(auth.DefaultServiceConfig(), enterpriseStore)
	authService.RegisterUser(ctx, "user", "debug-test", []byte("pass"), []string{auth.ScopeSearch})
	creds, _ := authService.Authenticate(ctx, "user", []byte("pass"))

	// Verify GT is in store
	gtBlobs, _ := store.GetBuckets(ctx, []string{gtBucketKey})
	gtInStore := false
	for _, b := range gtBlobs {
		if b.ID == gtID {
			gtInStore = true
			break
		}
	}
	t.Logf("GT in store (via GetBuckets): %v (%d blobs in bucket %s)", gtInStore, len(gtBlobs), gtBucketKey)

	// Verify GT via GetSuperBuckets
	superBlobs, _ := store.GetSuperBuckets(ctx, []int{gtCluster})
	gtInSuper := false
	for _, b := range superBlobs {
		if b.ID == gtID {
			gtInSuper = true
			break
		}
	}
	t.Logf("GT in store (via GetSuperBuckets): %v (%d blobs in super %d)", gtInSuper, len(superBlobs), gtCluster)

	// Manually decrypt GT
	testEncryptor, _ := encrypt.NewAESGCM(creds.AESKey)
	for _, b := range superBlobs {
		if b.ID == gtID {
			decVec, err := testEncryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
			if err != nil {
				t.Logf("Failed to decrypt GT: %v", err)
			} else {
				normDec := crypto.NormalizeVector(decVec)
				decScore := dotProduct(normalizedQuery, normDec)
				t.Logf("GT decrypted score: %.4f (expected %.4f)", decScore, bestScore)
			}
			break
		}
	}

	// Check if HE selection matches plaintext
	// Manually do what the client does
	t.Logf("\n=== HE vs Plaintext Verification ===")

	client, err := NewEnterpriseHierarchicalClient(cfg, creds, store)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Get what clusters will be fetched via GetSuperBuckets
	// The plaintext top-8 is [16 14 28 24 17 3 12 13]
	// Let's fetch cluster 14 specifically and see what we get
	cluster14Blobs, _ := store.GetSuperBuckets(ctx, []int{14})
	t.Logf("GetSuperBuckets([14]) returned %d blobs", len(cluster14Blobs))

	// Find GT and try to decrypt it
	gtFoundInFetch := false
	for _, b := range cluster14Blobs {
		if b.ID == gtID {
			gtFoundInFetch = true
			decVec, err := testEncryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
			if err != nil {
				t.Logf("GT decrypt failed in fetch: %v", err)
			} else {
				normDec := crypto.NormalizeVector(decVec)
				score := dotProduct(normalizedQuery, normDec)
				t.Logf("GT in cluster14 fetch: decrypt OK, score=%.4f", score)
			}
			break
		}
	}
	t.Logf("GT found in cluster 14 fetch: %v", gtFoundInFetch)

	// Now let's fetch all 8 clusters and see what happens
	allBlobs, _ := store.GetSuperBuckets(ctx, plainTopK)
	t.Logf("GetSuperBuckets(top-8) returned %d blobs", len(allBlobs))

	// Try to decrypt all and find GT
	decrypted := 0
	decryptFailed := 0
	gtDecryptedInAll := false
	var gtScoreInAll float64
	for _, b := range allBlobs {
		vec, err := testEncryptor.DecryptVectorWithID(b.Ciphertext, b.ID)
		if err != nil {
			decryptFailed++
			continue
		}
		decrypted++
		if b.ID == gtID {
			gtDecryptedInAll = true
			normVec := crypto.NormalizeVector(vec)
			gtScoreInAll = dotProduct(normalizedQuery, normVec)
		}
	}
	t.Logf("Decrypted: %d, Failed: %d", decrypted, decryptFailed)
	t.Logf("GT decrypted in all clusters: %v, score=%.4f", gtDecryptedInAll, gtScoreInAll)

	result, err := client.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	t.Logf("Search results: %d vectors scored", result.Stats.VectorsScored)
	t.Logf("Super-buckets selected: %d", result.Stats.SuperBucketsSelected)
	t.Logf("Blobs fetched: %d", result.Stats.BlobsFetched)

	// The discrepancy: test decrypted 2166 from plaintext clusters
	// but client scored fewer. This means HE selected different clusters!
	t.Logf("\n=== HE vs Plaintext Selection ===")
	t.Logf("Test (plaintext) clusters: %v", plainTopK)
	t.Logf("Test (plaintext) blobs: %d", len(allBlobs))
	t.Logf("Client blobs fetched: %d", result.Stats.BlobsFetched)
	t.Logf("Difference: %d blobs", len(allBlobs)-result.Stats.BlobsFetched)

	// Check if GT is in results
	gtInResults := false
	gtRank := -1
	for i, r := range result.Results {
		if r.ID == gtID {
			gtInResults = true
			gtRank = i + 1
			t.Logf("GT found at rank %d with score %.4f", gtRank, r.Score)
			break
		}
	}

	if !gtInResults {
		t.Logf("GT NOT in results!")
		t.Logf("Top-5 results:")
		for i := 0; i < min(5, len(result.Results)); i++ {
			t.Logf("  %d. %s (score=%.4f)", i+1, result.Results[i].ID, result.Results[i].Score)
		}

		// Check what cluster each top result is from
		t.Logf("Top-5 result clusters:")
		for i := 0; i < min(5, len(result.Results)); i++ {
			loc := idx.VectorLocations[result.Results[i].ID]
			if loc != nil {
				t.Logf("  %s -> cluster %d", result.Results[i].ID, loc.SuperID)
			}
		}
	}
}

// cosineSim and getTopKIndices are defined in other test files
