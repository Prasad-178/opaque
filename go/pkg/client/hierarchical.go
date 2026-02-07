// Hierarchical Private Search Client
//
// This implements a three-level privacy-preserving vector search:
//
//	Level 1: HE on super-bucket centroids
//	  - Encrypt query with HE
//	  - Compute HE scores for ALL centroids (server does this)
//	  - Client decrypts privately, selects top-K super-buckets
//	  - Server NEVER knows which super-buckets were selected
//
//	Level 2: Decoy-based sub-bucket fetch
//	  - Fetch real sub-buckets + decoy sub-buckets
//	  - Shuffle all requests - server can't tell real from decoy
//
//	Level 3: Local AES decrypt + scoring
//	  - Decrypt vectors with AES
//	  - Score locally with plaintext dot products
//
// Privacy guarantees:
//   - Query: hidden from server (HE)
//   - Super-bucket selection: hidden from server (client-side decrypt)
//   - Sub-bucket interest: hidden from server (decoys)
//   - Vectors: hidden from storage (AES)
//   - Scores: hidden from everyone (local)
package client

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sort"
	"sync"
	"time"

	"github.com/opaque/opaque/go/pkg/crypto"
	"github.com/opaque/opaque/go/pkg/hierarchical"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

// HierarchicalClient performs hierarchical private search.
type HierarchicalClient struct {
	index    *hierarchical.Index
	heEngine *crypto.Engine
}

// NewHierarchicalClient creates a new hierarchical search client.
func NewHierarchicalClient(index *hierarchical.Index) (*HierarchicalClient, error) {
	heEngine, err := crypto.NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create HE engine: %w", err)
	}

	return &HierarchicalClient{
		index:    index,
		heEngine: heEngine,
	}, nil
}

// Search performs the full hierarchical private search.
func (c *HierarchicalClient) Search(ctx context.Context, query []float64, topK int) (*hierarchical.SearchResult, error) {
	startTotal := time.Now()
	result := &hierarchical.SearchResult{
		Stats: hierarchical.SearchStats{},
	}

	if len(query) != c.index.Config.Dimension {
		return nil, fmt.Errorf("query has wrong dimension: %d (expected %d)", len(query), c.index.Config.Dimension)
	}

	// Normalize query
	normalizedQuery := normalizeVectorCopy(query)

	// ==========================================
	// LEVEL 1: HE Centroid Scoring
	// ==========================================

	// Step 1a: Encrypt query with HE
	startEncrypt := time.Now()
	encQuery, err := c.heEngine.EncryptVector(normalizedQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt query: %w", err)
	}
	result.Timing.HEEncryptQuery = time.Since(startEncrypt)

	// Step 1b: Compute HE scores for ALL centroids (parallel)
	// This simulates what a server would do - compute all scores blindly
	startHE := time.Now()
	centroids := c.index.GetCentroids()
	encScores := make([]*rlwe.Ciphertext, len(centroids))
	scoreErrs := make([]error, len(centroids))

	var wg sync.WaitGroup
	// Use limited parallelism to avoid overwhelming the system
	numWorkers := 4
	workChan := make(chan int, len(centroids))

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range workChan {
				encScores[i], scoreErrs[i] = c.heEngine.HomomorphicDotProduct(encQuery, centroids[i])
			}
		}()
	}

	for i := range centroids {
		workChan <- i
	}
	close(workChan)
	wg.Wait()

	// Check for errors
	for i, err := range scoreErrs {
		if err != nil {
			return nil, fmt.Errorf("failed to compute HE score for centroid %d: %w", i, err)
		}
	}
	result.Timing.HECentroidScores = time.Since(startHE)
	result.Stats.HEOperations = len(centroids)

	// Step 1c: Decrypt scores privately (only client sees this)
	startDecrypt := time.Now()
	scores := make([]float64, len(encScores))
	for i, encScore := range encScores {
		score, err := c.heEngine.DecryptScalar(encScore)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt score %d: %w", i, err)
		}
		scores[i] = score
	}
	result.Timing.HEDecryptScores = time.Since(startDecrypt)

	// Step 1d: Select top super-buckets (SERVER NEVER SEES THIS!)
	// Use multi-probe selection to capture clusters within threshold of top-K
	topSupers := selectClustersWithProbing(
		scores,
		c.index.Config.TopSuperBuckets,
		c.index.Config.ProbeThreshold,
		c.index.Config.MaxProbeClusters,
	)
	result.Stats.SuperBucketsSelected = len(topSupers)

	// ==========================================
	// LEVEL 2: Decoy-Based Sub-Bucket Fetch
	// ==========================================

	startSelection := time.Now()

	// Determine primary sub-bucket for query
	primarySubID := c.index.GetSubBucketID(normalizedQuery)

	// Build list of real sub-buckets to fetch
	realBuckets := make([]string, 0)
	for _, superID := range topSupers {
		// Get primary + neighbors for better recall
		subBuckets := c.index.GetNeighborSubBuckets(superID, primarySubID, c.index.Config.SubBucketsPerSuper-1)
		realBuckets = append(realBuckets, subBuckets...)
	}

	// Generate decoy buckets from OTHER super-buckets
	decoyBuckets := c.generateDecoyBuckets(topSupers, c.index.Config.NumDecoys)

	// Combine and shuffle
	allBuckets := append(realBuckets, decoyBuckets...)
	shuffleStrings(allBuckets)

	result.Stats.RealSubBuckets = len(realBuckets)
	result.Stats.DecoySubBuckets = len(decoyBuckets)
	result.Stats.TotalSubBuckets = len(allBuckets)
	result.Timing.BucketSelection = time.Since(startSelection)

	// Fetch all sub-buckets (server can't tell which are real)
	startFetch := time.Now()
	blobs, err := c.index.Store.GetBuckets(ctx, allBuckets)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch buckets: %w", err)
	}
	result.Timing.BucketFetch = time.Since(startFetch)
	result.Stats.BlobsFetched = len(blobs)

	// ==========================================
	// LEVEL 3: Local AES Decrypt + Scoring
	// ==========================================

	// Decrypt all vectors
	startAES := time.Now()
	type decryptedVec struct {
		id       string // Original ID (without _dupN suffix)
		blobID   string // Blob ID (may have _dupN suffix)
		vector   []float64
	}
	decrypted := make([]decryptedVec, 0, len(blobs))

	for _, blob := range blobs {
		// Extract original ID for decryption (handles redundant assignment)
		origID := extractOriginalID(blob.ID)
		vec, err := c.index.Encryptor.DecryptVectorWithID(blob.Ciphertext, origID)
		if err != nil {
			continue // Skip failed decryptions (likely decoy or corrupted)
		}
		decrypted = append(decrypted, decryptedVec{id: origID, blobID: blob.ID, vector: vec})
	}
	result.Timing.AESDecrypt = time.Since(startAES)

	// Score locally with deduplication for redundant assignment
	// Keep only the highest score for each unique original ID
	startScore := time.Now()

	// Map to track best score per original ID
	scoreMap := make(map[string]float64)

	for _, d := range decrypted {
		normalizedVec := normalizeVectorCopy(d.vector)
		score := dotProductVec(normalizedQuery, normalizedVec)

		// Keep highest score for each original ID
		if existing, ok := scoreMap[d.id]; !ok || score > existing {
			scoreMap[d.id] = score
		}
	}

	// Convert map to sorted slice
	type scored struct {
		id    string
		score float64
	}
	scoredResults := make([]scored, 0, len(scoreMap))
	for id, score := range scoreMap {
		scoredResults = append(scoredResults, scored{id: id, score: score})
	}

	// Sort by score descending
	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].score > scoredResults[j].score
	})

	result.Timing.LocalScoring = time.Since(startScore)
	result.Stats.VectorsScored = len(scoreMap) // Unique vectors scored

	// Return top-K
	n := min(topK, len(scoredResults))
	result.Results = make([]hierarchical.Result, n)
	for i := 0; i < n; i++ {
		result.Results[i] = hierarchical.Result{
			ID:    scoredResults[i].id,
			Score: scoredResults[i].score,
		}
	}

	result.Timing.Total = time.Since(startTotal)
	return result, nil
}

// generateDecoyBuckets generates decoy bucket keys from non-selected super-buckets.
func (c *HierarchicalClient) generateDecoyBuckets(selectedSupers []int, numDecoys int) []string {
	if numDecoys <= 0 {
		return nil
	}

	// Create set of selected super-buckets
	selected := make(map[int]bool)
	for _, s := range selectedSupers {
		selected[s] = true
	}

	// Collect all non-selected super-buckets
	var nonSelected []int
	for i := 0; i < c.index.Config.NumSuperBuckets; i++ {
		if !selected[i] {
			nonSelected = append(nonSelected, i)
		}
	}

	if len(nonSelected) == 0 {
		return nil
	}

	// Generate random decoy buckets
	decoys := make([]string, 0, numDecoys)
	for i := 0; i < numDecoys; i++ {
		// Pick random non-selected super-bucket
		superIdx, _ := rand.Int(rand.Reader, big.NewInt(int64(len(nonSelected))))
		superID := nonSelected[superIdx.Int64()]

		// Pick random sub-bucket
		subIdx, _ := rand.Int(rand.Reader, big.NewInt(int64(c.index.Config.NumSubBuckets)))
		subID := int(subIdx.Int64())

		decoys = append(decoys, fmt.Sprintf("%02d_%02d", superID, subID))
	}

	return decoys
}

// selectTopKIndices returns indices of the top-K values in scores.
func selectTopKIndices(scores []float64, k int) []int {
	type indexedScore struct {
		index int
		score float64
	}

	indexed := make([]indexedScore, len(scores))
	for i, s := range scores {
		indexed[i] = indexedScore{index: i, score: s}
	}

	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].score > indexed[j].score
	})

	n := min(k, len(indexed))
	result := make([]int, n)
	for i := 0; i < n; i++ {
		result[i] = indexed[i].index
	}

	return result
}

// selectClustersWithProbing returns cluster indices using multi-probe strategy.
// This improves recall by including clusters that scored close to the top-K threshold,
// addressing HE precision noise that can cause near-miss exclusions.
//
// Parameters:
//   - scores: HE-decrypted scores for each cluster
//   - minK: minimum clusters to always include (top minK by score)
//   - threshold: score ratio threshold for additional clusters (e.g., 0.95)
//   - maxK: maximum total clusters to return (hard cap)
//
// Returns indices of selected clusters (always includes top minK, plus additional
// clusters within threshold of the minK-th score, up to maxK total).
//
// Edge cases:
//   - threshold <= 0 or >= 1.0: falls back to strict top-K (no probing)
//   - maxK <= minK: returns exactly minK clusters
func selectClustersWithProbing(scores []float64, minK int, threshold float64, maxK int) []int {
	if len(scores) == 0 {
		return nil
	}

	// Handle edge cases - fall back to strict top-K
	if threshold <= 0 || threshold >= 1.0 {
		return selectTopKIndices(scores, minK)
	}
	if maxK <= 0 {
		maxK = minK
	}

	type indexedScore struct {
		index int
		score float64
	}

	indexed := make([]indexedScore, len(scores))
	for i, s := range scores {
		indexed[i] = indexedScore{index: i, score: s}
	}

	// Sort by score descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].score > indexed[j].score
	})

	// Always include top minK
	n := min(minK, len(indexed))
	result := make([]int, 0, maxK)
	for i := 0; i < n; i++ {
		result = append(result, indexed[i].index)
	}

	if n >= len(indexed) || n >= maxK {
		return result
	}

	// Calculate threshold score = score of rank minK * threshold
	// Note: indexed[n-1] is the minK-th element (0-indexed)
	kthScore := indexed[n-1].score

	// Handle negative scores properly
	// For positive scores: threshold 0.95 means include if >= 0.95 * kthScore
	// For negative scores: we need to be careful - if kthScore is -0.5,
	// threshold 0.95 gives -0.475, which is HIGHER (better) than -0.5.
	// We want scores WORSE than kth but close to it, so we need to invert for negatives.
	var thresholdScore float64
	if kthScore >= 0 {
		thresholdScore = kthScore * threshold
	} else {
		// For negative: if kth=-0.5, threshold=0.95, we want to include down to -0.5/0.95=-0.526
		thresholdScore = kthScore / threshold
	}

	// Add clusters that meet threshold up to maxK
	for i := n; i < len(indexed) && len(result) < maxK; i++ {
		if indexed[i].score >= thresholdScore {
			result = append(result, indexed[i].index)
		} else {
			// Scores are sorted descending, no need to check further
			break
		}
	}

	return result
}

// shuffleStrings shuffles a string slice in place.
func shuffleStrings(s []string) {
	for i := len(s) - 1; i > 0; i-- {
		jBig, _ := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		j := int(jBig.Int64())
		s[i], s[j] = s[j], s[i]
	}
}

// normalizeVectorCopy returns a normalized copy of the vector.
func normalizeVectorCopy(v []float64) []float64 {
	result := make([]float64, len(v))
	copy(result, v)

	var norm float64
	for _, val := range result {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return result
}

// dotProductVec computes the dot product of two vectors.
func dotProductVec(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// extractOriginalID is defined in enterprise_hierarchical.go
