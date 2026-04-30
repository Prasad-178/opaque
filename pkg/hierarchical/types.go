// Package hierarchical implements a three-level privacy-preserving vector search.
//
// Architecture:
//
//	Level 1: HE on super-bucket centroids (64 HE ops)
//	  - Client sends HE(query) to server
//	  - Server computes HE scores for ALL centroids
//	  - Client decrypts privately, selects top-K super-buckets
//	  - Server NEVER sees which super-buckets were selected
//
//	Level 2: Decoy-based sub-bucket fetch
//	  - Each super-bucket has 64 sub-buckets
//	  - Client fetches real sub-buckets + decoy sub-buckets
//	  - All requests shuffled - server can't tell real from decoy
//
//	Level 3: Local AES decrypt + scoring
//	  - Vectors encrypted with AES-256-GCM
//	  - Client decrypts and scores locally
//
// Privacy guarantees:
//   - Query vector: protected from server (HE)
//   - Super-bucket selection: protected from server (client-side decrypt)
//   - Sub-bucket interest: protected from server (decoys + shuffling)
//   - Vector values: protected from storage (AES)
//   - Final scores: protected from everyone (local computation)
package hierarchical

import (
	"time"
)

// Config holds configuration for the hierarchical index.
type Config struct {
	// Vector dimension
	Dimension int

	// Number of super-buckets (Level 1)
	// Each super-bucket has a centroid for HE scoring
	// Default: 64 (6-bit LSH)
	NumSuperBuckets int

	// Number of sub-buckets per super-bucket (Level 2)
	// Default: 64 (6-bit LSH with different seed)
	NumSubBuckets int

	// How many super-buckets to select after HE scoring
	// Higher = better recall, more sub-buckets to fetch
	// Default: 4
	TopSuperBuckets int

	// How many sub-buckets to fetch per selected super-bucket
	// Includes primary + neighbors for better recall
	// Default: 3
	SubBucketsPerSuper int

	// Number of decoy sub-buckets to fetch
	// Higher = better privacy, more data to process
	// Default: 8
	NumDecoys int

	// TargetEpsilon, when > 0, overrides NumDecoys at index-build / search
	// configuration time. The system derives NumDecoys from the requested
	// (ε)-DP-style upper bound on cluster-identity distinguishability:
	//   NumDecoys = ceil((NumSuperBuckets - TopSuperBuckets) * exp(-TargetEpsilon))
	// Smaller ε ⇒ stronger privacy ⇒ larger NumDecoys ⇒ more bandwidth.
	// Use ComputeDecoyCountForEpsilon to derive concrete NumDecoys values.
	// Default: 0 (unused; NumDecoys takes effect directly).
	//
	// Note: this is an informal upper bound for the uniform-K-from-non-selected
	// sampling scheme. A formal proof under the standard (ε,δ)-DP framework
	// requires switching to Bernoulli sampling per cluster — planned future work.
	TargetEpsilon float64

	// Multi-probe configuration for improved recall
	// Instead of hard top-K cutoff, use confidence-based selection

	// ProbeThreshold is the minimum score ratio for additional clusters.
	// Clusters with score >= (Kth cluster score * ProbeThreshold) are included.
	// Example: 0.95 means include if >= 95% of the Kth cluster's score.
	// Set to 1.0 to disable multi-probe (strict top-K only).
	// Default: 0.95
	ProbeThreshold float64

	// MaxProbeClusters is the maximum total clusters to probe.
	// This caps how many clusters can be selected even with multi-probe.
	// Default: 48 (allows up to 16 additional clusters beyond TopSuperBuckets)
	MaxProbeClusters int

	// RedundantAssignments controls soft cluster assignment for improved recall.
	// Each vector is assigned to the N nearest clusters during indexing.
	// This improves recall for boundary queries at the cost of increased storage.
	// Set to 1 for single assignment (default), 2 for redundant assignment.
	// Storage impact: ~N times the original index size.
	// Default: 1 (single assignment, no redundancy)
	RedundantAssignments int

	// LSH seeds for super and sub bucket assignment
	LSHSuperSeed int64
	LSHSubSeed   int64

	// NumKMeansInit is the number of k-means initializations to run during Build.
	// Multiple initializations run in parallel; the result with the lowest inertia wins.
	// Default: 1 (single initialization).
	NumKMeansInit int

	// NormalizedStorage indicates vectors are stored pre-normalized.
	// When true, the builder encrypts normalized vectors and the search client
	// skips per-vector normalization during local scoring.
	// Default: false (normalize on-the-fly during search for backwards compatibility).
	NormalizedStorage bool

	// ProbeStrategy selects the cluster probing method during search.
	// "threshold" (default) uses ProbeThreshold ratio.
	// "gap" uses adaptive score-gap detection.
	ProbeStrategy string

	// GapMultiplier controls gap-based probing sensitivity.
	// Stop expanding when gap between consecutive scores > GapMultiplier * median gap.
	// Only used when ProbeStrategy is "gap".
	// Default: 2.0.
	GapMultiplier float64
}

// DefaultConfig returns sensible defaults for 100K vectors.
// Tuned for good privacy with 64 HE ops.
// With 64 super-buckets and no sub-buckets: ~1562 vectors per bucket.
func DefaultConfig() Config {
	return Config{
		Dimension:          128,
		NumSuperBuckets:    64,  // 2^6, for HE scoring (64 HE ops)
		NumSubBuckets:      1,   // No sub-buckets = ~1562 vectors per bucket for 100K
		TopSuperBuckets:    32,  // Select top 32 after HE (gives ~90%+ recall)
		SubBucketsPerSuper: 1,   // Just the primary bucket
		NumDecoys:            8,   // 8 decoy buckets for privacy
		ProbeThreshold:       0.95, // Include clusters within 5% of Kth score
		MaxProbeClusters:     48,  // Allow up to 48 clusters with multi-probe
		RedundantAssignments: 1,   // Single assignment (no redundancy)
		LSHSuperSeed:         42,
		LSHSubSeed:           137, // Different seed (unused with NumSubBuckets=1)
	}
}

// HighPrivacyConfig returns config optimized for maximum privacy.
// Fewer, larger buckets = better k-anonymity.
func HighPrivacyConfig() Config {
	return Config{
		Dimension:          128,
		NumSuperBuckets:    32,  // Fewer buckets = ~3125 vectors per bucket for 100K
		NumSubBuckets:      1,   // No sub-buckets
		TopSuperBuckets:    16,  // Select top 16 after HE (~80% recall)
		SubBucketsPerSuper: 1,   // Just the primary bucket
		NumDecoys:            12,  // More decoys
		ProbeThreshold:       0.98, // Stricter threshold for privacy
		MaxProbeClusters:     24,  // Limit probe clusters for privacy
		RedundantAssignments: 1,   // Single assignment (no redundancy for privacy)
		LSHSuperSeed:         42,
		LSHSubSeed:           137,
	}
}

// HighRecallConfig returns config optimized for better recall.
// Uses more clusters for higher coverage.
func HighRecallConfig() Config {
	return Config{
		Dimension:          128,
		NumSuperBuckets:    64,  // 64 HE ops
		NumSubBuckets:      1,   // No sub-buckets
		TopSuperBuckets:    48,  // Select 48/64 = 75% of clusters (~96% recall)
		SubBucketsPerSuper: 1,   // Just the primary bucket
		NumDecoys:            8,   // 8 decoy buckets
		ProbeThreshold:       0.90, // More aggressive probing
		MaxProbeClusters:     56,  // Allow up to 56 clusters (87.5% coverage)
		RedundantAssignments: 2,   // Assign to 2 clusters for better recall
		LSHSuperSeed:         42,
		LSHSubSeed:           137,
	}
}

// ComputeDecoyCountForEpsilon returns the NumDecoys value that approximately
// satisfies an ε-DP-style upper bound on cluster-identity distinguishability
// for the uniform-K-from-non-selected decoy scheme.
//
// Bound derivation (informal): an HBC server observing a fetched cluster set
// S of size K_real + K_decoy cannot distinguish "real cluster is c" from
// "real cluster is c' ∈ S" with probability ratio greater than
//
//   (N - K_real) / K_decoy
//
// where N is the total number of super-buckets. Setting this ratio ≤ e^ε
// gives K_decoy ≥ (N - K_real) · e^(-ε), rounded up.
//
// Caller is responsible for checking that the result fits NumSuperBuckets.
// Returns 0 for non-positive ε. Returns at least 1 when ε > 0 and pool > 0.
//
// This is NOT a formally tight (ε,δ)-DP bound; it is a useful scaling guide.
// See SECURITY_MODEL.md §5 for the formal claim and limitations.
func ComputeDecoyCountForEpsilon(numSuperBuckets, topSuperBuckets int, epsilon float64) int {
	if epsilon <= 0 {
		return 0
	}
	pool := numSuperBuckets - topSuperBuckets
	if pool <= 0 {
		return 0
	}
	// math.Exp imported via stdlib in callers; inline here to avoid extra dep.
	exp := func(x float64) float64 {
		// 32-term Taylor sufficient for ε in [0, 10] domain.
		sum, term := 1.0, 1.0
		for i := 1; i < 32; i++ {
			term *= x / float64(i)
			sum += term
		}
		return sum
	}
	target := float64(pool) * exp(-epsilon)
	if target < 1 {
		return 1
	}
	// Ceil.
	t := int(target)
	if float64(t) < target {
		t++
	}
	return t
}

// SuperBucket represents a top-level bucket with a centroid.
type SuperBucket struct {
	// Unique identifier (0 to NumSuperBuckets-1)
	ID int

	// Centroid vector (mean of all vectors in this super-bucket)
	// Used for HE scoring at Level 1
	Centroid []float64

	// Number of vectors in this super-bucket
	VectorCount int

	// Running sum for incremental centroid updates (Welford's algorithm)
	sum []float64
}

// SubBucket represents a second-level bucket within a super-bucket.
type SubBucket struct {
	// Super-bucket this belongs to
	SuperID int

	// Sub-bucket ID within the super-bucket (0 to NumSubBuckets-1)
	SubID int

	// Full bucket key for storage (e.g., "07_23")
	BucketKey string

	// Number of vectors in this sub-bucket
	VectorCount int
}

// VectorLocation tracks where a vector is stored.
type VectorLocation struct {
	ID        string
	SuperID   int
	SubID     int    // Deprecated: Sub-buckets have been removed. Always 0.
	BucketKey string // Format: "XX" (super-bucket only)
}

// SearchResult contains the final search results with timing breakdown.
type SearchResult struct {
	// Top-K results
	Results []Result

	// Timing breakdown
	Timing SearchTiming

	// Statistics
	Stats SearchStats
}

// Result is a single search result.
type Result struct {
	ID    string
	Score float64
}

// SearchTiming provides detailed timing for each phase.
type SearchTiming struct {
	// Level 1: HE operations
	HEEncryptQuery   time.Duration
	HECentroidScores time.Duration // All 64 centroid dot products
	HEDecryptScores  time.Duration

	// Level 1 sub-phases (for GPU profiling analysis)
	HEQueryPack     time.Duration // Query replication into CKKS slots
	HEBatchMultiply time.Duration // HE multiply + rescale (NTT-dominated)
	HEBatchRotate   time.Duration // Rotation tree for partial sums
	HEBatchDecrypt  time.Duration // Decrypt packed results

	// Level 2: Sub-bucket fetch
	BucketSelection time.Duration // Selecting buckets + generating decoys
	BucketFetch     time.Duration // Fetching from storage

	// Level 3: Local scoring
	AESDecrypt   time.Duration
	LocalScoring time.Duration

	// Total
	Total time.Duration
}

// SearchStats provides statistics about the search.
type SearchStats struct {
	// Level 1
	HEOperations         int // Should be NumSuperBuckets (64)
	SuperBucketsSelected int

	// Level 2
	RealSubBuckets  int
	DecoySubBuckets int
	TotalSubBuckets int
	BlobsFetched    int

	// Level 3
	VectorsScored    int
	DecryptSucceeded int // Blobs successfully decrypted
	DecryptFailed    int // Blobs that failed decryption (expected for decoys)

	// Cluster selection diagnostics (for accuracy analysis)
	// These help identify WHERE recall is lost:
	// - If ClusterCoverage >> Recall: problem is in local scoring
	// - If ClusterCoverage ≈ Recall: problem is in cluster selection
	SelectedClusters []int // Which clusters were selected (for analysis)
}

// ClusterDiagnostics provides detailed cluster selection analysis.
// Used to diagnose whether recall issues are from cluster selection
// or from post-selection vector ranking.
type ClusterDiagnostics struct {
	// Ground truth analysis
	GTVectorCluster int // Which cluster contains the #1 GT vector
	GTClusterRank   int // Rank of GT cluster in HE scoring (1=best, 0=not scored)

	// Coverage analysis
	GTTop10Clusters   []int   // Clusters containing GT top-10 vectors
	GTInSelected      int     // How many GT top-10 are in selected clusters
	ClusterCoverage   float64 // GTInSelected / 10 (theoretical max recall)
	GTClusterSelected bool    // Was the #1 GT vector's cluster selected?

	// HE vs Plaintext comparison
	PlaintextTopK    []int   // Clusters that plaintext scoring would select
	HETopK           []int   // Clusters that HE scoring actually selected
	SelectionMatch   int     // How many clusters match between HE and plaintext
	SelectionDivergence int  // How many clusters differ

	// Per-cluster scores (for debugging)
	ClusterScoresHE       []float64 // HE-decrypted scores for each cluster
	ClusterScoresPlain    []float64 // Plaintext scores for each cluster
}

// IndexStats provides statistics about the index.
type IndexStats struct {
	TotalVectors      int
	NumSuperBuckets   int
	NumSubBuckets     int // Total across all super-buckets
	AvgVectorsPerSub  float64
	MinVectorsPerSub  int
	MaxVectorsPerSub  int
	EmptySubBuckets   int
}
