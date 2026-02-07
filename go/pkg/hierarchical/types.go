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
	VectorsScored int

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
