package hierarchical

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"

	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/cluster"
	"github.com/opaque/opaque/go/pkg/encrypt"
	"github.com/opaque/opaque/go/pkg/lsh"
)

// Index is the hierarchical search index.
type Index struct {
	Config Config

	// Super-buckets with centroids
	SuperBuckets []*SuperBucket
	Centroids    [][]float64 // Precomputed for efficient HE ops

	// LSH indices (used when ClusteringMode is LSH)
	LSHSuper *lsh.Index
	LSHSub   *lsh.Index

	// K-means clustering (used when ClusteringMode is KMeans)
	KMeans *cluster.KMeans

	// Storage
	Store     blob.Store
	Encryptor *encrypt.AESGCM

	// Tracking
	VectorLocations map[string]*VectorLocation
	SubBucketCounts map[string]int // bucketKey -> count
}

// GetCentroids returns all super-bucket centroids.
// These are used for HE scoring at Level 1.
func (idx *Index) GetCentroids() [][]float64 {
	return idx.Centroids
}

// GetSuperBucketID returns the super-bucket ID for a vector.
func (idx *Index) GetSuperBucketID(vector []float64) int {
	return idx.LSHSuper.HashToIndexFromVector(vector, idx.Config.NumSuperBuckets)
}

// GetSubBucketID returns the sub-bucket ID for a vector within a super-bucket.
func (idx *Index) GetSubBucketID(vector []float64) int {
	return idx.LSHSub.HashToIndexFromVector(vector, idx.Config.NumSubBuckets)
}

// GetBucketKey returns the full bucket key for a vector.
func (idx *Index) GetBucketKey(vector []float64) string {
	superID := idx.GetSuperBucketID(vector)
	subID := idx.GetSubBucketID(vector)
	return formatBucketKey(superID, subID)
}

// GetNeighborSubBuckets returns neighboring sub-bucket keys for better recall.
// Given a super-bucket ID and primary sub-bucket ID, returns keys for the primary
// and its LSH neighbors.
func (idx *Index) GetNeighborSubBuckets(superID, primarySubID, numNeighbors int) []string {
	keys := make([]string, 0, numNeighbors+1)

	// Always include primary
	keys = append(keys, formatBucketKey(superID, primarySubID))

	// Add neighbors by flipping bits in the sub-bucket ID
	// This is a simple heuristic - nearby LSH buckets likely have similar vectors
	for i := 0; i < numNeighbors && i < idx.Config.NumSubBuckets; i++ {
		neighborID := (primarySubID + i + 1) % idx.Config.NumSubBuckets
		key := formatBucketKey(superID, neighborID)
		if key != keys[0] { // Don't duplicate primary
			keys = append(keys, key)
		}
		if len(keys) >= numNeighbors+1 {
			break
		}
	}

	return keys
}

// GetAllSubBucketKeys returns all sub-bucket keys for a given super-bucket.
func (idx *Index) GetAllSubBucketKeys(superID int) []string {
	keys := make([]string, idx.Config.NumSubBuckets)
	for i := 0; i < idx.Config.NumSubBuckets; i++ {
		keys[i] = formatBucketKey(superID, i)
	}
	return keys
}

// GetVectorCount returns the total number of vectors in the index.
func (idx *Index) GetVectorCount() int {
	return len(idx.VectorLocations)
}

// GetStats returns statistics about the index.
func (idx *Index) GetStats() IndexStats {
	stats := IndexStats{
		TotalVectors:    len(idx.VectorLocations),
		NumSuperBuckets: idx.Config.NumSuperBuckets,
		NumSubBuckets:   len(idx.SubBucketCounts),
		MinVectorsPerSub: -1,
		MaxVectorsPerSub: 0,
	}

	// Count vectors per sub-bucket
	for _, count := range idx.SubBucketCounts {
		if stats.MinVectorsPerSub < 0 || count < stats.MinVectorsPerSub {
			stats.MinVectorsPerSub = count
		}
		if count > stats.MaxVectorsPerSub {
			stats.MaxVectorsPerSub = count
		}
	}

	if stats.MinVectorsPerSub < 0 {
		stats.MinVectorsPerSub = 0
	}

	// Calculate empty sub-buckets
	totalPossibleSubBuckets := idx.Config.NumSuperBuckets * idx.Config.NumSubBuckets
	stats.EmptySubBuckets = totalPossibleSubBuckets - len(idx.SubBucketCounts)

	// Average
	if len(idx.SubBucketCounts) > 0 {
		stats.AvgVectorsPerSub = float64(stats.TotalVectors) / float64(len(idx.SubBucketCounts))
	}

	return stats
}

// indexData holds the serializable parts of the index.
type indexData struct {
	Config          Config
	SuperBuckets    []*SuperBucket
	Centroids       [][]float64
	VectorLocations map[string]*VectorLocation
	SubBucketCounts map[string]int
	LSHSuperSeed    int64
	LSHSubSeed      int64
	LSHSuperBits    int
	LSHSubBits      int
}

// Save serializes the index metadata to a writer.
// Note: The store and encryptor must be provided separately when loading.
func (idx *Index) Save(w io.Writer) error {
	data := indexData{
		Config:          idx.Config,
		SuperBuckets:    idx.SuperBuckets,
		Centroids:       idx.Centroids,
		VectorLocations: idx.VectorLocations,
		SubBucketCounts: idx.SubBucketCounts,
		LSHSuperSeed:    idx.Config.LSHSuperSeed,
		LSHSubSeed:      idx.Config.LSHSubSeed,
	}

	enc := gob.NewEncoder(w)
	return enc.Encode(data)
}

// SaveToFile saves the index metadata to a file.
func (idx *Index) SaveToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return idx.Save(f)
}

// Load deserializes index metadata from a reader.
// The store and encryptor must be provided.
func Load(r io.Reader, store blob.Store, encryptor *encrypt.AESGCM) (*Index, error) {
	var data indexData
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&data); err != nil {
		return nil, err
	}

	// Recreate LSH indices
	lshSuper := lsh.NewIndex(lsh.Config{
		Dimension: data.Config.Dimension,
		NumBits:   8, // Same as builder
		Seed:      data.Config.LSHSuperSeed,
	})

	lshSub := lsh.NewIndex(lsh.Config{
		Dimension: data.Config.Dimension,
		NumBits:   8,
		Seed:      data.Config.LSHSubSeed,
	})

	return &Index{
		Config:          data.Config,
		SuperBuckets:    data.SuperBuckets,
		Centroids:       data.Centroids,
		LSHSuper:        lshSuper,
		LSHSub:          lshSub,
		Store:           store,
		Encryptor:       encryptor,
		VectorLocations: data.VectorLocations,
		SubBucketCounts: data.SubBucketCounts,
	}, nil
}

// LoadFromFile loads index metadata from a file.
func LoadFromFile(path string, store blob.Store, encryptor *encrypt.AESGCM) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f, store, encryptor)
}

// ValidateConfig validates the configuration.
func ValidateConfig(cfg Config) error {
	if cfg.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive")
	}
	if cfg.NumSuperBuckets <= 0 {
		return fmt.Errorf("NumSuperBuckets must be positive")
	}
	if cfg.NumSubBuckets <= 0 {
		return fmt.Errorf("NumSubBuckets must be positive")
	}
	if cfg.TopSuperBuckets <= 0 || cfg.TopSuperBuckets > cfg.NumSuperBuckets {
		return fmt.Errorf("TopSuperBuckets must be between 1 and NumSuperBuckets")
	}
	if cfg.SubBucketsPerSuper <= 0 {
		return fmt.Errorf("SubBucketsPerSuper must be positive")
	}
	if cfg.NumDecoys < 0 {
		return fmt.Errorf("NumDecoys cannot be negative")
	}
	return nil
}
