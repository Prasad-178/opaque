// Package embeddings provides utilities for loading pre-computed embedding datasets
// like SIFT1M, GloVe, etc. for benchmarking the private vector search system.
package embeddings

// Dataset represents a loaded embedding dataset with optional ground truth.
type Dataset struct {
	// Name of the dataset (e.g., "sift1m", "glove")
	Name string

	// Dimension of each vector
	Dimension int

	// Vectors is the main dataset (base vectors)
	Vectors [][]float64

	// IDs for each vector (optional, auto-generated if not provided)
	IDs []string

	// Queries is the query set (for benchmarking)
	Queries [][]float64

	// GroundTruth contains the true nearest neighbors for each query
	// GroundTruth[i] = indices of nearest neighbors for Queries[i]
	GroundTruth [][]int
}

// Stats returns statistics about the dataset.
func (d *Dataset) Stats() DatasetStats {
	return DatasetStats{
		Name:             d.Name,
		NumVectors:       len(d.Vectors),
		NumQueries:       len(d.Queries),
		Dimension:        d.Dimension,
		HasGroundTruth:   len(d.GroundTruth) > 0,
		GroundTruthDepth: d.groundTruthDepth(),
	}
}

func (d *Dataset) groundTruthDepth() int {
	if len(d.GroundTruth) == 0 {
		return 0
	}
	return len(d.GroundTruth[0])
}

// DatasetStats contains summary statistics about a dataset.
type DatasetStats struct {
	Name             string
	NumVectors       int
	NumQueries       int
	Dimension        int
	HasGroundTruth   bool
	GroundTruthDepth int // Number of ground truth neighbors per query
}

// Subset returns a subset of the dataset with the first n vectors.
// Useful for testing with smaller datasets.
func (d *Dataset) Subset(n int) *Dataset {
	if n > len(d.Vectors) {
		n = len(d.Vectors)
	}

	ids := d.IDs
	if len(ids) > n {
		ids = ids[:n]
	}

	return &Dataset{
		Name:        d.Name + "_subset",
		Dimension:   d.Dimension,
		Vectors:     d.Vectors[:n],
		IDs:         ids,
		Queries:     d.Queries,
		GroundTruth: d.GroundTruth,
	}
}
