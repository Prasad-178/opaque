// Package cluster provides clustering algorithms for vector indexing.
//
// Internal storage is float32 throughout — half the memory bandwidth of
// float64 with imperceptible precision impact for SIFT-class workloads
// (~1.2e-7 per element, ~6 orders of magnitude below recall sensitivity)
// and zero impact for ada-002-class embeddings (already float32-native).
// Conversion to float64 happens only at HE-encoding boundaries where
// Lattigo's encoder requires it.
package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// KMeans performs k-means clustering on float32 vectors.
type KMeans struct {
	K          int         // Number of clusters
	MaxIter    int         // Maximum iterations
	Tolerance  float64     // Convergence tolerance (still float64 — scalar accumulator)
	Seed       int64       // Random seed for reproducibility
	NumInit    int         // Number of initializations (best inertia wins)
	Centroids  [][]float32 // Cluster centroids (after Fit)
	Labels     []int       // Cluster assignment for each vector (after Fit)
	Iterations int         // Actual iterations run
	Inertia    float64     // Final inertia (sum of squared distances to centroids)
}

// Config holds k-means configuration.
type Config struct {
	K         int     // Number of clusters
	MaxIter   int     // Maximum iterations (default: 100)
	Tolerance float64 // Convergence tolerance (default: 1e-4)
	Seed      int64   // Random seed
	NumInit   int     // Number of initializations to try (default: 1). Best inertia wins.
}

// DefaultConfig returns default k-means configuration.
func DefaultConfig(k int) Config {
	return Config{
		K:         k,
		MaxIter:   100,
		Tolerance: 1e-4,
		Seed:      42,
	}
}

// NewKMeans creates a new k-means clusterer.
func NewKMeans(cfg Config) *KMeans {
	if cfg.MaxIter <= 0 {
		cfg.MaxIter = 100
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-4
	}
	return &KMeans{
		K:         cfg.K,
		MaxIter:   cfg.MaxIter,
		Tolerance: cfg.Tolerance,
		Seed:      cfg.Seed,
		NumInit:   cfg.NumInit,
	}
}

// Fit runs k-means clustering on the given vectors.
// If NumInit > 1, runs multiple initializations in parallel and keeps the best result.
func (km *KMeans) Fit(vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}
	if km.K > len(vectors) {
		return fmt.Errorf("k (%d) cannot be larger than number of vectors (%d)", km.K, len(vectors))
	}

	numInit := km.numInit()
	if numInit <= 1 {
		return km.fitOnce(vectors, km.Seed)
	}

	// Run multiple initializations in parallel, keep the best (lowest inertia)
	type initResult struct {
		centroids  [][]float32
		labels     []int
		iterations int
		inertia    float64
		err        error
	}
	results := make([]initResult, numInit)

	var wg sync.WaitGroup
	for n := 0; n < numInit; n++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			candidate := &KMeans{
				K:         km.K,
				MaxIter:   km.MaxIter,
				Tolerance: km.Tolerance,
				Seed:      km.Seed + int64(n),
			}
			if err := candidate.fitOnce(vectors, candidate.Seed); err != nil {
				results[n] = initResult{err: err}
				return
			}
			results[n] = initResult{
				centroids:  candidate.Centroids,
				labels:     candidate.Labels,
				iterations: candidate.Iterations,
				inertia:    candidate.Inertia,
			}
		}(n)
	}
	wg.Wait()

	// Pick the best result (lowest inertia)
	bestIdx := -1
	bestInertia := math.MaxFloat64
	for i, r := range results {
		if r.err != nil {
			return r.err
		}
		if r.inertia < bestInertia {
			bestInertia = r.inertia
			bestIdx = i
		}
	}

	best := results[bestIdx]
	km.Centroids = best.centroids
	km.Labels = best.labels
	km.Iterations = best.iterations
	km.Inertia = best.inertia
	return nil
}

// numInit returns the effective number of initializations.
func (km *KMeans) numInit() int {
	if km.NumInit > 1 {
		return km.NumInit
	}
	return 1
}

// fitOnce runs a single k-means initialization and iteration cycle.
func (km *KMeans) fitOnce(vectors [][]float32, seed int64) error {
	dim := len(vectors[0])
	rng := rand.New(rand.NewSource(seed))

	// Initialize centroids using k-means++ for better starting points
	km.Centroids = kmeansppInit(vectors, km.K, rng)
	km.Labels = make([]int, len(vectors))

	var prevInertia float64 = math.MaxFloat64

	for iter := 0; iter < km.MaxIter; iter++ {
		// Assignment step: assign each vector to nearest centroid
		inertia := km.assignClusters(vectors)

		// Check convergence
		if math.Abs(prevInertia-inertia) < km.Tolerance*float64(len(vectors)) {
			km.Iterations = iter + 1
			km.Inertia = inertia
			break
		}
		prevInertia = inertia

		// Update step: recompute centroids
		km.updateCentroids(vectors, dim)

		km.Iterations = iter + 1
		km.Inertia = inertia
	}

	return nil
}

// kmeansppInit initializes centroids using k-means++ algorithm.
// This provides better starting points than random initialization.
func kmeansppInit(vectors [][]float32, k int, rng *rand.Rand) [][]float32 {
	n := len(vectors)
	dim := len(vectors[0])
	centroids := make([][]float32, k)

	// Choose first centroid randomly
	firstIdx := rng.Intn(n)
	centroids[0] = make([]float32, dim)
	copy(centroids[0], vectors[firstIdx])

	// Distance from each point to nearest centroid (float64 accumulator;
	// individual distances accumulate over many vectors).
	distances := make([]float64, n)
	for i := range distances {
		distances[i] = math.MaxFloat64
	}

	// Choose remaining centroids
	for c := 1; c < k; c++ {
		// Update distances to nearest centroid
		var totalDist float64
		for i, vec := range vectors {
			d := squaredEuclidean(vec, centroids[c-1])
			if d < distances[i] {
				distances[i] = d
			}
			totalDist += distances[i]
		}

		// Choose next centroid with probability proportional to distance squared
		threshold := rng.Float64() * totalDist
		var cumulative float64
		chosen := 0
		for i, d := range distances {
			cumulative += d
			if cumulative >= threshold {
				chosen = i
				break
			}
		}

		centroids[c] = make([]float32, dim)
		copy(centroids[c], vectors[chosen])
	}

	return centroids
}

// assignClusters assigns each vector to its nearest centroid.
// Returns total inertia (sum of squared distances to centroids).
func (km *KMeans) assignClusters(vectors [][]float32) float64 {
	var totalInertia float64
	var mu sync.Mutex

	// Parallel assignment for large datasets
	numWorkers := 4
	chunkSize := (len(vectors) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			var localInertia float64

			for i := start; i < end; i++ {
				minDist := math.MaxFloat64
				minIdx := 0
				for c, centroid := range km.Centroids {
					d := squaredEuclidean(vectors[i], centroid)
					if d < minDist {
						minDist = d
						minIdx = c
					}
				}
				km.Labels[i] = minIdx
				localInertia += minDist
			}

			mu.Lock()
			totalInertia += localInertia
			mu.Unlock()
		}(start, end)
	}
	wg.Wait()

	return totalInertia
}

// updateCentroids recomputes centroids as means of assigned vectors.
// Empty clusters are recovered by reassigning the farthest vector from the largest cluster.
// Centroids are accumulated in float64 for numerical stability, then downcast to float32.
func (km *KMeans) updateCentroids(vectors [][]float32, dim int) {
	// Reset centroids and counts
	counts := make([]int, km.K)
	newCentroids64 := make([][]float64, km.K)
	for c := 0; c < km.K; c++ {
		newCentroids64[c] = make([]float64, dim)
	}

	// Sum vectors in each cluster (in float64 to avoid catastrophic
	// cancellation when many small float32 contributions accumulate).
	for i, vec := range vectors {
		c := km.Labels[i]
		counts[c]++
		for j, v := range vec {
			newCentroids64[c][j] += float64(v)
		}
	}

	// Compute means and downcast back to float32 for storage.
	for c := 0; c < km.K; c++ {
		if counts[c] > 0 {
			scale := 1.0 / float64(counts[c])
			centroid := make([]float32, dim)
			for j := range newCentroids64[c] {
				centroid[j] = float32(newCentroids64[c][j] * scale)
			}
			km.Centroids[c] = centroid
		}
	}

	// Recover empty clusters: reassign the farthest vector from the largest cluster
	for c := 0; c < km.K; c++ {
		if counts[c] > 0 {
			continue
		}

		// Find the largest cluster
		largestCluster := 0
		for j := 1; j < km.K; j++ {
			if counts[j] > counts[largestCluster] {
				largestCluster = j
			}
		}
		if counts[largestCluster] <= 1 {
			continue // Can't split a cluster with only 1 vector
		}

		// Find the vector farthest from the largest cluster's centroid
		maxDist := -1.0
		farthestIdx := -1
		for i, label := range km.Labels {
			if label == largestCluster {
				d := squaredEuclidean(vectors[i], km.Centroids[largestCluster])
				if d > maxDist {
					maxDist = d
					farthestIdx = i
				}
			}
		}
		if farthestIdx < 0 {
			continue
		}

		// Reassign the farthest vector to the empty cluster
		copy(km.Centroids[c], vectors[farthestIdx])
		km.Labels[farthestIdx] = c
		counts[c] = 1
		counts[largestCluster]--
	}
}

// Predict returns the nearest centroid index for a query vector.
func (km *KMeans) Predict(query []float32) int {
	minDist := math.MaxFloat64
	minIdx := 0
	for c, centroid := range km.Centroids {
		d := squaredEuclidean(query, centroid)
		if d < minDist {
			minDist = d
			minIdx = c
		}
	}
	return minIdx
}

// PredictTopK returns the indices of the K nearest centroids.
func (km *KMeans) PredictTopK(query []float32, k int) []int {
	type centroidDist struct {
		idx  int
		dist float64
	}

	distances := make([]centroidDist, len(km.Centroids))
	for c, centroid := range km.Centroids {
		distances[c] = centroidDist{c, squaredEuclidean(query, centroid)}
	}

	// Partial sort to get top-k
	for i := 0; i < k && i < len(distances); i++ {
		minIdx := i
		for j := i + 1; j < len(distances); j++ {
			if distances[j].dist < distances[minIdx].dist {
				minIdx = j
			}
		}
		distances[i], distances[minIdx] = distances[minIdx], distances[i]
	}

	result := make([]int, min(k, len(distances)))
	for i := range result {
		result[i] = distances[i].idx
	}
	return result
}

// GetClusterSizes returns the number of vectors in each cluster.
func (km *KMeans) GetClusterSizes() []int {
	sizes := make([]int, km.K)
	for _, label := range km.Labels {
		sizes[label]++
	}
	return sizes
}

// squaredEuclidean computes squared Euclidean distance between two float32 vectors.
// Accumulator is float64 to avoid catastrophic cancellation for large dim.
func squaredEuclidean(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return sum
}

// CosineSimilarity computes cosine similarity between two float32 vectors.
// Accumulator is float64 to maintain precision over high dimensions.
func CosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// NormalizeVector returns a unit-length copy of the float32 vector.
// The norm computation is float64 for numerical stability; the output is
// downcast to float32 for storage.
func NormalizeVector(v []float32) []float32 {
	var norm float64
	for _, val := range v {
		fv := float64(val)
		norm += fv * fv
	}
	norm = math.Sqrt(norm)

	result := make([]float32, len(v))
	if norm > 0 {
		invNorm := 1.0 / norm
		for i, val := range v {
			result[i] = float32(float64(val) * invNorm)
		}
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// AsFloat32 converts a [][]float64 slice to [][]float32 (used at API boundaries
// where the caller still passes float64). Allocates a fresh outer + inner slice.
func AsFloat32(vectors [][]float64) [][]float32 {
	out := make([][]float32, len(vectors))
	for i, vec := range vectors {
		out[i] = make([]float32, len(vec))
		for j, v := range vec {
			out[i][j] = float32(v)
		}
	}
	return out
}

// AsFloat32One converts a single []float64 vector to []float32.
func AsFloat32One(v []float64) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

// AsFloat64 converts [][]float32 back to [][]float64 (used at HE-encoding
// boundary where Lattigo requires float64 inputs).
func AsFloat64(vectors [][]float32) [][]float64 {
	out := make([][]float64, len(vectors))
	for i, vec := range vectors {
		out[i] = make([]float64, len(vec))
		for j, v := range vec {
			out[i][j] = float64(v)
		}
	}
	return out
}

// AsFloat64One converts a single []float32 to []float64.
func AsFloat64One(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}
