// Package cluster provides clustering algorithms for vector indexing.
package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// KMeans performs k-means clustering on vectors.
type KMeans struct {
	K          int         // Number of clusters
	MaxIter    int         // Maximum iterations
	Tolerance  float64     // Convergence tolerance
	Seed       int64       // Random seed for reproducibility
	Centroids  [][]float64 // Cluster centroids (after Fit)
	Labels     []int       // Cluster assignment for each vector (after Fit)
	Iterations int         // Actual iterations run
}

// Config holds k-means configuration.
type Config struct {
	K         int     // Number of clusters
	MaxIter   int     // Maximum iterations (default: 100)
	Tolerance float64 // Convergence tolerance (default: 1e-4)
	Seed      int64   // Random seed
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
	}
}

// Fit runs k-means clustering on the given vectors.
// Returns cluster centroids and labels for each vector.
func (km *KMeans) Fit(vectors [][]float64) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided")
	}
	if km.K > len(vectors) {
		return fmt.Errorf("k (%d) cannot be larger than number of vectors (%d)", km.K, len(vectors))
	}

	dim := len(vectors[0])
	rng := rand.New(rand.NewSource(km.Seed))

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
			break
		}
		prevInertia = inertia

		// Update step: recompute centroids
		km.updateCentroids(vectors, dim)

		km.Iterations = iter + 1
	}

	return nil
}

// kmeansppInit initializes centroids using k-means++ algorithm.
// This provides better starting points than random initialization.
func kmeansppInit(vectors [][]float64, k int, rng *rand.Rand) [][]float64 {
	n := len(vectors)
	dim := len(vectors[0])
	centroids := make([][]float64, k)

	// Choose first centroid randomly
	firstIdx := rng.Intn(n)
	centroids[0] = make([]float64, dim)
	copy(centroids[0], vectors[firstIdx])

	// Distance from each point to nearest centroid
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

		centroids[c] = make([]float64, dim)
		copy(centroids[c], vectors[chosen])
	}

	return centroids
}

// assignClusters assigns each vector to its nearest centroid.
// Returns total inertia (sum of squared distances to centroids).
func (km *KMeans) assignClusters(vectors [][]float64) float64 {
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
func (km *KMeans) updateCentroids(vectors [][]float64, dim int) {
	// Reset centroids and counts
	counts := make([]int, km.K)
	newCentroids := make([][]float64, km.K)
	for c := 0; c < km.K; c++ {
		newCentroids[c] = make([]float64, dim)
	}

	// Sum vectors in each cluster
	for i, vec := range vectors {
		c := km.Labels[i]
		counts[c]++
		for j, v := range vec {
			newCentroids[c][j] += v
		}
	}

	// Compute means
	for c := 0; c < km.K; c++ {
		if counts[c] > 0 {
			for j := range newCentroids[c] {
				newCentroids[c][j] /= float64(counts[c])
			}
			km.Centroids[c] = newCentroids[c]
		}
		// If cluster is empty, keep old centroid
	}
}

// Predict returns the nearest centroid index for a query vector.
func (km *KMeans) Predict(query []float64) int {
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
func (km *KMeans) PredictTopK(query []float64, k int) []int {
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

// squaredEuclidean computes squared Euclidean distance between two vectors.
func squaredEuclidean(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// CosineSimilarity computes cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// NormalizeVector returns a unit-length copy of the vector.
func NormalizeVector(v []float64) []float64 {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	result := make([]float64, len(v))
	if norm > 0 {
		for i, val := range v {
			result[i] = val / norm
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
