package cluster

import (
	"math"
	"math/rand"
	"testing"
)

func TestKMeansBasic(t *testing.T) {
	// Create 3 obvious clusters
	vectors := [][]float64{
		// Cluster 0: around (0, 0)
		{0.1, 0.1}, {-0.1, 0.1}, {0.1, -0.1}, {-0.1, -0.1},
		// Cluster 1: around (10, 0)
		{10.1, 0.1}, {9.9, 0.1}, {10.1, -0.1}, {9.9, -0.1},
		// Cluster 2: around (5, 10)
		{5.1, 10.1}, {4.9, 10.1}, {5.1, 9.9}, {4.9, 9.9},
	}

	km := NewKMeans(Config{K: 3, MaxIter: 100, Seed: 42})
	err := km.Fit(vectors)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	t.Logf("Converged in %d iterations", km.Iterations)
	t.Logf("Centroids: %v", km.Centroids)
	t.Logf("Labels: %v", km.Labels)

	// Verify each cluster has 4 vectors
	sizes := km.GetClusterSizes()
	for i, size := range sizes {
		if size != 4 {
			t.Errorf("Cluster %d has %d vectors, expected 4", i, size)
		}
	}

	// Verify vectors in same original group have same label
	// Cluster 0: indices 0-3
	label0 := km.Labels[0]
	for i := 1; i < 4; i++ {
		if km.Labels[i] != label0 {
			t.Errorf("Vector %d has label %d, expected %d (same as vector 0)", i, km.Labels[i], label0)
		}
	}
}

func TestKMeansPredict(t *testing.T) {
	// Create clusters
	vectors := [][]float64{
		{0, 0}, {0.1, 0.1},
		{10, 0}, {10.1, 0.1},
		{5, 10}, {5.1, 10.1},
	}

	km := NewKMeans(Config{K: 3, MaxIter: 100, Seed: 42})
	km.Fit(vectors)

	// Test prediction - should assign to nearest cluster
	testCases := []struct {
		query    []float64
		expected int // The cluster containing similar training vectors
	}{
		{[]float64{0.05, 0.05}, km.Labels[0]},  // Near cluster 0
		{[]float64{10.05, 0.05}, km.Labels[2]}, // Near cluster 1
		{[]float64{5.05, 10.05}, km.Labels[4]}, // Near cluster 2
	}

	for i, tc := range testCases {
		predicted := km.Predict(tc.query)
		if predicted != tc.expected {
			t.Errorf("Case %d: predicted %d, expected %d", i, predicted, tc.expected)
		}
	}
}

func TestKMeansPredictTopK(t *testing.T) {
	// Create well-separated clusters
	vectors := make([][]float64, 0)
	for i := 0; i < 5; i++ {
		// Cluster i centered at (i*10, 0)
		vectors = append(vectors, []float64{float64(i*10) + 0.1, 0.1})
		vectors = append(vectors, []float64{float64(i*10) - 0.1, -0.1})
	}

	km := NewKMeans(Config{K: 5, MaxIter: 100, Seed: 42})
	km.Fit(vectors)

	// Query near cluster 1 (centered around 10, 0)
	query := []float64{10, 0}
	topK := km.PredictTopK(query, 3)

	t.Logf("Centroids: %v", km.Centroids)
	t.Logf("Top-3 for query %v: %v", query, topK)

	// The first result should be the cluster containing (10, 0)
	// Find which cluster that is
	expectedFirst := km.Labels[2] // vectors[2] is {10.1, 0.1}
	if topK[0] != expectedFirst {
		t.Errorf("Top-1 should be cluster %d, got %d", expectedFirst, topK[0])
	}
}

func TestKMeansLargeDataset(t *testing.T) {
	// Test with larger dataset
	rng := rand.New(rand.NewSource(42))
	n := 10000
	dim := 128
	k := 64

	vectors := make([][]float64, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}

	km := NewKMeans(Config{K: k, MaxIter: 50, Seed: 42})
	err := km.Fit(vectors)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	t.Logf("Converged in %d iterations", km.Iterations)

	// Check cluster sizes
	sizes := km.GetClusterSizes()
	minSize, maxSize := n, 0
	for _, s := range sizes {
		if s < minSize {
			minSize = s
		}
		if s > maxSize {
			maxSize = s
		}
	}
	avgSize := n / k

	t.Logf("Cluster sizes: min=%d, max=%d, avg=%d", minSize, maxSize, avgSize)

	// Verify all vectors are assigned
	totalAssigned := 0
	for _, s := range sizes {
		totalAssigned += s
	}
	if totalAssigned != n {
		t.Errorf("Total assigned %d != %d", totalAssigned, n)
	}
}

func TestKMeansQuality(t *testing.T) {
	// Test that k-means produces better centroids than random assignment
	rng := rand.New(rand.NewSource(42))
	n := 1000
	dim := 32
	k := 16

	// Generate vectors with some structure (mix of Gaussians)
	vectors := make([][]float64, n)
	for i := 0; i < n; i++ {
		center := i % k // Assign to one of k "true" clusters
		vectors[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			// Cluster centers spread out, with noise
			vectors[i][j] = float64(center)*2 + rng.NormFloat64()*0.5
		}
	}

	km := NewKMeans(Config{K: k, MaxIter: 100, Seed: 42})
	km.Fit(vectors)

	// Measure: for each vector, distance to its centroid vs distance to random centroid
	var avgDistToCentroid, avgDistToRandom float64
	for i, vec := range vectors {
		assignedCentroid := km.Centroids[km.Labels[i]]
		randomCentroid := km.Centroids[rng.Intn(k)]

		avgDistToCentroid += math.Sqrt(squaredEuclidean(vec, assignedCentroid))
		avgDistToRandom += math.Sqrt(squaredEuclidean(vec, randomCentroid))
	}
	avgDistToCentroid /= float64(n)
	avgDistToRandom /= float64(n)

	t.Logf("Avg distance to assigned centroid: %.4f", avgDistToCentroid)
	t.Logf("Avg distance to random centroid: %.4f", avgDistToRandom)

	// K-means centroids should be much closer than random
	if avgDistToCentroid >= avgDistToRandom*0.8 {
		t.Errorf("K-means centroids not significantly better than random")
	}
}

func BenchmarkKMeansFit(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	n := 10000
	dim := 128
	k := 64

	vectors := make([][]float64, n)
	for i := 0; i < n; i++ {
		vectors[i] = make([]float64, dim)
		for j := 0; j < dim; j++ {
			vectors[i][j] = rng.NormFloat64()
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km := NewKMeans(Config{K: k, MaxIter: 20, Seed: 42})
		km.Fit(vectors)
	}
}
