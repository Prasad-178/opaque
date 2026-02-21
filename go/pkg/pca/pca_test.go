package pca

import (
	"math"
	"math/rand"
	"testing"
)

func generateTestVectors(n, d int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float64, n)
	for i := range vectors {
		vec := make([]float64, d)
		for j := range vec {
			vec[j] = rng.NormFloat64()
		}
		vectors[i] = vec
	}
	return vectors
}

func normalizeVector(v []float64) []float64 {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

func TestFitBasic(t *testing.T) {
	vectors := generateTestVectors(100, 32, 42)

	pca, err := Fit(vectors, 16)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	if pca.OriginalDim != 32 {
		t.Errorf("expected OriginalDim=32, got %d", pca.OriginalDim)
	}
	if pca.ReducedDim != 16 {
		t.Errorf("expected ReducedDim=16, got %d", pca.ReducedDim)
	}
	if len(pca.Mean) != 32 {
		t.Errorf("expected Mean length=32, got %d", len(pca.Mean))
	}
	if len(pca.SingularValues) != 16 {
		t.Errorf("expected 16 singular values, got %d", len(pca.SingularValues))
	}
	if len(pca.VarianceExplained) != 16 {
		t.Errorf("expected 16 variance values, got %d", len(pca.VarianceExplained))
	}

	// Singular values should be in descending order
	for i := 1; i < len(pca.SingularValues); i++ {
		if pca.SingularValues[i] > pca.SingularValues[i-1]+1e-10 {
			t.Errorf("singular values not descending at index %d: %.4f > %.4f",
				i, pca.SingularValues[i], pca.SingularValues[i-1])
		}
	}
}

func TestFitValidation(t *testing.T) {
	vectors := generateTestVectors(10, 8, 42)

	tests := []struct {
		name      string
		vectors   [][]float64
		targetDim int
		wantErr   bool
	}{
		{"valid", vectors, 4, false},
		{"targetDim=originalDim", vectors, 8, false},
		{"targetDim=1", vectors, 1, false},
		{"empty vectors", nil, 4, true},
		{"targetDim too large", vectors, 9, true},
		{"targetDim zero", vectors, 0, true},
		{"targetDim negative", vectors, -1, true},
		{"too few vectors", vectors[:2], 4, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Fit(tt.vectors, tt.targetDim)
			if (err != nil) != tt.wantErr {
				t.Errorf("Fit() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTransformDimension(t *testing.T) {
	vectors := generateTestVectors(50, 16, 42)

	pca, err := Fit(vectors, 8)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	transformed, err := pca.Transform(vectors[0])
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	if len(transformed) != 8 {
		t.Errorf("expected output dim=8, got %d", len(transformed))
	}
}

func TestTransformBatch(t *testing.T) {
	vectors := generateTestVectors(50, 16, 42)

	pca, err := Fit(vectors, 8)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	batch, err := pca.TransformBatch(vectors)
	if err != nil {
		t.Fatalf("TransformBatch failed: %v", err)
	}

	if len(batch) != len(vectors) {
		t.Errorf("expected %d results, got %d", len(vectors), len(batch))
	}

	// Verify batch results match individual transforms
	for i, v := range vectors {
		single, err := pca.Transform(v)
		if err != nil {
			t.Fatalf("Transform failed for vector %d: %v", i, err)
		}
		for j := range single {
			if math.Abs(single[j]-batch[i][j]) > 1e-10 {
				t.Errorf("batch[%d][%d] = %.10f, single = %.10f", i, j, batch[i][j], single[j])
			}
		}
	}
}

func TestInverseTransform(t *testing.T) {
	vectors := generateTestVectors(100, 32, 42)

	// With all components, reconstruction should be near-perfect
	pca, err := Fit(vectors, 32)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	for i := 0; i < 5; i++ {
		reduced, err := pca.Transform(vectors[i])
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}

		reconstructed, err := pca.InverseTransform(reduced)
		if err != nil {
			t.Fatalf("InverseTransform failed: %v", err)
		}

		var maxErr float64
		for j := range vectors[i] {
			diff := math.Abs(vectors[i][j] - reconstructed[j])
			if diff > maxErr {
				maxErr = diff
			}
		}

		if maxErr > 1e-8 {
			t.Errorf("vector %d: max reconstruction error = %.2e (expected < 1e-8)", i, maxErr)
		}
	}
}

func TestVarianceExplained(t *testing.T) {
	// Create data with clear variance structure:
	// First dimension has high variance, others low
	rng := rand.New(rand.NewSource(42))
	n := 200
	d := 16
	vectors := make([][]float64, n)
	for i := range vectors {
		vec := make([]float64, d)
		vec[0] = rng.NormFloat64() * 10 // High variance
		vec[1] = rng.NormFloat64() * 5  // Medium variance
		for j := 2; j < d; j++ {
			vec[j] = rng.NormFloat64() * 0.1 // Low variance
		}
		vectors[i] = vec
	}

	pca, err := Fit(vectors, 4)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// First component should explain most variance
	if pca.VarianceExplained[0] < 0.5 {
		t.Errorf("first component should explain >50%% variance, got %.2f%%",
			pca.VarianceExplained[0]*100)
	}

	// Total variance with 4 components should be high
	total := pca.TotalVarianceExplained()
	if total < 0.95 {
		t.Errorf("4 components should explain >95%% variance, got %.2f%%", total*100)
	}

	t.Logf("Variance explained per component: %.2f%%, %.2f%%, %.2f%%, %.2f%%",
		pca.VarianceExplained[0]*100, pca.VarianceExplained[1]*100,
		pca.VarianceExplained[2]*100, pca.VarianceExplained[3]*100)
	t.Logf("Total variance explained: %.2f%%", total*100)
}

func TestReconstructionError(t *testing.T) {
	vectors := generateTestVectors(100, 32, 42)

	// More components should give lower reconstruction error
	pcaFull, _ := Fit(vectors, 32)
	pcaHalf, _ := Fit(vectors, 16)
	pcaQuarter, _ := Fit(vectors, 8)

	errFull, _ := pcaFull.ReconstructionError(vectors)
	errHalf, _ := pcaHalf.ReconstructionError(vectors)
	errQuarter, _ := pcaQuarter.ReconstructionError(vectors)

	if errFull > 1e-10 {
		t.Errorf("full PCA should have near-zero error, got %.2e", errFull)
	}
	if errHalf >= errQuarter {
		t.Errorf("half PCA (%.6f) should have less error than quarter PCA (%.6f)",
			errHalf, errQuarter)
	}

	t.Logf("Reconstruction error: full=%.2e, half=%.6f, quarter=%.6f",
		errFull, errHalf, errQuarter)
}

func TestCosineSimilarityPreservation(t *testing.T) {
	// Generate normalized vectors (unit length)
	rng := rand.New(rand.NewSource(42))
	n := 100
	d := 64
	vectors := make([][]float64, n)
	for i := range vectors {
		vec := make([]float64, d)
		for j := range vec {
			vec[j] = rng.NormFloat64()
		}
		vectors[i] = normalizeVector(vec)
	}

	pca, err := Fit(vectors, 32)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Transform and normalize
	reduced := make([][]float64, n)
	for i, v := range vectors {
		r, err := pca.Transform(v)
		if err != nil {
			t.Fatalf("Transform failed: %v", err)
		}
		reduced[i] = NormalizeTransformed(r)
	}

	// Compare cosine similarities in original vs reduced space
	var totalCorrelation float64
	numPairs := 0

	for i := 0; i < 20; i++ {
		for j := i + 1; j < 20; j++ {
			// Original cosine similarity
			var origSim float64
			for k := 0; k < d; k++ {
				origSim += vectors[i][k] * vectors[j][k]
			}

			// Reduced cosine similarity
			var redSim float64
			for k := 0; k < 32; k++ {
				redSim += reduced[i][k] * reduced[j][k]
			}

			totalCorrelation += origSim * redSim
			numPairs++
		}
	}

	// The correlation between original and reduced similarities should be positive
	avgCorrelation := totalCorrelation / float64(numPairs)
	if avgCorrelation < 0 {
		t.Errorf("expected positive correlation, got %.4f", avgCorrelation)
	}

	t.Logf("Average similarity correlation: %.4f (positive = good)", avgCorrelation)
	t.Logf("Total variance explained: %.2f%%", pca.TotalVarianceExplained()*100)
}

func TestNormalizeTransformed(t *testing.T) {
	v := []float64{3.0, 4.0}
	normalized := NormalizeTransformed(v)

	var norm float64
	for _, val := range normalized {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("expected unit norm, got %.10f", norm)
	}

	// Zero vector should be returned unchanged
	zero := NormalizeTransformed([]float64{0, 0, 0})
	for _, val := range zero {
		if val != 0 {
			t.Errorf("expected zero vector, got %v", zero)
		}
	}
}

func TestTransformInvalidDimension(t *testing.T) {
	vectors := generateTestVectors(50, 16, 42)
	pca, _ := Fit(vectors, 8)

	_, err := pca.Transform([]float64{1, 2, 3}) // wrong dimension
	if err == nil {
		t.Error("expected error for wrong input dimension")
	}
}
