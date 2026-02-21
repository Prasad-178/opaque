// Package pca provides Principal Component Analysis for dimensionality reduction.
//
// PCA reduces high-dimensional vectors to a lower dimension while preserving
// the most variance. This is useful for reducing CKKS encryption/decryption
// latency and bandwidth, with minimal impact on search accuracy.
//
// The PCA transform is applied client-side before encryption, so it has no
// impact on privacy — the server never sees original or reduced vectors.
package pca

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// PCA holds a fitted PCA model for dimensionality reduction.
type PCA struct {
	// Components are the top-k principal component vectors (k x originalDim).
	// Each row is a unit eigenvector of the covariance matrix.
	Components *mat.Dense

	// Mean is the per-dimension mean of the training data.
	Mean []float64

	// SingularValues holds the singular values corresponding to each component.
	SingularValues []float64

	// VarianceExplained holds the fraction of total variance explained by each component.
	VarianceExplained []float64

	// OriginalDim is the input dimension.
	OriginalDim int

	// ReducedDim is the output dimension (number of components kept).
	ReducedDim int
}

// Fit computes a PCA model from the given training vectors.
// targetDim specifies how many principal components to keep.
// All input vectors must have the same length.
func Fit(vectors [][]float64, targetDim int) (*PCA, error) {
	n := len(vectors)
	if n == 0 {
		return nil, fmt.Errorf("no vectors provided")
	}

	d := len(vectors[0])
	if targetDim <= 0 || targetDim > d {
		return nil, fmt.Errorf("targetDim must be in [1, %d], got %d", d, targetDim)
	}
	if n < targetDim {
		return nil, fmt.Errorf("need at least %d vectors, got %d", targetDim, n)
	}

	// Compute mean
	mean := make([]float64, d)
	for _, v := range vectors {
		if len(v) != d {
			return nil, fmt.Errorf("inconsistent vector dimensions: expected %d, got %d", d, len(v))
		}
		for j, val := range v {
			mean[j] += val
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}

	// Build centered data matrix (n x d)
	data := mat.NewDense(n, d, nil)
	for i, v := range vectors {
		for j, val := range v {
			data.Set(i, j, val-mean[j])
		}
	}

	// SVD of centered data: X = U * S * V^T
	// The columns of V are the principal components (eigenvectors of X^T X).
	var svd mat.SVD
	ok := svd.Factorize(data, mat.SVDThin)
	if !ok {
		return nil, fmt.Errorf("SVD factorization failed")
	}

	// Extract singular values
	sv := svd.Values(nil)

	// Compute total variance (proportional to sum of squared singular values)
	var totalVar float64
	for _, s := range sv {
		totalVar += s * s
	}

	// Extract top-k components from V (right singular vectors)
	var vt mat.Dense
	svd.VTo(&vt)

	// vt is (d x min(n,d)); take first targetDim rows
	components := mat.NewDense(targetDim, d, nil)
	for i := 0; i < targetDim; i++ {
		for j := 0; j < d; j++ {
			components.Set(i, j, vt.At(i, j))
		}
	}

	// Compute variance explained per component
	varExplained := make([]float64, targetDim)
	singularValues := make([]float64, targetDim)
	for i := 0; i < targetDim; i++ {
		singularValues[i] = sv[i]
		if totalVar > 0 {
			varExplained[i] = (sv[i] * sv[i]) / totalVar
		}
	}

	return &PCA{
		Components:        components,
		Mean:              mean,
		SingularValues:    singularValues,
		VarianceExplained: varExplained,
		OriginalDim:       d,
		ReducedDim:        targetDim,
	}, nil
}

// Transform projects a single vector into the reduced PCA space.
// Input vector must have length OriginalDim. Output has length ReducedDim.
func (p *PCA) Transform(vector []float64) ([]float64, error) {
	if len(vector) != p.OriginalDim {
		return nil, fmt.Errorf("expected dimension %d, got %d", p.OriginalDim, len(vector))
	}

	// Center the vector
	centered := make([]float64, p.OriginalDim)
	for i, v := range vector {
		centered[i] = v - p.Mean[i]
	}

	// Project: result = Components * centered^T
	// Components is (k x d), centered is (d x 1), result is (k x 1)
	result := make([]float64, p.ReducedDim)
	for i := 0; i < p.ReducedDim; i++ {
		var sum float64
		for j := 0; j < p.OriginalDim; j++ {
			sum += p.Components.At(i, j) * centered[j]
		}
		result[i] = sum
	}

	return result, nil
}

// TransformBatch projects multiple vectors into the reduced PCA space.
func (p *PCA) TransformBatch(vectors [][]float64) ([][]float64, error) {
	result := make([][]float64, len(vectors))
	for i, v := range vectors {
		r, err := p.Transform(v)
		if err != nil {
			return nil, fmt.Errorf("vector %d: %w", i, err)
		}
		result[i] = r
	}
	return result, nil
}

// TotalVarianceExplained returns the cumulative variance explained by all components.
func (p *PCA) TotalVarianceExplained() float64 {
	var total float64
	for _, v := range p.VarianceExplained {
		total += v
	}
	return total
}

// InverseTransform maps a reduced vector back to the original space (approximate).
// Useful for debugging and visualization.
func (p *PCA) InverseTransform(reduced []float64) ([]float64, error) {
	if len(reduced) != p.ReducedDim {
		return nil, fmt.Errorf("expected dimension %d, got %d", p.ReducedDim, len(reduced))
	}

	// Reconstruct: result = Components^T * reduced + mean
	result := make([]float64, p.OriginalDim)
	for j := 0; j < p.OriginalDim; j++ {
		var sum float64
		for i := 0; i < p.ReducedDim; i++ {
			sum += p.Components.At(i, j) * reduced[i]
		}
		result[j] = sum + p.Mean[j]
	}

	return result, nil
}

// ReconstructionError computes the mean squared reconstruction error for given vectors.
// Lower values indicate better preservation of information.
func (p *PCA) ReconstructionError(vectors [][]float64) (float64, error) {
	if len(vectors) == 0 {
		return 0, nil
	}

	var totalErr float64
	for _, v := range vectors {
		reduced, err := p.Transform(v)
		if err != nil {
			return 0, err
		}
		reconstructed, err := p.InverseTransform(reduced)
		if err != nil {
			return 0, err
		}

		var sqErr float64
		for j := range v {
			diff := v[j] - reconstructed[j]
			sqErr += diff * diff
		}
		totalErr += sqErr / float64(len(v))
	}

	return totalErr / float64(len(vectors)), nil
}

// NormalizeTransformed normalizes a PCA-transformed vector to unit length.
// This preserves the cosine similarity interpretation after dimensionality reduction.
func NormalizeTransformed(vector []float64) []float64 {
	var norm float64
	for _, v := range vector {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vector
	}

	result := make([]float64, len(vector))
	for i, v := range vector {
		result[i] = v / norm
	}
	return result
}
