// This file defines the HEProvider interface that abstracts the homomorphic
// encryption backend. Two implementations exist:
//   - DirectHEProvider: single-key mode (current default)
//   - ThresholdHEProvider: t-of-N threshold CKKS mode
//
// The client uses HEProvider so it works identically with either backend.

package crypto

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// HEProvider abstracts the homomorphic encryption backend.
// Implementations handle key management, encryption, and decryption differently
// (single-key vs threshold) while exposing the same interface for HE computation.
type HEProvider interface {
	// EncryptVector encrypts a float64 vector (values should be in [-1, 1]).
	EncryptVector(vector []float64) (*rlwe.Ciphertext, error)

	// DecryptScalar decrypts a ciphertext containing a single scalar (e.g., dot product).
	// In threshold mode, this goes through the committee protocol.
	DecryptScalar(ct *rlwe.Ciphertext) (float64, error)

	// DecryptBatchScalars decrypts multiple dot product results from a batch ciphertext.
	DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error)

	// HomomorphicBatchDotProduct computes multiple packed dot products in one HE op.
	HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error)

	// Acquire returns an EvalEngine from the internal pool for parallel HE computation.
	// The caller MUST call Release when done.
	Acquire() EvalEngine

	// Release returns an EvalEngine to the pool.
	Release(EvalEngine)

	// Size returns the number of parallel evaluators available.
	Size() int

	// GetParams returns the shared CKKS parameters.
	GetParams() hefloat.Parameters

	// GetEncoder returns an HE encoder.
	GetEncoder() *hefloat.Encoder

	// Close zeros all key material and releases resources.
	Close()
}

// ProfiledEvalEngine extends EvalEngine with sub-phase timing for GPU analysis.
// Implementations that support profiling can be detected via type assertion.
type ProfiledEvalEngine interface {
	EvalEngine
	HomomorphicBatchDotProductProfiled(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, HEProfile, error)
}

// EvalEngine provides homomorphic evaluation operations.
// Acquired from HEProvider.Acquire(), must be returned via Release().
// Not thread-safe — each goroutine should acquire its own.
type EvalEngine interface {
	// HomomorphicDotProduct computes E(q · v) from E(q) and plaintext v.
	HomomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error)

	// HomomorphicDotProductCached computes E(q · v) using a pre-encoded plaintext.
	HomomorphicDotProductCached(encQuery *rlwe.Ciphertext, encodedVector *rlwe.Plaintext) (*rlwe.Ciphertext, error)

	// HomomorphicBatchDotProduct computes packed dot products.
	HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error)

	// DecryptScalar decrypts a scalar result.
	// In threshold mode, this delegates to the committee protocol.
	DecryptScalar(ct *rlwe.Ciphertext) (float64, error)
}
