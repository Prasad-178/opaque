// Package crypto provides homomorphic encryption operations using Lattigo CKKS scheme.
// CKKS is ideal for approximate arithmetic on encrypted real numbers.
package crypto

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// Engine provides homomorphic encryption operations using CKKS scheme.
// CKKS allows approximate arithmetic on encrypted floating-point numbers,
// which is ideal for computing dot products for similarity search.
type Engine struct {
	params    hefloat.Parameters
	encoder   *hefloat.Encoder
	evaluator *hefloat.Evaluator

	// Only set on client side
	secretKey *rlwe.SecretKey
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor

	// Only set on server side (for homomorphic ops)
	publicKey *rlwe.PublicKey

	mu sync.RWMutex
}

// NewParameters creates CKKS parameters optimized for vector dot products.
// Uses 128-bit security with LogN=14 supporting vectors up to 8192 dimensions.
func NewParameters() (hefloat.Parameters, error) {
	// CKKS parameters for 128-bit security
	// LogN=14 gives 2^14 = 16384 slots (enough for 8192-dim complex or 16384-dim real)
	// LogDefaultScale=45 provides good precision for normalized vectors
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,                                    // Ring degree 2^14 = 16384
		LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45}, // Ciphertext modulus chain
		LogP:            []int{61, 61},                         // Special primes for key-switching
		LogDefaultScale: 45,                                    // Scale for encoding (2^45)
	})
	if err != nil {
		return hefloat.Parameters{}, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}
	return params, nil
}

// NewClientEngine creates an encryption engine for client-side operations.
// Generates a new key pair for encryption/decryption.
func NewClientEngine() (*Engine, error) {
	params, err := NewParameters()
	if err != nil {
		return nil, fmt.Errorf("failed to create parameters: %w", err)
	}

	// Generate key pair
	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()

	// Create evaluation key for rotations (needed for dot product sum)
	evk := rlwe.NewMemEvaluationKeySet(nil, kgen.GenGaloisKeysNew(galoisElements(params), sk)...)

	return &Engine{
		params:    params,
		encoder:   hefloat.NewEncoder(params),
		evaluator: hefloat.NewEvaluator(params, evk),
		secretKey: sk,
		publicKey: pk,
		encryptor: rlwe.NewEncryptor(params, pk),
		decryptor: rlwe.NewDecryptor(params, sk),
	}, nil
}

// NewServerEngine creates an encryption engine for server-side operations.
// Does not have access to the secret key - can only perform homomorphic operations.
func NewServerEngine(publicKeyBytes []byte) (*Engine, error) {
	params, err := NewParameters()
	if err != nil {
		return nil, fmt.Errorf("failed to create parameters: %w", err)
	}

	// Deserialize public key
	pk := rlwe.NewPublicKey(params)
	if _, err := pk.ReadFrom(bytes.NewReader(publicKeyBytes)); err != nil {
		return nil, fmt.Errorf("failed to deserialize public key: %w", err)
	}

	return &Engine{
		params:    params,
		encoder:   hefloat.NewEncoder(params),
		evaluator: hefloat.NewEvaluator(params, nil),
		publicKey: pk,
	}, nil
}

// galoisElements returns the Galois elements needed for rotations in dot product
func galoisElements(params hefloat.Parameters) []uint64 {
	// We need rotations by powers of 2 for the tree-based summation
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}

// GetPublicKeyBytes returns the serialized public key for distribution
func (e *Engine) GetPublicKeyBytes() ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.publicKey == nil {
		return nil, errors.New("no public key available")
	}

	buf := new(bytes.Buffer)
	if _, err := e.publicKey.WriteTo(buf); err != nil {
		return nil, fmt.Errorf("failed to serialize public key: %w", err)
	}
	return buf.Bytes(), nil
}

// EncryptVector encrypts a float64 vector using CKKS.
// Values should be normalized to [-1, 1] range for best precision.
// CKKS directly handles floating-point values without offset encoding.
func (e *Engine) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.encryptor == nil {
		return nil, errors.New("encryptor not available (server-side engine?)")
	}

	// CKKS uses complex slots, but we only use the real part
	// Pad vector to max slots with 0 (zeros don't affect dot product sum)
	maxSlots := e.params.MaxSlots()
	paddedVector := make([]float64, maxSlots)
	copy(paddedVector, vector)
	// Remaining slots are 0 by default - no contribution to dot product

	// Encode directly as floating-point values (no offset needed in CKKS!)
	pt := hefloat.NewPlaintext(e.params, e.params.MaxLevel())
	if err := e.encoder.Encode(paddedVector, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	// Encrypt
	ct, err := e.encryptor.EncryptNew(pt)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt: %w", err)
	}

	return ct, nil
}

// DecryptVector decrypts a ciphertext back to a float64 vector.
func (e *Engine) DecryptVector(ct *rlwe.Ciphertext, length int) ([]float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.decryptor == nil {
		return nil, errors.New("decryptor not available (server-side engine?)")
	}

	// Decrypt
	pt := e.decryptor.DecryptNew(ct)

	// Decode to float64 slice
	decoded := make([]float64, length)
	if err := e.encoder.Decode(pt, decoded); err != nil {
		return nil, fmt.Errorf("failed to decode: %w", err)
	}

	return decoded, nil
}

// DecryptScalar decrypts a ciphertext containing a single scalar value (dot product result).
func (e *Engine) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.decryptor == nil {
		return 0, errors.New("decryptor not available (server-side engine?)")
	}

	// Decrypt
	pt := e.decryptor.DecryptNew(ct)

	// Decode - the result is in the first slot
	decoded := make([]float64, 1)
	if err := e.encoder.Decode(pt, decoded); err != nil {
		return 0, fmt.Errorf("failed to decode: %w", err)
	}

	// CKKS returns the actual floating-point value directly!
	return decoded[0], nil
}

// HomomorphicDotProduct computes E(q · v) from E(q) and plaintext v.
// This is the core operation for privacy-preserving similarity search.
// Returns encrypted dot product that only the client can decrypt.
// Note: Uses full lock because Lattigo evaluator is not thread-safe.
func (e *Engine) HomomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Pad vector to max slots with 0 (same as EncryptVector)
	maxSlots := e.params.MaxSlots()
	paddedVector := make([]float64, maxSlots)
	copy(paddedVector, vector)

	// Encode vector as plaintext
	pt := hefloat.NewPlaintext(e.params, encQuery.Level())
	if err := e.encoder.Encode(paddedVector, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	// Multiply: E(q) * v = E(q * v) component-wise
	result, err := e.evaluator.MulNew(encQuery, pt)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}

	// Rescale after multiplication to manage scale growth
	if err := e.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("failed to rescale: %w", err)
	}

	// Sum all slots using tree-based rotation and addition
	// This computes: slot[0] = sum(q[i] * v[i]) for all i
	for i := 1; i < maxSlots; i *= 2 {
		rotated, err := e.evaluator.RotateNew(result, i)
		if err != nil {
			return nil, fmt.Errorf("failed to rotate by %d: %w", i, err)
		}
		if err := e.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
	}

	return result, nil
}

// HomomorphicDotProductCached computes E(q · v) using a pre-encoded plaintext.
// This is faster than HomomorphicDotProduct when centroids are cached as plaintexts.
// Note: Uses full lock because Lattigo evaluator is not thread-safe.
func (e *Engine) HomomorphicDotProductCached(encQuery *rlwe.Ciphertext, encodedVector *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if encodedVector == nil {
		return nil, errors.New("encoded vector is nil")
	}

	// Multiply: E(q) * v = E(q * v) component-wise
	result, err := e.evaluator.MulNew(encQuery, encodedVector)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}

	// Rescale after multiplication to manage scale growth
	if err := e.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("failed to rescale: %w", err)
	}

	// Sum all slots using tree-based rotation and addition
	maxSlots := e.params.MaxSlots()
	for i := 1; i < maxSlots; i *= 2 {
		rotated, err := e.evaluator.RotateNew(result, i)
		if err != nil {
			return nil, fmt.Errorf("failed to rotate by %d: %w", i, err)
		}
		if err := e.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
	}

	return result, nil
}

// GetEncoder returns the HE encoder for external caching purposes.
func (e *Engine) GetEncoder() *hefloat.Encoder {
	return e.encoder
}

// SerializeCiphertext serializes a ciphertext to bytes for transmission
func (e *Engine) SerializeCiphertext(ct *rlwe.Ciphertext) ([]byte, error) {
	buf := new(bytes.Buffer)
	if _, err := ct.WriteTo(buf); err != nil {
		return nil, fmt.Errorf("failed to serialize ciphertext: %w", err)
	}
	return buf.Bytes(), nil
}

// DeserializeCiphertext deserializes bytes to a ciphertext
func (e *Engine) DeserializeCiphertext(data []byte) (*rlwe.Ciphertext, error) {
	ct := rlwe.NewCiphertext(e.params, 1, e.params.MaxLevel())
	if _, err := ct.ReadFrom(bytes.NewReader(data)); err != nil {
		return nil, fmt.Errorf("failed to deserialize ciphertext: %w", err)
	}
	return ct, nil
}

// GetParams returns the CKKS parameters
func (e *Engine) GetParams() hefloat.Parameters {
	return e.params
}

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(vector []float64) []float64 {
	var norm float64
	for _, v := range vector {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return vector
	}

	normalized := make([]float64, len(vector))
	for i, v := range vector {
		normalized[i] = v / norm
	}
	return normalized
}
