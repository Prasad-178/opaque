// Package crypto provides homomorphic encryption operations using Lattigo BFV scheme.
package crypto

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/schemes/bfv"
)

const (
	// PlaintextModulus is the BFV plaintext modulus (from ExampleParameters128BitLogN14LogQP438)
	PlaintextModulus = 65537 // 0x10001

	// ScaleFactor for fixed-point encoding
	// With values in [-1, 1], offset of 1 gives [0, 2]
	// Scale of 30000 gives [0, 60000] which fits in plaintext modulus 65537
	ScaleFactor = float64(30000)

	// Offset is added to make negative values positive before encoding
	Offset = 1.0
)

// Parameters holds the BFV parameters for the encryption scheme
type Parameters struct {
	bfv.Parameters
}

// KeyPair contains the secret and public keys
type KeyPair struct {
	SecretKey *rlwe.SecretKey
	PublicKey *rlwe.PublicKey
}

// Engine provides homomorphic encryption operations
type Engine struct {
	params    bfv.Parameters
	encoder   *bfv.Encoder
	evaluator *bfv.Evaluator

	// Only set on client side
	secretKey *rlwe.SecretKey
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor

	// Only set on server side (for homomorphic ops)
	publicKey *rlwe.PublicKey

	mu sync.RWMutex
}

// NewParameters creates BFV parameters with the specified security level.
// Uses ExampleParameters128BitLogN14LogQP438 for 128-bit security with good performance.
func NewParameters() (bfv.Parameters, error) {
	// ExampleParameters128BitLogN14LogQP438 provides:
	// - 128-bit security
	// - LogN=14 supports vectors up to 8192 dimensions
	// - Good balance of speed and capability
	return bfv.NewParametersFromLiteral(bfv.ExampleParameters128BitLogN14LogQP438)
}

// NewClientEngine creates an encryption engine for client-side operations.
// Generates a new key pair.
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
		encoder:   bfv.NewEncoder(params),
		evaluator: bfv.NewEvaluator(params, evk),
		secretKey: sk,
		publicKey: pk,
		encryptor: rlwe.NewEncryptor(params, pk),
		decryptor: rlwe.NewDecryptor(params, sk),
	}, nil
}

// NewServerEngine creates an encryption engine for server-side operations.
// Does not have access to the secret key.
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
		encoder:   bfv.NewEncoder(params),
		evaluator: bfv.NewEvaluator(params, nil),
		publicKey: pk,
	}, nil
}

// galoisElements returns the Galois elements needed for rotations
func galoisElements(params bfv.Parameters) []uint64 {
	// We need rotations by powers of 2 for the summation
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}

// GetPublicKeyBytes returns the serialized public key
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

// EncryptVector encrypts a float64 vector using fixed-point encoding.
// Values are expected to be in range [-1, 1] (normalized vectors).
func (e *Engine) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.encryptor == nil {
		return nil, errors.New("encryptor not available (server-side engine?)")
	}

	// Encode to fixed-point integers
	encoded := encodeVector(vector)

	// Create plaintext
	pt := bfv.NewPlaintext(e.params, e.params.MaxLevel())
	if err := e.encoder.Encode(encoded, pt); err != nil {
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

	// Decode
	encoded := make([]uint64, length)
	if err := e.encoder.Decode(pt, encoded); err != nil {
		return nil, fmt.Errorf("failed to decode: %w", err)
	}

	return decodeVector(encoded), nil
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
	encoded := make([]uint64, 1)
	if err := e.encoder.Decode(pt, encoded); err != nil {
		return 0, fmt.Errorf("failed to decode: %w", err)
	}

	// Decode from fixed-point
	// The dot product result has been squared in scale, so we need to adjust
	return decodeScalar(encoded[0]), nil
}

// HomomorphicDotProduct computes E(q · v) from E(q) and plaintext v.
// This is the core operation for privacy-preserving similarity search.
func (e *Engine) HomomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Encode vector as plaintext
	encoded := encodeVector(vector)

	pt := bfv.NewPlaintext(e.params, encQuery.Level())
	if err := e.encoder.Encode(encoded, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	// Multiply: E(q) * v = E(q * v) component-wise
	result, err := e.evaluator.MulNew(encQuery, pt)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}

	// Sum components using rotations
	// result = E(q[0]*v[0] + q[1]*v[1] + ... + q[n]*v[n])
	n := len(vector)
	for i := 1; i < n; i *= 2 {
		rotated, err := e.evaluator.RotateColumnsNew(result, i)
		if err != nil {
			return nil, fmt.Errorf("failed to rotate: %w", err)
		}
		if err := e.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
	}

	return result, nil
}

// SerializeCiphertext serializes a ciphertext to bytes
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

// GetParams returns the BFV parameters
func (e *Engine) GetParams() bfv.Parameters {
	return e.params
}

// encodeVector converts float64 values to fixed-point uint64
func encodeVector(vector []float64) []uint64 {
	encoded := make([]uint64, len(vector))
	for i, v := range vector {
		// Add offset to make positive, then scale
		encoded[i] = uint64((v + Offset) * ScaleFactor)
	}
	return encoded
}

// decodeVector converts fixed-point uint64 back to float64
func decodeVector(encoded []uint64) []float64 {
	decoded := make([]float64, len(encoded))
	for i, v := range encoded {
		decoded[i] = (float64(v) / ScaleFactor) - Offset
	}
	return decoded
}

// decodeScalar decodes a single scalar from a dot product result
func decodeScalar(encoded uint64) float64 {
	// Dot product result: sum of (qi + offset) * (vi + offset) * scale^2
	// We need to adjust for the double scaling
	return float64(encoded) / (ScaleFactor * ScaleFactor)
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
