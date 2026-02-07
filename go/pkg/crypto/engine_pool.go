package crypto

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// EnginePool manages a pool of HE engines for parallel operations.
// Each engine has its own evaluator (not thread-safe), but shares keys.
// This allows parallel HE operations without the serialization bottleneck.
type EnginePool struct {
	engines []*Engine
	free    chan *Engine
}

// NewEnginePool creates a pool of n HE engines.
// All engines share the same keys but have independent evaluators.
// Recommended: n = runtime.NumCPU() for optimal parallelism.
func NewEnginePool(n int) (*EnginePool, error) {
	if n < 1 {
		n = 1
	}

	// Create the first engine which generates the keys
	primary, err := NewClientEngine()
	if err != nil {
		return nil, fmt.Errorf("failed to create primary engine: %w", err)
	}

	pool := &EnginePool{
		engines: make([]*Engine, n),
		free:    make(chan *Engine, n),
	}

	// First engine is the primary
	pool.engines[0] = primary
	pool.free <- primary

	// Create additional engines sharing the same keys
	for i := 1; i < n; i++ {
		engine, err := newEngineWithKeys(primary.params, primary.secretKey, primary.publicKey)
		if err != nil {
			return nil, fmt.Errorf("failed to create engine %d: %w", i, err)
		}
		pool.engines[i] = engine
		pool.free <- engine
	}

	return pool, nil
}

// newEngineWithKeys creates an engine sharing existing keys but with independent evaluator.
// This is the key to parallelization - each evaluator can run concurrently.
func newEngineWithKeys(params hefloat.Parameters, sk *rlwe.SecretKey, pk *rlwe.PublicKey) (*Engine, error) {
	// Create new key generator to generate Galois keys for this evaluator
	kgen := rlwe.NewKeyGenerator(params)
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

// Acquire gets an engine from the pool. Blocks if none available.
// The caller MUST call Release when done.
func (p *EnginePool) Acquire() *Engine {
	return <-p.free
}

// Release returns an engine to the pool.
func (p *EnginePool) Release(e *Engine) {
	p.free <- e
}

// Size returns the number of engines in the pool.
func (p *EnginePool) Size() int {
	return len(p.engines)
}

// GetParams returns the CKKS parameters (same for all engines).
func (p *EnginePool) GetParams() hefloat.Parameters {
	return p.engines[0].params
}

// GetEncoder returns an encoder (from the primary engine).
// Note: Encoder may not be thread-safe for encoding, but is fine for reading params.
func (p *EnginePool) GetEncoder() *hefloat.Encoder {
	return p.engines[0].encoder
}

// GetPrimaryEngine returns the first engine (useful for encryption/decryption).
func (p *EnginePool) GetPrimaryEngine() *Engine {
	return p.engines[0]
}

// EncryptVector encrypts a vector using the primary engine.
// This is typically called once per query, so serialization is acceptable.
func (p *EnginePool) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	engine := p.Acquire()
	defer p.Release(engine)
	return engine.EncryptVector(vector)
}

// DecryptScalar decrypts a scalar result using the primary engine.
func (p *EnginePool) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	engine := p.Acquire()
	defer p.Release(engine)
	return engine.DecryptScalar(ct)
}

// HomomorphicBatchDotProduct computes multiple dot products using SIMD.
func (p *EnginePool) HomomorphicBatchDotProduct(encPackedQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	engine := p.Acquire()
	defer p.Release(engine)
	return engine.HomomorphicBatchDotProduct(encPackedQuery, packedCentroids, numCentroids, dimension)
}

// DecryptBatchScalars decrypts and extracts multiple dot product results.
func (p *EnginePool) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	engine := p.Acquire()
	defer p.Release(engine)
	return engine.DecryptBatchScalars(ct, numCentroids, dimension)
}
