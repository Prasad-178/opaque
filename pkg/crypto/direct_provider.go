package crypto

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// DirectHEProvider wraps EnginePool to implement HEProvider.
// This is the default single-key mode where one entity holds the full secret key.
// Each engine in the pool can encrypt, decrypt, and evaluate independently.
type DirectHEProvider struct {
	pool *EnginePool
}

// Verify interface compliance at compile time.
var _ HEProvider = (*DirectHEProvider)(nil)
var _ EvalEngine = (*Engine)(nil)

// NewDirectHEProvider creates a single-key HE provider with the given pool size.
func NewDirectHEProvider(poolSize int) (*DirectHEProvider, error) {
	pool, err := NewEnginePool(poolSize)
	if err != nil {
		return nil, err
	}
	return &DirectHEProvider{pool: pool}, nil
}

func (d *DirectHEProvider) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	return d.pool.EncryptVector(vector)
}

func (d *DirectHEProvider) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	return d.pool.DecryptScalar(ct)
}

func (d *DirectHEProvider) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	return d.pool.DecryptBatchScalars(ct, numCentroids, dimension)
}

func (d *DirectHEProvider) HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	return d.pool.HomomorphicBatchDotProduct(encQuery, packedCentroids, numCentroids, dimension)
}

func (d *DirectHEProvider) Acquire() EvalEngine {
	return d.pool.Acquire()
}

func (d *DirectHEProvider) Release(e EvalEngine) {
	d.pool.Release(e.(*Engine))
}

func (d *DirectHEProvider) Size() int {
	return d.pool.Size()
}

func (d *DirectHEProvider) GetParams() hefloat.Parameters {
	return d.pool.GetParams()
}

func (d *DirectHEProvider) GetEncoder() *hefloat.Encoder {
	return d.pool.GetEncoder()
}

func (d *DirectHEProvider) Close() {
	d.pool.Close()
}
