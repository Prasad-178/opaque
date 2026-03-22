package crypto

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"

	"github.com/Prasad-178/opaque/pkg/crypto/threshold"
)

// ThresholdHEProvider implements HEProvider using threshold CKKS.
// Encryption uses the committee's collective public key. Decryption goes through
// the t-of-N threshold protocol, which re-encrypts results under the client's
// ephemeral public key. No single party ever sees the full secret key.
type ThresholdHEProvider struct {
	committee *threshold.Committee
	session   *threshold.ClientSession
	evalPool  []*thresholdEvalEngine
	free      chan *thresholdEvalEngine
	params    hefloat.Parameters
}

// Verify interface compliance at compile time.
var _ HEProvider = (*ThresholdHEProvider)(nil)

// NewThresholdHEProvider creates a threshold HE provider from an existing committee.
// Creates an internal client session with an ephemeral keypair for receiving decrypted results.
func NewThresholdHEProvider(committee *threshold.Committee, poolSize int) (*ThresholdHEProvider, error) {
	if committee == nil {
		return nil, errors.New("committee is required")
	}
	if poolSize < 1 {
		poolSize = 1
	}

	session, err := committee.NewClientSession()
	if err != nil {
		return nil, fmt.Errorf("failed to create client session: %w", err)
	}

	params := committee.GetParams()
	evkSet := committee.GetEvalKeySet()

	p := &ThresholdHEProvider{
		committee: committee,
		session:   session,
		evalPool:  make([]*thresholdEvalEngine, poolSize),
		free:      make(chan *thresholdEvalEngine, poolSize),
		params:    params,
	}

	for i := 0; i < poolSize; i++ {
		eng := newThresholdEvalEngine(params, evkSet, committee, session)
		p.evalPool[i] = eng
		p.free <- eng
	}

	return p, nil
}

func (t *ThresholdHEProvider) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	return t.committee.EncryptVector(vector)
}

func (t *ThresholdHEProvider) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	eng := <-t.free
	defer func() { t.free <- eng }()
	return eng.DecryptScalar(ct)
}

func (t *ThresholdHEProvider) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	eng := <-t.free
	defer func() { t.free <- eng }()
	return eng.DecryptBatchScalars(ct, numCentroids, dimension)
}

func (t *ThresholdHEProvider) HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	eng := <-t.free
	defer func() { t.free <- eng }()
	return eng.HomomorphicBatchDotProduct(encQuery, packedCentroids, numCentroids, dimension)
}

func (t *ThresholdHEProvider) Acquire() EvalEngine {
	return <-t.free
}

func (t *ThresholdHEProvider) Release(e EvalEngine) {
	t.free <- e.(*thresholdEvalEngine)
}

func (t *ThresholdHEProvider) Size() int {
	return len(t.evalPool)
}

func (t *ThresholdHEProvider) GetParams() hefloat.Parameters {
	return t.params
}

func (t *ThresholdHEProvider) GetEncoder() *hefloat.Encoder {
	return hefloat.NewEncoder(t.params)
}

func (t *ThresholdHEProvider) Close() {
	// Drain pool
	for i := 0; i < len(t.evalPool); i++ {
		select {
		case <-t.free:
		default:
		}
	}
	if t.session != nil {
		t.session.Close()
	}
}

// thresholdEvalEngine provides HE evaluation using the committee's eval keys.
// Decryption is delegated to the threshold protocol. Each engine has its own
// encoder, evaluator, and decryptor to avoid data races in concurrent use.
type thresholdEvalEngine struct {
	params    hefloat.Parameters
	encoder   *hefloat.Encoder
	evaluator *hefloat.Evaluator
	committee *threshold.Committee
	session   *threshold.ClientSession
	// Per-engine decryptor and decoder for client-side decryption (not thread-safe).
	clientDecryptor *rlwe.Decryptor
	clientEncoder   *hefloat.Encoder
	mu              sync.Mutex
}

var _ EvalEngine = (*thresholdEvalEngine)(nil)

func newThresholdEvalEngine(params hefloat.Parameters, evkSet rlwe.EvaluationKeySet, committee *threshold.Committee, session *threshold.ClientSession) *thresholdEvalEngine {
	return &thresholdEvalEngine{
		params:          params,
		encoder:         hefloat.NewEncoder(params),
		evaluator:       hefloat.NewEvaluator(params, evkSet),
		committee:       committee,
		session:         session,
		clientDecryptor: rlwe.NewDecryptor(params, session.SK),
		clientEncoder:   hefloat.NewEncoder(params),
	}
}

func (e *thresholdEvalEngine) HomomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	maxSlots := e.params.MaxSlots()
	padded := make([]float64, maxSlots)
	copy(padded, vector)

	pt := hefloat.NewPlaintext(e.params, encQuery.Level())
	if err := e.encoder.Encode(padded, pt); err != nil {
		return nil, fmt.Errorf("failed to encode vector: %w", err)
	}

	result, err := e.evaluator.MulNew(encQuery, pt)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}
	if err := e.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("failed to rescale: %w", err)
	}

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

func (e *thresholdEvalEngine) HomomorphicDotProductCached(encQuery *rlwe.Ciphertext, encodedVector *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if encodedVector == nil {
		return nil, errors.New("encoded vector is nil")
	}

	result, err := e.evaluator.MulNew(encQuery, encodedVector)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}
	if err := e.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("failed to rescale: %w", err)
	}

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

func (e *thresholdEvalEngine) HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if packedCentroids == nil {
		return nil, errors.New("packed centroids is nil")
	}

	result, err := e.evaluator.MulNew(encQuery, packedCentroids)
	if err != nil {
		return nil, fmt.Errorf("failed to multiply: %w", err)
	}
	if err := e.evaluator.Rescale(result, result); err != nil {
		return nil, fmt.Errorf("failed to rescale: %w", err)
	}

	for stride := 1; stride < dimension; stride *= 2 {
		rotated, err := e.evaluator.RotateNew(result, stride)
		if err != nil {
			return nil, fmt.Errorf("failed to rotate by %d: %w", stride, err)
		}
		if err := e.evaluator.Add(result, rotated, result); err != nil {
			return nil, fmt.Errorf("failed to add: %w", err)
		}
	}

	return result, nil
}

func (e *thresholdEvalEngine) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	ctClient, err := e.committee.ThresholdDecrypt(ct, e.session.PK)
	if err != nil {
		return 0, fmt.Errorf("threshold decrypt failed: %w", err)
	}
	// Use per-engine decryptor/encoder to avoid data races on shared session.
	pt := e.clientDecryptor.DecryptNew(ctClient)
	decoded := make([]float64, 1)
	if err := e.clientEncoder.Decode(pt, decoded); err != nil {
		return 0, fmt.Errorf("failed to decode: %w", err)
	}
	return decoded[0], nil
}

func (e *thresholdEvalEngine) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	ctClient, err := e.committee.ThresholdDecrypt(ct, e.session.PK)
	if err != nil {
		return nil, fmt.Errorf("threshold decrypt failed: %w", err)
	}
	pt := e.clientDecryptor.DecryptNew(ctClient)
	maxSlots := e.params.MaxSlots()
	decoded := make([]float64, maxSlots)
	if err := e.clientEncoder.Decode(pt, decoded); err != nil {
		return nil, fmt.Errorf("failed to decode: %w", err)
	}
	results := make([]float64, numCentroids)
	for i := 0; i < numCentroids; i++ {
		pos := i * dimension
		if pos < len(decoded) {
			results[i] = decoded[pos]
		}
	}
	return results, nil
}
