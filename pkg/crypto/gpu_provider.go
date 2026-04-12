// GPUHEProvider offloads expensive HE operations to a remote GPU server
// while keeping key management, encryption, and decryption local.
//
// Architecture:
//   - Encryption/decryption: local EnginePool (secret key never leaves client)
//   - HomomorphicBatchDotProduct: sent to GPU server via gRPC
//   - Eval keys (Galois + relin): sent once at setup, stored on GPU server
//
// The existing DirectHEProvider and ThresholdHEProvider are NOT modified.
// GPUHEProvider is a new, additive option enabled via Config.GPUServerAddress.
package crypto

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"time"

	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// GPUHEProviderConfig configures the GPU HE provider.
type GPUHEProviderConfig struct {
	// ServerAddress is the GPU HE server address (e.g., "localhost:50052").
	ServerAddress string

	// LocalPoolSize is the number of local HE engines for encrypt/decrypt.
	// These never touch the GPU — only used for client-side operations.
	// Default: 2.
	LocalPoolSize int

	// Timeout for GPU RPC calls.
	// Default: 30s.
	Timeout time.Duration

	// SessionID identifies this client to the GPU server.
	// Default: auto-generated.
	SessionID string
}

// GPUHEProvider implements HEProvider by offloading batch HE operations to a GPU server.
// Encryption, decryption, and key management remain local.
type GPUHEProvider struct {
	localPool *EnginePool   // For encrypt/decrypt (local, has secret key)
	conn      *grpc.ClientConn
	client    pb.GPUHEServiceClient
	sessionID string
	timeout   time.Duration

	// Track whether eval keys have been registered with the server.
	keysRegistered bool
	nttConverter   *NTTConverter // For ciphertext conversion (set after InitContext)
	mu             sync.Mutex
}

// NewGPUHEProvider creates a provider that offloads HE computation to a GPU server.
// It creates a local engine pool for encrypt/decrypt and connects to the GPU server
// for batch dot product operations.
func NewGPUHEProvider(cfg GPUHEProviderConfig) (*GPUHEProvider, error) {
	if cfg.ServerAddress == "" {
		return nil, fmt.Errorf("GPU server address is required")
	}
	if cfg.LocalPoolSize <= 0 {
		cfg.LocalPoolSize = 2
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.SessionID == "" {
		cfg.SessionID = fmt.Sprintf("gpu-client-%d", time.Now().UnixNano())
	}

	// Create local engine pool (for encrypt/decrypt only).
	localPool, err := NewEnginePool(cfg.LocalPoolSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create local engine pool: %w", err)
	}

	// Connect to GPU server.
	// Use 256MB max message size for eval key registration (~200MB for Galois keys).
	conn, err := grpc.NewClient(cfg.ServerAddress,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(256*1024*1024)),
		grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(256*1024*1024)),
	)
	if err != nil {
		localPool.Close()
		return nil, fmt.Errorf("failed to connect to GPU server: %w", err)
	}

	p := &GPUHEProvider{
		localPool: localPool,
		conn:      conn,
		client:    pb.NewGPUHEServiceClient(conn),
		sessionID: cfg.SessionID,
		timeout:   cfg.Timeout,
	}

	return p, nil
}

// RegisterEvalKeys initializes the GPU server context and sends evaluation keys.
// This is called automatically on the first HomomorphicBatchDotProduct if not called explicitly.
//
// The registration is a two-step process:
//  1. InitContext: server creates HEonGPU context and returns NTT roots
//  2. RegisterEvalKeys: client converts keys to server's NTT domain and sends them
func (p *GPUHEProvider) RegisterEvalKeys() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.keysRegistered {
		return nil
	}

	engine := p.localPool.GetPrimaryEngine()

	// Extract exact modulus primes.
	ringQ := engine.params.RingQ()
	qModuli := ringQ.ModuliChain()
	var pModuli []uint64
	if ringP := engine.params.RingP(); ringP != nil {
		pModuli = ringP.ModuliChain()
	}

	ckksParams := &pb.CKKSParams{
		LogN:            int32(engine.params.LogN()),
		QModuli:         qModuli,
		PModuli:         pModuli,
		LogDefaultScale: 45,
	}

	// Step 1: InitContext — get server's NTT roots.
	initCtx, initCancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer initCancel()

	initResp, err := p.client.InitContext(initCtx, &pb.InitContextRequest{
		SessionId: p.sessionID,
		Params:    ckksParams,
	})
	if err != nil {
		return fmt.Errorf("InitContext RPC failed: %w", err)
	}
	if !initResp.Success {
		return fmt.Errorf("InitContext failed: %s", initResp.Error)
	}

	serverRoots := initResp.NttRoots

	// Step 2: Serialize eval keys with server's NTT roots for domain conversion.
	galoisElems := galoisElements(engine.params)

	// Get Galois keys from eval key set
	evk := engine.GetEvalKeys()
	if evk == nil {
		return fmt.Errorf("no evaluation keys available")
	}

	gkeys := make([]*rlwe.GaloisKey, len(galoisElems))
	for i, el := range galoisElems {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			return fmt.Errorf("GetGaloisKey(%d): %w", el, err)
		}
		gkeys[i] = gk
	}

	// Serialize in HEonGPU format with NTT domain conversion
	galoisKeysHEonGPU, err := SerializeGaloisKeysHEonGPU(engine.params, gkeys, galoisElems, serverRoots)
	if err != nil {
		return fmt.Errorf("SerializeGaloisKeysHEonGPU: %w", err)
	}

	// Also serialize Lattigo format for CPU stub fallback
	paramsBytes, err := engine.params.MarshalBinary()
	if err != nil {
		return fmt.Errorf("failed to serialize params: %w", err)
	}
	galoisKeysLattigo, err := serializeGaloisKeys(engine, galoisElems)
	if err != nil {
		return fmt.Errorf("failed to serialize Lattigo Galois keys: %w", err)
	}
	relinBytes, err := serializeRelinKey(engine)
	if err != nil {
		return fmt.Errorf("failed to serialize relin key: %w", err)
	}

	// Step 3: Send eval keys to GPU server.
	regCtx, regCancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer regCancel()

	resp, err := p.client.RegisterEvalKeys(regCtx, &pb.RegisterEvalKeysRequest{
		SessionId:             p.sessionID,
		Params:                ckksParams,
		CkksParamsBinary:      paramsBytes,
		RelinKeySerialized:    relinBytes,
		GaloisKeysSerialized:  galoisKeysLattigo,
		GaloisElements:        galoisElems,
		GaloisKeysHeongpu:     galoisKeysHEonGPU, // HEonGPU-format keys with correct NTT domain
	})
	if err != nil {
		return fmt.Errorf("RegisterEvalKeys RPC failed: %w", err)
	}
	if !resp.Success {
		return fmt.Errorf("RegisterEvalKeys failed: %s", resp.Error)
	}

	// Store the converter for ciphertext conversion during search
	p.nttConverter = NewNTTConverterWithRoots(engine.params, serverRoots)

	p.keysRegistered = true
	return nil
}

// --- HEProvider interface implementation ---

func (p *GPUHEProvider) EncryptVector(vector []float64) (*rlwe.Ciphertext, error) {
	return p.localPool.EncryptVector(vector)
}

func (p *GPUHEProvider) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	return p.localPool.DecryptScalar(ct)
}

func (p *GPUHEProvider) DecryptBatchScalars(ct *rlwe.Ciphertext, numCentroids, dimension int) ([]float64, error) {
	return p.localPool.DecryptBatchScalars(ct, numCentroids, dimension)
}

func (p *GPUHEProvider) HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	// Ensure eval keys are registered.
	p.mu.Lock()
	if !p.keysRegistered {
		p.mu.Unlock()
		if err := p.RegisterEvalKeys(); err != nil {
			return nil, fmt.Errorf("auto-register eval keys failed: %w", err)
		}
	} else {
		p.mu.Unlock()
	}

	// Serialize ciphertext and plaintext (legacy format for CPU stub).
	queryBytes, err := serializeCiphertext(encQuery)
	if err != nil {
		return nil, fmt.Errorf("serialize query: %w", err)
	}

	centroidBytes, err := serializePlaintext(packedCentroids)
	if err != nil {
		return nil, fmt.Errorf("serialize centroids: %w", err)
	}

	// Extract raw coefficients with NTT domain conversion for HEonGPU.
	var rawQuery *pb.RawCiphertext
	if p.nttConverter != nil {
		// Convert ciphertext to HEonGPU's NTT domain (Montgomery removal + NTT conversion)
		rawQuery = CiphertextToHEonGPU(encQuery, p.nttConverter)
	} else {
		rawQuery = extractRawCiphertext(encQuery)
	}
	rawCentroids := extractRawPlaintext(packedCentroids)

	// Send to GPU server.
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	resp, err := p.client.BatchDotProduct(ctx, &pb.BatchDotProductRequest{
		SessionId:       p.sessionID,
		RawQuery:        rawQuery,
		RawCentroids:    rawCentroids,
		EncryptedQuery:  queryBytes,
		PackedCentroids: centroidBytes,
		NumCentroids:    int32(numCentroids),
		Dimension:       int32(dimension),
	})
	if err != nil {
		return nil, fmt.Errorf("BatchDotProduct RPC failed: %w", err)
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("BatchDotProduct server error: %s", resp.Error)
	}

	// Reconstruct result — convert from HEonGPU domain back to Lattigo domain.
	engine := p.localPool.GetPrimaryEngine()
	var result *rlwe.Ciphertext
	if resp.RawResult != nil && len(resp.RawResult.Polynomials) > 0 && p.nttConverter != nil {
		// Convert from HEonGPU NTT domain back to Lattigo Montgomery-NTT domain
		result, err = CiphertextFromHEonGPU(resp.RawResult, engine.params, p.nttConverter)
		if err != nil {
			return nil, fmt.Errorf("convert HEonGPU result to Lattigo: %w", err)
		}
	} else if resp.RawResult != nil && len(resp.RawResult.Polynomials) > 0 {
		result, err = reconstructCiphertext(engine.params, resp.RawResult)
		if err != nil {
			return nil, fmt.Errorf("reconstruct raw result: %w", err)
		}
	} else {
		result, err = engine.DeserializeCiphertext(resp.EncryptedResult)
		if err != nil {
			return nil, fmt.Errorf("deserialize result: %w", err)
		}
	}

	return result, nil
}

// Acquire returns a gpuEvalEngine that delegates HE ops to the GPU server
// and decrypt ops to the local pool.
func (p *GPUHEProvider) Acquire() EvalEngine {
	localEngine := p.localPool.Acquire()
	return &gpuEvalEngine{
		local:    localEngine,
		provider: p,
	}
}

func (p *GPUHEProvider) Release(e EvalEngine) {
	if ge, ok := e.(*gpuEvalEngine); ok {
		p.localPool.Release(ge.local)
	}
}

func (p *GPUHEProvider) Size() int {
	return p.localPool.Size()
}

func (p *GPUHEProvider) GetParams() hefloat.Parameters {
	return p.localPool.GetParams()
}

func (p *GPUHEProvider) GetEncoder() *hefloat.Encoder {
	return p.localPool.GetEncoder()
}

func (p *GPUHEProvider) Close() {
	if p.conn != nil {
		p.conn.Close()
	}
	p.localPool.Close()
}

// --- gpuEvalEngine wraps a local engine + GPU provider ---

type gpuEvalEngine struct {
	local    *Engine
	provider *GPUHEProvider
}

func (g *gpuEvalEngine) HomomorphicDotProduct(encQuery *rlwe.Ciphertext, vector []float64) (*rlwe.Ciphertext, error) {
	// Single dot product: use local engine (not worth GPU round-trip).
	return g.local.HomomorphicDotProduct(encQuery, vector)
}

func (g *gpuEvalEngine) HomomorphicDotProductCached(encQuery *rlwe.Ciphertext, encodedVector *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	// Single cached dot product: use local engine.
	return g.local.HomomorphicDotProductCached(encQuery, encodedVector)
}

func (g *gpuEvalEngine) HomomorphicBatchDotProduct(encQuery *rlwe.Ciphertext, packedCentroids *rlwe.Plaintext, numCentroids, dimension int) (*rlwe.Ciphertext, error) {
	// Batch dot product: offload to GPU server.
	return g.provider.HomomorphicBatchDotProduct(encQuery, packedCentroids, numCentroids, dimension)
}

func (g *gpuEvalEngine) DecryptScalar(ct *rlwe.Ciphertext) (float64, error) {
	// Decryption: always local (secret key never leaves client).
	return g.local.DecryptScalar(ct)
}

// --- Raw coefficient extraction and reconstruction ---
// These functions bridge Lattigo's internal representation with the
// cross-library raw coefficient format used for GPU communication.

// extractRawCiphertext extracts raw uint64 coefficients from a Lattigo ciphertext.
// The coefficients are in RNS+NTT domain — the server can reconstruct an
// equivalent ciphertext in any CKKS library that uses the same modulus primes.
func extractRawCiphertext(ct *rlwe.Ciphertext) *pb.RawCiphertext {
	polys := make([]*pb.RawPolynomial, len(ct.Value))
	for i, poly := range ct.Value {
		numLevels := poly.Level() + 1
		ringSize := poly.N()
		flat := make([]uint64, 0, numLevels*ringSize)
		for lvl := 0; lvl < numLevels; lvl++ {
			flat = append(flat, poly.Coeffs[lvl]...)
		}
		polys[i] = &pb.RawPolynomial{
			Coefficients: flat,
			NumLevels:    int32(numLevels),
			RingSize:     int32(ringSize),
		}
	}

	return &pb.RawCiphertext{
		Polynomials: polys,
		Scale:       ct.Scale.Float64(),
		IsNtt:       ct.IsNTT,
		Depth:       int32(ct.Value[0].Level()),
	}
}

// extractRawPlaintext extracts raw uint64 coefficients from a Lattigo plaintext.
func extractRawPlaintext(pt *rlwe.Plaintext) *pb.RawPlaintext {
	poly := pt.Value
	numLevels := poly.Level() + 1
	ringSize := poly.N()
	flat := make([]uint64, 0, numLevels*ringSize)
	for lvl := 0; lvl < numLevels; lvl++ {
		flat = append(flat, poly.Coeffs[lvl]...)
	}

	return &pb.RawPlaintext{
		Polynomial: &pb.RawPolynomial{
			Coefficients: flat,
			NumLevels:    int32(numLevels),
			RingSize:     int32(ringSize),
		},
		Scale: pt.Scale.Float64(),
		IsNtt: pt.IsNTT,
	}
}

// reconstructCiphertext rebuilds a Lattigo ciphertext from raw coefficients.
func reconstructCiphertext(params hefloat.Parameters, raw *pb.RawCiphertext) (*rlwe.Ciphertext, error) {
	if len(raw.Polynomials) < 2 {
		return nil, fmt.Errorf("need at least 2 polynomials, got %d", len(raw.Polynomials))
	}

	numLevels := int(raw.Polynomials[0].NumLevels)
	ringSize := int(raw.Polynomials[0].RingSize)
	level := numLevels - 1

	ct := rlwe.NewCiphertext(params, len(raw.Polynomials)-1, level)
	ct.IsNTT = raw.IsNtt
	ct.Scale = rlwe.NewScale(raw.Scale)

	for i, rawPoly := range raw.Polynomials {
		if len(rawPoly.Coefficients) != numLevels*ringSize {
			return nil, fmt.Errorf("poly %d: expected %d coeffs, got %d",
				i, numLevels*ringSize, len(rawPoly.Coefficients))
		}
		for lvl := 0; lvl < numLevels; lvl++ {
			start := lvl * ringSize
			copy(ct.Value[i].Coeffs[lvl], rawPoly.Coefficients[start:start+ringSize])
		}
	}

	return ct, nil
}

// --- Serialization helpers ---

func serializeCiphertext(ct *rlwe.Ciphertext) ([]byte, error) {
	buf := new(bytes.Buffer)
	if _, err := ct.WriteTo(buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func serializePlaintext(pt *rlwe.Plaintext) ([]byte, error) {
	buf := new(bytes.Buffer)
	if _, err := pt.WriteTo(buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func serializeGaloisKeys(engine *Engine, elements []uint64) ([][]byte, error) {
	engine.mu.RLock()
	defer engine.mu.RUnlock()

	if engine.evalKeys == nil {
		return nil, fmt.Errorf("no evaluation keys available")
	}

	keys := make([][]byte, len(elements))
	for i, el := range elements {
		gk, err := engine.evalKeys.GetGaloisKey(el)
		if err != nil {
			return nil, fmt.Errorf("Galois key for element %d: %w", el, err)
		}
		buf := new(bytes.Buffer)
		if _, err := gk.WriteTo(buf); err != nil {
			return nil, fmt.Errorf("serialize Galois key %d: %w", el, err)
		}
		keys[i] = buf.Bytes()
	}
	return keys, nil
}

func serializeRelinKey(engine *Engine) ([]byte, error) {
	engine.mu.RLock()
	defer engine.mu.RUnlock()

	if engine.evalKeys == nil {
		return nil, nil
	}

	rlk, err := engine.evalKeys.GetRelinearizationKey()
	if err != nil {
		// No relin key available (not all configs have one).
		return nil, nil
	}
	buf := new(bytes.Buffer)
	if _, err := rlk.WriteTo(buf); err != nil {
		return nil, fmt.Errorf("serialize relin key: %w", err)
	}
	return buf.Bytes(), nil
}
