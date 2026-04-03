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

// RegisterEvalKeys sends the evaluation keys (Galois + relin) to the GPU server.
// This is called automatically on the first HomomorphicBatchDotProduct if not called explicitly.
func (p *GPUHEProvider) RegisterEvalKeys() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.keysRegistered {
		return nil
	}

	engine := p.localPool.GetPrimaryEngine()

	// Serialize CKKS parameters.
	paramsBytes, err := engine.params.MarshalBinary()
	if err != nil {
		return fmt.Errorf("failed to serialize params: %w", err)
	}

	// Serialize Galois keys.
	galoisElems := galoisElements(engine.params)
	galoisKeys, err := serializeGaloisKeys(engine, galoisElems)
	if err != nil {
		return fmt.Errorf("failed to serialize Galois keys: %w", err)
	}

	// Serialize relinearization key.
	relinBytes, err := serializeRelinKey(engine)
	if err != nil {
		return fmt.Errorf("failed to serialize relin key: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := p.client.RegisterEvalKeys(ctx, &pb.RegisterEvalKeysRequest{
		CkksParams:     paramsBytes,
		RelinKey:       relinBytes,
		GaloisKeys:     galoisKeys,
		GaloisElements: galoisElems,
		SessionId:      p.sessionID,
	})
	if err != nil {
		return fmt.Errorf("RegisterEvalKeys RPC failed: %w", err)
	}
	if !resp.Success {
		return fmt.Errorf("RegisterEvalKeys failed: %s", resp.Error)
	}

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

	// Serialize ciphertext and plaintext.
	queryBytes, err := serializeCiphertext(encQuery)
	if err != nil {
		return nil, fmt.Errorf("serialize query: %w", err)
	}

	centroidBytes, err := serializePlaintext(packedCentroids)
	if err != nil {
		return nil, fmt.Errorf("serialize centroids: %w", err)
	}

	// Send to GPU server.
	ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
	defer cancel()

	resp, err := p.client.BatchDotProduct(ctx, &pb.BatchDotProductRequest{
		SessionId:       p.sessionID,
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

	// Deserialize result.
	engine := p.localPool.GetPrimaryEngine()
	result, err := engine.DeserializeCiphertext(resp.EncryptedResult)
	if err != nil {
		return nil, fmt.Errorf("deserialize result: %w", err)
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
