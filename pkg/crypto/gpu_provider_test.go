package crypto

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"sync"
	"testing"
	"time"

	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"google.golang.org/grpc"
)

// inProcessGPUServer runs a GPU HE stub server in the test process.
type inProcessGPUServer struct {
	pb.UnimplementedGPUHEServiceServer
	sessions map[string]*testHESession
	mu       sync.RWMutex
}

type testHESession struct {
	params    hefloat.Parameters
	evaluator *hefloat.Evaluator
	mu        sync.Mutex
}

func newInProcessGPUServer() *inProcessGPUServer {
	return &inProcessGPUServer{sessions: make(map[string]*testHESession)}
}

func (s *inProcessGPUServer) RegisterEvalKeys(_ context.Context, req *pb.RegisterEvalKeysRequest) (*pb.RegisterEvalKeysResponse, error) {
	var params hefloat.Parameters
	if err := params.UnmarshalBinary(req.CkksParams); err != nil {
		return &pb.RegisterEvalKeysResponse{Success: false, Error: err.Error()}, nil
	}

	galoisKeys := make([]*rlwe.GaloisKey, len(req.GaloisKeys))
	for i, kb := range req.GaloisKeys {
		gk := &rlwe.GaloisKey{}
		if _, err := gk.ReadFrom(bytes.NewReader(kb)); err != nil {
			return &pb.RegisterEvalKeysResponse{Success: false, Error: err.Error()}, nil
		}
		galoisKeys[i] = gk
	}

	var rlk *rlwe.RelinearizationKey
	if len(req.RelinKey) > 0 {
		rlk = &rlwe.RelinearizationKey{}
		if _, err := rlk.ReadFrom(bytes.NewReader(req.RelinKey)); err != nil {
			return &pb.RegisterEvalKeysResponse{Success: false, Error: err.Error()}, nil
		}
	}

	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
	s.mu.Lock()
	s.sessions[req.SessionId] = &testHESession{
		params:    params,
		evaluator: hefloat.NewEvaluator(params, evk),
	}
	s.mu.Unlock()

	return &pb.RegisterEvalKeysResponse{Success: true}, nil
}

func (s *inProcessGPUServer) BatchDotProduct(_ context.Context, req *pb.BatchDotProductRequest) (*pb.BatchDotProductResponse, error) {
	s.mu.RLock()
	session, ok := s.sessions[req.SessionId]
	s.mu.RUnlock()
	if !ok {
		return &pb.BatchDotProductResponse{Error: "session not found"}, nil
	}

	session.mu.Lock()
	defer session.mu.Unlock()

	encQuery := rlwe.NewCiphertext(session.params, 1, session.params.MaxLevel())
	if _, err := encQuery.ReadFrom(bytes.NewReader(req.EncryptedQuery)); err != nil {
		return &pb.BatchDotProductResponse{Error: err.Error()}, nil
	}

	pt := rlwe.NewPlaintext(session.params, session.params.MaxLevel())
	if _, err := pt.ReadFrom(bytes.NewReader(req.PackedCentroids)); err != nil {
		return &pb.BatchDotProductResponse{Error: err.Error()}, nil
	}

	dim := int(req.Dimension)

	result, err := session.evaluator.MulNew(encQuery, pt)
	if err != nil {
		return &pb.BatchDotProductResponse{Error: err.Error()}, nil
	}
	if err := session.evaluator.Rescale(result, result); err != nil {
		return &pb.BatchDotProductResponse{Error: err.Error()}, nil
	}

	for stride := 1; stride < dim; stride *= 2 {
		rotated, err := session.evaluator.RotateNew(result, stride)
		if err != nil {
			return &pb.BatchDotProductResponse{Error: err.Error()}, nil
		}
		if err := session.evaluator.Add(result, rotated, result); err != nil {
			return &pb.BatchDotProductResponse{Error: err.Error()}, nil
		}
	}

	buf := new(bytes.Buffer)
	if _, err := result.WriteTo(buf); err != nil {
		return &pb.BatchDotProductResponse{Error: err.Error()}, nil
	}

	return &pb.BatchDotProductResponse{EncryptedResult: buf.Bytes()}, nil
}

func (s *inProcessGPUServer) HealthCheck(_ context.Context, _ *pb.GPUHealthCheckRequest) (*pb.GPUHealthCheckResponse, error) {
	return &pb.GPUHealthCheckResponse{
		Status:  pb.GPUHealthCheckResponse_SERVING,
		Backend: pb.GPUHealthCheckResponse_CPU_STUB,
	}, nil
}

// startTestServer starts an in-process GPU stub server and returns the address.
func startTestServer(t *testing.T) string {
	t.Helper()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	srv := grpc.NewServer(
		grpc.MaxRecvMsgSize(256*1024*1024),
		grpc.MaxSendMsgSize(256*1024*1024),
	)
	pb.RegisterGPUHEServiceServer(srv, newInProcessGPUServer())

	go srv.Serve(lis)
	t.Cleanup(func() { srv.Stop() })

	return lis.Addr().String()
}

func TestGPUProvider_EncryptDecrypt(t *testing.T) {
	addr := startTestServer(t)

	provider, err := NewGPUHEProvider(GPUHEProviderConfig{
		ServerAddress: addr,
		LocalPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer provider.Close()

	// Encrypt and decrypt a vector — this should work locally without GPU.
	rng := rand.New(rand.NewSource(42))
	dim := 128
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rng.Float64()*2 - 1
	}

	ct, err := provider.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	result, err := provider.DecryptScalar(ct)
	if err != nil {
		t.Fatalf("DecryptScalar: %v", err)
	}

	// The scalar result is the first slot of the encrypted vector.
	if math.Abs(result-vec[0]) > 0.001 {
		t.Errorf("DecryptScalar: got %.6f, want %.6f (diff %.6f)", result, vec[0], math.Abs(result-vec[0]))
	}
}

func TestGPUProvider_BatchDotProduct(t *testing.T) {
	addr := startTestServer(t)

	provider, err := NewGPUHEProvider(GPUHEProviderConfig{
		ServerAddress: addr,
		LocalPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer provider.Close()

	dim := 128
	rng := rand.New(rand.NewSource(42))

	// Create a query vector and a centroid.
	query := make([]float64, dim)
	centroid := make([]float64, dim)
	for i := 0; i < dim; i++ {
		query[i] = rng.Float64()*2 - 1
		centroid[i] = rng.Float64()*2 - 1
	}

	// Compute expected dot product.
	expectedDot := 0.0
	for i := 0; i < dim; i++ {
		expectedDot += query[i] * centroid[i]
	}

	// Pack query (replicate across slots) and centroid.
	params := provider.GetParams()
	maxSlots := params.MaxSlots()
	centroidsPerPack := maxSlots / dim

	// Pack query: replicate dim-sized query across all centroid positions.
	packedQuery := make([]float64, maxSlots)
	for c := 0; c < centroidsPerPack; c++ {
		copy(packedQuery[c*dim:(c+1)*dim], query)
	}

	// Pack centroids: place centroid in first position, zeros elsewhere.
	packedCentroids := make([]float64, maxSlots)
	copy(packedCentroids[:dim], centroid)

	// Encode centroids as plaintext.
	encoder := provider.GetEncoder()
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(packedCentroids, pt); err != nil {
		t.Fatalf("Encode centroids: %v", err)
	}

	// Encrypt packed query.
	ct, err := provider.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	// Run batch dot product via GPU server.
	result, err := provider.HomomorphicBatchDotProduct(ct, pt, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("HomomorphicBatchDotProduct: %v", err)
	}

	// Decrypt and check.
	scores, err := provider.DecryptBatchScalars(result, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("DecryptBatchScalars: %v", err)
	}

	if len(scores) < 1 {
		t.Fatal("no scores returned")
	}

	diff := math.Abs(scores[0] - expectedDot)
	t.Logf("GPU BatchDotProduct: expected=%.6f, got=%.6f, diff=%.6f", expectedDot, scores[0], diff)

	if diff > 0.01 {
		t.Errorf("dot product mismatch: diff=%.6f (want < 0.01)", diff)
	}
}

func TestGPUProvider_MatchesDirect(t *testing.T) {
	// Verify GPU provider gives same results as DirectHEProvider.
	addr := startTestServer(t)

	gpuProvider, err := NewGPUHEProvider(GPUHEProviderConfig{
		ServerAddress: addr,
		LocalPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer gpuProvider.Close()

	directProvider, err := NewDirectHEProvider(2)
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	defer directProvider.Close()

	dim := 128
	rng := rand.New(rand.NewSource(99))

	// Create query and centroids.
	query := make([]float64, dim)
	centroids := make([][]float64, 4)
	for i := range query {
		query[i] = rng.Float64()*2 - 1
	}
	for c := range centroids {
		centroids[c] = make([]float64, dim)
		for i := range centroids[c] {
			centroids[c][i] = rng.Float64()*2 - 1
		}
	}

	// Pack query and centroids identically for both providers.
	params := gpuProvider.GetParams()
	maxSlots := params.MaxSlots()
	centroidsPerPack := maxSlots / dim

	packedQuery := make([]float64, maxSlots)
	for c := 0; c < centroidsPerPack; c++ {
		copy(packedQuery[c*dim:(c+1)*dim], query)
	}

	packedCentroidVals := make([]float64, maxSlots)
	for c := 0; c < len(centroids) && c < centroidsPerPack; c++ {
		copy(packedCentroidVals[c*dim:(c+1)*dim], centroids[c])
	}

	encoder := gpuProvider.GetEncoder()
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(packedCentroidVals, pt); err != nil {
		t.Fatalf("Encode: %v", err)
	}

	// Encrypt query (same for both — use GPU provider's encryption).
	ct, err := gpuProvider.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("Encrypt: %v", err)
	}

	// Run through GPU provider.
	gpuResult, err := gpuProvider.HomomorphicBatchDotProduct(ct, pt, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("GPU BatchDotProduct: %v", err)
	}
	gpuScores, err := gpuProvider.DecryptBatchScalars(gpuResult, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("GPU DecryptBatchScalars: %v", err)
	}

	// Run through direct provider (need to re-encrypt with direct provider's keys).
	directCt, err := directProvider.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("Direct Encrypt: %v", err)
	}

	// Encode with direct provider's encoder.
	directEncoder := directProvider.GetEncoder()
	directPt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := directEncoder.Encode(packedCentroidVals, directPt); err != nil {
		t.Fatalf("Direct Encode: %v", err)
	}

	directResult, err := directProvider.HomomorphicBatchDotProduct(directCt, directPt, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("Direct BatchDotProduct: %v", err)
	}
	directScores, err := directProvider.DecryptBatchScalars(directResult, centroidsPerPack, dim)
	if err != nil {
		t.Fatalf("Direct DecryptBatchScalars: %v", err)
	}

	// Compare results for the first 4 centroids.
	for i := 0; i < 4; i++ {
		// They use different keys, so exact values differ. But both should match
		// the plaintext dot product within CKKS precision.
		expectedDot := 0.0
		for j := 0; j < dim; j++ {
			expectedDot += query[j] * centroids[i][j]
		}

		gpuDiff := math.Abs(gpuScores[i] - expectedDot)
		directDiff := math.Abs(directScores[i] - expectedDot)

		t.Logf("Centroid %d: expected=%.6f, gpu=%.6f (diff=%.6f), direct=%.6f (diff=%.6f)",
			i, expectedDot, gpuScores[i], gpuDiff, directScores[i], directDiff)

		if gpuDiff > 0.01 {
			t.Errorf("GPU centroid %d: diff=%.6f too large", i, gpuDiff)
		}
		if directDiff > 0.01 {
			t.Errorf("Direct centroid %d: diff=%.6f too large", i, directDiff)
		}
	}
}

func TestGPUProvider_AcquireRelease(t *testing.T) {
	addr := startTestServer(t)

	provider, err := NewGPUHEProvider(GPUHEProviderConfig{
		ServerAddress: addr,
		LocalPoolSize: 2,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer provider.Close()

	if provider.Size() != 2 {
		t.Errorf("Size: got %d, want 2", provider.Size())
	}

	// Acquire and release engines.
	e1 := provider.Acquire()
	e2 := provider.Acquire()

	if _, ok := e1.(*gpuEvalEngine); !ok {
		t.Error("Acquire should return *gpuEvalEngine")
	}

	provider.Release(e1)
	provider.Release(e2)
}

func TestGPUProvider_HealthCheck(t *testing.T) {
	addr := startTestServer(t)

	provider, err := NewGPUHEProvider(GPUHEProviderConfig{
		ServerAddress: addr,
	})
	if err != nil {
		t.Fatalf("NewGPUHEProvider: %v", err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := provider.client.HealthCheck(ctx, &pb.GPUHealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck: %v", err)
	}

	if resp.Status != pb.GPUHealthCheckResponse_SERVING {
		t.Errorf("Status: got %v, want SERVING", resp.Status)
	}
	if resp.Backend != pb.GPUHealthCheckResponse_CPU_STUB {
		t.Errorf("Backend: got %v, want CPU_STUB", resp.Backend)
	}
}

// Suppress unused import warnings.
var _ = fmt.Sprintf
