// Command gpu-server runs a GPU HE acceleration server.
//
// This server receives encrypted data via gRPC and performs homomorphic
// encryption operations (batch dot products). The current implementation
// uses Lattigo CPU as a stub backend; a future version will use HEonGPU
// or custom CUDA kernels for GPU acceleration.
//
// The server never has access to secret keys — it only receives evaluation
// keys (Galois + relinearization) for performing HE computation. All
// encryption and decryption happens on the client side.
//
// Usage:
//
//	go run ./cmd/gpu-server/ -addr :50052
//	go run ./cmd/gpu-server/ -addr :50052 -backend cpu    # explicit CPU stub
//	go run ./cmd/gpu-server/ -addr :50052 -backend cuda   # future: CUDA backend
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"google.golang.org/grpc"
)

func main() {
	addr := flag.String("addr", ":50052", "Server listen address")
	backend := flag.String("backend", "cpu", "Compute backend: cpu (Lattigo stub) or cuda (future)")
	maxMsgSize := flag.Int("max-msg-size", 256*1024*1024, "Max gRPC message size in bytes")
	flag.Parse()

	if *backend != "cpu" {
		log.Fatalf("Backend %q not implemented. Only 'cpu' (Lattigo stub) is available.", *backend)
	}

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", *addr, err)
	}

	srv := grpc.NewServer(
		grpc.MaxRecvMsgSize(*maxMsgSize),
		grpc.MaxSendMsgSize(*maxMsgSize),
	)
	pb.RegisterGPUHEServiceServer(srv, newGPUHEServer(*backend))

	log.Printf("GPU HE Server starting on %s (backend: %s)", *addr, *backend)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// gpuHEServer implements the GPUHEService gRPC interface.
type gpuHEServer struct {
	pb.UnimplementedGPUHEServiceServer

	backend  string
	sessions map[string]*heSession
	mu       sync.RWMutex
}

// heSession holds per-client state: CKKS params + evaluator with loaded keys.
type heSession struct {
	params    hefloat.Parameters
	evaluator *hefloat.Evaluator
	encoder   *hefloat.Encoder
	createdAt time.Time
	mu        sync.Mutex // Protects evaluator (not thread-safe)
}

func newGPUHEServer(backend string) *gpuHEServer {
	return &gpuHEServer{
		backend:  backend,
		sessions: make(map[string]*heSession),
	}
}

func (s *gpuHEServer) RegisterEvalKeys(_ context.Context, req *pb.RegisterEvalKeysRequest) (*pb.RegisterEvalKeysResponse, error) {
	if req.SessionId == "" {
		return &pb.RegisterEvalKeysResponse{Success: false, Error: "session_id required"}, nil
	}

	log.Printf("[%s] RegisterEvalKeys: %d Galois keys, relin key %d bytes",
		req.SessionId, len(req.GaloisKeysSerialized), len(req.RelinKeySerialized))

	if req.Params != nil {
		log.Printf("[%s] CKKS params: LogN=%d, %d Q primes, %d P primes",
			req.SessionId, req.Params.LogN, len(req.Params.QModuli), len(req.Params.PModuli))
	}

	// Deserialize CKKS parameters (use legacy binary format for CPU stub).
	var params hefloat.Parameters
	if err := params.UnmarshalBinary(req.CkksParamsBinary); err != nil {
		return &pb.RegisterEvalKeysResponse{Success: false, Error: fmt.Sprintf("unmarshal params: %v", err)}, nil
	}

	// Deserialize Galois keys.
	galoisKeys := make([]*rlwe.GaloisKey, len(req.GaloisKeysSerialized))
	for i, keyBytes := range req.GaloisKeysSerialized {
		gk := &rlwe.GaloisKey{}
		if _, err := gk.ReadFrom(bytes.NewReader(keyBytes)); err != nil {
			return &pb.RegisterEvalKeysResponse{
				Success: false,
				Error:   fmt.Sprintf("deserialize Galois key %d: %v", i, err),
			}, nil
		}
		galoisKeys[i] = gk
	}

	// Deserialize relinearization key (optional).
	var rlk *rlwe.RelinearizationKey
	if len(req.RelinKeySerialized) > 0 {
		rlk = &rlwe.RelinearizationKey{}
		if _, err := rlk.ReadFrom(bytes.NewReader(req.RelinKeySerialized)); err != nil {
			return &pb.RegisterEvalKeysResponse{
				Success: false,
				Error:   fmt.Sprintf("deserialize relin key: %v", err),
			}, nil
		}
	}

	// Create evaluation key set and evaluator.
	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
	evaluator := hefloat.NewEvaluator(params, evk)

	session := &heSession{
		params:    params,
		evaluator: evaluator,
		encoder:   hefloat.NewEncoder(params),
		createdAt: time.Now(),
	}

	s.mu.Lock()
	s.sessions[req.SessionId] = session
	s.mu.Unlock()

	log.Printf("[%s] Session created (LogN=%d, %d Galois keys)",
		req.SessionId, params.LogN(), len(galoisKeys))

	return &pb.RegisterEvalKeysResponse{Success: true}, nil
}

func (s *gpuHEServer) BatchDotProduct(_ context.Context, req *pb.BatchDotProductRequest) (*pb.BatchDotProductResponse, error) {
	s.mu.RLock()
	session, ok := s.sessions[req.SessionId]
	s.mu.RUnlock()

	if !ok {
		return &pb.BatchDotProductResponse{Error: "session not found; call RegisterEvalKeys first"}, nil
	}

	// Lock the session's evaluator (Lattigo evaluators are not thread-safe).
	session.mu.Lock()
	defer session.mu.Unlock()

	// Deserialize encrypted query.
	encQuery := rlwe.NewCiphertext(session.params, 1, session.params.MaxLevel())
	if _, err := encQuery.ReadFrom(bytes.NewReader(req.EncryptedQuery)); err != nil {
		return &pb.BatchDotProductResponse{Error: fmt.Sprintf("deserialize query: %v", err)}, nil
	}

	// Deserialize packed centroids plaintext.
	packedCentroids := rlwe.NewPlaintext(session.params, session.params.MaxLevel())
	if _, err := packedCentroids.ReadFrom(bytes.NewReader(req.PackedCentroids)); err != nil {
		return &pb.BatchDotProductResponse{Error: fmt.Sprintf("deserialize centroids: %v", err)}, nil
	}

	dimension := int(req.Dimension)
	totalStart := time.Now()

	// --- CPU stub: run the same batch dot product as Lattigo ---
	// Future: replace this block with HEonGPU/CUDA calls.

	// Multiply: E(packed_q) * packed_centroids
	mulStart := time.Now()
	result, err := session.evaluator.MulNew(encQuery, packedCentroids)
	if err != nil {
		return &pb.BatchDotProductResponse{Error: fmt.Sprintf("multiply: %v", err)}, nil
	}
	mulTime := time.Since(mulStart)

	// Rescale
	rescaleStart := time.Now()
	if err := session.evaluator.Rescale(result, result); err != nil {
		return &pb.BatchDotProductResponse{Error: fmt.Sprintf("rescale: %v", err)}, nil
	}
	rescaleTime := time.Since(rescaleStart)

	// Partial sum with rotations
	var rotateTime, addTime time.Duration
	for stride := 1; stride < dimension; stride *= 2 {
		rotStart := time.Now()
		rotated, err := session.evaluator.RotateNew(result, stride)
		if err != nil {
			return &pb.BatchDotProductResponse{Error: fmt.Sprintf("rotate by %d: %v", stride, err)}, nil
		}
		rotateTime += time.Since(rotStart)

		addStart := time.Now()
		if err := session.evaluator.Add(result, rotated, result); err != nil {
			return &pb.BatchDotProductResponse{Error: fmt.Sprintf("add: %v", err)}, nil
		}
		addTime += time.Since(addStart)
	}

	totalTime := time.Since(totalStart)

	// Serialize result.
	buf := new(bytes.Buffer)
	if _, err := result.WriteTo(buf); err != nil {
		return &pb.BatchDotProductResponse{Error: fmt.Sprintf("serialize result: %v", err)}, nil
	}

	return &pb.BatchDotProductResponse{
		EncryptedResult: buf.Bytes(),
		ComputeTimeUs:   totalTime.Microseconds(),
		MultiplyUs:      mulTime.Microseconds(),
		RescaleUs:       rescaleTime.Microseconds(),
		RotateUs:        rotateTime.Microseconds(),
		AddUs:           addTime.Microseconds(),
	}, nil
}

func (s *gpuHEServer) HealthCheck(_ context.Context, _ *pb.GPUHealthCheckRequest) (*pb.GPUHealthCheckResponse, error) {
	s.mu.RLock()
	numSessions := len(s.sessions)
	s.mu.RUnlock()

	backendType := pb.GPUHealthCheckResponse_CPU_STUB
	deviceName := "Lattigo CPU (stub)"
	if s.backend == "cuda" {
		backendType = pb.GPUHealthCheckResponse_CUDA
		deviceName = "CUDA (not implemented)"
	}

	return &pb.GPUHealthCheckResponse{
		Status:         pb.GPUHealthCheckResponse_SERVING,
		Backend:        backendType,
		DeviceName:     deviceName,
		ActiveSessions: int32(numSessions),
	}, nil
}
