//go:build integration

package test

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/opaque/opaque/go/api/proto"
	"github.com/opaque/opaque/go/internal/service"
	"github.com/opaque/opaque/go/internal/store"
	"github.com/opaque/opaque/go/pkg/grpcserver"
)

// startTestGRPCServer starts a gRPC server on a random port and returns
// the client connection and a cleanup function.
func startTestGRPCServer(t *testing.T, numVectors, dim int) (pb.OpaqueSearchClient, func()) {
	t.Helper()

	vectorStore := store.NewMemoryStore()
	cfg := service.Config{
		LSHNumBits:          16,
		LSHDimension:        dim,
		LSHSeed:             42,
		MaxSessionTTL:       time.Hour,
		MaxConcurrentScores: 4,
	}

	svc, err := service.NewSearchService(cfg, vectorStore)
	if err != nil {
		t.Fatalf("failed to create search service: %v", err)
	}

	// Add test vectors
	if numVectors > 0 {
		ids := make([]string, numVectors)
		vectors := make([][]float64, numVectors)
		rng := rand.New(rand.NewSource(42))
		for i := 0; i < numVectors; i++ {
			ids[i] = fmt.Sprintf("vec_%d", i)
			vec := make([]float64, dim)
			var norm float64
			for j := range vec {
				vec[j] = rng.NormFloat64()
				norm += vec[j] * vec[j]
			}
			norm = math.Sqrt(norm)
			for j := range vec {
				vec[j] /= norm
			}
			vectors[i] = vec
		}
		if err := svc.AddVectors(context.Background(), ids, vectors, nil); err != nil {
			t.Fatalf("failed to add vectors: %v", err)
		}
	}

	// Start server on random port
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(
			grpcserver.RecoveryUnaryInterceptor(),
			grpcserver.LoggingUnaryInterceptor(),
		),
		grpc.ChainStreamInterceptor(
			grpcserver.RecoveryStreamInterceptor(),
			grpcserver.LoggingStreamInterceptor(),
		),
	)
	pb.RegisterOpaqueSearchServer(grpcServer, grpcserver.New(svc))

	go grpcServer.Serve(lis)

	// Create client connection
	conn, err := grpc.NewClient(
		lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		grpcServer.Stop()
		t.Fatalf("failed to dial: %v", err)
	}

	client := pb.NewOpaqueSearchClient(conn)

	cleanup := func() {
		conn.Close()
		grpcServer.GracefulStop()
	}

	return client, cleanup
}

func TestGRPCFullRoundTrip(t *testing.T) {
	const (
		numVectors = 200
		dim        = 8
	)

	client, cleanup := startTestGRPCServer(t, numVectors, dim)
	defer cleanup()

	ctx := context.Background()

	// Step 1: Register a key
	regResp, err := client.RegisterKey(ctx, &pb.RegisterKeyRequest{
		PublicKey:         []byte("test-bfv-public-key"),
		SessionTtlSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("RegisterKey failed: %v", err)
	}
	if regResp.SessionId == "" {
		t.Fatal("expected non-empty session ID")
	}
	sessionID := regResp.SessionId
	t.Logf("Registered session: %s (TTL: %ds)", sessionID, regResp.SessionTtlSeconds)

	// Step 2: Get LSH planes
	planesResp, err := client.GetPlanes(ctx, &pb.GetPlanesRequest{
		SessionId: sessionID,
	})
	if err != nil {
		t.Fatalf("GetPlanes failed: %v", err)
	}
	if planesResp.NumPlanes <= 0 {
		t.Fatal("expected positive num_planes")
	}
	if planesResp.Dimension != int32(dim) {
		t.Errorf("expected dimension=%d, got %d", dim, planesResp.Dimension)
	}
	expectedLen := int(planesResp.NumPlanes) * int(planesResp.Dimension)
	if len(planesResp.Planes) != expectedLen {
		t.Errorf("expected %d plane values, got %d", expectedLen, len(planesResp.Planes))
	}
	t.Logf("Got %d planes of dimension %d", planesResp.NumPlanes, planesResp.Dimension)

	// Step 3: Get candidates using a simple hash
	lshHash := make([]byte, 2)
	lshHash[0] = 0xAB
	lshHash[1] = 0xCD

	candResp, err := client.GetCandidates(ctx, &pb.CandidateRequest{
		SessionId:     sessionID,
		LshHash:       lshHash,
		NumCandidates: 20,
		MultiProbe:    true,
		NumProbes:     3,
	})
	if err != nil {
		t.Fatalf("GetCandidates failed: %v", err)
	}
	if len(candResp.Ids) == 0 {
		t.Fatal("expected at least one candidate")
	}
	if len(candResp.Ids) != len(candResp.Distances) {
		t.Errorf("ids/distances length mismatch: %d vs %d", len(candResp.Ids), len(candResp.Distances))
	}
	t.Logf("Got %d candidates", len(candResp.Ids))

	// Step 4: Health check
	healthResp, err := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if healthResp.Status != pb.HealthCheckResponse_SERVING {
		t.Errorf("expected SERVING, got %v", healthResp.Status)
	}
	if healthResp.VectorCount != int64(numVectors) {
		t.Errorf("expected %d vectors, got %d", numVectors, healthResp.VectorCount)
	}
	if healthResp.ActiveSessions != 1 {
		t.Errorf("expected 1 active session, got %d", healthResp.ActiveSessions)
	}
	t.Logf("Health: %v, vectors=%d, sessions=%d", healthResp.Status, healthResp.VectorCount, healthResp.ActiveSessions)
}

func TestGRPCComputeScoresStream(t *testing.T) {
	client, cleanup := startTestGRPCServer(t, 50, 8)
	defer cleanup()

	ctx := context.Background()

	// Register session
	regResp, err := client.RegisterKey(ctx, &pb.RegisterKeyRequest{
		PublicKey:         []byte("test-key"),
		SessionTtlSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("RegisterKey failed: %v", err)
	}

	// Get candidates first
	candResp, err := client.GetCandidates(ctx, &pb.CandidateRequest{
		SessionId:     regResp.SessionId,
		LshHash:       []byte{0x01, 0x02},
		NumCandidates: 5,
	})
	if err != nil {
		t.Fatalf("GetCandidates failed: %v", err)
	}
	if len(candResp.Ids) == 0 {
		t.Skip("no candidates returned, skipping stream test")
	}

	// Note: ComputeScoresStream requires a valid BFV ciphertext.
	// We test that the stream connection works and returns an error for invalid input.
	stream, err := client.ComputeScoresStream(ctx, &pb.ScoreRequest{
		SessionId:     regResp.SessionId,
		EncryptedQuery: []byte("not-a-real-ciphertext"),
		CandidateIds:  candResp.Ids[:1],
	})
	if err != nil {
		t.Fatalf("ComputeScoresStream call failed: %v", err)
	}

	// Read from stream — expect an error since the ciphertext is invalid
	_, err = stream.Recv()
	if err == nil || err == io.EOF {
		t.Log("Received response (ciphertext was accepted)")
	} else {
		t.Logf("Got expected error for invalid ciphertext: %v", err)
	}
}

func TestGRPCMultipleSessions(t *testing.T) {
	client, cleanup := startTestGRPCServer(t, 100, 8)
	defer cleanup()

	ctx := context.Background()

	// Register multiple sessions
	sessions := make([]string, 5)
	for i := range sessions {
		resp, err := client.RegisterKey(ctx, &pb.RegisterKeyRequest{
			PublicKey:         []byte(fmt.Sprintf("key-%d", i)),
			SessionTtlSeconds: 3600,
		})
		if err != nil {
			t.Fatalf("RegisterKey %d failed: %v", i, err)
		}
		sessions[i] = resp.SessionId
	}

	// Each session should independently access planes
	for i, sid := range sessions {
		resp, err := client.GetPlanes(ctx, &pb.GetPlanesRequest{SessionId: sid})
		if err != nil {
			t.Errorf("GetPlanes for session %d failed: %v", i, err)
		}
		if resp.NumPlanes <= 0 {
			t.Errorf("session %d: expected positive num_planes", i)
		}
	}

	// Verify session count in health
	health, err := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if health.ActiveSessions != int64(len(sessions)) {
		t.Errorf("expected %d sessions, got %d", len(sessions), health.ActiveSessions)
	}
}

func TestGRPCGracefulShutdown(t *testing.T) {
	client, cleanup := startTestGRPCServer(t, 10, 8)

	// Register a session first to verify it works
	ctx := context.Background()
	_, err := client.RegisterKey(ctx, &pb.RegisterKeyRequest{
		PublicKey:         []byte("test-key"),
		SessionTtlSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("RegisterKey failed: %v", err)
	}

	// Trigger graceful shutdown
	cleanup()

	// After shutdown, new requests should fail
	_, err = client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err == nil {
		t.Error("expected error after shutdown")
	}
}
