//go:build integration

package test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	opaque "github.com/Prasad-178/opaque"
	pb "github.com/Prasad-178/opaque/api/proto"
	"github.com/Prasad-178/opaque/pkg/grpcserver"
)

// startTestGRPCServer starts a gRPC server backed by an opaque.DB on a random port.
func startTestGRPCServer(t *testing.T, numVectors, dim int) (pb.OpaqueSearchClient, *opaque.DB, func()) {
	t.Helper()

	db, err := opaque.NewDB(opaque.Config{
		Dimension:   dim,
		NumClusters: 4,
	})
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}

	// Add test vectors.
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
		if err := db.AddBatch(context.Background(), ids, vectors); err != nil {
			t.Fatalf("failed to add vectors: %v", err)
		}
		if err := db.Build(context.Background()); err != nil {
			t.Fatalf("failed to build index: %v", err)
		}
	}

	// Start gRPC server.
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
	pb.RegisterOpaqueSearchServer(grpcServer, grpcserver.New(db))

	go grpcServer.Serve(lis)

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
		db.Close()
	}

	return client, db, cleanup
}

func TestGRPCHealthCheck(t *testing.T) {
	const (
		numVectors = 200
		dim        = 8
	)

	client, _, cleanup := startTestGRPCServer(t, numVectors, dim)
	defer cleanup()

	ctx := context.Background()

	resp, err := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if resp.Status != pb.HealthCheckResponse_SERVING {
		t.Errorf("expected SERVING, got %v", resp.Status)
	}
	if resp.VectorCount != int64(numVectors) {
		t.Errorf("expected %d vectors, got %d", numVectors, resp.VectorCount)
	}
	if resp.Message != "healthy" {
		t.Errorf("expected 'healthy', got %q", resp.Message)
	}
	t.Logf("Health: %v, vectors=%d", resp.Status, resp.VectorCount)
}

func TestGRPCDeprecatedRPCs(t *testing.T) {
	client, _, cleanup := startTestGRPCServer(t, 10, 8)
	defer cleanup()

	ctx := context.Background()

	// All legacy LSH RPCs should return Unimplemented.
	tests := []struct {
		name string
		call func() error
	}{
		{"RegisterKey", func() error {
			_, err := client.RegisterKey(ctx, &pb.RegisterKeyRequest{PublicKey: []byte("key")})
			return err
		}},
		{"GetPlanes", func() error {
			_, err := client.GetPlanes(ctx, &pb.GetPlanesRequest{SessionId: "s"})
			return err
		}},
		{"GetCandidates", func() error {
			_, err := client.GetCandidates(ctx, &pb.CandidateRequest{SessionId: "s", LshHash: []byte{1}})
			return err
		}},
		{"ComputeScores", func() error {
			_, err := client.ComputeScores(ctx, &pb.ScoreRequest{SessionId: "s", EncryptedQuery: []byte{1}})
			return err
		}},
		{"Search", func() error {
			_, err := client.Search(ctx, &pb.SearchRequest{SessionId: "s"})
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if err == nil {
				t.Fatal("expected error for deprecated RPC")
			}
			s, ok := status.FromError(err)
			if !ok || s.Code() != codes.Unimplemented {
				t.Errorf("expected Unimplemented, got code=%v err=%v", s.Code(), err)
			}
		})
	}
}

func TestGRPCGracefulShutdown(t *testing.T) {
	client, _, cleanup := startTestGRPCServer(t, 10, 8)

	// Verify server is up.
	ctx := context.Background()
	_, err := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}

	// Trigger graceful shutdown.
	cleanup()

	// After shutdown, new requests should fail.
	_, err = client.HealthCheck(ctx, &pb.HealthCheckRequest{})
	if err == nil {
		t.Error("expected error after shutdown")
	}
}
