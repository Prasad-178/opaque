package grpcserver

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	pb "github.com/opaque/opaque/go/api/proto"
	"github.com/opaque/opaque/go/internal/service"
	"github.com/opaque/opaque/go/internal/store"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func setupTestServer(t *testing.T, numVectors int) *Server {
	t.Helper()

	vectorStore := store.NewMemoryStore()
	cfg := service.Config{
		LSHNumBits:          16,
		LSHDimension:        8,
		LSHSeed:             42,
		MaxSessionTTL:       time.Hour,
		MaxConcurrentScores: 4,
	}

	svc, err := service.NewSearchService(cfg, vectorStore)
	if err != nil {
		t.Fatalf("failed to create search service: %v", err)
	}

	if numVectors > 0 {
		ids := make([]string, numVectors)
		vectors := make([][]float64, numVectors)
		rng := rand.New(rand.NewSource(42))
		for i := 0; i < numVectors; i++ {
			ids[i] = fmt.Sprintf("vec_%d", i)
			vec := make([]float64, 8)
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

	return New(svc)
}

func registerSession(t *testing.T, srv *Server) string {
	t.Helper()
	resp, err := srv.RegisterKey(context.Background(), &pb.RegisterKeyRequest{
		PublicKey:         []byte("test-public-key"),
		SessionTtlSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("RegisterKey failed: %v", err)
	}
	return resp.SessionId
}

func TestRegisterKey(t *testing.T) {
	srv := setupTestServer(t, 0)

	resp, err := srv.RegisterKey(context.Background(), &pb.RegisterKeyRequest{
		PublicKey:         []byte("my-public-key"),
		SessionTtlSeconds: 300,
	})
	if err != nil {
		t.Fatalf("RegisterKey failed: %v", err)
	}
	if resp.SessionId == "" {
		t.Error("expected non-empty session ID")
	}
	if resp.SessionTtlSeconds <= 0 {
		t.Errorf("expected positive TTL, got %d", resp.SessionTtlSeconds)
	}
}

func TestRegisterKey_EmptyPublicKey(t *testing.T) {
	srv := setupTestServer(t, 0)

	_, err := srv.RegisterKey(context.Background(), &pb.RegisterKeyRequest{
		PublicKey: nil,
	})
	if err == nil {
		t.Fatal("expected error for empty public key")
	}
	if s, ok := status.FromError(err); !ok || s.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestGetPlanes(t *testing.T) {
	srv := setupTestServer(t, 10)
	sessionID := registerSession(t, srv)

	resp, err := srv.GetPlanes(context.Background(), &pb.GetPlanesRequest{
		SessionId: sessionID,
	})
	if err != nil {
		t.Fatalf("GetPlanes failed: %v", err)
	}
	if resp.NumPlanes <= 0 {
		t.Error("expected positive num_planes")
	}
	if resp.Dimension <= 0 {
		t.Error("expected positive dimension")
	}
	expectedLen := int(resp.NumPlanes) * int(resp.Dimension)
	if len(resp.Planes) != expectedLen {
		t.Errorf("expected %d plane values, got %d", expectedLen, len(resp.Planes))
	}
}

func TestGetPlanes_InvalidSession(t *testing.T) {
	srv := setupTestServer(t, 0)

	_, err := srv.GetPlanes(context.Background(), &pb.GetPlanesRequest{
		SessionId: "non-existent-session",
	})
	if err == nil {
		t.Fatal("expected error for invalid session")
	}
	if s, ok := status.FromError(err); !ok || s.Code() != codes.Unauthenticated {
		t.Errorf("expected Unauthenticated, got %v", err)
	}
}

func TestGetCandidates(t *testing.T) {
	srv := setupTestServer(t, 100)
	sessionID := registerSession(t, srv)

	// Use a simple LSH hash
	resp, err := srv.GetCandidates(context.Background(), &pb.CandidateRequest{
		SessionId:     sessionID,
		LshHash:       []byte{0x01, 0x02},
		NumCandidates: 10,
	})
	if err != nil {
		t.Fatalf("GetCandidates failed: %v", err)
	}
	if len(resp.Ids) == 0 {
		t.Error("expected at least one candidate")
	}
	if len(resp.Ids) != len(resp.Distances) {
		t.Errorf("ids and distances length mismatch: %d vs %d", len(resp.Ids), len(resp.Distances))
	}
}

func TestGetCandidates_MissingHash(t *testing.T) {
	srv := setupTestServer(t, 10)
	sessionID := registerSession(t, srv)

	_, err := srv.GetCandidates(context.Background(), &pb.CandidateRequest{
		SessionId:     sessionID,
		LshHash:       nil,
		NumCandidates: 10,
	})
	if err == nil {
		t.Fatal("expected error for missing LSH hash")
	}
	if s, ok := status.FromError(err); !ok || s.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument, got %v", err)
	}
}

func TestHealthCheck(t *testing.T) {
	srv := setupTestServer(t, 50)

	resp, err := srv.HealthCheck(context.Background(), &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if resp.Status != pb.HealthCheckResponse_SERVING {
		t.Errorf("expected SERVING, got %v", resp.Status)
	}
	if resp.VectorCount != 50 {
		t.Errorf("expected 50 vectors, got %d", resp.VectorCount)
	}
	if resp.Message != "healthy" {
		t.Errorf("expected 'healthy' message, got %q", resp.Message)
	}
}

func TestInvalidSession_AllRPCs(t *testing.T) {
	srv := setupTestServer(t, 10)
	badSession := "bad-session-id"

	tests := []struct {
		name string
		call func() error
	}{
		{"GetPlanes", func() error {
			_, err := srv.GetPlanes(context.Background(), &pb.GetPlanesRequest{SessionId: badSession})
			return err
		}},
		{"GetCandidates", func() error {
			_, err := srv.GetCandidates(context.Background(), &pb.CandidateRequest{
				SessionId: badSession, LshHash: []byte{0x01}, NumCandidates: 5,
			})
			return err
		}},
		{"ComputeScores", func() error {
			_, err := srv.ComputeScores(context.Background(), &pb.ScoreRequest{
				SessionId: badSession, EncryptedQuery: []byte{0x01}, CandidateIds: []string{"a"},
			})
			return err
		}},
		{"Search", func() error {
			_, err := srv.Search(context.Background(), &pb.SearchRequest{
				SessionId: badSession, LshHash: []byte{0x01}, EncryptedQuery: []byte{0x01},
			})
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if err == nil {
				t.Fatal("expected error for invalid session")
			}
			s, ok := status.FromError(err)
			if !ok || s.Code() != codes.Unauthenticated {
				t.Errorf("expected Unauthenticated, got code=%v err=%v", s.Code(), err)
			}
		})
	}
}

func TestEmptySessionID(t *testing.T) {
	srv := setupTestServer(t, 0)

	tests := []struct {
		name string
		call func() error
	}{
		{"GetPlanes", func() error {
			_, err := srv.GetPlanes(context.Background(), &pb.GetPlanesRequest{})
			return err
		}},
		{"GetCandidates", func() error {
			_, err := srv.GetCandidates(context.Background(), &pb.CandidateRequest{})
			return err
		}},
		{"ComputeScores", func() error {
			_, err := srv.ComputeScores(context.Background(), &pb.ScoreRequest{})
			return err
		}},
		{"Search", func() error {
			_, err := srv.Search(context.Background(), &pb.SearchRequest{})
			return err
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if err == nil {
				t.Fatal("expected error for empty session ID")
			}
			s, ok := status.FromError(err)
			if !ok || s.Code() != codes.InvalidArgument {
				t.Errorf("expected InvalidArgument, got code=%v err=%v", s.Code(), err)
			}
		})
	}
}
