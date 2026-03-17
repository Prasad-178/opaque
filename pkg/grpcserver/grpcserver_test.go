package grpcserver

import (
	"context"
	"testing"

	pb "github.com/Prasad-178/opaque/api/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// mockDB implements the DB interface for testing.
type mockDB struct {
	ready bool
	size  int
}

func (m *mockDB) IsReady() bool { return m.ready }
func (m *mockDB) Size() int     { return m.size }

func TestHealthCheck_Serving(t *testing.T) {
	srv := New(&mockDB{ready: true, size: 100})

	resp, err := srv.HealthCheck(context.Background(), &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if resp.Status != pb.HealthCheckResponse_SERVING {
		t.Errorf("expected SERVING, got %v", resp.Status)
	}
	if resp.VectorCount != 100 {
		t.Errorf("expected 100 vectors, got %d", resp.VectorCount)
	}
	if resp.Message != "healthy" {
		t.Errorf("expected 'healthy', got %q", resp.Message)
	}
}

func TestHealthCheck_NotServing(t *testing.T) {
	srv := New(&mockDB{ready: false, size: 0})

	resp, err := srv.HealthCheck(context.Background(), &pb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("HealthCheck failed: %v", err)
	}
	if resp.Status != pb.HealthCheckResponse_NOT_SERVING {
		t.Errorf("expected NOT_SERVING, got %v", resp.Status)
	}
}

func TestDeprecatedRPCs_ReturnUnimplemented(t *testing.T) {
	srv := New(&mockDB{})
	ctx := context.Background()

	tests := []struct {
		name string
		call func() error
	}{
		{"RegisterKey", func() error {
			_, err := srv.RegisterKey(ctx, &pb.RegisterKeyRequest{})
			return err
		}},
		{"GetPlanes", func() error {
			_, err := srv.GetPlanes(ctx, &pb.GetPlanesRequest{})
			return err
		}},
		{"GetCandidates", func() error {
			_, err := srv.GetCandidates(ctx, &pb.CandidateRequest{})
			return err
		}},
		{"ComputeScores", func() error {
			_, err := srv.ComputeScores(ctx, &pb.ScoreRequest{})
			return err
		}},
		{"Search", func() error {
			_, err := srv.Search(ctx, &pb.SearchRequest{})
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
