// Package grpcserver implements the gRPC service for privacy-preserving vector search.
//
// The server wraps an opaque.DB instance, exposing health checks and vector count
// via gRPC. The legacy LSH-specific RPCs (RegisterKey, GetPlanes, GetCandidates,
// ComputeScores) return Unimplemented — the k-means SDK handles search client-side.
package grpcserver

import (
	"context"

	pb "github.com/Prasad-178/opaque/api/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// DB is the interface that grpcserver requires from an opaque.DB.
// This avoids a direct import cycle and keeps the package testable.
type DB interface {
	IsReady() bool
	Size() int
}

// Server implements the OpaqueSearchServer gRPC interface.
type Server struct {
	pb.UnimplementedOpaqueSearchServer
	db DB
}

// New creates a new gRPC server backed by the given DB.
func New(db DB) *Server {
	return &Server{db: db}
}

// HealthCheck reports service health based on the underlying DB state.
func (s *Server) HealthCheck(ctx context.Context, _ *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	st := pb.HealthCheckResponse_NOT_SERVING
	msg := "index not built"
	if s.db.IsReady() {
		st = pb.HealthCheckResponse_SERVING
		msg = "healthy"
	}

	return &pb.HealthCheckResponse{
		Status:      st,
		Message:     msg,
		VectorCount: int64(s.db.Size()),
	}, nil
}

// --- Legacy LSH RPCs (deprecated) ---
// These RPCs were part of the old LSH-based search protocol.
// The k-means SDK now handles the full search flow client-side.

func (s *Server) RegisterKey(context.Context, *pb.RegisterKeyRequest) (*pb.RegisterKeyResponse, error) {
	return nil, status.Error(codes.Unimplemented, "RegisterKey is deprecated: use the opaque SDK for client-side search")
}

func (s *Server) GetPlanes(context.Context, *pb.GetPlanesRequest) (*pb.GetPlanesResponse, error) {
	return nil, status.Error(codes.Unimplemented, "GetPlanes is deprecated: k-means replaced LSH")
}

func (s *Server) GetCandidates(context.Context, *pb.CandidateRequest) (*pb.CandidateResponse, error) {
	return nil, status.Error(codes.Unimplemented, "GetCandidates is deprecated: k-means replaced LSH")
}

func (s *Server) ComputeScores(context.Context, *pb.ScoreRequest) (*pb.ScoreResponse, error) {
	return nil, status.Error(codes.Unimplemented, "ComputeScores is deprecated: search is now client-side")
}

func (s *Server) ComputeScoresStream(_ *pb.ScoreRequest, _ pb.OpaqueSearch_ComputeScoresStreamServer) error {
	return status.Error(codes.Unimplemented, "ComputeScoresStream is deprecated: search is now client-side")
}

func (s *Server) Search(context.Context, *pb.SearchRequest) (*pb.SearchResponse, error) {
	return nil, status.Error(codes.Unimplemented, "Search is deprecated: use the opaque SDK directly")
}
