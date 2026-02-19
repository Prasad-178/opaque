// Package grpcserver implements the gRPC service for privacy-preserving vector search.
//
// It delegates all business logic to internal/service.SearchService, translating
// between protobuf messages and service-layer types.
package grpcserver

import (
	"context"
	"strings"

	pb "github.com/opaque/opaque/go/api/proto"
	"github.com/opaque/opaque/go/internal/service"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Server implements the OpaqueSearchServer gRPC interface.
type Server struct {
	pb.UnimplementedOpaqueSearchServer
	svc *service.SearchService
}

// New creates a new gRPC server backed by the given SearchService.
func New(svc *service.SearchService) *Server {
	return &Server{svc: svc}
}

func (s *Server) RegisterKey(ctx context.Context, req *pb.RegisterKeyRequest) (*pb.RegisterKeyResponse, error) {
	if len(req.PublicKey) == 0 {
		return nil, status.Error(codes.InvalidArgument, "public_key is required")
	}

	sessionID, ttl, err := s.svc.RegisterKey(ctx, req.PublicKey, req.SessionTtlSeconds)
	if err != nil {
		return nil, mapError(err)
	}

	return &pb.RegisterKeyResponse{
		SessionId:         sessionID,
		SessionTtlSeconds: ttl,
	}, nil
}

func (s *Server) GetPlanes(ctx context.Context, req *pb.GetPlanesRequest) (*pb.GetPlanesResponse, error) {
	if req.SessionId == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id is required")
	}

	planes, numPlanes, dimension, err := s.svc.GetPlanes(ctx, req.SessionId)
	if err != nil {
		return nil, mapError(err)
	}

	return &pb.GetPlanesResponse{
		Planes:    planes,
		NumPlanes: numPlanes,
		Dimension: dimension,
	}, nil
}

func (s *Server) GetCandidates(ctx context.Context, req *pb.CandidateRequest) (*pb.CandidateResponse, error) {
	if req.SessionId == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id is required")
	}
	if len(req.LshHash) == 0 {
		return nil, status.Error(codes.InvalidArgument, "lsh_hash is required")
	}

	ids, distances, err := s.svc.GetCandidates(ctx, req.SessionId, req.LshHash, req.NumCandidates, req.MultiProbe, req.NumProbes)
	if err != nil {
		return nil, mapError(err)
	}

	return &pb.CandidateResponse{
		Ids:       ids,
		Distances: distances,
	}, nil
}

func (s *Server) ComputeScores(ctx context.Context, req *pb.ScoreRequest) (*pb.ScoreResponse, error) {
	if req.SessionId == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id is required")
	}
	if len(req.EncryptedQuery) == 0 {
		return nil, status.Error(codes.InvalidArgument, "encrypted_query is required")
	}

	encScores, ids, err := s.svc.ComputeScores(ctx, req.SessionId, req.EncryptedQuery, req.CandidateIds)
	if err != nil {
		return nil, mapError(err)
	}

	return &pb.ScoreResponse{
		EncryptedScores: encScores,
		Ids:             ids,
	}, nil
}

func (s *Server) ComputeScoresStream(req *pb.ScoreRequest, stream pb.OpaqueSearch_ComputeScoresStreamServer) error {
	if req.SessionId == "" {
		return status.Error(codes.InvalidArgument, "session_id is required")
	}
	if len(req.EncryptedQuery) == 0 {
		return status.Error(codes.InvalidArgument, "encrypted_query is required")
	}

	encScores, ids, err := s.svc.ComputeScores(stream.Context(), req.SessionId, req.EncryptedQuery, req.CandidateIds)
	if err != nil {
		return mapError(err)
	}

	for i := range ids {
		if err := stream.Send(&pb.ScoreChunk{
			Id:             ids[i],
			EncryptedScore: encScores[i],
		}); err != nil {
			return err
		}
	}

	return nil
}

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResponse, error) {
	if req.SessionId == "" {
		return nil, status.Error(codes.InvalidArgument, "session_id is required")
	}

	encScores, ids, distances, err := s.svc.Search(ctx, req.SessionId, req.LshHash, req.EncryptedQuery, req.MaxCandidates, req.TopN)
	if err != nil {
		return nil, mapError(err)
	}

	return &pb.SearchResponse{
		EncryptedScores: encScores,
		Ids:             ids,
		LshDistances:    distances,
	}, nil
}

func (s *Server) HealthCheck(ctx context.Context, _ *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
	healthy, msg, sessions, vectors := s.svc.HealthCheck(ctx)

	st := pb.HealthCheckResponse_NOT_SERVING
	if healthy {
		st = pb.HealthCheckResponse_SERVING
	}

	return &pb.HealthCheckResponse{
		Status:         st,
		Message:        msg,
		ActiveSessions: sessions,
		VectorCount:    vectors,
	}, nil
}

// mapError translates service-layer errors to appropriate gRPC status codes.
func mapError(err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	switch {
	case strings.Contains(msg, "invalid session"),
		strings.Contains(msg, "session not found"),
		strings.Contains(msg, "session expired"):
		return status.Errorf(codes.Unauthenticated, "%v", err)
	case strings.Contains(msg, "failed to deserialize"),
		strings.Contains(msg, "invalid"):
		return status.Errorf(codes.InvalidArgument, "%v", err)
	default:
		return status.Errorf(codes.Internal, "%v", err)
	}
}
