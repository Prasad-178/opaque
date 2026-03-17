package service

import (
	"context"
	"testing"
	"time"

	"github.com/Prasad-178/opaque/internal/store"
)

func newTestSearchService(t *testing.T, dim int) *SearchService {
	t.Helper()
	svc, err := NewSearchService(Config{
		LSHNumBits:          16,
		LSHDimension:        dim,
		LSHSeed:             42,
		MaxSessionTTL:       time.Minute,
		MaxConcurrentScores: 2,
	}, store.NewMemoryStore())
	if err != nil {
		t.Fatalf("NewSearchService: %v", err)
	}
	return svc
}

func TestSearchService_RegisterAndGetPlanes(t *testing.T) {
	svc := newTestSearchService(t, 8)
	ctx := context.Background()

	sessionID, _, err := svc.RegisterKey(ctx, []byte("pk"), 60)
	if err != nil {
		t.Fatalf("RegisterKey: %v", err)
	}

	planes, numPlanes, dim, err := svc.GetPlanes(ctx, sessionID)
	if err != nil {
		t.Fatalf("GetPlanes: %v", err)
	}
	if numPlanes <= 0 {
		t.Fatalf("numPlanes = %d, want > 0", numPlanes)
	}
	if dim != 8 {
		t.Fatalf("dim = %d, want 8", dim)
	}
	if len(planes) != int(numPlanes*dim) {
		t.Fatalf("planes len = %d, want %d", len(planes), int(numPlanes*dim))
	}
}

func TestSearchService_AddVectorsAndCandidates(t *testing.T) {
	svc := newTestSearchService(t, 4)
	ctx := context.Background()

	ids := []string{"v1", "v2", "v3"}
	vectors := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	if err := svc.AddVectors(ctx, ids, vectors, nil); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	sessionID, _, err := svc.RegisterKey(ctx, []byte("pk"), 60)
	if err != nil {
		t.Fatalf("RegisterKey: %v", err)
	}

	cands, distances, err := svc.GetCandidates(ctx, sessionID, []byte{0x01, 0x02}, 5, false, 0)
	if err != nil {
		t.Fatalf("GetCandidates: %v", err)
	}
	if len(cands) == 0 {
		t.Fatal("expected at least one candidate")
	}
	if len(cands) != len(distances) {
		t.Fatalf("candidate/distance mismatch: %d vs %d", len(cands), len(distances))
	}
}

func TestSearchService_ComputeScoresRejectsInvalidCiphertext(t *testing.T) {
	svc := newTestSearchService(t, 4)
	ctx := context.Background()

	if err := svc.AddVectors(ctx, []string{"v1"}, [][]float64{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	sessionID, _, err := svc.RegisterKey(ctx, []byte("pk"), 60)
	if err != nil {
		t.Fatalf("RegisterKey: %v", err)
	}

	_, _, err = svc.ComputeScores(ctx, sessionID, []byte("bad"), []string{"v1"})
	if err == nil {
		t.Fatal("expected error for invalid ciphertext")
	}
}

func TestSearchService_RemoveVectors(t *testing.T) {
	svc := newTestSearchService(t, 4)
	ctx := context.Background()

	if err := svc.AddVectors(ctx, []string{"v1"}, [][]float64{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatalf("AddVectors: %v", err)
	}

	if err := svc.RemoveVectors(ctx, []string{"v1"}); err != nil {
		t.Fatalf("RemoveVectors: %v", err)
	}

	count, err := svc.GetVectorCount(ctx)
	if err != nil {
		t.Fatalf("GetVectorCount: %v", err)
	}
	if count != 0 {
		t.Fatalf("count = %d, want 0", count)
	}
}
