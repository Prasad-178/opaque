package store

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestMemoryStore_AddGetRoundtrip(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()

	ids := []string{"a", "b"}
	vecs := [][]float64{{1, 2, 3}, {4, 5, 6}}
	meta := []map[string]any{{"k": "v1"}, {"k": "v2"}}

	if err := s.Add(ctx, ids, vecs, meta); err != nil {
		t.Fatalf("Add: %v", err)
	}

	got, err := s.GetByIDs(ctx, ids)
	if err != nil {
		t.Fatalf("GetByIDs: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	for i := range vecs {
		for j := range vecs[i] {
			if got[i][j] != vecs[i][j] {
				t.Fatalf("got[%d][%d]=%v want %v", i, j, got[i][j], vecs[i][j])
			}
		}
	}

	one, err := s.GetByID(ctx, "a")
	if err != nil {
		t.Fatalf("GetByID(a): %v", err)
	}
	if len(one) != 3 {
		t.Fatalf("GetByID len=%d want 3", len(one))
	}

	count, err := s.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 2 {
		t.Fatalf("Count=%d want 2", count)
	}

	m, ok := s.GetMetadata("a")
	if !ok || m["k"] != "v1" {
		t.Fatalf("GetMetadata(a)=%v ok=%v", m, ok)
	}
	if _, ok := s.GetMetadata("missing"); ok {
		t.Fatal("GetMetadata(missing) returned ok=true")
	}

	all := s.GetAll()
	if len(all) != 2 {
		t.Fatalf("GetAll len=%d want 2", len(all))
	}
}

func TestMemoryStore_GetByID_NotFound(t *testing.T) {
	s := NewMemoryStore()
	_, err := s.GetByID(context.Background(), "nope")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err=%v want ErrNotFound", err)
	}
}

func TestMemoryStore_GetByIDs_NotFound(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	if err := s.Add(ctx, []string{"a"}, [][]float64{{1}}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := s.GetByIDs(ctx, []string{"a", "missing"}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err=%v want ErrNotFound", err)
	}
}

func TestMemoryStore_Add_LengthMismatch(t *testing.T) {
	s := NewMemoryStore()
	err := s.Add(context.Background(), []string{"a", "b"}, [][]float64{{1}}, nil)
	if err == nil {
		t.Fatal("expected error on ids/vectors length mismatch")
	}
}

func TestMemoryStore_Add_NilMetadataOK(t *testing.T) {
	s := NewMemoryStore()
	if err := s.Add(context.Background(), []string{"a"}, [][]float64{{1, 2}}, nil); err != nil {
		t.Fatalf("Add with nil meta: %v", err)
	}
	if _, ok := s.GetMetadata("a"); ok {
		t.Fatal("metadata recorded when none was supplied")
	}
}

func TestMemoryStore_Add_PartialMetadata(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	err := s.Add(ctx, []string{"a", "b"}, [][]float64{{1}, {2}}, []map[string]any{{"k": "v1"}})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if m, ok := s.GetMetadata("a"); !ok || m["k"] != "v1" {
		t.Fatalf("metadata(a)=%v ok=%v", m, ok)
	}
	if _, ok := s.GetMetadata("b"); ok {
		t.Fatal("metadata(b) recorded when none supplied for index 1")
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()
	_ = s.Add(ctx, []string{"a", "b"}, [][]float64{{1}, {2}}, nil)
	if err := s.Delete(ctx, []string{"a"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := s.GetByID(ctx, "a"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("after delete: err=%v want ErrNotFound", err)
	}
	count, _ := s.Count(ctx)
	if count != 1 {
		t.Fatalf("Count=%d want 1", count)
	}
}

func TestMemoryStore_Delete_Missing(t *testing.T) {
	// Deleting a missing id is a no-op, not an error.
	s := NewMemoryStore()
	if err := s.Delete(context.Background(), []string{"nope"}); err != nil {
		t.Fatalf("Delete missing: %v", err)
	}
}

func TestMemoryStore_Close(t *testing.T) {
	s := NewMemoryStore()
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestMemoryStore_ConcurrentAccess(t *testing.T) {
	s := NewMemoryStore()
	ctx := context.Background()

	const workers = 8
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		w := w
		go func() {
			defer wg.Done()
			for i := 0; i < 32; i++ {
				id := []string{string(rune('a' + w))}
				_ = s.Add(ctx, id, [][]float64{{float64(i)}}, nil)
				_, _ = s.GetByID(ctx, id[0])
				_, _ = s.Count(ctx)
			}
		}()
	}
	wg.Wait()
}

// Compile-time check: MemoryStore satisfies VectorStore.
var _ VectorStore = (*MemoryStore)(nil)
