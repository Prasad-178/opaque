package store

import (
	"context"
	"path/filepath"
	"testing"
)

func TestFileStore_Persistence(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "vectors.json")

	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}

	ids := []string{"a", "b"}
	vectors := [][]float64{{1, 2}, {3, 4}}
	meta := []map[string]any{{"type": "x"}, {"type": "y"}}
	if err := s.Add(ctx, ids, vectors, meta); err != nil {
		t.Fatalf("Add: %v", err)
	}

	s2, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore(reopen): %v", err)
	}

	got, err := s2.GetByID(ctx, "a")
	if err != nil {
		t.Fatalf("GetByID: %v", err)
	}
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("unexpected vector: %v", got)
	}

	count, err := s2.Count(ctx)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 2 {
		t.Fatalf("count = %d, want 2", count)
	}
}

func TestFileStore_Delete(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "vectors.json")

	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}

	if err := s.Add(ctx, []string{"a"}, [][]float64{{1, 2}}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if err := s.Delete(ctx, []string{"a"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, err := s.GetByID(ctx, "a"); err == nil {
		t.Fatal("expected ErrNotFound after delete")
	}
}
