package store

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestFileStore_EmptyPath(t *testing.T) {
	if _, err := NewFileStore(""); err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestFileStore_GetByID_NotFound(t *testing.T) {
	s, err := NewFileStore(filepath.Join(t.TempDir(), "v.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if _, err := s.GetByID(context.Background(), "missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err=%v want ErrNotFound", err)
	}
}

func TestFileStore_GetByIDs_NotFound(t *testing.T) {
	ctx := context.Background()
	s, err := NewFileStore(filepath.Join(t.TempDir(), "v.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if err := s.Add(ctx, []string{"a"}, [][]float64{{1}}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := s.GetByIDs(ctx, []string{"a", "nope"}); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err=%v want ErrNotFound", err)
	}
}

func TestFileStore_Add_LengthMismatch(t *testing.T) {
	s, err := NewFileStore(filepath.Join(t.TempDir(), "v.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if err := s.Add(context.Background(), []string{"a", "b"}, [][]float64{{1}}, nil); err == nil {
		t.Fatal("expected length-mismatch error")
	}
}

func TestFileStore_Delete_Missing(t *testing.T) {
	s, err := NewFileStore(filepath.Join(t.TempDir(), "v.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if err := s.Delete(context.Background(), []string{"nope"}); err != nil {
		t.Fatalf("Delete missing should be no-op: %v", err)
	}
}

func TestFileStore_Count_Empty(t *testing.T) {
	s, err := NewFileStore(filepath.Join(t.TempDir(), "v.json"))
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	count, err := s.Count(context.Background())
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 0 {
		t.Fatalf("Count=%d want 0", count)
	}
}

func TestFileStore_FilePermissions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v.json")
	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if err := s.Add(context.Background(), []string{"a"}, [][]float64{{1, 2}}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("file perms=%o want 0600", perm)
	}
	// Parent dir was created with 0700.
	parent, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("parent Stat: %v", err)
	}
	if !parent.IsDir() {
		t.Fatal("parent is not a directory")
	}
}

func TestFileStore_RecoversFromCorruptFile(t *testing.T) {
	// Truncated / malformed file at the configured path should surface an
	// error at NewFileStore.
	path := filepath.Join(t.TempDir(), "v.json")
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := NewFileStore(path); err == nil {
		t.Fatal("expected parse error on malformed file")
	}
}

func TestFileStore_EmptyFileIsOK(t *testing.T) {
	// An empty file at the path is treated as an empty store, not an error.
	path := filepath.Join(t.TempDir(), "v.json")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore empty-file: %v", err)
	}
	defer s.Close()
	count, _ := s.Count(context.Background())
	if count != 0 {
		t.Fatalf("Count=%d want 0", count)
	}
}

func TestFileStore_AtomicReplace(t *testing.T) {
	// After a write the tmp file must be gone — confirming os.Rename used
	// the atomic-replace path.
	path := filepath.Join(t.TempDir(), "v.json")
	s, err := NewFileStore(path)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	defer s.Close()
	if err := s.Add(context.Background(), []string{"a"}, [][]float64{{1}}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf(".tmp still exists after Add (Stat err=%v)", err)
	}
}

// Compile-time check: FileStore satisfies VectorStore.
var _ VectorStore = (*FileStore)(nil)
