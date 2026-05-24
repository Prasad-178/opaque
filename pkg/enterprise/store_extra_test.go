package enterprise

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func mustConfig(t *testing.T, id string) *Config {
	t.Helper()
	cfg, err := NewConfig(id, 16, 8)
	if err != nil {
		t.Fatalf("NewConfig(%s): %v", id, err)
	}
	return cfg
}

func TestMemoryStore_PutClonesInput(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	defer store.Close()

	cfg := mustConfig(t, "ent")
	if err := store.Put(ctx, cfg); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Mutate caller's struct after Put — stored value must be unaffected.
	cfg.AESKey[0] ^= 0xAA

	loaded, err := store.Get(ctx, "ent")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if loaded.AESKey[0] == cfg.AESKey[0] {
		t.Fatal("MemoryStore stored a reference to caller's slice")
	}
}

func TestMemoryStore_PutInvalidConfig(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()

	cfg := mustConfig(t, "ent")
	cfg.EnterpriseID = "" // make invalid
	if err := store.Put(ctx, cfg); err == nil {
		t.Fatal("expected validation error for empty enterprise ID")
	}
}

func TestMemoryStore_DeleteMissing(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	if err := store.Delete(ctx, "nope"); err != ErrEnterpriseNotFound {
		t.Fatalf("Delete missing: got %v want ErrEnterpriseNotFound", err)
	}
}

func TestMemoryStore_ListEmpty(t *testing.T) {
	store := NewMemoryStore()
	ids, err := store.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(ids) != 0 {
		t.Fatalf("len=%d want 0", len(ids))
	}
}

func TestMemoryStore_ConcurrentPutGet(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	const writers = 8
	var wg sync.WaitGroup
	wg.Add(writers)
	for i := 0; i < writers; i++ {
		i := i
		go func() {
			defer wg.Done()
			for j := 0; j < 32; j++ {
				cfg := mustConfig(t, "ent")
				cfg.EnterpriseID = "ent"
				_ = store.Put(ctx, cfg)
				_, _ = store.Get(ctx, "ent")
				_ = store.Exists(ctx, "ent")
				_, _ = store.List(ctx)
				_ = i
				_ = j
			}
		}()
	}
	wg.Wait()
}

func TestFileStore_PutInvalidConfig(t *testing.T) {
	store, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	cfg := mustConfig(t, "ent")
	cfg.EnterpriseID = ""
	if err := store.Put(context.Background(), cfg); err == nil {
		t.Fatal("expected validation error")
	}
}

func TestFileStore_DeleteMissing(t *testing.T) {
	store, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	if err := store.Delete(context.Background(), "missing"); err != ErrEnterpriseNotFound {
		t.Fatalf("Delete missing: got %v want ErrEnterpriseNotFound", err)
	}
}

func TestFileStore_GetMissing(t *testing.T) {
	store, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	if _, err := store.Get(context.Background(), "missing"); err != ErrEnterpriseNotFound {
		t.Fatalf("Get missing: got %v want ErrEnterpriseNotFound", err)
	}
}

func TestFileStore_ListIgnoresUnrelatedFiles(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	cfg := mustConfig(t, "ent")
	if err := store.Put(context.Background(), cfg); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Drop some noise into the dir.
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatalf("noise: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "subdir"), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	ids, err := store.List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(ids) != 1 || ids[0] != "ent" {
		t.Fatalf("ids=%v want [ent]", ids)
	}
}

func TestFileStore_PathTraversalSanitized(t *testing.T) {
	// EnterpriseID containing "/.." should not escape the base directory —
	// filepath.Base() trims directory parts.
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	cfg := mustConfig(t, "../escape")
	cfg.EnterpriseID = "../escape"
	if err := store.Put(context.Background(), cfg); err != nil {
		t.Fatalf("Put: %v", err)
	}
	// Confirm no file ended up outside dir.
	parent := filepath.Dir(dir)
	if _, err := os.Stat(filepath.Join(parent, "escape.gob")); err == nil {
		t.Fatal("path traversal succeeded — file written outside base dir")
	}
	// The sanitized file should be inside dir.
	if _, err := os.Stat(filepath.Join(dir, "escape.gob")); err != nil {
		t.Fatalf("expected sanitized file at base dir: %v", err)
	}
}

func TestFileStore_GetCorruptFile(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	// Drop a malformed gob file directly.
	bad := filepath.Join(dir, "ent.gob")
	if err := os.WriteFile(bad, []byte("not a gob"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := store.Get(context.Background(), "ent"); err == nil {
		t.Fatal("expected decode error on corrupt file")
	}
}

func TestFileStore_PutOverwrites(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	ctx := context.Background()
	cfg := mustConfig(t, "ent")
	if err := store.Put(ctx, cfg); err != nil {
		t.Fatalf("Put 1: %v", err)
	}
	// Second put with a different LSHSeed must overwrite atomically.
	cfg2 := mustConfig(t, "ent")
	if err := store.Put(ctx, cfg2); err != nil {
		t.Fatalf("Put 2: %v", err)
	}
	loaded, err := store.Get(ctx, "ent")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	// Loaded LSHSeed should match the second put — not the first.
	matchSecond := true
	matchFirst := true
	for i := range loaded.LSHSeed {
		if loaded.LSHSeed[i] != cfg2.LSHSeed[i] {
			matchSecond = false
		}
		if loaded.LSHSeed[i] != cfg.LSHSeed[i] {
			matchFirst = false
		}
	}
	if !matchSecond {
		t.Fatal("Put did not overwrite previous config")
	}
	if matchFirst && matchSecond {
		// Different seeds should be... different. If they collided we
		// can't tell — re-generate.
		t.Skip("LSH seeds collided across two NewConfig calls; cannot verify overwrite")
	}
}

func TestFileStore_NewFileStore_BadPath(t *testing.T) {
	// Trying to create a store inside a file (not a dir) should fail.
	tmp := t.TempDir()
	bogus := filepath.Join(tmp, "file")
	if err := os.WriteFile(bogus, []byte("x"), 0o600); err != nil {
		t.Fatalf("setup: %v", err)
	}
	if _, err := NewFileStore(filepath.Join(bogus, "child")); err == nil {
		t.Fatal("expected error creating store inside a file path")
	}
}

func TestFileStore_ExistsFalseForMissing(t *testing.T) {
	store, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	if store.Exists(context.Background(), "missing") {
		t.Fatal("Exists true for missing enterprise")
	}
}
