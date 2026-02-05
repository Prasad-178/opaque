package blob

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestFileStore_PutGet(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "blob_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	store, err := NewFileStore(tmpDir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Put
	blob := NewBlob("doc-1", "bucket-a", []byte("encrypted-data"), 128)
	if err := store.Put(ctx, blob); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(filepath.Join(tmpDir, "blobs", "doc-1.json")); err != nil {
		t.Error("blob file should exist")
	}

	// Get
	retrieved, err := store.Get(ctx, "doc-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != blob.ID {
		t.Errorf("ID mismatch: got %s, want %s", retrieved.ID, blob.ID)
	}
	if string(retrieved.Ciphertext) != string(blob.Ciphertext) {
		t.Error("ciphertext mismatch")
	}
}

func TestFileStore_Persistence(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "blob_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()

	// Create store and add data
	store1, _ := NewFileStore(tmpDir)
	store1.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store1.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))

	// Create new store instance (simulates restart)
	store2, _ := NewFileStore(tmpDir)

	// Verify data persisted
	stats, _ := store2.Stats(ctx)
	if stats.TotalBlobs != 2 {
		t.Errorf("expected 2 blobs after reload, got %d", stats.TotalBlobs)
	}

	// Verify bucket index persisted
	blobs, _ := store2.GetBucket(ctx, "bucket-a")
	if len(blobs) != 2 {
		t.Errorf("expected 2 blobs in bucket after reload, got %d", len(blobs))
	}
}

func TestFileStore_Delete(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "blob_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))

	// Delete
	store.Delete(ctx, "doc-1")

	// Verify deleted
	_, err = store.Get(ctx, "doc-1")
	if err != ErrBlobNotFound {
		t.Error("blob should be deleted")
	}

	// Verify file removed
	if _, err := os.Stat(filepath.Join(tmpDir, "blobs", "doc-1.json")); !os.IsNotExist(err) {
		t.Error("blob file should be deleted")
	}

	// Verify index updated
	blobs, _ := store.GetBucket(ctx, "bucket-a")
	if len(blobs) != 1 {
		t.Errorf("expected 1 blob in bucket after delete, got %d", len(blobs))
	}
}

func TestFileStore_GetBucket(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "blob_test_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-b", []byte("data3"), 128))

	blobs, err := store.GetBucket(ctx, "bucket-a")
	if err != nil {
		t.Fatalf("get bucket failed: %v", err)
	}

	if len(blobs) != 2 {
		t.Errorf("expected 2 blobs in bucket-a, got %d", len(blobs))
	}
}

func TestFileStore_ListBuckets(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "blob_test_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-b", []byte("data"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-c", []byte("data"), 128))

	buckets, err := store.ListBuckets(ctx)
	if err != nil {
		t.Fatalf("list buckets failed: %v", err)
	}

	if len(buckets) != 3 {
		t.Errorf("expected 3 buckets, got %d", len(buckets))
	}
}

func TestFileStore_IDSanitization(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "blob_test_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	// IDs with special characters should be sanitized
	blob := NewBlob("doc/../../../etc/passwd", "bucket", []byte("data"), 128)
	err := store.Put(ctx, blob)
	if err != nil {
		t.Fatalf("put failed: %v", err)
	}

	// Should be stored with sanitized name
	retrieved, err := store.Get(ctx, "doc/../../../etc/passwd")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != blob.ID {
		t.Error("ID should be preserved in blob data")
	}
}

func TestFileStore_DuplicateID(t *testing.T) {
	tmpDir, _ := os.MkdirTemp("", "blob_test_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	blob1 := NewBlob("doc-1", "bucket-a", []byte("data1"), 128)
	blob2 := NewBlob("doc-1", "bucket-b", []byte("data2"), 128)

	store.Put(ctx, blob1)
	err := store.Put(ctx, blob2)

	if err != ErrBlobExists {
		t.Error("should reject duplicate ID")
	}
}

func BenchmarkFileStore_Put(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "blob_bench_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blob := NewBlob(string(rune(i)), "bucket", make([]byte, 1024), 128)
		_ = store.Put(ctx, blob)
	}
}

func BenchmarkFileStore_Get(b *testing.B) {
	tmpDir, _ := os.MkdirTemp("", "blob_bench_*")
	defer os.RemoveAll(tmpDir)

	store, _ := NewFileStore(tmpDir)
	ctx := context.Background()

	// Pre-populate
	for i := 0; i < 100; i++ {
		blob := NewBlob(string(rune(i)), "bucket", make([]byte, 1024), 128)
		store.Put(ctx, blob)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = store.Get(ctx, string(rune(i%100)))
	}
}
