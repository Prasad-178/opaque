package blob

import (
	"context"
	"testing"
)

func TestBlob_Serialize(t *testing.T) {
	blob := NewBlob("doc-123", "bucket-abc", []byte("encrypted-data"), 128)
	blob.WithMetadata([]byte("encrypted-metadata"))

	// Serialize
	data, err := blob.Serialize()
	if err != nil {
		t.Fatalf("serialization failed: %v", err)
	}

	// Deserialize
	recovered, err := Deserialize(data)
	if err != nil {
		t.Fatalf("deserialization failed: %v", err)
	}

	// Verify fields
	if recovered.ID != blob.ID {
		t.Errorf("ID mismatch: got %s, want %s", recovered.ID, blob.ID)
	}
	if recovered.LSHBucket != blob.LSHBucket {
		t.Errorf("LSHBucket mismatch: got %s, want %s", recovered.LSHBucket, blob.LSHBucket)
	}
	if string(recovered.Ciphertext) != string(blob.Ciphertext) {
		t.Error("Ciphertext mismatch")
	}
	if string(recovered.MetadataCiphertext) != string(blob.MetadataCiphertext) {
		t.Error("MetadataCiphertext mismatch")
	}
	if recovered.Dimension != blob.Dimension {
		t.Errorf("Dimension mismatch: got %d, want %d", recovered.Dimension, blob.Dimension)
	}
}

func TestMemoryStore_PutGet(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	blob := NewBlob("doc-1", "bucket-a", []byte("data"), 128)

	// Put
	if err := store.Put(ctx, blob); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	// Get
	retrieved, err := store.Get(ctx, "doc-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}

	if retrieved.ID != blob.ID {
		t.Errorf("ID mismatch")
	}

	// Get non-existent
	_, err = store.Get(ctx, "non-existent")
	if err != ErrBlobNotFound {
		t.Error("should return ErrBlobNotFound for non-existent blob")
	}
}

func TestMemoryStore_PutDuplicate(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	blob := NewBlob("doc-1", "bucket-a", []byte("data"), 128)

	// First put should succeed
	if err := store.Put(ctx, blob); err != nil {
		t.Fatalf("first put failed: %v", err)
	}

	// Second put should fail
	if err := store.Put(ctx, blob); err != ErrBlobExists {
		t.Error("should return ErrBlobExists for duplicate")
	}
}

func TestMemoryStore_PutBatch(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	blobs := []*Blob{
		NewBlob("doc-1", "bucket-a", []byte("data1"), 128),
		NewBlob("doc-2", "bucket-a", []byte("data2"), 128),
		NewBlob("doc-3", "bucket-b", []byte("data3"), 128),
	}

	if err := store.PutBatch(ctx, blobs); err != nil {
		t.Fatalf("batch put failed: %v", err)
	}

	// Verify all stored
	stats, _ := store.Stats(ctx)
	if stats.TotalBlobs != 3 {
		t.Errorf("expected 3 blobs, got %d", stats.TotalBlobs)
	}
	if stats.TotalBuckets != 2 {
		t.Errorf("expected 2 buckets, got %d", stats.TotalBuckets)
	}
}

func TestMemoryStore_GetBucket(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	// Add blobs to different buckets
	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-b", []byte("data3"), 128))

	// Get bucket-a
	blobs, err := store.GetBucket(ctx, "bucket-a")
	if err != nil {
		t.Fatalf("get bucket failed: %v", err)
	}

	if len(blobs) != 2 {
		t.Errorf("expected 2 blobs in bucket-a, got %d", len(blobs))
	}

	// Get non-existent bucket (should return empty, not error)
	blobs, err = store.GetBucket(ctx, "non-existent")
	if err != nil {
		t.Fatalf("get non-existent bucket failed: %v", err)
	}
	if len(blobs) != 0 {
		t.Error("expected empty slice for non-existent bucket")
	}
}

func TestMemoryStore_GetBuckets(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-b", []byte("data2"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-c", []byte("data3"), 128))

	// Get multiple buckets
	blobs, err := store.GetBuckets(ctx, []string{"bucket-a", "bucket-c"})
	if err != nil {
		t.Fatalf("get buckets failed: %v", err)
	}

	if len(blobs) != 2 {
		t.Errorf("expected 2 blobs, got %d", len(blobs))
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))

	// Delete one
	if err := store.Delete(ctx, "doc-1"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	// Verify deleted
	_, err := store.Get(ctx, "doc-1")
	if err != ErrBlobNotFound {
		t.Error("blob should be deleted")
	}

	// Verify other still exists
	_, err = store.Get(ctx, "doc-2")
	if err != nil {
		t.Error("doc-2 should still exist")
	}

	// Delete non-existent should not error
	if err := store.Delete(ctx, "non-existent"); err != nil {
		t.Error("delete non-existent should not error")
	}
}

func TestMemoryStore_ListBuckets(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-b", []byte("data2"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-c", []byte("data3"), 128))

	buckets, err := store.ListBuckets(ctx)
	if err != nil {
		t.Fatalf("list buckets failed: %v", err)
	}

	if len(buckets) != 3 {
		t.Errorf("expected 3 buckets, got %d", len(buckets))
	}
}

func TestMemoryStore_BucketInfo(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data22"), 128))

	info, err := store.BucketInfo(ctx, "bucket-a")
	if err != nil {
		t.Fatalf("bucket info failed: %v", err)
	}

	if info.Count != 2 {
		t.Errorf("expected count 2, got %d", info.Count)
	}
	if info.TotalSize != 11 { // "data1" + "data22" = 5 + 6 = 11
		t.Errorf("expected total size 11, got %d", info.TotalSize)
	}

	// Non-existent bucket
	_, err = store.BucketInfo(ctx, "non-existent")
	if err != ErrBucketNotFound {
		t.Error("should return ErrBucketNotFound")
	}
}

func TestMemoryStore_Stats(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-a", []byte("data2"), 128))
	store.Put(ctx, NewBlob("doc-3", "bucket-b", []byte("data3"), 128))

	stats, err := store.Stats(ctx)
	if err != nil {
		t.Fatalf("stats failed: %v", err)
	}

	if stats.TotalBlobs != 3 {
		t.Errorf("expected 3 blobs, got %d", stats.TotalBlobs)
	}
	if stats.TotalBuckets != 2 {
		t.Errorf("expected 2 buckets, got %d", stats.TotalBuckets)
	}
	if stats.AvgBlobsPerBucket != 1.5 {
		t.Errorf("expected avg 1.5, got %f", stats.AvgBlobsPerBucket)
	}
}

func TestMemoryStore_Clear(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Put(ctx, NewBlob("doc-1", "bucket-a", []byte("data1"), 128))
	store.Put(ctx, NewBlob("doc-2", "bucket-b", []byte("data2"), 128))

	store.Clear()

	stats, _ := store.Stats(ctx)
	if stats.TotalBlobs != 0 {
		t.Error("store should be empty after clear")
	}
}

func BenchmarkMemoryStore_Put(b *testing.B) {
	store := NewMemoryStore()
	ctx := context.Background()
	blob := NewBlob("", "bucket", make([]byte, 1024), 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blob.ID = string(rune(i))
		_ = store.Put(ctx, blob)
	}
}

func BenchmarkMemoryStore_GetBucket(b *testing.B) {
	store := NewMemoryStore()
	ctx := context.Background()

	// Pre-populate with 1000 blobs in one bucket
	for i := 0; i < 1000; i++ {
		blob := NewBlob(string(rune(i)), "bucket-a", make([]byte, 1024), 128)
		store.Put(ctx, blob)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = store.GetBucket(ctx, "bucket-a")
	}
}
