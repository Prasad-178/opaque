package blob

import (
	"context"
	"testing"
)

func TestGetSuperBuckets(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()

	// Add blobs with new format (super-bucket only)
	b1 := NewBlob("vec1", "05", []byte("data1"), 128)
	b2 := NewBlob("vec2", "05", []byte("data2"), 128)
	b3 := NewBlob("vec3", "12", []byte("data3"), 128)

	store.Put(ctx, b1)
	store.Put(ctx, b2)
	store.Put(ctx, b3)

	// Test GetSuperBuckets
	blobs, err := store.GetSuperBuckets(ctx, []int{5, 12})
	if err != nil {
		t.Fatalf("GetSuperBuckets error: %v", err)
	}

	if len(blobs) != 3 {
		t.Errorf("Expected 3 blobs, got %d", len(blobs))
	}

	// Verify all expected IDs are present
	ids := make(map[string]bool)
	for _, b := range blobs {
		ids[b.ID] = true
	}
	for _, expected := range []string{"vec1", "vec2", "vec3"} {
		if !ids[expected] {
			t.Errorf("Missing blob ID: %s", expected)
		}
	}

	t.Logf("GetSuperBuckets([5, 12]) returned %d blobs", len(blobs))
}

func TestGetSuperBucketsLegacyFormat(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()

	// Add blobs with old format (super_sub)
	b1 := NewBlob("vec1", "05_03", []byte("data1"), 128)
	b2 := NewBlob("vec2", "05_07", []byte("data2"), 128)
	b3 := NewBlob("vec3", "12_00", []byte("data3"), 128)

	store.Put(ctx, b1)
	store.Put(ctx, b2)
	store.Put(ctx, b3)

	// Test GetSuperBuckets should still work
	blobs, err := store.GetSuperBuckets(ctx, []int{5})
	if err != nil {
		t.Fatalf("GetSuperBuckets error: %v", err)
	}

	if len(blobs) != 2 {
		t.Errorf("Expected 2 blobs from super-bucket 5, got %d", len(blobs))
	}

	t.Logf("GetSuperBuckets([5]) with legacy format returned %d blobs", len(blobs))
}

func TestGetSuperBucketsMixed(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()

	// Mix of new and old format
	b1 := NewBlob("vec1", "05", []byte("data1"), 128)    // new format
	b2 := NewBlob("vec2", "05_03", []byte("data2"), 128) // old format
	b3 := NewBlob("vec3", "05_07", []byte("data3"), 128) // old format

	store.Put(ctx, b1)
	store.Put(ctx, b2)
	store.Put(ctx, b3)

	blobs, err := store.GetSuperBuckets(ctx, []int{5})
	if err != nil {
		t.Fatalf("GetSuperBuckets error: %v", err)
	}

	if len(blobs) != 3 {
		t.Errorf("Expected 3 blobs from super-bucket 5, got %d", len(blobs))
	}

	t.Logf("GetSuperBuckets([5]) with mixed format returned %d blobs", len(blobs))
}
