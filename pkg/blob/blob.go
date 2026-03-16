// Package blob provides encrypted blob storage for Tier 2 data-private search.
// Blobs are encrypted vectors that can be stored on untrusted storage backends.
package blob

import (
	"encoding/json"
	"time"
)

// Blob represents an encrypted vector with metadata.
// The vector contents are encrypted; only the LSH bucket and ID are visible to storage.
type Blob struct {
	// ID is the unique identifier for this blob.
	ID string `json:"id"`

	// LSHBucket is the LSH hash bucket this vector belongs to.
	// This is visible to storage (enables bucket-based retrieval).
	LSHBucket string `json:"lsh_bucket"`

	// Ciphertext is the encrypted vector data.
	// Format: nonce (12 bytes) || ciphertext || tag (16 bytes)
	Ciphertext []byte `json:"ciphertext"`

	// MetadataCiphertext is optional encrypted metadata.
	// Can store additional info like document title, source, etc.
	MetadataCiphertext []byte `json:"metadata_ciphertext,omitempty"`

	// Dimension is the vector dimension (visible, needed for decryption).
	Dimension int `json:"dimension"`

	// CreatedAt is when this blob was created.
	CreatedAt time.Time `json:"created_at"`

	// Version for future schema changes.
	Version int `json:"version"`
}

// NewBlob creates a new encrypted blob.
func NewBlob(id, lshBucket string, ciphertext []byte, dimension int) *Blob {
	return &Blob{
		ID:         id,
		LSHBucket:  lshBucket,
		Ciphertext: ciphertext,
		Dimension:  dimension,
		CreatedAt:  time.Now(),
		Version:    1,
	}
}

// WithMetadata adds encrypted metadata to the blob.
func (b *Blob) WithMetadata(metadataCiphertext []byte) *Blob {
	b.MetadataCiphertext = metadataCiphertext
	return b
}

// Serialize converts the blob to JSON bytes for storage.
func (b *Blob) Serialize() ([]byte, error) {
	return json.Marshal(b)
}

// Deserialize parses JSON bytes into a Blob.
func Deserialize(data []byte) (*Blob, error) {
	var blob Blob
	if err := json.Unmarshal(data, &blob); err != nil {
		return nil, err
	}
	return &blob, nil
}

// BucketInfo contains information about a bucket.
type BucketInfo struct {
	// Bucket is the LSH bucket identifier.
	Bucket string `json:"bucket"`

	// Count is the number of blobs in this bucket.
	Count int `json:"count"`

	// TotalSize is the total size of all blobs in bytes.
	TotalSize int64 `json:"total_size"`
}

// StoreStats contains statistics about the blob store.
type StoreStats struct {
	// TotalBlobs is the total number of blobs.
	TotalBlobs int64 `json:"total_blobs"`

	// TotalBuckets is the number of unique buckets.
	TotalBuckets int64 `json:"total_buckets"`

	// TotalSize is the total storage size in bytes.
	TotalSize int64 `json:"total_size"`

	// AvgBlobsPerBucket is the average number of blobs per bucket.
	AvgBlobsPerBucket float64 `json:"avg_blobs_per_bucket"`
}
