package blob

import (
	"context"
	"errors"
)

// Common errors for blob stores.
var (
	ErrBlobNotFound   = errors.New("blob not found")
	ErrBucketNotFound = errors.New("bucket not found")
	ErrBlobExists     = errors.New("blob already exists")
)

// Store is the interface for encrypted blob storage backends.
// Implementations can store blobs in memory, filesystem, S3, IPFS, blockchain, etc.
type Store interface {
	// Put stores a blob. Returns ErrBlobExists if ID already exists.
	Put(ctx context.Context, blob *Blob) error

	// PutBatch stores multiple blobs atomically.
	PutBatch(ctx context.Context, blobs []*Blob) error

	// Get retrieves a blob by ID. Returns ErrBlobNotFound if not found.
	Get(ctx context.Context, id string) (*Blob, error)

	// GetBatch retrieves multiple blobs by ID.
	// Returns nil for IDs that don't exist.
	GetBatch(ctx context.Context, ids []string) ([]*Blob, error)

	// GetBucket retrieves all blobs in a bucket.
	// Returns empty slice if bucket doesn't exist.
	GetBucket(ctx context.Context, bucket string) ([]*Blob, error)

	// GetBuckets retrieves blobs from multiple buckets.
	GetBuckets(ctx context.Context, buckets []string) ([]*Blob, error)

	// GetSuperBuckets retrieves all blobs from the specified super-bucket IDs.
	// Super-bucket keys are formatted as "XX" (2-digit zero-padded).
	// This is the primary method for fetching vectors after removing sub-buckets.
	GetSuperBuckets(ctx context.Context, superBucketIDs []int) ([]*Blob, error)

	// Delete removes a blob by ID. No error if blob doesn't exist.
	Delete(ctx context.Context, id string) error

	// DeleteBatch removes multiple blobs by ID.
	DeleteBatch(ctx context.Context, ids []string) error

	// ListBuckets returns all bucket identifiers.
	ListBuckets(ctx context.Context) ([]string, error)

	// BucketInfo returns information about a specific bucket.
	BucketInfo(ctx context.Context, bucket string) (*BucketInfo, error)

	// Stats returns overall store statistics.
	Stats(ctx context.Context) (*StoreStats, error)

	// Close closes the store connection.
	Close() error
}

// ReadOnlyStore is a read-only view of a blob store.
// Useful for search operations that don't need write access.
type ReadOnlyStore interface {
	Get(ctx context.Context, id string) (*Blob, error)
	GetBatch(ctx context.Context, ids []string) ([]*Blob, error)
	GetBucket(ctx context.Context, bucket string) ([]*Blob, error)
	GetBuckets(ctx context.Context, buckets []string) ([]*Blob, error)
	GetSuperBuckets(ctx context.Context, superBucketIDs []int) ([]*Blob, error)
	ListBuckets(ctx context.Context) ([]string, error)
	Stats(ctx context.Context) (*StoreStats, error)
}
