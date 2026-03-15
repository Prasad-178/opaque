package opaque

import "errors"

// Sentinel errors for programmatic error handling.
// Use [errors.Is] to check:
//
//	if errors.Is(err, opaque.ErrNotBuilt) { ... }
var (
	// ErrNotBuilt is returned when Search is called before Build.
	ErrNotBuilt = errors.New("opaque: index not built")

	// ErrAlreadyBuilt is returned when Add/AddBatch is called after Build.
	ErrAlreadyBuilt = errors.New("opaque: index already built; use Rebuild to add vectors")

	// ErrDimensionMismatch is returned when a vector has the wrong dimension.
	ErrDimensionMismatch = errors.New("opaque: dimension mismatch")

	// ErrNotFound is returned when a vector ID is not found.
	ErrNotFound = errors.New("opaque: vector not found")

	// ErrEmptyID is returned when an empty vector ID is provided.
	ErrEmptyID = errors.New("opaque: empty vector ID")

	// ErrNoVectors is returned when Build is called with no buffered vectors.
	ErrNoVectors = errors.New("opaque: no vectors added")

	// ErrNotReady is returned when an operation requires a built index but
	// the DB is not in the ready state (e.g., Save before Build).
	ErrNotReady = errors.New("opaque: database not ready")

	// ErrClosed is returned when an operation is attempted on a closed DB.
	ErrClosed = errors.New("opaque: database is closed")
)
