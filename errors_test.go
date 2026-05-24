package opaque

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestSentinelErrors_Distinct verifies each sentinel is its own value —
// catches accidental aliasing or rename collisions during refactors.
func TestSentinelErrors_Distinct(t *testing.T) {
	sentinels := map[string]error{
		"ErrNotBuilt":          ErrNotBuilt,
		"ErrAlreadyBuilt":      ErrAlreadyBuilt,
		"ErrDimensionMismatch": ErrDimensionMismatch,
		"ErrNotFound":          ErrNotFound,
		"ErrEmptyID":           ErrEmptyID,
		"ErrNoVectors":         ErrNoVectors,
		"ErrNotReady":          ErrNotReady,
		"ErrClosed":            ErrClosed,
	}

	for name, e := range sentinels {
		if e == nil {
			t.Fatalf("%s is nil", name)
		}
		if !strings.HasPrefix(e.Error(), "opaque:") {
			t.Errorf("%s message %q must start with 'opaque:' namespace", name, e.Error())
		}
	}

	// Pairwise distinctness — two sentinels must never satisfy errors.Is
	// against each other.
	for nameA, a := range sentinels {
		for nameB, b := range sentinels {
			if nameA == nameB {
				continue
			}
			if errors.Is(a, b) {
				t.Errorf("%s and %s are not distinct (errors.Is matched)", nameA, nameB)
			}
		}
	}
}

// TestSentinelErrors_WrappingPreservesIs verifies that wrapping a sentinel
// with fmt.Errorf("...%w") still lets callers use errors.Is to match. This
// is how the production code returns these errors (e.g. ErrDimensionMismatch
// is wrapped with the actual dim numbers).
func TestSentinelErrors_WrappingPreservesIs(t *testing.T) {
	for _, s := range []error{
		ErrNotBuilt,
		ErrAlreadyBuilt,
		ErrDimensionMismatch,
		ErrNotFound,
		ErrEmptyID,
		ErrNoVectors,
		ErrNotReady,
		ErrClosed,
	} {
		wrapped := fmt.Errorf("context: %w", s)
		if !errors.Is(wrapped, s) {
			t.Errorf("errors.Is failed for wrapped %q", s.Error())
		}
	}
}
