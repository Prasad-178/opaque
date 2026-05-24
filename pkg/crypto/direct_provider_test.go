package crypto

import (
	"math"
	"testing"
)

// TestDirectHEProvider_Surface verifies the simple accessor wrappers and
// Acquire/Release contract on DirectHEProvider. The deeper Encrypt /
// DotProduct correctness is covered by the shared testProviderEncryptDecrypt /
// testProviderBatchDotProduct in provider_test.go.
func TestDirectHEProvider_Surface(t *testing.T) {
	p, err := NewDirectHEProvider(2)
	if err != nil {
		t.Fatalf("NewDirectHEProvider: %v", err)
	}
	defer p.Close()

	if p.Size() != 2 {
		t.Fatalf("Size=%d want 2", p.Size())
	}
	if p.GetParams().LogN() == 0 {
		t.Fatal("GetParams returned zero-value params")
	}
	if p.GetEncoder() == nil {
		t.Fatal("GetEncoder nil")
	}

	e1 := p.Acquire()
	e2 := p.Acquire()
	if e1 == nil || e2 == nil {
		t.Fatal("Acquire returned nil")
	}
	p.Release(e1)
	p.Release(e2)

	// One more encrypt/decrypt to confirm Close doesn't leave the pool half-
	// usable from the smoke tests.
	vec := []float64{0.25, 0.5, 0.75}
	ct, err := p.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}
	scalar, err := p.DecryptScalar(ct)
	if err != nil {
		t.Fatalf("DecryptScalar: %v", err)
	}
	if math.Abs(scalar-vec[0]) > 0.01 {
		t.Fatalf("scalar=%v want=%v", scalar, vec[0])
	}
}

// TestDirectHEProvider_ImplementsHEProvider is a compile-time check that
// DirectHEProvider satisfies the HEProvider interface.
func TestDirectHEProvider_ImplementsHEProvider(t *testing.T) {
	var _ HEProvider = (*DirectHEProvider)(nil)
}
