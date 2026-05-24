package lsh

import (
	"bytes"
	"math"
	"testing"
)

func TestEnterpriseHasher_HashDeterministic(t *testing.T) {
	planes := GenerateHyperplanes(7, 32, 16)
	h := NewEnterpriseHasher(planes)

	if h.NumBits() != 32 {
		t.Fatalf("NumBits=%d want 32", h.NumBits())
	}
	if h.Dimension() != 16 {
		t.Fatalf("Dimension=%d want 16", h.Dimension())
	}

	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = float64(i)*0.1 - 0.5
	}
	a := h.Hash(vec)
	b := h.Hash(vec)
	if !bytes.Equal(a, b) {
		t.Fatalf("hash not deterministic: %x vs %x", a, b)
	}
	want := (len(planes) + 7) / 8
	if len(a) != want {
		t.Fatalf("len=%d want %d", len(a), want)
	}
}

func TestEnterpriseHasher_HashMatchesIndex(t *testing.T) {
	// EnterpriseHasher must agree with Index.HashBytes when both use the same
	// planes — this is the cross-component invariant for client/server agreement.
	idx := NewIndex(Config{Dimension: 16, NumBits: 32, Seed: 99})
	planes := idx.GetPlanes()
	h := NewEnterpriseHasher(planes)

	vec := make([]float64, 16)
	for i := range vec {
		vec[i] = float64(i+1) * 0.137
	}
	if got, want := h.Hash(vec), idx.HashBytes(vec); !bytes.Equal(got, want) {
		t.Fatalf("hash mismatch:\n  enterprise: %x\n  index:      %x", got, want)
	}
}

func TestEnterpriseHasher_GetPlanesIsCopy(t *testing.T) {
	planes := GenerateHyperplanes(1, 8, 4)
	h := NewEnterpriseHasher(planes)

	copyOut := h.GetPlanes()
	if len(copyOut) != len(planes) {
		t.Fatalf("len=%d want %d", len(copyOut), len(planes))
	}
	// Mutate the returned copy; original must be unchanged.
	copyOut[0][0] = 999.0
	if h.GetPlanes()[0][0] == 999.0 {
		t.Fatal("GetPlanes returns mutable view; expected defensive copy")
	}
}

func TestEnterpriseHasher_HashToIndexInRange(t *testing.T) {
	planes := GenerateHyperplanes(11, 64, 8)
	h := NewEnterpriseHasher(planes)

	for trial := 0; trial < 50; trial++ {
		vec := make([]float64, 8)
		for i := range vec {
			vec[i] = float64(trial*8+i) * 0.013
		}
		idx := h.HashToIndex(vec, 128)
		if idx < 0 || idx >= 128 {
			t.Fatalf("bucket index out of range: %d", idx)
		}
	}
}

func TestEnterpriseHasher_EmptyPlanes(t *testing.T) {
	h := NewEnterpriseHasher(nil)
	if h.NumBits() != 0 || h.Dimension() != 0 {
		t.Fatalf("empty hasher: bits=%d dim=%d", h.NumBits(), h.Dimension())
	}
	// Hashing with no planes returns an empty hash and must not panic.
	got := h.Hash([]float64{1, 2, 3})
	if len(got) != 0 {
		t.Fatalf("expected zero-length hash, got %d bytes", len(got))
	}
}

func TestGenerateHyperplanes_NormalizedAndDeterministic(t *testing.T) {
	planes := GenerateHyperplanes(123, 16, 32)
	if len(planes) != 16 {
		t.Fatalf("len=%d want 16", len(planes))
	}
	for i, p := range planes {
		var norm float64
		for _, v := range p {
			norm += v * v
		}
		if math.Abs(math.Sqrt(norm)-1.0) > 1e-9 {
			t.Errorf("plane %d not unit-norm: %f", i, math.Sqrt(norm))
		}
	}

	again := GenerateHyperplanes(123, 16, 32)
	for i := range planes {
		for j := range planes[i] {
			if planes[i][j] != again[i][j] {
				t.Fatalf("not reproducible at [%d][%d]", i, j)
			}
		}
	}

	other := GenerateHyperplanes(456, 16, 32)
	same := true
	for i := range planes {
		for j := range planes[i] {
			if planes[i][j] != other[i][j] {
				same = false
				break
			}
		}
		if !same {
			break
		}
	}
	if same {
		t.Fatal("different seeds produced identical planes")
	}
}

func TestGenerateHyperplanesFromBytes_Deterministic(t *testing.T) {
	seed := []byte("opaque-test-seed-31415926535897932")
	a := GenerateHyperplanesFromBytes(seed, 8, 16)
	b := GenerateHyperplanesFromBytes(seed, 8, 16)
	if len(a) != 8 || len(b) != 8 {
		t.Fatalf("expected 8 planes, got %d/%d", len(a), len(b))
	}
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("not deterministic at [%d][%d]", i, j)
			}
		}
		var norm float64
		for _, v := range a[i] {
			norm += v * v
		}
		if math.Abs(math.Sqrt(norm)-1.0) > 1e-9 {
			t.Errorf("plane %d not normalized: %f", i, math.Sqrt(norm))
		}
	}
}

func TestGenerateHyperplanesFromBytes_DifferentSeed(t *testing.T) {
	a := GenerateHyperplanesFromBytes([]byte("seed-one"), 4, 8)
	b := GenerateHyperplanesFromBytes([]byte("seed-two"), 4, 8)
	diff := false
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				diff = true
				break
			}
		}
	}
	if !diff {
		t.Fatal("different seeds produced identical output")
	}
}
