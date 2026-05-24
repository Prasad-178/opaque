package cluster

import (
	"math"
	"testing"
)

func TestNormalizeVector_UnitNorm(t *testing.T) {
	v := []float32{3, 4} // norm=5
	got := NormalizeVector(v)
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	var norm float64
	for _, x := range got {
		norm += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(norm)-1.0) > 1e-6 {
		t.Fatalf("norm=%v want 1", math.Sqrt(norm))
	}
	// Direction must be preserved.
	if math.Abs(float64(got[0])-3.0/5.0) > 1e-6 || math.Abs(float64(got[1])-4.0/5.0) > 1e-6 {
		t.Fatalf("direction wrong: %v", got)
	}
}

func TestNormalizeVector_ZeroInputReturnsZeros(t *testing.T) {
	v := []float32{0, 0, 0}
	got := NormalizeVector(v)
	for _, x := range got {
		if x != 0 {
			t.Fatalf("zero vector should normalize to zeros; got %v", got)
		}
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	got := CosineSimilarity(a, b)
	if math.Abs(got) > 1e-6 {
		t.Fatalf("orthogonal sim=%v want 0", got)
	}
}

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	got := CosineSimilarity(a, a)
	if math.Abs(got-1.0) > 1e-6 {
		t.Fatalf("identical sim=%v want 1", got)
	}
}

func TestCosineSimilarity_Antiparallel(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{-1, -2}
	got := CosineSimilarity(a, b)
	if math.Abs(got+1.0) > 1e-6 {
		t.Fatalf("antiparallel sim=%v want -1", got)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	if got := CosineSimilarity([]float32{0, 0}, []float32{1, 2}); got != 0 {
		t.Fatalf("zero vector should yield 0, got %v", got)
	}
	if got := CosineSimilarity([]float32{1, 2}, []float32{0, 0}); got != 0 {
		t.Fatalf("zero vector should yield 0, got %v", got)
	}
}

func TestAsFloat32_Float64_Roundtrip(t *testing.T) {
	original := [][]float64{
		{0.5, -1.5, 2.25},
		{1, 2, 3},
	}
	f32 := AsFloat32(original)
	back := AsFloat64(f32)
	if len(back) != len(original) {
		t.Fatalf("len=%d want %d", len(back), len(original))
	}
	for i := range original {
		for j := range original[i] {
			if back[i][j] != original[i][j] {
				t.Fatalf("vec[%d][%d] = %v want %v", i, j, back[i][j], original[i][j])
			}
		}
	}
}

func TestAsFloat32One_Float64One_Roundtrip(t *testing.T) {
	v := []float64{0.1, 0.2, 0.3, 0.5}
	got := AsFloat64One(AsFloat32One(v))
	if len(got) != len(v) {
		t.Fatalf("len=%d want %d", len(got), len(v))
	}
	for i := range v {
		if math.Abs(got[i]-v[i]) > 1e-6 {
			t.Fatalf("vec[%d] = %v want %v", i, got[i], v[i])
		}
	}
}

func TestAsFloat32_PreservesShape(t *testing.T) {
	in := make([][]float64, 5)
	for i := range in {
		in[i] = make([]float64, 7)
	}
	out := AsFloat32(in)
	if len(out) != 5 {
		t.Fatalf("len=%d", len(out))
	}
	for i := range out {
		if len(out[i]) != 7 {
			t.Fatalf("len[%d]=%d", i, len(out[i]))
		}
	}
}
