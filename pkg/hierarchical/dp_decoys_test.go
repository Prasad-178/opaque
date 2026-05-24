package hierarchical

import "testing"

func TestComputeDecoyCountForEpsilon_Zero(t *testing.T) {
	if got := ComputeDecoyCountForEpsilon(128, 8, 0); got != 0 {
		t.Errorf("ε=0: got %d, want 0", got)
	}
	if got := ComputeDecoyCountForEpsilon(128, 8, -1); got != 0 {
		t.Errorf("ε<0: got %d, want 0", got)
	}
}

func TestComputeDecoyCountForEpsilon_Monotonic(t *testing.T) {
	// Larger ε → smaller decoy count.
	prev := ComputeDecoyCountForEpsilon(128, 8, 0.5)
	for _, eps := range []float64{1.0, 1.5, 2.0, 2.5, 3.0} {
		got := ComputeDecoyCountForEpsilon(128, 8, eps)
		if got > prev {
			t.Errorf("ε=%.1f produced %d > previous %d (non-monotonic)", eps, got, prev)
		}
		prev = got
	}
}

func TestComputeDecoyCountForEpsilon_KnownValues(t *testing.T) {
	// Pool = 128 - 8 = 120.
	// ε=1: 120 * exp(-1) ≈ 44.1 → ceil 45
	// ε=2: 120 * exp(-2) ≈ 16.2 → ceil 17
	// ε=3: 120 * exp(-3) ≈ 5.97 → ceil 6
	cases := []struct {
		eps     float64
		wantMin int
		wantMax int
	}{
		{1.0, 44, 46},
		{2.0, 16, 18},
		{3.0, 5, 7},
	}
	for _, c := range cases {
		got := ComputeDecoyCountForEpsilon(128, 8, c.eps)
		if got < c.wantMin || got > c.wantMax {
			t.Errorf("ε=%.1f: got %d, want %d-%d", c.eps, got, c.wantMin, c.wantMax)
		}
	}
}

func TestComputeDecoyCountForEpsilon_PoolEdges(t *testing.T) {
	// pool = 0 → 0
	if got := ComputeDecoyCountForEpsilon(8, 8, 1.0); got != 0 {
		t.Errorf("pool=0: got %d, want 0", got)
	}
	// huge ε → at least 1 (because returns ≥1 when ε > 0 and pool > 0)
	if got := ComputeDecoyCountForEpsilon(128, 8, 100); got != 1 {
		t.Errorf("ε=100: got %d, want 1", got)
	}
}

func TestResolveDecoyCount_TargetEpsilonOverrides(t *testing.T) {
	cfg := Config{
		NumSuperBuckets: 128,
		TopSuperBuckets: 8,
		NumDecoys:       8,   // would be used if TargetEpsilon == 0
		TargetEpsilon:   2.0, // overrides → ~17
	}
	got := ResolveDecoyCount(cfg)
	if got < 16 || got > 18 {
		t.Errorf("got %d, want ε=2.0-derived ~17", got)
	}
}

func TestResolveDecoyCount_NoTargetEpsilon(t *testing.T) {
	cfg := Config{
		NumSuperBuckets: 128,
		TopSuperBuckets: 8,
		NumDecoys:       8,
	}
	if got := ResolveDecoyCount(cfg); got != 8 {
		t.Errorf("got %d, want 8 (NumDecoys passthrough)", got)
	}
}

func TestResolveDecoyCount_ClampsToPool(t *testing.T) {
	cfg := Config{
		NumSuperBuckets: 16,
		TopSuperBuckets: 8,
		NumDecoys:       100, // exceeds pool of 8
	}
	if got := ResolveDecoyCount(cfg); got != 8 {
		t.Errorf("got %d, want clamped to pool=8", got)
	}
}

func TestResolveDecoyCount_NegativeNumDecoys(t *testing.T) {
	cfg := Config{
		NumSuperBuckets: 128,
		TopSuperBuckets: 8,
		NumDecoys:       -5,
	}
	if got := ResolveDecoyCount(cfg); got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}
