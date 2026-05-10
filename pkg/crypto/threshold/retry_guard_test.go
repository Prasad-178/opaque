package threshold

import (
	"errors"
	"sync"
	"testing"
)

func TestRetryGuard_FirstAdmit(t *testing.T) {
	g := NewRetryGuard()
	if err := g.Admit([]byte("instance-1"), []byte("ct"), []byte("pk")); err != nil {
		t.Fatalf("first admit must succeed, got %v", err)
	}
	if g.Size() != 1 {
		t.Errorf("expected size=1 after one admit, got %d", g.Size())
	}
}

func TestRetryGuard_RefusesDuplicate(t *testing.T) {
	g := NewRetryGuard()
	if err := g.Admit([]byte("instance-1"), []byte("ct"), []byte("pk")); err != nil {
		t.Fatalf("first admit failed: %v", err)
	}
	err := g.Admit([]byte("instance-1"), []byte("ct"), []byte("pk"))
	if !errors.Is(err, ErrShareAlreadyEmitted) {
		t.Fatalf("second admit must return ErrShareAlreadyEmitted, got %v", err)
	}
	if g.Size() != 1 {
		t.Errorf("size must not grow after refusal, got %d", g.Size())
	}
}

// Different instance IDs => different fingerprints, both admitted.
// This is exactly the "abort and restart with fresh instance_id" path.
func TestRetryGuard_DifferentInstanceAdmitted(t *testing.T) {
	g := NewRetryGuard()
	if err := g.Admit([]byte("instance-1"), []byte("ct"), []byte("pk")); err != nil {
		t.Fatalf("instance-1 admit failed: %v", err)
	}
	if err := g.Admit([]byte("instance-2"), []byte("ct"), []byte("pk")); err != nil {
		t.Fatalf("instance-2 admit must succeed (fresh instance), got %v", err)
	}
	if g.Size() != 2 {
		t.Errorf("expected size=2 across two instances, got %d", g.Size())
	}
}

// Different ciphertexts under the same instance must be admitted independently
// (the protocol instance can decrypt many ciphertexts in batched mode).
func TestRetryGuard_DifferentCiphertextAdmitted(t *testing.T) {
	g := NewRetryGuard()
	if err := g.Admit([]byte("instance-1"), []byte("ct-A"), []byte("pk")); err != nil {
		t.Fatalf("ct-A admit failed: %v", err)
	}
	if err := g.Admit([]byte("instance-1"), []byte("ct-B"), []byte("pk")); err != nil {
		t.Fatalf("ct-B admit failed: %v", err)
	}
	if g.Size() != 2 {
		t.Errorf("expected size=2, got %d", g.Size())
	}
}

// Length-prefixing must prevent (a||b) and (a'||b') from collapsing when the
// concatenations happen to coincide. Tests the writeLP discipline.
func TestRetryGuard_LengthPrefixingPreventsCollision(t *testing.T) {
	g := NewRetryGuard()
	// Naïve concat would make these equal: "a" + "bc" == "ab" + "c"
	// With length prefixing the fingerprints differ.
	if err := g.Admit([]byte("a"), []byte("bc"), []byte("pk")); err != nil {
		t.Fatalf("first admit failed: %v", err)
	}
	if err := g.Admit([]byte("ab"), []byte("c"), []byte("pk")); err != nil {
		t.Fatalf("second admit must succeed (different fingerprints), got %v", err)
	}
	if g.Size() != 2 {
		t.Errorf("expected size=2, got %d", g.Size())
	}
}

// Concurrent Admit calls for the same fingerprint must yield exactly one
// successful admit and the rest refused — no race condition allowing two
// shares to be emitted under the same fingerprint.
func TestRetryGuard_ConcurrentAdmitOnceWins(t *testing.T) {
	g := NewRetryGuard()
	const n = 64
	var wg sync.WaitGroup
	results := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = g.Admit([]byte("instance-1"), []byte("ct"), []byte("pk"))
		}(i)
	}
	wg.Wait()

	successes := 0
	for _, err := range results {
		if err == nil {
			successes++
		} else if !errors.Is(err, ErrShareAlreadyEmitted) {
			t.Errorf("unexpected error type: %v", err)
		}
	}
	if successes != 1 {
		t.Fatalf("exactly one concurrent admit must succeed, got %d", successes)
	}
}

func TestFingerprintHex_Deterministic(t *testing.T) {
	a := FingerprintHex([]byte("i"), []byte("c"), []byte("p"))
	b := FingerprintHex([]byte("i"), []byte("c"), []byte("p"))
	if a != b {
		t.Errorf("fingerprint must be deterministic across calls, got %s vs %s", a, b)
	}
	if len(a) != 64 {
		t.Errorf("expected 64-char hex (sha256), got %d chars", len(a))
	}
}

// TestDerivePerRoundCRS_DistinctRounds confirms the Phase-3 keygen-CRS
// derivation yields a different PRNG sequence for each (epochSeed, roundLabel)
// pair. CPK / Relin / Galois rounds within the same epoch must never collide;
// distinct epochs must produce entirely different CRS families.
func TestDerivePerRoundCRS_DistinctRounds(t *testing.T) {
	epochA := []byte("epoch-A-test-seed-32-bytes-padding")
	epochB := []byte("epoch-B-test-seed-32-bytes-padding")

	read := func(seed []byte, round string, n int) []byte {
		prng, err := derivePerRoundCRS(seed, round)
		if err != nil {
			t.Fatalf("derivePerRoundCRS(%s): %v", round, err)
		}
		buf := make([]byte, n)
		if _, err := prng.Read(buf); err != nil {
			t.Fatalf("prng.Read: %v", err)
		}
		return buf
	}

	// 1. Determinism: same input → same output.
	a1 := read(epochA, "cpk", 32)
	a2 := read(epochA, "cpk", 32)
	if !bytesEqual(a1, a2) {
		t.Errorf("derivePerRoundCRS must be deterministic for same inputs")
	}

	// 2. Distinct round labels → distinct PRNG output.
	cpk := read(epochA, "cpk", 32)
	relin := read(epochA, "relin", 32)
	if bytesEqual(cpk, relin) {
		t.Errorf("CPK and Relin rounds must produce distinct CRS streams")
	}

	// 3. Distinct epochs → distinct PRNG output for the same round.
	a := read(epochA, "cpk", 32)
	b := read(epochB, "cpk", 32)
	if bytesEqual(a, b) {
		t.Errorf("distinct epoch seeds must produce distinct CRS streams")
	}

	// 4. Galois sub-derivations: different element → different stream.
	g1 := read(epochA, "galois/3", 32)
	g2 := read(epochA, "galois/5", 32)
	if bytesEqual(g1, g2) {
		t.Errorf("distinct Galois element labels must produce distinct CRS streams")
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
