package hierarchical

import (
	"sort"
	"testing"
)

func TestGenerateBlobIDPermutation_IsBijection(t *testing.T) {
	cases := []int{1, 2, 16, 64, 128, 256, 1024}
	for _, n := range cases {
		π, err := generateBlobIDPermutation(n)
		if err != nil {
			t.Fatalf("n=%d: unexpected error: %v", n, err)
		}
		if len(π) != n {
			t.Fatalf("n=%d: got len(π)=%d, want %d", n, len(π), n)
		}

		// Verify every element of [0, n) appears exactly once.
		sorted := make([]int, n)
		copy(sorted, π)
		sort.Ints(sorted)
		for i := 0; i < n; i++ {
			if sorted[i] != i {
				t.Fatalf("n=%d: not a bijection (sorted[%d]=%d, π=%v)", n, i, sorted[i], π)
			}
		}
	}
}

func TestGenerateBlobIDPermutation_NotIdentity(t *testing.T) {
	// A random permutation of size 128 should almost never be the identity.
	// Probability of identity = 1/128! ≈ 0. If this fails, RNG is broken.
	π, err := generateBlobIDPermutation(128)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	identity := true
	for i, v := range π {
		if i != v {
			identity = false
			break
		}
	}
	if identity {
		t.Fatalf("got identity permutation; RNG broken or improbable event (1/128!)")
	}
}

func TestGenerateBlobIDPermutation_Distinct(t *testing.T) {
	// Two independent calls should produce different permutations
	// (probability of collision ≈ 1/n! for n=128).
	a, _ := generateBlobIDPermutation(128)
	b, _ := generateBlobIDPermutation(128)

	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("two independent permutations identical; RNG broken")
	}
}

func TestGenerateBlobIDPermutation_ZeroAndOne(t *testing.T) {
	π0, err := generateBlobIDPermutation(0)
	if err != nil {
		t.Fatalf("n=0: unexpected error: %v", err)
	}
	if len(π0) != 0 {
		t.Fatalf("n=0: got len=%d, want 0", len(π0))
	}

	π1, err := generateBlobIDPermutation(1)
	if err != nil {
		t.Fatalf("n=1: unexpected error: %v", err)
	}
	if len(π1) != 1 || π1[0] != 0 {
		t.Fatalf("n=1: got %v, want [0]", π1)
	}
}
