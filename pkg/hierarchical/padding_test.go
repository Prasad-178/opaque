package hierarchical

import (
	"testing"

	"github.com/Prasad-178/opaque/pkg/enterprise"
)

func TestNextPow2(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 1},
		{1, 1},
		{2, 2},
		{3, 4},
		{7, 8},
		{8, 8},
		{9, 16},
		{1023, 1024},
		{1024, 1024},
		{1025, 2048},
	}
	for _, c := range cases {
		if got := nextPow2(c.in); got != c.want {
			t.Errorf("nextPow2(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestComputePaddingTargets_PerBucket(t *testing.T) {
	counts := map[string]int{
		"00_00": 100,
		"00_01": 200,
		"01_00": 175,
	}

	if got := computePaddingTargets(counts, enterprise.PaddingNone); got != nil {
		t.Errorf("PaddingNone: got %v, want nil", got)
	}

	// PaddingMaxFixed: every bucket padded to global max.
	mf := computePaddingTargets(counts, enterprise.PaddingMaxFixed)
	if len(mf) != 3 {
		t.Errorf("PaddingMaxFixed: got %d targets, want 3", len(mf))
	}
	for k, v := range mf {
		if v != 200 {
			t.Errorf("PaddingMaxFixed[%s]: got %d, want 200", k, v)
		}
	}

	// PaddingBucketed: per-cluster nextPow2 (NOT global max nextPow2).
	// This is the fix landed in commit 03b095f follow-up — earlier the
	// implementation padded every bucket to nextPow2(global_max), which
	// at imbalanced clusters could ~2× total blob count and caused the
	// DBpedia 1M @ 1536-dim OOM saga (May 2026).
	bk := computePaddingTargets(counts, enterprise.PaddingBucketed)
	expected := map[string]int{
		"00_00": 128, // nextPow2(100)
		"00_01": 256, // nextPow2(200)
		"01_00": 256, // nextPow2(175)
	}
	for k, want := range expected {
		if bk[k] != want {
			t.Errorf("PaddingBucketed[%s]: got %d, want %d", k, bk[k], want)
		}
	}

	// Empty counts → nil
	if got := computePaddingTargets(nil, enterprise.PaddingMaxFixed); got != nil {
		t.Errorf("empty counts: got %v, want nil", got)
	}
}

func TestGeneratePaddingBlobs_None(t *testing.T) {
	counts := map[string]int{"00_00": 5, "01_00": 10}
	blobs, err := generatePaddingBlobs(counts, enterprise.PaddingNone, 128)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(blobs) != 0 {
		t.Errorf("PaddingNone: got %d blobs, want 0", len(blobs))
	}
}

func TestGeneratePaddingBlobs_MaxFixed(t *testing.T) {
	counts := map[string]int{
		"00_00": 100,
		"00_01": 50,
		"01_00": 75,
	}
	dim := 128
	blobs, err := generatePaddingBlobs(counts, enterprise.PaddingMaxFixed, dim)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// Target = 100. To-add: 0 + 50 + 25 = 75.
	if len(blobs) != 75 {
		t.Errorf("got %d padding blobs, want 75", len(blobs))
	}
	// Padding now matches real-blob size — float32-encoded ciphertext is
	// dim*4 + 28 bytes (12-byte nonce + 16-byte GCM tag) since commit
	// 7a9a369 and the padding fix in this commit.
	expectedCTLen := dim*4 + 28
	for i, b := range blobs {
		if len(b.Ciphertext) != expectedCTLen {
			t.Errorf("blob[%d] ciphertext len = %d, want %d (float32-encoded vector + 28-byte AES-GCM overhead)", i, len(b.Ciphertext), expectedCTLen)
		}
		if len(b.ID) != 64 { // 32 bytes hex-encoded
			t.Errorf("blob[%d] ID len = %d, want 64", i, len(b.ID))
		}
	}
}

func TestGeneratePaddingBlobs_Bucketed_PerCluster(t *testing.T) {
	// Verifies per-cluster nextPow2 semantics. Pre-fix this would have
	// padded both buckets to nextPow2(global_max=100) = 128. Post-fix
	// each bucket pads to its own nextPow2 — 50 → 64, 100 → 128.
	counts := map[string]int{
		"00_00": 100, // → per-cluster target nextPow2(100) = 128, add 28
		"00_01": 50,  // → per-cluster target nextPow2(50) = 64, add 14
	}
	blobs, err := generatePaddingBlobs(counts, enterprise.PaddingBucketed, 128)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := 28 + 14
	if len(blobs) != want {
		t.Errorf("got %d padding blobs, want %d (per-cluster nextPow2)", len(blobs), want)
	}
}

func TestGeneratePaddingBlobs_DistributedCorrectly(t *testing.T) {
	counts := map[string]int{
		"00_00": 5,
		"00_01": 10,
		"01_00": 3,
	}
	blobs, err := generatePaddingBlobs(counts, enterprise.PaddingMaxFixed, 64)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// Target = 10. Per-bucket distribution should top each up to 10.
	perBucket := make(map[string]int)
	for _, b := range blobs {
		perBucket[b.LSHBucket]++
	}
	expected := map[string]int{"00_00": 5, "00_01": 0, "01_00": 7}
	for k, want := range expected {
		if got := perBucket[k]; got != want {
			t.Errorf("bucket %s: got %d padding, want %d", k, got, want)
		}
	}
}

func TestRandomHexID(t *testing.T) {
	id1, err := randomHexID(32)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	id2, _ := randomHexID(32)
	if len(id1) != 64 {
		t.Errorf("len = %d, want 64", len(id1))
	}
	if id1 == id2 {
		t.Errorf("two crypto-random IDs collided; RNG broken")
	}
	for _, c := range id1 {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("non-hex char in ID: %q", c)
			break
		}
	}
}
