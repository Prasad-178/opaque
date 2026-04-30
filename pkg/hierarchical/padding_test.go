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

func TestComputePaddingTarget(t *testing.T) {
	counts := map[string]int{
		"00_00": 100,
		"00_01": 200,
		"01_00": 175,
	}
	if got := computePaddingTarget(counts, enterprise.PaddingNone); got != 0 {
		t.Errorf("PaddingNone: got %d, want 0", got)
	}
	if got := computePaddingTarget(counts, enterprise.PaddingMaxFixed); got != 200 {
		t.Errorf("PaddingMaxFixed: got %d, want 200", got)
	}
	if got := computePaddingTarget(counts, enterprise.PaddingBucketed); got != 256 {
		t.Errorf("PaddingBucketed: got %d, want 256 (next pow2 of 200)", got)
	}
	// Empty counts → 0
	if got := computePaddingTarget(nil, enterprise.PaddingMaxFixed); got != 0 {
		t.Errorf("empty counts: got %d, want 0", got)
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
	expectedCTLen := dim*8 + 28
	for i, b := range blobs {
		if len(b.Ciphertext) != expectedCTLen {
			t.Errorf("blob[%d] ciphertext len = %d, want %d", i, len(b.Ciphertext), expectedCTLen)
		}
		if len(b.ID) != 64 { // 32 bytes hex-encoded
			t.Errorf("blob[%d] ID len = %d, want 64", i, len(b.ID))
		}
	}
}

func TestGeneratePaddingBlobs_Bucketed(t *testing.T) {
	counts := map[string]int{
		"00_00": 100, // → target 128, add 28
		"00_01": 50,  // → target 128, add 78
	}
	blobs, err := generatePaddingBlobs(counts, enterprise.PaddingBucketed, 128)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := 28 + 78
	if len(blobs) != want {
		t.Errorf("got %d padding blobs, want %d", len(blobs), want)
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
