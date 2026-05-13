package crypto

import (
	"bytes"
	"testing"
)

// ============================================================================
// Ephemeral key rotation (direct-mode Engine).
// Parallel to pkg/crypto/threshold/rotation_test.go for the threshold path.
// Closes the "Planned key rotation" item in docs/SECURITY_MODEL.md §8 by
// providing the mechanism — policy (cadence) is caller-driven.
// ============================================================================

func TestEngine_DecryptCountInitiallyZero(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()
	if got := engine.DecryptCount(); got != 0 {
		t.Errorf("initial DecryptCount: got %d, want 0", got)
	}
}

func TestEngine_DecryptScalarIncrementsCount(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	vec := make([]float64, 128)
	vec[0] = 0.42
	ct, err := engine.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	const N = 5
	for i := 0; i < N; i++ {
		if _, err := engine.DecryptScalar(ct); err != nil {
			t.Fatalf("DecryptScalar #%d: %v", i, err)
		}
	}
	if got := engine.DecryptCount(); got != N {
		t.Errorf("DecryptCount after %d decrypts: got %d, want %d", N, got, N)
	}
}

func TestEngine_ShouldRotateGate(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	if engine.ShouldRotate() {
		t.Error("ShouldRotate with limit=0: got true, want false")
	}

	engine.SetRotationLimit(3)
	if engine.ShouldRotate() {
		t.Error("ShouldRotate at count=0/limit=3: got true, want false")
	}

	vec := make([]float64, 128)
	vec[0] = 0.42
	ct, err := engine.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := engine.DecryptScalar(ct); err != nil {
			t.Fatalf("DecryptScalar #%d: %v", i, err)
		}
	}
	if !engine.ShouldRotate() {
		t.Errorf("ShouldRotate at count=3/limit=3: got false, want true (count=%d)",
			engine.DecryptCount())
	}
}

func TestEngine_RotateKeysProducesFreshKeys(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	oldPK, err := engine.GetPublicKeyBytes()
	if err != nil {
		t.Fatalf("old GetPublicKeyBytes: %v", err)
	}

	if err := engine.RotateKeys(); err != nil {
		t.Fatalf("RotateKeys: %v", err)
	}

	newPK, err := engine.GetPublicKeyBytes()
	if err != nil {
		t.Fatalf("new GetPublicKeyBytes: %v", err)
	}
	if bytes.Equal(oldPK, newPK) {
		t.Error("RotateKeys did not change public key — fresh PK expected")
	}
}

func TestEngine_RotateKeysResetsCounter(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	vec := make([]float64, 128)
	vec[0] = 0.42
	ct, err := engine.EncryptVector(vec)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := engine.DecryptScalar(ct); err != nil {
			t.Fatalf("DecryptScalar #%d: %v", i, err)
		}
	}
	if got := engine.DecryptCount(); got != 3 {
		t.Fatalf("pre-rotate count: got %d, want 3", got)
	}

	if err := engine.RotateKeys(); err != nil {
		t.Fatalf("RotateKeys: %v", err)
	}
	if got := engine.DecryptCount(); got != 0 {
		t.Errorf("post-rotate count: got %d, want 0", got)
	}
}

func TestEngine_RotateKeysFreshEncryptDecrypt(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	if err := engine.RotateKeys(); err != nil {
		t.Fatalf("RotateKeys: %v", err)
	}

	vec := make([]float64, 128)
	vec[0] = 0.42
	ct, err := engine.EncryptVector(vec)
	if err != nil {
		t.Fatalf("post-rotate EncryptVector: %v", err)
	}
	got, err := engine.DecryptScalar(ct)
	if err != nil {
		t.Fatalf("post-rotate DecryptScalar: %v", err)
	}
	if got < 0.42-0.01 || got > 0.42+0.01 {
		t.Errorf("post-rotate decrypt value: got %.4f, want 0.42 ± 0.01", got)
	}
}

func TestEngine_RotateKeysOnServerSideFails(t *testing.T) {
	// Server-side engine has no secret key — rotation must fail cleanly.
	client, err := NewClientEngine()
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	pkBytes, err := client.GetPublicKeyBytes()
	if err != nil {
		t.Fatalf("GetPublicKeyBytes: %v", err)
	}
	client.Close()

	server, err := NewServerEngine(pkBytes)
	if err != nil {
		t.Fatalf("NewServerEngine: %v", err)
	}
	defer server.Close()

	err = server.RotateKeys()
	if err == nil {
		t.Error("RotateKeys on server-side engine: expected error, got nil")
	}
}
