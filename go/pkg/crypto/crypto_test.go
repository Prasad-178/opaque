package crypto

import (
	"math"
	"math/rand"
	"testing"
)

func TestNewClientEngine(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("failed to create client engine: %v", err)
	}

	// Check we have keys
	pubKey, err := engine.GetPublicKeyBytes()
	if err != nil {
		t.Fatalf("failed to get public key: %v", err)
	}

	if len(pubKey) == 0 {
		t.Error("public key is empty")
	}

	t.Logf("Public key size: %d bytes", len(pubKey))
}

func TestEncryptDecryptVector(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	// Test with a small vector
	original := []float64{0.1, 0.2, 0.3, 0.4, -0.1, -0.2}

	// Encrypt
	ct, err := engine.EncryptVector(original)
	if err != nil {
		t.Fatalf("failed to encrypt: %v", err)
	}

	// Decrypt
	decrypted, err := engine.DecryptVector(ct, len(original))
	if err != nil {
		t.Fatalf("failed to decrypt: %v", err)
	}

	// Check values (with tolerance for fixed-point encoding)
	tolerance := 1e-4
	for i := range original {
		if math.Abs(decrypted[i]-original[i]) > tolerance {
			t.Errorf("value %d mismatch: got %.6f, want %.6f", i, decrypted[i], original[i])
		}
	}
}

func TestCiphertextSerialization(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}

	original := []float64{0.5, -0.5, 0.25, -0.25}

	ct, err := engine.EncryptVector(original)
	if err != nil {
		t.Fatalf("failed to encrypt: %v", err)
	}

	// Serialize
	data, err := engine.SerializeCiphertext(ct)
	if err != nil {
		t.Fatalf("failed to serialize: %v", err)
	}

	t.Logf("Ciphertext size: %d bytes", len(data))

	// Deserialize
	ct2, err := engine.DeserializeCiphertext(data)
	if err != nil {
		t.Fatalf("failed to deserialize: %v", err)
	}

	// Decrypt and verify
	decrypted, err := engine.DecryptVector(ct2, len(original))
	if err != nil {
		t.Fatalf("failed to decrypt: %v", err)
	}

	tolerance := 1e-4
	for i := range original {
		if math.Abs(decrypted[i]-original[i]) > tolerance {
			t.Errorf("value %d mismatch: got %.6f, want %.6f", i, decrypted[i], original[i])
		}
	}
}

func TestNormalizeVector(t *testing.T) {
	vec := []float64{3, 4, 0}
	normalized := NormalizeVector(vec)

	// Should be [0.6, 0.8, 0]
	expected := []float64{0.6, 0.8, 0}

	for i := range expected {
		if math.Abs(normalized[i]-expected[i]) > 1e-10 {
			t.Errorf("value %d: got %.6f, want %.6f", i, normalized[i], expected[i])
		}
	}

	// Check norm is 1
	var norm float64
	for _, v := range normalized {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("norm should be 1, got %f", norm)
	}
}

func BenchmarkEncryption(b *testing.B) {
	engine, _ := NewClientEngine()
	vector := make([]float64, 128)
	rand.Seed(42)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1
	}
	vector = NormalizeVector(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.EncryptVector(vector)
	}
}

func BenchmarkDecryption(b *testing.B) {
	engine, _ := NewClientEngine()
	vector := make([]float64, 128)
	rand.Seed(42)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1
	}
	vector = NormalizeVector(vector)

	ct, _ := engine.EncryptVector(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.DecryptVector(ct, len(vector))
	}
}

func BenchmarkSerialization(b *testing.B) {
	engine, _ := NewClientEngine()
	vector := make([]float64, 128)
	rand.Seed(42)
	for i := range vector {
		vector[i] = rand.Float64()*2 - 1
	}

	ct, _ := engine.EncryptVector(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.SerializeCiphertext(ct)
	}
}
