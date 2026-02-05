package encrypt

import (
	"bytes"
	"testing"
)

func TestAESGCM_EncryptDecrypt(t *testing.T) {
	key, err := GenerateKey()
	if err != nil {
		t.Fatalf("failed to generate key: %v", err)
	}

	enc, err := NewAESGCM(key)
	if err != nil {
		t.Fatalf("failed to create encryptor: %v", err)
	}

	plaintext := []byte("Hello, World! This is a test message for encryption.")

	// Encrypt
	ciphertext, err := enc.Encrypt(plaintext)
	if err != nil {
		t.Fatalf("encryption failed: %v", err)
	}

	// Ciphertext should be longer than plaintext (nonce + tag)
	if len(ciphertext) <= len(plaintext) {
		t.Error("ciphertext should be longer than plaintext")
	}

	// Decrypt
	decrypted, err := enc.Decrypt(ciphertext)
	if err != nil {
		t.Fatalf("decryption failed: %v", err)
	}

	if !bytes.Equal(plaintext, decrypted) {
		t.Errorf("decrypted text doesn't match original\ngot: %s\nwant: %s", decrypted, plaintext)
	}
}

func TestAESGCM_EncryptDecryptWithAAD(t *testing.T) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)

	plaintext := []byte("Secret data")
	aad := []byte("document-id-12345")

	// Encrypt with AAD
	ciphertext, err := enc.EncryptWithAAD(plaintext, aad)
	if err != nil {
		t.Fatalf("encryption failed: %v", err)
	}

	// Decrypt with correct AAD
	decrypted, err := enc.DecryptWithAAD(ciphertext, aad)
	if err != nil {
		t.Fatalf("decryption failed: %v", err)
	}

	if !bytes.Equal(plaintext, decrypted) {
		t.Error("decrypted text doesn't match original")
	}

	// Decrypt with wrong AAD should fail
	_, err = enc.DecryptWithAAD(ciphertext, []byte("wrong-id"))
	if err != ErrDecryptionFailed {
		t.Error("decryption should fail with wrong AAD")
	}
}

func TestAESGCM_WrongKey(t *testing.T) {
	key1, _ := GenerateKey()
	key2, _ := GenerateKey()

	enc1, _ := NewAESGCM(key1)
	enc2, _ := NewAESGCM(key2)

	plaintext := []byte("Secret message")
	ciphertext, _ := enc1.Encrypt(plaintext)

	// Decrypting with wrong key should fail
	_, err := enc2.Decrypt(ciphertext)
	if err != ErrDecryptionFailed {
		t.Error("decryption should fail with wrong key")
	}
}

func TestAESGCM_TamperedCiphertext(t *testing.T) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)

	plaintext := []byte("Secret message")
	ciphertext, _ := enc.Encrypt(plaintext)

	// Tamper with ciphertext
	ciphertext[len(ciphertext)-1] ^= 0xFF

	// Decryption should fail
	_, err := enc.Decrypt(ciphertext)
	if err != ErrDecryptionFailed {
		t.Error("decryption should fail with tampered ciphertext")
	}
}

func TestAESGCM_InvalidKeySize(t *testing.T) {
	// Too short
	_, err := NewAESGCM([]byte("short"))
	if err != ErrInvalidKey {
		t.Error("should reject short key")
	}

	// Too long
	_, err = NewAESGCM(make([]byte, 64))
	if err != ErrInvalidKey {
		t.Error("should reject long key")
	}
}

func TestDeriveKey(t *testing.T) {
	password := "my-secret-password"
	salt := []byte("random-salt-1234")

	key1 := DeriveKey(password, salt)
	key2 := DeriveKey(password, salt)

	// Same password and salt should produce same key
	if !bytes.Equal(key1, key2) {
		t.Error("same password and salt should produce same key")
	}

	// Different salt should produce different key
	key3 := DeriveKey(password, []byte("different-salt!!"))
	if bytes.Equal(key1, key3) {
		t.Error("different salt should produce different key")
	}

	// Key should be correct size
	if len(key1) != KeySize {
		t.Errorf("key size should be %d, got %d", KeySize, len(key1))
	}
}

func TestDeriveKeyWithSalt(t *testing.T) {
	password := "my-secret-password"

	key1, salt1, err := DeriveKeyWithSalt(password)
	if err != nil {
		t.Fatalf("failed to derive key: %v", err)
	}

	key2, salt2, err := DeriveKeyWithSalt(password)
	if err != nil {
		t.Fatalf("failed to derive key: %v", err)
	}

	// Salts should be different (random)
	if bytes.Equal(salt1, salt2) {
		t.Error("salts should be randomly generated")
	}

	// Keys should be different (different salts)
	if bytes.Equal(key1, key2) {
		t.Error("keys should be different due to different salts")
	}

	// Re-deriving with same salt should give same key
	keyCheck := DeriveKey(password, salt1)
	if !bytes.Equal(key1, keyCheck) {
		t.Error("re-derived key should match original")
	}
}

func TestKeyFingerprint(t *testing.T) {
	key1, _ := GenerateKey()
	key2, _ := GenerateKey()

	enc1, _ := NewAESGCM(key1)
	enc2, _ := NewAESGCM(key2)

	fp1 := enc1.KeyFingerprint()
	fp2 := enc2.KeyFingerprint()

	// Fingerprints should be different for different keys
	if fp1 == fp2 {
		t.Error("fingerprints should be different for different keys")
	}

	// Same key should have same fingerprint
	enc1b, _ := NewAESGCM(key1)
	if enc1.KeyFingerprint() != enc1b.KeyFingerprint() {
		t.Error("same key should have same fingerprint")
	}

	// Fingerprint should be 16 hex chars (8 bytes)
	if len(fp1) != 16 {
		t.Errorf("fingerprint should be 16 chars, got %d", len(fp1))
	}
}

func TestEncryptVector(t *testing.T) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)

	vector := []float64{0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.0}

	// Encrypt
	ciphertext, err := enc.EncryptVector(vector)
	if err != nil {
		t.Fatalf("vector encryption failed: %v", err)
	}

	// Decrypt
	decrypted, err := enc.DecryptVector(ciphertext)
	if err != nil {
		t.Fatalf("vector decryption failed: %v", err)
	}

	// Compare
	if len(decrypted) != len(vector) {
		t.Fatalf("vector length mismatch: got %d, want %d", len(decrypted), len(vector))
	}

	for i := range vector {
		if decrypted[i] != vector[i] {
			t.Errorf("vector[%d] mismatch: got %f, want %f", i, decrypted[i], vector[i])
		}
	}
}

func TestEncryptVectorWithID(t *testing.T) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)

	vector := []float64{1.0, 2.0, 3.0}
	id := "doc-123"

	// Encrypt with ID
	ciphertext, err := enc.EncryptVectorWithID(vector, id)
	if err != nil {
		t.Fatalf("encryption failed: %v", err)
	}

	// Decrypt with correct ID
	decrypted, err := enc.DecryptVectorWithID(ciphertext, id)
	if err != nil {
		t.Fatalf("decryption failed: %v", err)
	}

	for i := range vector {
		if decrypted[i] != vector[i] {
			t.Errorf("vector[%d] mismatch", i)
		}
	}

	// Decrypt with wrong ID should fail
	_, err = enc.DecryptVectorWithID(ciphertext, "wrong-id")
	if err != ErrDecryptionFailed {
		t.Error("decryption should fail with wrong ID")
	}
}

func TestVectorToBytes(t *testing.T) {
	vector := []float64{1.0, 2.0, 3.0, -1.5, 0.0}

	bytes := VectorToBytes(vector)
	if len(bytes) != len(vector)*8 {
		t.Errorf("bytes length should be %d, got %d", len(vector)*8, len(bytes))
	}

	recovered := BytesToVector(bytes)
	if len(recovered) != len(vector) {
		t.Errorf("recovered vector length should be %d, got %d", len(vector), len(recovered))
	}

	for i := range vector {
		if recovered[i] != vector[i] {
			t.Errorf("vector[%d] mismatch: got %f, want %f", i, recovered[i], vector[i])
		}
	}
}

func BenchmarkEncrypt(b *testing.B) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)
	plaintext := make([]byte, 1024) // 1KB

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = enc.Encrypt(plaintext)
	}
}

func BenchmarkDecrypt(b *testing.B) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)
	plaintext := make([]byte, 1024)
	ciphertext, _ := enc.Encrypt(plaintext)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = enc.Decrypt(ciphertext)
	}
}

func BenchmarkEncryptVector128D(b *testing.B) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)
	vector := make([]float64, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = enc.EncryptVector(vector)
	}
}

func BenchmarkDecryptVector128D(b *testing.B) {
	key, _ := GenerateKey()
	enc, _ := NewAESGCM(key)
	vector := make([]float64, 128)
	ciphertext, _ := enc.EncryptVector(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = enc.DecryptVector(ciphertext)
	}
}
