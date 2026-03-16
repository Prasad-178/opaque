// Package encrypt provides symmetric encryption for Tier 2 data-private storage.
// Uses AES-256-GCM for authenticated encryption.
package encrypt

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"

	"golang.org/x/crypto/argon2"
)

const (
	// KeySize is the size of AES-256 keys in bytes.
	KeySize = 32

	// NonceSize is the size of GCM nonces in bytes.
	NonceSize = 12

	// SaltSize is the size of salts for key derivation.
	SaltSize = 16

	// Argon2Time is the time parameter for Argon2id.
	Argon2Time = 1

	// Argon2Memory is the memory parameter for Argon2id (64 MB).
	Argon2Memory = 64 * 1024

	// Argon2Threads is the parallelism parameter for Argon2id.
	Argon2Threads = 4
)

var (
	// ErrInvalidKey is returned when the encryption key is invalid.
	ErrInvalidKey = errors.New("invalid encryption key: must be 32 bytes")

	// ErrInvalidCiphertext is returned when ciphertext is too short.
	ErrInvalidCiphertext = errors.New("invalid ciphertext: too short")

	// ErrDecryptionFailed is returned when decryption fails (wrong key or tampered data).
	ErrDecryptionFailed = errors.New("decryption failed: authentication error")
)

// Encryptor provides symmetric encryption operations.
type Encryptor interface {
	// Encrypt encrypts plaintext and returns ciphertext (nonce prepended).
	Encrypt(plaintext []byte) ([]byte, error)

	// Decrypt decrypts ciphertext and returns plaintext.
	Decrypt(ciphertext []byte) ([]byte, error)

	// EncryptWithAAD encrypts with additional authenticated data.
	EncryptWithAAD(plaintext, aad []byte) ([]byte, error)

	// DecryptWithAAD decrypts with additional authenticated data.
	DecryptWithAAD(ciphertext, aad []byte) ([]byte, error)

	// KeyFingerprint returns a fingerprint of the current key (for verification).
	KeyFingerprint() string
}

// AESGCM implements Encryptor using AES-256-GCM.
type AESGCM struct {
	key    []byte
	cipher cipher.AEAD
}

// NewAESGCM creates a new AES-256-GCM encryptor with the given key.
// Key must be exactly 32 bytes (256 bits).
func NewAESGCM(key []byte) (*AESGCM, error) {
	if len(key) != KeySize {
		return nil, ErrInvalidKey
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	// Copy key to prevent external modification
	keyCopy := make([]byte, KeySize)
	copy(keyCopy, key)

	return &AESGCM{
		key:    keyCopy,
		cipher: gcm,
	}, nil
}

// Encrypt encrypts plaintext using AES-256-GCM.
// Returns: nonce (12 bytes) || ciphertext || tag (16 bytes)
func (e *AESGCM) Encrypt(plaintext []byte) ([]byte, error) {
	return e.EncryptWithAAD(plaintext, nil)
}

// EncryptWithAAD encrypts with additional authenticated data.
// AAD is authenticated but not encrypted (useful for metadata).
func (e *AESGCM) EncryptWithAAD(plaintext, aad []byte) ([]byte, error) {
	// Generate random nonce
	nonce := make([]byte, NonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt: Seal appends ciphertext+tag to nonce
	ciphertext := e.cipher.Seal(nonce, nonce, plaintext, aad)

	return ciphertext, nil
}

// Decrypt decrypts ciphertext encrypted with Encrypt.
func (e *AESGCM) Decrypt(ciphertext []byte) ([]byte, error) {
	return e.DecryptWithAAD(ciphertext, nil)
}

// DecryptWithAAD decrypts with additional authenticated data.
func (e *AESGCM) DecryptWithAAD(ciphertext, aad []byte) ([]byte, error) {
	if len(ciphertext) < NonceSize+e.cipher.Overhead() {
		return nil, ErrInvalidCiphertext
	}

	// Extract nonce and actual ciphertext
	nonce := ciphertext[:NonceSize]
	encryptedData := ciphertext[NonceSize:]

	// Decrypt and verify
	plaintext, err := e.cipher.Open(nil, nonce, encryptedData, aad)
	if err != nil {
		return nil, ErrDecryptionFailed
	}

	return plaintext, nil
}

// KeyFingerprint returns a SHA-256 fingerprint of the key (first 8 bytes, hex encoded).
// Useful for verifying key matches without exposing the key.
func (e *AESGCM) KeyFingerprint() string {
	hash := sha256.Sum256(e.key)
	return fmt.Sprintf("%x", hash[:8])
}

// DeriveKey derives a 256-bit key from a password and salt using Argon2id.
// This is suitable for user-provided passwords.
func DeriveKey(password string, salt []byte) []byte {
	return argon2.IDKey(
		[]byte(password),
		salt,
		Argon2Time,
		Argon2Memory,
		Argon2Threads,
		KeySize,
	)
}

// DeriveKeyWithSalt derives a key and returns both the key and a new random salt.
// Use this when creating a new encryption key from a password.
func DeriveKeyWithSalt(password string) (key []byte, salt []byte, err error) {
	salt = make([]byte, SaltSize)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return nil, nil, fmt.Errorf("failed to generate salt: %w", err)
	}

	key = DeriveKey(password, salt)
	return key, salt, nil
}

// GenerateKey generates a cryptographically secure random 256-bit key.
// Use this when you don't need password-based key derivation.
func GenerateKey() ([]byte, error) {
	key := make([]byte, KeySize)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}
	return key, nil
}

// EncryptVector encrypts a float64 vector for storage.
// Converts to bytes, encrypts, and returns ciphertext.
func (e *AESGCM) EncryptVector(vector []float64) ([]byte, error) {
	// Convert float64 slice to bytes
	plaintext := VectorToBytes(vector)
	return e.Encrypt(plaintext)
}

// DecryptVector decrypts a ciphertext back to a float64 vector.
func (e *AESGCM) DecryptVector(ciphertext []byte) ([]float64, error) {
	plaintext, err := e.Decrypt(ciphertext)
	if err != nil {
		return nil, err
	}
	return BytesToVector(plaintext), nil
}

// EncryptVectorWithID encrypts a vector with its ID as additional authenticated data.
// This binds the ciphertext to the ID, preventing ID swapping attacks.
func (e *AESGCM) EncryptVectorWithID(vector []float64, id string) ([]byte, error) {
	plaintext := VectorToBytes(vector)
	return e.EncryptWithAAD(plaintext, []byte(id))
}

// DecryptVectorWithID decrypts a vector and verifies the ID.
func (e *AESGCM) DecryptVectorWithID(ciphertext []byte, id string) ([]float64, error) {
	plaintext, err := e.DecryptWithAAD(ciphertext, []byte(id))
	if err != nil {
		return nil, err
	}
	return BytesToVector(plaintext), nil
}
