package crypto

import (
	"testing"
)

// The deserialization boundary takes attacker-controlled bytes (public keys and
// ciphertexts arrive over the wire) and hands them to Lattigo's ReadFrom, which
// can panic on malformed input. NewServerEngine and DeserializeCiphertext wrap
// those calls in size checks + recover guards. These fuzz targets assert the
// guard invariant holds across arbitrary inputs: the functions must never panic,
// and must never return a non-nil result together with an error (or vice versa).
//
// Under a plain `go test` the targets run only their seed corpus, so they double
// as fast regression tests against someone removing a recover guard. Run the
// fuzzing loop explicitly with `make fuzz` (or `go test -fuzz=...`).

// seedCiphertexts returns a valid serialized ciphertext plus malformed samples
// for use as the fuzz seed corpus.
func seedCiphertexts(t testing.TB) (*Engine, [][]byte) {
	t.Helper()
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("failed to create client engine: %v", err)
	}
	ct, err := engine.EncryptVector([]float64{0.5, -0.5, 0.25, -0.25})
	if err != nil {
		t.Fatalf("failed to encrypt: %v", err)
	}
	valid, err := engine.SerializeCiphertext(ct)
	if err != nil {
		t.Fatalf("failed to serialize: %v", err)
	}
	seeds := [][]byte{
		valid,
		{},
		{0x00},
		{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
		valid[:len(valid)/2], // truncated
	}
	return engine, seeds
}

// FuzzDeserializeCiphertext feeds arbitrary bytes to the ciphertext deserializer
// and asserts it never panics and keeps the (result, error) invariant.
func FuzzDeserializeCiphertext(f *testing.F) {
	engine, seeds := seedCiphertexts(f) // setup runs once
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		ct, err := engine.DeserializeCiphertext(data) // must not panic
		if err != nil && ct != nil {
			t.Fatalf("error returned with non-nil ciphertext (len=%d)", len(data))
		}
		if err == nil && ct == nil {
			t.Fatalf("nil ciphertext returned with nil error (len=%d)", len(data))
		}
	})
}

// FuzzNewServerEngine feeds arbitrary bytes to the public-key deserializer and
// asserts it never panics and keeps the (engine, error) invariant.
func FuzzNewServerEngine(f *testing.F) {
	client, err := NewClientEngine()
	if err != nil {
		f.Fatalf("failed to create client engine: %v", err)
	}
	pkBytes, err := client.GetPublicKeyBytes()
	if err != nil {
		f.Fatalf("failed to get public key bytes: %v", err)
	}
	for _, s := range [][]byte{
		pkBytes,
		{},
		{0x01, 0x02, 0x03},
		{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
		pkBytes[:len(pkBytes)/2], // truncated
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		eng, err := NewServerEngine(data) // must not panic
		if err != nil && eng != nil {
			t.Fatalf("error returned with non-nil engine (len=%d)", len(data))
		}
		if err == nil && eng == nil {
			t.Fatalf("nil engine returned with nil error (len=%d)", len(data))
		}
	})
}
