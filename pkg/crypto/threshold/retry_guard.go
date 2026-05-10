package threshold

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
)

// NewInstanceID returns a fresh 16-byte random nonce suitable for use as the
// `instanceID` parameter to ThresholdDecrypt. Callers that don't have a
// domain-specific protocol-instance identifier should call this once per
// logical threshold-decrypt invocation. A retry of the same logical operation
// should keep the same instanceID — the per-participant RetryGuard then
// refuses the second emission, closing the Mouchet'24 retry-attack family.
func NewInstanceID() ([]byte, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return nil, fmt.Errorf("crypto/rand: %w", err)
	}
	return b, nil
}

// mustInstanceID returns a fresh 16-byte random nonce or panics. Test-only
// helper; production callers should use NewInstanceID and handle the error.
func mustInstanceID() []byte {
	b, err := NewInstanceID()
	if err != nil {
		panic(err)
	}
	return b
}

// ErrShareAlreadyEmitted is returned when a participant is asked to emit a
// partial-decryption share for an (instance, ciphertext, clientPK) tuple that
// it has already served. Refusing the second emission closes the
// Mouchet'24 / Okada'25 / Colin de Verdière 2026 retry-attack family —
// see docs/THRESHOLD_RETRY_FIX.md §2.1.
var ErrShareAlreadyEmitted = errors.New("threshold: share already emitted for this fingerprint — refusing replay")

// fingerprint is a stable identifier for a single (instance, ciphertext, clientPK)
// triple. SHA-256 is sufficient: collision-resistance below 2^128 is enough since
// we only need uniqueness across at most ~2^τ emissions per key (τ ≤ 2^20 in our
// IND-CPA^D budget — see docs/SECURITY_MODEL.md).
type fingerprint [sha256.Size]byte

// computeFingerprint hashes the inputs that uniquely identify a single share
// emission. instanceID separates protocol instances — a coordinator MUST mint
// a fresh instanceID per protocol invocation; aborted instances are NOT retried
// under the same ID.
func computeFingerprint(instanceID, ctBytes, clientPKBytes []byte) fingerprint {
	h := sha256.New()
	// Length-prefix every field to avoid collisions across (a||b) vs (a'||b')
	// where the concatenations happen to match.
	writeLP(h, instanceID)
	writeLP(h, ctBytes)
	writeLP(h, clientPKBytes)
	var fp fingerprint
	copy(fp[:], h.Sum(nil))
	return fp
}

func writeLP(h interface {
	Write(p []byte) (n int, err error)
}, b []byte) {
	var lenBuf [8]byte
	n := uint64(len(b))
	for i := 0; i < 8; i++ {
		lenBuf[i] = byte(n >> (8 * i))
	}
	_, _ = h.Write(lenBuf[:])
	_, _ = h.Write(b)
}

// RetryGuard tracks which fingerprints a participant has already served a
// share for, and refuses to emit a second share for the same fingerprint.
//
// The guard is safe for concurrent use — every committee node would in
// practice run independent instances, but a single Go test or simulator
// shares a process and may invoke participants concurrently.
//
// Memory grows linearly with the number of distinct (instance, ct, pk)
// triples a node has served. For Opaque's τ ≤ 2^20 budget per key this is at
// most ~32 MB at sha256.Size per entry — acceptable. A future Phase 4 may add
// an LRU cap once we wire epoch-based key rotation.
type RetryGuard struct {
	mu   sync.Mutex
	seen map[fingerprint]struct{}
}

// NewRetryGuard returns an empty guard.
func NewRetryGuard() *RetryGuard {
	return &RetryGuard{seen: make(map[fingerprint]struct{})}
}

// Admit checks whether this fingerprint has been served before. On first
// observation it records the fingerprint and returns nil (the caller may
// proceed to emit a share). On a repeat observation it returns
// ErrShareAlreadyEmitted and the caller MUST NOT emit a second share.
//
// instanceID — coordinator-supplied protocol-instance nonce. Different
// instances of "the same query" must use different IDs.
// ctBytes — the canonical-serialised ciphertext for which a share is requested.
// clientPKBytes — the canonical-serialised client public key.
func (g *RetryGuard) Admit(instanceID, ctBytes, clientPKBytes []byte) error {
	fp := computeFingerprint(instanceID, ctBytes, clientPKBytes)

	g.mu.Lock()
	defer g.mu.Unlock()

	if _, replay := g.seen[fp]; replay {
		return ErrShareAlreadyEmitted
	}
	g.seen[fp] = struct{}{}
	return nil
}

// Size returns the number of distinct fingerprints recorded — useful for
// tests and operational metrics. Not part of the security argument.
func (g *RetryGuard) Size() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.seen)
}

// FingerprintHex is a stable string representation of the fingerprint for an
// (instance, ct, pk) triple. Exposed for diagnostics + log-correlation. Not
// part of the security argument.
func FingerprintHex(instanceID, ctBytes, clientPKBytes []byte) string {
	fp := computeFingerprint(instanceID, ctBytes, clientPKBytes)
	return hex.EncodeToString(fp[:])
}
