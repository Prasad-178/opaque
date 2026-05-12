package threshold

import (
	"encoding/base64"
	"errors"
	"fmt"
	"sync"
)

// instanceState is the coordinator-side lifecycle of a single threshold-
// decrypt protocol instance. Each instanceID follows:
//
//	unknown → active → committed (success, terminal)
//	              ↓
//	           aborted (failure, terminal)
//
// Terminal states are absorbing — an instanceID can never be reused for
// another protocol invocation. This is the Phase 3b piece of the threshold
// retry-attack fix; see docs/THRESHOLD_RETRY_FIX.md §2.3.
type instanceState int

const (
	stateUnknown instanceState = iota
	// stateReserved: BeginInstance has minted the ID but ThresholdDecrypt
	// has not yet started. AbortInstance moves it to aborted; the first
	// ThresholdDecrypt moves it to running.
	stateReserved
	// stateRunning: ThresholdDecrypt is actively executing. A second
	// ThresholdDecrypt with the same ID is refused (ErrInstanceConcurrent).
	stateRunning
	stateCommitted
	stateAborted
)

var (
	// ErrInstanceCommitted is returned when ThresholdDecrypt is called with
	// an instanceID that already completed a successful decryption. Re-using
	// an instanceID is forbidden — the coordinator MUST mint a fresh
	// instanceID for every protocol invocation, even on retry. See
	// docs/THRESHOLD_RETRY_FIX.md §2.3 for the threat-model rationale.
	ErrInstanceCommitted = errors.New("threshold: instance already committed — must mint a fresh instanceID for retry")

	// ErrInstanceAborted is returned when ThresholdDecrypt is called with an
	// instanceID that was previously aborted (either by AbortInstance or by
	// a prior failed ThresholdDecrypt). Aborted instances are terminal;
	// the coordinator MUST mint a fresh instanceID for any retry attempt.
	ErrInstanceAborted = errors.New("threshold: instance already aborted — must mint a fresh instanceID for retry")

	// ErrInstanceConcurrent is returned when two goroutines simultaneously
	// invoke ThresholdDecrypt with the same instanceID. The first call wins
	// activation; the second is refused. The protocol design requires one
	// instanceID per logical decrypt, and concurrent reuse is treated as a
	// caller bug.
	ErrInstanceConcurrent = errors.New("threshold: instance is concurrently active — instanceID must be unique per decrypt invocation")

	// ErrInstanceEmpty is returned when an empty instanceID is supplied.
	// Empty IDs would all collide in the tracker state map, defeating the
	// whole point of per-instance state.
	ErrInstanceEmpty = errors.New("threshold: instanceID must be non-empty")
)

// instanceTracker is the coordinator-side state machine that prevents
// instanceID reuse. Together with the per-participant RetryGuard (Phase 2),
// this is defense in depth:
//
//   - At the coordinator level, instanceTracker refuses any second
//     ThresholdDecrypt call with the same instanceID — before any
//     participant is contacted.
//   - At the participant level, RetryGuard refuses any second share
//     emission for the same (instanceID, ct, clientPK) fingerprint — even
//     if a malicious coordinator bypasses its own tracker.
//
// In Opaque the coordinator is honest-but-curious today, so the tracker is
// the load-bearing piece in practice; the RetryGuard remains the security
// fallback if the coordinator is ever compromised.
type instanceTracker struct {
	mu    sync.Mutex
	state map[string]instanceState
}

func newInstanceTracker() *instanceTracker {
	return &instanceTracker{state: make(map[string]instanceState)}
}

// key returns the canonical map-key encoding of an instanceID. Base64 is
// chosen over hex purely to keep the map keys short — both are
// collision-free for any byte input.
func (t *instanceTracker) key(instanceID []byte) string {
	return base64.StdEncoding.EncodeToString(instanceID)
}

// reserve transitions an instanceID into the reserved state. Called by
// BeginInstance to claim an ID before ThresholdDecrypt is invoked. A
// reserved ID can still be aborted (AbortInstance) or activated (the next
// ThresholdDecrypt call). Returns an error if the ID is already in any
// other state — which for a fresh 16-byte random nonce is astronomically
// unlikely and indicates a structural bug.
func (t *instanceTracker) reserve(instanceID []byte) error {
	if len(instanceID) == 0 {
		return ErrInstanceEmpty
	}
	k := t.key(instanceID)
	t.mu.Lock()
	defer t.mu.Unlock()
	switch t.state[k] {
	case stateUnknown:
		t.state[k] = stateReserved
		return nil
	case stateReserved, stateRunning:
		return ErrInstanceConcurrent
	case stateCommitted:
		return ErrInstanceCommitted
	case stateAborted:
		return ErrInstanceAborted
	default:
		return fmt.Errorf("threshold: instance in unknown state %d", t.state[k])
	}
}

// activate transitions an instanceID into the running state. Auto-registers
// from unknown for callers that didn't go through BeginInstance.
// Returns an error if the ID is already running (concurrent) OR terminal
// (committed/aborted).
//
// On success, the caller MUST eventually call commit or abort to release
// the state — the tracker has no GC pass to recover orphaned running
// entries, but a process restart resets all state, so this is acceptable
// for the current threat model.
func (t *instanceTracker) activate(instanceID []byte) error {
	if len(instanceID) == 0 {
		return ErrInstanceEmpty
	}
	k := t.key(instanceID)
	t.mu.Lock()
	defer t.mu.Unlock()
	switch t.state[k] {
	case stateUnknown, stateReserved:
		t.state[k] = stateRunning
		return nil
	case stateRunning:
		return ErrInstanceConcurrent
	case stateCommitted:
		return ErrInstanceCommitted
	case stateAborted:
		return ErrInstanceAborted
	default:
		return fmt.Errorf("threshold: instance in unknown state %d", t.state[k])
	}
}

// commit marks an instance as successfully completed. Called by
// ThresholdDecrypt on the happy path. Idempotent: calling commit twice is
// a no-op (terminal state stays terminal).
func (t *instanceTracker) commit(instanceID []byte) {
	k := t.key(instanceID)
	t.mu.Lock()
	if t.state[k] == stateRunning {
		t.state[k] = stateCommitted
	}
	t.mu.Unlock()
}

// abort marks an instance as failed. Called by ThresholdDecrypt on any
// error path AND by the public AbortInstance method. Idempotent: aborting
// an already-aborted or already-committed instance is a no-op. Aborts a
// reserved or running instance — terminal once set.
func (t *instanceTracker) abort(instanceID []byte) {
	if len(instanceID) == 0 {
		return
	}
	k := t.key(instanceID)
	t.mu.Lock()
	switch t.state[k] {
	case stateUnknown, stateReserved, stateRunning:
		t.state[k] = stateAborted
	}
	t.mu.Unlock()
}

// peek returns the current state of an instanceID. Used by tests + metrics
// — not part of the security argument.
func (t *instanceTracker) peek(instanceID []byte) instanceState {
	if len(instanceID) == 0 {
		return stateUnknown
	}
	k := t.key(instanceID)
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.state[k]
}

// size returns the number of distinct instanceIDs the tracker has seen
// (across all states). Useful for capacity-planning tests.
func (t *instanceTracker) size() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.state)
}
