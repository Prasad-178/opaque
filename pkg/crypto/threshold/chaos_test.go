package threshold

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// ============================================================================
// Phase 3c — chaos-monkey tests for the threshold retry-attack fix.
//
// Each test simulates an adversarial pattern the design is meant to defend
// against, and asserts the empirical safety invariant: no second share is
// ever emitted for the same (instanceID, ct, clientPK) fingerprint, under
// any failure mode or concurrent-attack pattern. See
// docs/THRESHOLD_RETRY_FIX.md §3 Phase 3c.
// ============================================================================

// makeChaosFixture builds a Committee + ClientSession + a ready-to-decrypt
// ciphertext. Element-wise multiplies query=[0.5, ...] by centroid=[0.4, ...]
// so decoded[0] = 0.20 — minimal HE machinery, no rotation/sum tree. The
// chaos tests don't care about the cryptographic content; they only need a
// real ciphertext to send through the full ThresholdDecrypt path.
func makeChaosFixture(t *testing.T, n, threshold int) (*Committee, *ClientSession, *rlwe.Ciphertext, float64) {
	t.Helper()
	committee, err := NewCommittee(n, threshold)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}
	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("NewClientSession: %v", err)
	}
	query := make([]float64, 128)
	query[0] = 0.5
	ct, err := committee.EncryptVector(query)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}
	eval := hefloat.NewEvaluator(committee.params, committee.GetEvalKeySet())
	encoder := committee.GetEncoder()
	centroid := make([]float64, committee.params.MaxSlots())
	centroid[0] = 0.4
	ptCentroid := hefloat.NewPlaintext(committee.params, ct.Level())
	if err := encoder.Encode(centroid, ptCentroid); err != nil {
		t.Fatalf("Encode centroid: %v", err)
	}
	result, err := eval.MulNew(ct, ptCentroid)
	if err != nil {
		t.Fatalf("MulNew: %v", err)
	}
	if err := eval.Rescale(result, result); err != nil {
		t.Fatalf("Rescale: %v", err)
	}
	return committee, session, result, 0.5 * 0.4
}

// TestChaos_ConcurrentRetrySameID hammers ThresholdDecrypt from N goroutines
// with the SAME instanceID on the SAME ciphertext. Expected behaviour:
// exactly one goroutine receives a successful decrypt; all others receive
// either ErrInstanceConcurrent (caught at coordinator state machine) or
// ErrInstanceCommitted/ErrInstanceAborted (also coordinator-side, after
// the first call commits/aborts). The per-participant RetryGuard never
// gets a chance to emit a second share — that's the headline invariant.
func TestChaos_ConcurrentRetrySameID(t *testing.T) {
	committee, session, fixture, want := makeChaosFixture(t, 3, 2)
	defer session.Close()

	id := mustInstanceID()
	const N = 50

	var successes int64
	var concurrentRefusals int64
	var terminalRefusals int64
	var otherErrors int64

	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := committee.ThresholdDecrypt(fixture, session.PK, id)
			switch {
			case err == nil:
				atomic.AddInt64(&successes, 1)
				got, derr := session.DecryptScalar(out)
				if derr != nil {
					t.Errorf("DecryptScalar: %v", derr)
				}
				if !nearlyEqual(got, want, 0.02) {
					t.Errorf("decrypt value: got %.4f, want %.4f", got, want)
				}
			case errors.Is(err, ErrInstanceConcurrent):
				atomic.AddInt64(&concurrentRefusals, 1)
			case errors.Is(err, ErrInstanceCommitted), errors.Is(err, ErrInstanceAborted):
				atomic.AddInt64(&terminalRefusals, 1)
			default:
				atomic.AddInt64(&otherErrors, 1)
				t.Errorf("unexpected error: %v", err)
			}
		}()
	}
	wg.Wait()

	if successes != 1 {
		t.Errorf("successes: got %d, want exactly 1", successes)
	}
	if otherErrors != 0 {
		t.Errorf("unexpected errors: %d", otherErrors)
	}
	total := successes + concurrentRefusals + terminalRefusals
	if total != N {
		t.Errorf("accounted %d/%d outcomes", total, N)
	}

	// CRITICAL invariant: the per-participant RetryGuard must have seen
	// exactly 1 fingerprint per participant — proving no second emission
	// reached the per-participant guard either. With the coordinator-side
	// tracker doing the bulk of the rejection, only the single winning
	// decrypt should have admitted a fingerprint.
	for i, p := range committee.Participants[:committee.Threshold] {
		if got := p.guard.Size(); got != 1 {
			t.Errorf("participant %d: guard.Size() = %d, want 1 — possible double-emission",
				i, got)
		}
	}
}

// TestChaos_ManyDistinctIDsAllSucceed is the dual of the concurrent-same-id
// chaos test: N goroutines call ThresholdDecrypt with N DIFFERENT
// instanceIDs on the same ciphertext. All N should succeed — fresh IDs are
// always valid. This proves the state machine doesn't accidentally serialize
// unrelated decrypts.
func TestChaos_ManyDistinctIDsAllSucceed(t *testing.T) {
	committee, session, fixture, want := makeChaosFixture(t, 3, 2)
	defer session.Close()

	const N = 20 // each decrypt is ~50-100 ms, keep total runtime sane

	ids := make([][]byte, N)
	for i := range ids {
		var err error
		ids[i], err = committee.BeginInstance()
		if err != nil {
			t.Fatalf("BeginInstance #%d: %v", i, err)
		}
	}

	var successes int64
	var failures int64
	outs := make([]*rlwe.Ciphertext, N)

	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := committee.ThresholdDecrypt(fixture, session.PK, ids[i])
			if err != nil {
				atomic.AddInt64(&failures, 1)
				t.Errorf("decrypt #%d: %v", i, err)
				return
			}
			atomic.AddInt64(&successes, 1)
			outs[i] = out
		}()
	}
	wg.Wait()

	// CKKS encoder is not safe for concurrent use; decode results serially.
	for i, out := range outs {
		if out == nil {
			continue
		}
		got, derr := session.DecryptScalar(out)
		if derr != nil {
			t.Errorf("DecryptScalar #%d: %v", i, derr)
			continue
		}
		if !nearlyEqual(got, want, 0.02) {
			t.Errorf("decrypt #%d value: got %.4f, want %.4f", i, got, want)
		}
	}

	if successes != N {
		t.Errorf("successes: got %d, want %d (all distinct IDs should succeed)", successes, N)
	}
	if failures != 0 {
		t.Errorf("failures: got %d, want 0", failures)
	}
	// Each participant should have admitted exactly N fingerprints (one per
	// distinct instanceID × the one ciphertext).
	for i, p := range committee.Participants[:committee.Threshold] {
		if got := p.guard.Size(); got != N {
			t.Errorf("participant %d: guard.Size() = %d, want %d", i, got, N)
		}
	}
}

// TestChaos_AbortMidFlightBlocksRetry simulates the coordinator deciding to
// abort an instance (e.g., upstream timeout) AFTER calling BeginInstance but
// BEFORE invoking ThresholdDecrypt. The subsequent decrypt with the same ID
// must fail with ErrInstanceAborted; a retry must use a fresh ID.
func TestChaos_AbortMidFlightBlocksRetry(t *testing.T) {
	committee, session, fixture, want := makeChaosFixture(t, 3, 2)
	defer session.Close()

	// Coordinator begins an instance then changes its mind.
	id, err := committee.BeginInstance()
	if err != nil {
		t.Fatalf("BeginInstance: %v", err)
	}
	committee.AbortInstance(id)

	// Any subsequent decrypt with the aborted id must fail.
	_, err = committee.ThresholdDecrypt(fixture, session.PK, id)
	if !errors.Is(err, ErrInstanceAborted) {
		t.Errorf("retry on aborted id: got %v, want ErrInstanceAborted", err)
	}

	// A fresh id, meanwhile, succeeds.
	id2, err := committee.BeginInstance()
	if err != nil {
		t.Fatalf("BeginInstance #2: %v", err)
	}
	out, err := committee.ThresholdDecrypt(fixture, session.PK, id2)
	if err != nil {
		t.Fatalf("decrypt with fresh id: %v", err)
	}
	got, err := session.DecryptScalar(out)
	if err != nil {
		t.Fatalf("DecryptScalar: %v", err)
	}
	if !nearlyEqual(got, want, 0.02) {
		t.Errorf("decrypt value: got %.4f, want %.4f", got, want)
	}
}

// TestChaos_GuardCatchesEvenIfCoordinatorBypassed is the defense-in-depth
// proof: directly bypass the Committee.tracker (simulating a malicious
// coordinator that ignores its own state machine) and call into the
// participants' RetryGuard with the same fingerprint twice. The second call
// MUST be refused by the per-participant guard. This is the Phase 2 piece
// (per-emission guard), exercised under the Phase 3c chaos framing.
func TestChaos_GuardCatchesEvenIfCoordinatorBypassed(t *testing.T) {
	guard := NewRetryGuard()
	instanceID := mustInstanceID()
	ct := []byte("ciphertext-bytes")
	pk := []byte("client-pk-bytes")

	if err := guard.Admit(instanceID, ct, pk); err != nil {
		t.Fatalf("first Admit: %v", err)
	}
	if err := guard.Admit(instanceID, ct, pk); !errors.Is(err, ErrShareAlreadyEmitted) {
		t.Errorf("second Admit with same fingerprint: got %v, want ErrShareAlreadyEmitted", err)
	}
	if got := guard.Size(); got != 1 {
		t.Errorf("guard.Size() after replay attempt: got %d, want 1", got)
	}
}

// TestChaos_ConcurrentGuardAdmitOneWins hammers a SINGLE guard from N
// goroutines with the same fingerprint. Exactly one should succeed; all
// others should be refused. This is the participant-level analogue of
// TestChaos_ConcurrentRetrySameID — the per-participant guard must hold
// even under racy concurrent emission attempts.
func TestChaos_ConcurrentGuardAdmitOneWins(t *testing.T) {
	guard := NewRetryGuard()
	instanceID := mustInstanceID()
	ct := []byte("ciphertext-bytes")
	pk := []byte("client-pk-bytes")

	const N = 200
	var admits int64
	var refusals int64

	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := guard.Admit(instanceID, ct, pk); err == nil {
				atomic.AddInt64(&admits, 1)
			} else if errors.Is(err, ErrShareAlreadyEmitted) {
				atomic.AddInt64(&refusals, 1)
			}
		}()
	}
	wg.Wait()

	if admits != 1 {
		t.Errorf("admits: got %d, want exactly 1", admits)
	}
	if refusals != N-1 {
		t.Errorf("refusals: got %d, want %d", refusals, N-1)
	}
}

// TestChaos_DifferentFingerprintsAllAdmitted is the guard's positive case
// under chaos: each distinct (instanceID, ct, pk) triple gets through. The
// guard must not over-refuse on unrelated fingerprints.
func TestChaos_DifferentFingerprintsAllAdmitted(t *testing.T) {
	guard := NewRetryGuard()
	const N = 200
	var admits int64
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := mustInstanceID()
			ct := []byte{byte(i), byte(i >> 8)}
			pk := []byte("pk-shared")
			if err := guard.Admit(id, ct, pk); err == nil {
				atomic.AddInt64(&admits, 1)
			} else {
				t.Errorf("unexpected refusal for fresh fingerprint: %v", err)
			}
		}(i)
	}
	wg.Wait()
	if admits != N {
		t.Errorf("admits: got %d, want %d (all distinct fingerprints)", admits, N)
	}
	if got := guard.Size(); got != N {
		t.Errorf("guard.Size(): got %d, want %d", got, N)
	}
}
