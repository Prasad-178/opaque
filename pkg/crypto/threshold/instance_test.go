package threshold

import (
	"bytes"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// nearlyEqual is shared with chaos_test.go.
func nearlyEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// ============================================================================
// Phase 3b — coordinator state-machine tests (instanceTracker isolated).
// ============================================================================

func TestInstanceTracker_RejectsEmptyID(t *testing.T) {
	tr := newInstanceTracker()
	if err := tr.activate(nil); !errors.Is(err, ErrInstanceEmpty) {
		t.Errorf("activate(nil): got %v, want ErrInstanceEmpty", err)
	}
	if err := tr.activate([]byte{}); !errors.Is(err, ErrInstanceEmpty) {
		t.Errorf("activate(empty): got %v, want ErrInstanceEmpty", err)
	}
}

func TestInstanceTracker_HappyPath(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	if got := tr.peek(id); got != stateUnknown {
		t.Errorf("initial state: got %d, want %d (unknown)", got, stateUnknown)
	}
	if err := tr.activate(id); err != nil {
		t.Fatalf("first activate: %v", err)
	}
	if got := tr.peek(id); got != stateRunning {
		t.Errorf("post-activate state: got %d, want %d (running)", got, stateRunning)
	}
	tr.commit(id)
	if got := tr.peek(id); got != stateCommitted {
		t.Errorf("post-commit state: got %d, want %d (committed)", got, stateCommitted)
	}
}

func TestInstanceTracker_RefusesReuseAfterCommit(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	if err := tr.activate(id); err != nil {
		t.Fatalf("activate: %v", err)
	}
	tr.commit(id)

	if err := tr.activate(id); !errors.Is(err, ErrInstanceCommitted) {
		t.Errorf("activate after commit: got %v, want ErrInstanceCommitted", err)
	}
}

func TestInstanceTracker_RefusesReuseAfterAbort(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	if err := tr.activate(id); err != nil {
		t.Fatalf("activate: %v", err)
	}
	tr.abort(id)

	if err := tr.activate(id); !errors.Is(err, ErrInstanceAborted) {
		t.Errorf("activate after abort: got %v, want ErrInstanceAborted", err)
	}
}

func TestInstanceTracker_AbortBeforeActivate(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	// Abort an instance that was never activated (e.g., coordinator decides
	// to refuse before initiating decrypt).
	tr.abort(id)
	if got := tr.peek(id); got != stateAborted {
		t.Errorf("post-abort-from-unknown state: got %d, want %d (aborted)", got, stateAborted)
	}
	if err := tr.activate(id); !errors.Is(err, ErrInstanceAborted) {
		t.Errorf("activate after pre-activation abort: got %v, want ErrInstanceAborted", err)
	}
}

func TestInstanceTracker_IdempotentTerminalTransitions(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	if err := tr.activate(id); err != nil {
		t.Fatalf("activate: %v", err)
	}
	tr.commit(id)
	// commit-twice no-op
	tr.commit(id)
	if got := tr.peek(id); got != stateCommitted {
		t.Errorf("double-commit changed state: got %d", got)
	}
	// abort-after-commit no-op (committed wins)
	tr.abort(id)
	if got := tr.peek(id); got != stateCommitted {
		t.Errorf("abort-after-commit changed committed state: got %d", got)
	}
}

func TestInstanceTracker_ConcurrentActivateExactlyOneWins(t *testing.T) {
	tr := newInstanceTracker()
	id := []byte("test-instance-1")

	const N = 200
	var winners int64
	var concurrentRefusals int64
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := tr.activate(id); err == nil {
				atomic.AddInt64(&winners, 1)
			} else if errors.Is(err, ErrInstanceConcurrent) {
				atomic.AddInt64(&concurrentRefusals, 1)
			}
		}()
	}
	wg.Wait()

	if winners != 1 {
		t.Errorf("concurrent activate winners: got %d, want exactly 1", winners)
	}
	if concurrentRefusals != N-1 {
		t.Errorf("concurrent activate refusals: got %d, want %d", concurrentRefusals, N-1)
	}
}

func TestInstanceTracker_DistinctIDsIndependent(t *testing.T) {
	tr := newInstanceTracker()
	a := []byte("instance-A")
	b := []byte("instance-B")

	if err := tr.activate(a); err != nil {
		t.Fatalf("activate A: %v", err)
	}
	if err := tr.activate(b); err != nil {
		t.Fatalf("activate B: %v", err)
	}
	tr.commit(a)
	tr.abort(b)

	if got := tr.peek(a); got != stateCommitted {
		t.Errorf("A state: got %d, want committed", got)
	}
	if got := tr.peek(b); got != stateAborted {
		t.Errorf("B state: got %d, want aborted", got)
	}
}

// ============================================================================
// Phase 3b — Committee-level integration tests.
// ============================================================================

// TestCommittee_BeginInstanceMintsUniqueIDs covers the BeginInstance happy path
// and confirms the tracker is wired into the Committee.
func TestCommittee_BeginInstanceMintsUniqueIDs(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}

	const N = 100
	ids := make(map[string]struct{}, N)
	for i := 0; i < N; i++ {
		id, err := committee.BeginInstance()
		if err != nil {
			t.Fatalf("BeginInstance #%d: %v", i, err)
		}
		if len(id) != 16 {
			t.Errorf("BeginInstance id length: got %d, want 16", len(id))
		}
		key := string(id)
		if _, dup := ids[key]; dup {
			t.Errorf("BeginInstance returned duplicate id at iter %d", i)
		}
		ids[key] = struct{}{}
		// Each ID lands in stateReserved — a second reserve should fail
		// with ErrInstanceConcurrent (only one reservation per ID).
		if err := committee.tracker.reserve(id); !errors.Is(err, ErrInstanceConcurrent) {
			t.Errorf("post-BeginInstance second reserve: got %v, want ErrInstanceConcurrent", err)
		}
		// But ThresholdDecrypt with this id WOULD succeed (reserved → running).
		// We don't exercise it here to keep the test scoped — covered by
		// TestCommittee_ThresholdDecryptRefusesInstanceReuse.
	}
}

// TestCommittee_ThresholdDecryptRefusesInstanceReuse runs a real decrypt and
// asserts that re-using the instanceID afterwards is refused at the
// coordinator level (Phase 3b) — even before the participant-level
// RetryGuard would catch a duplicate fingerprint.
func TestCommittee_ThresholdDecryptRefusesInstanceReuse(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}

	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("NewClientSession: %v", err)
	}
	defer session.Close()

	// Build a real ciphertext to decrypt.
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
		t.Fatalf("Encode: %v", err)
	}
	result, err := eval.MulNew(ct, ptCentroid)
	if err != nil {
		t.Fatalf("MulNew: %v", err)
	}
	if err := eval.Rescale(result, result); err != nil {
		t.Fatalf("Rescale: %v", err)
	}

	instanceID := mustInstanceID()

	// First decrypt succeeds.
	out, err := committee.ThresholdDecrypt(result, session.PK, instanceID)
	if err != nil {
		t.Fatalf("first ThresholdDecrypt: %v", err)
	}
	if out == nil {
		t.Fatal("first ThresholdDecrypt returned nil ciphertext")
	}
	got, err := session.DecryptScalar(out)
	if err != nil {
		t.Fatalf("DecryptScalar: %v", err)
	}
	const want = 0.5 * 0.4
	if !nearlyEqual(got, want, 0.02) {
		t.Errorf("first decrypt value: got %.4f, want %.4f", got, want)
	}

	// Second decrypt with the SAME instanceID must be refused at the
	// coordinator BEFORE reaching the participants. This is the Phase 3b
	// guarantee — defense in depth on top of the per-participant guard.
	_, err = committee.ThresholdDecrypt(result, session.PK, instanceID)
	if !errors.Is(err, ErrInstanceCommitted) {
		t.Errorf("second ThresholdDecrypt with reused id: got %v, want ErrInstanceCommitted", err)
	}
}

// TestCommittee_AbortInstanceBlocksRetry verifies that an explicit
// AbortInstance call from the coordinator (e.g., upstream timeout detection)
// permanently blocks any subsequent ThresholdDecrypt with the same ID.
func TestCommittee_AbortInstanceBlocksRetry(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}

	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("NewClientSession: %v", err)
	}
	defer session.Close()

	query := make([]float64, 128)
	query[0] = 0.5
	ct, err := committee.EncryptVector(query)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	id, err := committee.BeginInstance()
	if err != nil {
		t.Fatalf("BeginInstance: %v", err)
	}
	// Simulate the coordinator deciding to abort (e.g., upstream timeout).
	committee.AbortInstance(id)

	_, err = committee.ThresholdDecrypt(ct, session.PK, id)
	if !errors.Is(err, ErrInstanceAborted) {
		t.Errorf("ThresholdDecrypt after AbortInstance: got %v, want ErrInstanceAborted", err)
	}
}

// TestCommittee_EmptyInstanceIDRejected guards against the empty-ID footgun.
func TestCommittee_EmptyInstanceIDRejected(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}
	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("NewClientSession: %v", err)
	}
	defer session.Close()

	query := make([]float64, 128)
	query[0] = 0.5
	ct, err := committee.EncryptVector(query)
	if err != nil {
		t.Fatalf("EncryptVector: %v", err)
	}

	_, err = committee.ThresholdDecrypt(ct, session.PK, nil)
	if !errors.Is(err, ErrInstanceEmpty) {
		t.Errorf("ThresholdDecrypt with nil id: got %v, want ErrInstanceEmpty", err)
	}
	_, err = committee.ThresholdDecrypt(ct, session.PK, []byte{})
	if !errors.Is(err, ErrInstanceEmpty) {
		t.Errorf("ThresholdDecrypt with empty id: got %v, want ErrInstanceEmpty", err)
	}
}

// TestCommittee_InstanceIDsDistinctNoCollision verifies the tracker stores
// each ID separately — no collision in the base64 encoding.
func TestCommittee_InstanceIDsDistinctNoCollision(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee: %v", err)
	}

	ids := make([][]byte, 50)
	for i := range ids {
		id, err := committee.BeginInstance()
		if err != nil {
			t.Fatalf("BeginInstance #%d: %v", i, err)
		}
		ids[i] = id
	}
	// Verify all 50 IDs are pairwise distinct as byte slices.
	for i := range ids {
		for j := i + 1; j < len(ids); j++ {
			if bytes.Equal(ids[i], ids[j]) {
				t.Errorf("ids[%d] == ids[%d] = %x", i, j, ids[i])
			}
		}
	}
	if got := committee.tracker.size(); got != 50 {
		t.Errorf("tracker size after 50 BeginInstance: got %d, want 50", got)
	}
}
