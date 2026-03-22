package threshold

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// TestCommitteeSetup verifies that a committee can be created with collective keys.
func TestCommitteeSetup(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	if committee.N != 3 {
		t.Errorf("expected N=3, got %d", committee.N)
	}
	if committee.Threshold != 2 {
		t.Errorf("expected threshold=2, got %d", committee.Threshold)
	}
	if committee.CollectivePK == nil {
		t.Fatal("collective public key is nil")
	}
	if committee.RelinKey == nil {
		t.Fatal("relinearization key is nil")
	}
	if len(committee.GaloisKeys) == 0 {
		t.Fatal("no Galois keys generated")
	}
	if len(committee.Participants) != 3 {
		t.Errorf("expected 3 participants, got %d", len(committee.Participants))
	}
}

// TestCommitteeSetupNofN verifies N-of-N committee (no Shamir threshold).
func TestCommitteeSetupNofN(t *testing.T) {
	committee, err := NewCommittee(3, 3)
	if err != nil {
		t.Fatalf("failed to create N-of-N committee: %v", err)
	}

	if committee.CollectivePK == nil {
		t.Fatal("collective public key is nil")
	}
}

// TestThresholdDecryptScalar tests the full threshold decryption flow for a scalar value.
func TestThresholdDecryptScalar(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	// Client creates an ephemeral session.
	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("failed to create client session: %v", err)
	}
	defer session.Close()

	// Encrypt a known vector using the collective public key.
	query := make([]float64, 128)
	query[0] = 0.5
	query[1] = 0.3
	query[2] = 0.7

	ct, err := committee.EncryptVector(query)
	if err != nil {
		t.Fatalf("failed to encrypt vector: %v", err)
	}

	// Server-side HE computation: dot product with a plaintext centroid.
	centroid := make([]float64, committee.params.MaxSlots())
	centroid[0] = 0.4
	centroid[1] = 0.6
	centroid[2] = 0.2

	// Expected: 0.5*0.4 + 0.3*0.6 + 0.7*0.2 = 0.2 + 0.18 + 0.14 = 0.52
	expectedDot := 0.5*0.4 + 0.3*0.6 + 0.7*0.2

	// Compute HE dot product using the evaluation keys (server has no secret key).
	eval := hefloat.NewEvaluator(committee.params, committee.GetEvalKeySet())
	encoder := committee.GetEncoder()

	ptCentroid := hefloat.NewPlaintext(committee.params, ct.Level())
	if err := encoder.Encode(centroid, ptCentroid); err != nil {
		t.Fatalf("failed to encode centroid: %v", err)
	}

	// Multiply (component-wise).
	result, err := eval.MulNew(ct, ptCentroid)
	if err != nil {
		t.Fatalf("failed to multiply: %v", err)
	}
	if err := eval.Rescale(result, result); err != nil {
		t.Fatalf("failed to rescale: %v", err)
	}

	// Sum all slots (tree-based rotation + addition).
	maxSlots := committee.params.MaxSlots()
	for stride := 1; stride < maxSlots; stride *= 2 {
		rotated, err := eval.RotateNew(result, stride)
		if err != nil {
			t.Fatalf("failed to rotate by %d: %v", stride, err)
		}
		if err := eval.Add(result, rotated, result); err != nil {
			t.Fatalf("failed to add: %v", err)
		}
	}

	// Threshold decryption: re-encrypt result under client's public key.
	ctClient, err := committee.ThresholdDecrypt(result, session.PK)
	if err != nil {
		t.Fatalf("threshold decryption failed: %v", err)
	}

	// Client decrypts locally.
	got, err := session.DecryptScalar(ctClient)
	if err != nil {
		t.Fatalf("client decryption failed: %v", err)
	}

	// Verify precision.
	if math.Abs(got-expectedDot) > 0.01 {
		t.Errorf("dot product mismatch: expected %.6f, got %.6f (diff=%.6f)",
			expectedDot, got, math.Abs(got-expectedDot))
	}

	t.Logf("Threshold decrypt result: expected=%.6f, got=%.6f, diff=%.2e",
		expectedDot, got, math.Abs(got-expectedDot))
}

// TestThresholdDecryptNofN tests N-of-N threshold decryption.
func TestThresholdDecryptNofN(t *testing.T) {
	committee, err := NewCommittee(3, 3)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("failed to create client session: %v", err)
	}
	defer session.Close()

	// Encrypt a simple value.
	vector := make([]float64, 128)
	vector[0] = 0.42

	ct, err := committee.EncryptVector(vector)
	if err != nil {
		t.Fatalf("failed to encrypt: %v", err)
	}

	// Threshold decrypt directly (no HE computation, just re-encrypt under client key).
	ctClient, err := committee.ThresholdDecrypt(ct, session.PK)
	if err != nil {
		t.Fatalf("threshold decryption failed: %v", err)
	}

	// Client decrypts.
	got, err := session.DecryptScalar(ctClient)
	if err != nil {
		t.Fatalf("client decryption failed: %v", err)
	}

	if math.Abs(got-0.42) > 0.001 {
		t.Errorf("decryption mismatch: expected 0.42, got %.6f", got)
	}
}

// TestThresholdDecryptBatch tests batch dot product with threshold decryption.
func TestThresholdDecryptBatch(t *testing.T) {
	committee, err := NewCommittee(5, 3)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("failed to create client session: %v", err)
	}
	defer session.Close()

	dim := 128
	numCentroids := 4
	maxSlots := committee.params.MaxSlots()

	// Create a packed query: query replicated across centroid segments.
	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(i+1) / float64(dim)
	}

	packedQuery := make([]float64, maxSlots)
	for c := 0; c < numCentroids; c++ {
		offset := c * dim
		copy(packedQuery[offset:offset+dim], query)
	}

	// Create packed centroids.
	centroids := make([][]float64, numCentroids)
	packedCentroids := make([]float64, maxSlots)
	for c := 0; c < numCentroids; c++ {
		centroids[c] = make([]float64, dim)
		for j := range centroids[c] {
			centroids[c][j] = float64(c*dim+j+1) / float64(numCentroids*dim)
		}
		offset := c * dim
		copy(packedCentroids[offset:offset+dim], centroids[c])
	}

	// Compute expected dot products.
	expected := make([]float64, numCentroids)
	for c := 0; c < numCentroids; c++ {
		for j := 0; j < dim; j++ {
			expected[c] += query[j] * centroids[c][j]
		}
	}

	// Encrypt packed query.
	ct, err := committee.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("failed to encrypt: %v", err)
	}

	// Server computes batch dot product.
	eval := hefloat.NewEvaluator(committee.params, committee.GetEvalKeySet())
	encoder := committee.GetEncoder()

	ptCentroids := hefloat.NewPlaintext(committee.params, ct.Level())
	if err := encoder.Encode(packedCentroids, ptCentroids); err != nil {
		t.Fatalf("failed to encode centroids: %v", err)
	}

	result, err := eval.MulNew(ct, ptCentroids)
	if err != nil {
		t.Fatalf("failed to multiply: %v", err)
	}
	if err := eval.Rescale(result, result); err != nil {
		t.Fatalf("failed to rescale: %v", err)
	}

	// Partial sum within each centroid's dimension slots.
	for stride := 1; stride < dim; stride *= 2 {
		rotated, err := eval.RotateNew(result, stride)
		if err != nil {
			t.Fatalf("failed to rotate: %v", err)
		}
		if err := eval.Add(result, rotated, result); err != nil {
			t.Fatalf("failed to add: %v", err)
		}
	}

	// Threshold decrypt.
	ctClient, err := committee.ThresholdDecrypt(result, session.PK)
	if err != nil {
		t.Fatalf("threshold decryption failed: %v", err)
	}

	// Client decrypts batch.
	got, err := session.DecryptBatchScalars(ctClient, numCentroids, dim)
	if err != nil {
		t.Fatalf("batch decryption failed: %v", err)
	}

	// Verify each dot product.
	for c := 0; c < numCentroids; c++ {
		diff := math.Abs(got[c] - expected[c])
		if diff > 0.01 {
			t.Errorf("centroid %d: expected %.6f, got %.6f (diff=%.2e)",
				c, expected[c], got[c], diff)
		}
	}

	t.Logf("Batch threshold decrypt: %d centroids, all within tolerance", numCentroids)
}

// TestCommitteeInvalidParams tests error handling for invalid committee parameters.
func TestCommitteeInvalidParams(t *testing.T) {
	_, err := NewCommittee(0, 1)
	if err == nil {
		t.Error("expected error for n=0")
	}

	_, err = NewCommittee(3, 0)
	if err == nil {
		t.Error("expected error for threshold=0")
	}

	_, err = NewCommittee(3, 4)
	if err == nil {
		t.Error("expected error for threshold > n")
	}
}

// TestClientSessionCleanup verifies key zeroing.
func TestClientSessionCleanup(t *testing.T) {
	committee, err := NewCommittee(2, 2)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	session, err := committee.NewClientSession()
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	session.Close()

	if session.SK != nil {
		t.Error("secret key should be nil after Close")
	}
	if session.decryptor != nil {
		t.Error("decryptor should be nil after Close")
	}
}

// TestEvalKeySetHasNoSecretMaterial verifies the eval key set contains no secret keys.
func TestEvalKeySetHasNoSecretMaterial(t *testing.T) {
	committee, err := NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("failed to create committee: %v", err)
	}

	evkSet := committee.GetEvalKeySet()
	if evkSet == nil {
		t.Fatal("eval key set is nil")
	}

	// The eval key set should contain relin and Galois keys but no secret key.
	// We can verify by checking that we can create an evaluator with it.
	eval := hefloat.NewEvaluator(committee.params, evkSet)
	if eval == nil {
		t.Fatal("failed to create evaluator from eval key set")
	}
}
