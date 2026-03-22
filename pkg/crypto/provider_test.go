package crypto_test

import (
	"math"
	"testing"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/Prasad-178/opaque/pkg/crypto/threshold"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// testProviderEncryptDecrypt is a shared test for any HEProvider implementation.
func testProviderEncryptDecrypt(t *testing.T, provider crypto.HEProvider) {
	t.Helper()

	vector := make([]float64, 128)
	vector[0] = 0.5
	vector[1] = 0.3
	vector[2] = 0.7

	ct, err := provider.EncryptVector(vector)
	if err != nil {
		t.Fatalf("EncryptVector failed: %v", err)
	}

	got, err := provider.DecryptScalar(ct)
	if err != nil {
		t.Fatalf("DecryptScalar failed: %v", err)
	}

	// Slot 0 should contain vector[0] = 0.5 (no HE operation, just encrypt/decrypt).
	// Threshold mode adds noise flooding (~30-bit Gaussian), so tolerance is wider.
	if math.Abs(got-0.5) > 0.01 {
		t.Errorf("expected ~0.5, got %.6f", got)
	}
	t.Logf("encrypt/decrypt: expected=0.5, got=%.6f, diff=%.2e", got, math.Abs(got-0.5))
}

// testProviderDotProduct tests HE dot product through Acquire/Release pattern.
func testProviderDotProduct(t *testing.T, provider crypto.HEProvider) {
	t.Helper()

	query := make([]float64, 128)
	query[0] = 0.5
	query[1] = 0.3
	query[2] = 0.7

	centroid := make([]float64, provider.GetParams().MaxSlots())
	centroid[0] = 0.4
	centroid[1] = 0.6
	centroid[2] = 0.2

	// Expected: 0.5*0.4 + 0.3*0.6 + 0.7*0.2 = 0.52
	expected := 0.5*0.4 + 0.3*0.6 + 0.7*0.2

	ct, err := provider.EncryptVector(query)
	if err != nil {
		t.Fatalf("EncryptVector failed: %v", err)
	}

	engine := provider.Acquire()
	defer provider.Release(engine)

	result, err := engine.HomomorphicDotProduct(ct, centroid)
	if err != nil {
		t.Fatalf("HomomorphicDotProduct failed: %v", err)
	}

	got, err := engine.DecryptScalar(result)
	if err != nil {
		t.Fatalf("DecryptScalar failed: %v", err)
	}

	if math.Abs(got-expected) > 0.01 {
		t.Errorf("dot product: expected %.6f, got %.6f (diff=%.2e)", expected, got, math.Abs(got-expected))
	}
	t.Logf("dot product: expected=%.6f, got=%.6f, diff=%.2e", expected, got, math.Abs(got-expected))
}

// testProviderBatchDotProduct tests batch (SIMD) dot product.
func testProviderBatchDotProduct(t *testing.T, provider crypto.HEProvider) {
	t.Helper()

	dim := 128
	numCentroids := 4
	maxSlots := provider.GetParams().MaxSlots()

	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(i+1) / float64(dim)
	}

	packedQuery := make([]float64, maxSlots)
	for c := 0; c < numCentroids; c++ {
		copy(packedQuery[c*dim:(c+1)*dim], query)
	}

	centroids := make([][]float64, numCentroids)
	packedCentroids := make([]float64, maxSlots)
	for c := 0; c < numCentroids; c++ {
		centroids[c] = make([]float64, dim)
		for j := range centroids[c] {
			centroids[c][j] = float64(c*dim+j+1) / float64(numCentroids*dim)
		}
		copy(packedCentroids[c*dim:(c+1)*dim], centroids[c])
	}

	expected := make([]float64, numCentroids)
	for c := 0; c < numCentroids; c++ {
		for j := 0; j < dim; j++ {
			expected[c] += query[j] * centroids[c][j]
		}
	}

	ct, err := provider.EncryptVector(packedQuery)
	if err != nil {
		t.Fatalf("EncryptVector failed: %v", err)
	}

	// Encode packed centroids
	encoder := provider.GetEncoder()
	ptCentroids := hefloat.NewPlaintext(provider.GetParams(), ct.Level())
	if err := encoder.Encode(packedCentroids, ptCentroids); err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	result, err := provider.HomomorphicBatchDotProduct(ct, ptCentroids, numCentroids, dim)
	if err != nil {
		t.Fatalf("HomomorphicBatchDotProduct failed: %v", err)
	}

	got, err := provider.DecryptBatchScalars(result, numCentroids, dim)
	if err != nil {
		t.Fatalf("DecryptBatchScalars failed: %v", err)
	}

	for c := 0; c < numCentroids; c++ {
		diff := math.Abs(got[c] - expected[c])
		// Use relative tolerance for large values; threshold mode adds noise flooding.
		tol := 0.02 + 0.001*math.Abs(expected[c])
		if diff > tol {
			t.Errorf("centroid %d: expected %.6f, got %.6f (diff=%.2e, tol=%.2e)", c, expected[c], got[c], diff, tol)
		}
	}
}

// TestDirectProviderEncryptDecrypt tests the single-key provider.
func TestDirectProviderEncryptDecrypt(t *testing.T) {
	provider, err := crypto.NewDirectHEProvider(2)
	if err != nil {
		t.Fatalf("NewDirectHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderEncryptDecrypt(t, provider)
}

// TestDirectProviderDotProduct tests dot product with single-key provider.
func TestDirectProviderDotProduct(t *testing.T) {
	provider, err := crypto.NewDirectHEProvider(2)
	if err != nil {
		t.Fatalf("NewDirectHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderDotProduct(t, provider)
}

// TestDirectProviderBatchDotProduct tests batch dot product with single-key provider.
func TestDirectProviderBatchDotProduct(t *testing.T) {
	provider, err := crypto.NewDirectHEProvider(2)
	if err != nil {
		t.Fatalf("NewDirectHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderBatchDotProduct(t, provider)
}

// TestThresholdProviderEncryptDecrypt tests the threshold CKKS provider.
func TestThresholdProviderEncryptDecrypt(t *testing.T) {
	committee, err := threshold.NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee failed: %v", err)
	}
	provider, err := crypto.NewThresholdHEProvider(committee, 2)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderEncryptDecrypt(t, provider)
}

// TestThresholdProviderDotProduct tests dot product with threshold CKKS provider.
func TestThresholdProviderDotProduct(t *testing.T) {
	committee, err := threshold.NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee failed: %v", err)
	}
	provider, err := crypto.NewThresholdHEProvider(committee, 2)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderDotProduct(t, provider)
}

// TestThresholdProviderBatchDotProduct tests batch dot product with threshold provider.
func TestThresholdProviderBatchDotProduct(t *testing.T) {
	committee, err := threshold.NewCommittee(5, 3)
	if err != nil {
		t.Fatalf("NewCommittee failed: %v", err)
	}
	provider, err := crypto.NewThresholdHEProvider(committee, 2)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderBatchDotProduct(t, provider)
}

// TestThresholdProviderNofN tests N-of-N threshold mode via the provider.
func TestThresholdProviderNofN(t *testing.T) {
	committee, err := threshold.NewCommittee(3, 3)
	if err != nil {
		t.Fatalf("NewCommittee failed: %v", err)
	}
	provider, err := crypto.NewThresholdHEProvider(committee, 2)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider failed: %v", err)
	}
	defer provider.Close()
	testProviderDotProduct(t, provider)
}

// --- Benchmarks ---

// benchEncryptVector benchmarks encryption for a given provider.
func benchEncryptVector(b *testing.B, provider crypto.HEProvider) {
	b.Helper()
	vec := make([]float64, 128)
	for i := range vec {
		vec[i] = float64(i+1) / 128.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = provider.EncryptVector(vec)
	}
}

// benchDecryptScalar benchmarks scalar decryption for a given provider.
func benchDecryptScalar(b *testing.B, provider crypto.HEProvider) {
	b.Helper()
	vec := make([]float64, 128)
	vec[0] = 0.42
	ct, err := provider.EncryptVector(vec)
	if err != nil {
		b.Fatalf("EncryptVector failed: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = provider.DecryptScalar(ct)
	}
}

// benchDotProduct benchmarks HE dot product for a given provider.
func benchDotProduct(b *testing.B, provider crypto.HEProvider) {
	b.Helper()
	query := make([]float64, 128)
	centroid := make([]float64, provider.GetParams().MaxSlots())
	for i := range query {
		query[i] = float64(i+1) / 128.0
	}
	for i := range centroid {
		centroid[i] = float64(i+1) / float64(len(centroid))
	}

	ct, err := provider.EncryptVector(query)
	if err != nil {
		b.Fatalf("EncryptVector failed: %v", err)
	}
	engine := provider.Acquire()
	defer provider.Release(engine)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = engine.HomomorphicDotProduct(ct, centroid)
	}
}

// benchDotProductAndDecrypt benchmarks the full HE dot product + decrypt cycle.
func benchDotProductAndDecrypt(b *testing.B, provider crypto.HEProvider) {
	b.Helper()
	query := make([]float64, 128)
	centroid := make([]float64, provider.GetParams().MaxSlots())
	for i := range query {
		query[i] = float64(i+1) / 128.0
	}
	for i := range centroid {
		centroid[i] = float64(i+1) / float64(len(centroid))
	}

	ct, err := provider.EncryptVector(query)
	if err != nil {
		b.Fatalf("EncryptVector failed: %v", err)
	}
	engine := provider.Acquire()
	defer provider.Release(engine)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, _ := engine.HomomorphicDotProduct(ct, centroid)
		_, _ = engine.DecryptScalar(result)
	}
}

// Direct mode benchmarks

func BenchmarkDirectEncrypt128D(b *testing.B) {
	p, _ := crypto.NewDirectHEProvider(1)
	defer p.Close()
	benchEncryptVector(b, p)
}

func BenchmarkDirectDecryptScalar(b *testing.B) {
	p, _ := crypto.NewDirectHEProvider(1)
	defer p.Close()
	benchDecryptScalar(b, p)
}

func BenchmarkDirectDotProduct128D(b *testing.B) {
	p, _ := crypto.NewDirectHEProvider(1)
	defer p.Close()
	benchDotProduct(b, p)
}

func BenchmarkDirectDotProductAndDecrypt(b *testing.B) {
	p, _ := crypto.NewDirectHEProvider(1)
	defer p.Close()
	benchDotProductAndDecrypt(b, p)
}

// Threshold mode benchmarks (3-of-5 committee)

func BenchmarkThresholdEncrypt128D(b *testing.B) {
	c, _ := threshold.NewCommittee(5, 3)
	p, _ := crypto.NewThresholdHEProvider(c, 1)
	defer p.Close()
	benchEncryptVector(b, p)
}

func BenchmarkThresholdDecryptScalar(b *testing.B) {
	c, _ := threshold.NewCommittee(5, 3)
	p, _ := crypto.NewThresholdHEProvider(c, 1)
	defer p.Close()
	benchDecryptScalar(b, p)
}

func BenchmarkThresholdDotProduct128D(b *testing.B) {
	c, _ := threshold.NewCommittee(5, 3)
	p, _ := crypto.NewThresholdHEProvider(c, 1)
	defer p.Close()
	benchDotProduct(b, p)
}

func BenchmarkThresholdDotProductAndDecrypt(b *testing.B) {
	c, _ := threshold.NewCommittee(5, 3)
	p, _ := crypto.NewThresholdHEProvider(c, 1)
	defer p.Close()
	benchDotProductAndDecrypt(b, p)
}

// Committee setup benchmark

func BenchmarkCommitteeSetup3of5(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = threshold.NewCommittee(5, 3)
	}
}

func BenchmarkCommitteeSetup2of3(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = threshold.NewCommittee(3, 2)
	}
}

// TestProviderPoolSize verifies pool size is reported correctly.
func TestProviderPoolSize(t *testing.T) {
	direct, err := crypto.NewDirectHEProvider(3)
	if err != nil {
		t.Fatalf("NewDirectHEProvider failed: %v", err)
	}
	defer direct.Close()
	if direct.Size() != 3 {
		t.Errorf("direct: expected size 3, got %d", direct.Size())
	}

	committee, err := threshold.NewCommittee(3, 2)
	if err != nil {
		t.Fatalf("NewCommittee failed: %v", err)
	}
	thresh, err := crypto.NewThresholdHEProvider(committee, 4)
	if err != nil {
		t.Fatalf("NewThresholdHEProvider failed: %v", err)
	}
	defer thresh.Close()
	if thresh.Size() != 4 {
		t.Errorf("threshold: expected size 4, got %d", thresh.Size())
	}
}
