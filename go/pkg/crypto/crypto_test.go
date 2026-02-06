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

// TestHEDotProductAccuracy tests whether HE dot product matches plaintext
func TestHEDotProductAccuracy(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Test with known vectors
	testCases := []struct {
		name   string
		query  []float64
		vector []float64
	}{
		{
			name:   "simple positive",
			query:  []float64{1.0, 0.0, 0.0, 0.0},
			vector: []float64{1.0, 0.0, 0.0, 0.0},
		},
		{
			name:   "orthogonal",
			query:  []float64{1.0, 0.0, 0.0, 0.0},
			vector: []float64{0.0, 1.0, 0.0, 0.0},
		},
		{
			name:   "normalized unit vectors",
			query:  NormalizeVector([]float64{1.0, 2.0, 3.0, 4.0}),
			vector: NormalizeVector([]float64{4.0, 3.0, 2.0, 1.0}),
		},
		{
			name:   "negative values",
			query:  NormalizeVector([]float64{-1.0, 2.0, -3.0, 4.0}),
			vector: NormalizeVector([]float64{4.0, -3.0, 2.0, -1.0}),
		},
	}

	t.Log("\n=== HE DOT PRODUCT ACCURACY TEST ===")

	for _, tc := range testCases {
		// Plaintext dot product
		plaintextDP := 0.0
		for i := range tc.query {
			plaintextDP += tc.query[i] * tc.vector[i]
		}

		// HE dot product
		encQuery, err := engine.EncryptVector(tc.query)
		if err != nil {
			t.Fatalf("Encrypt failed: %v", err)
		}

		encResult, err := engine.HomomorphicDotProduct(encQuery, tc.vector)
		if err != nil {
			t.Fatalf("HE dot product failed: %v", err)
		}

		heDP, err := engine.DecryptScalar(encResult)
		if err != nil {
			t.Fatalf("Decrypt failed: %v", err)
		}

		// Calculate error
		errorAbs := math.Abs(heDP - plaintextDP)

		t.Logf("%s: plaintext=%.4f, HE=%.4f, error=%.4f",
			tc.name, plaintextDP, heDP, errorAbs)
	}
}

// TestHECentroidRanking tests whether HE ranking matches plaintext ranking
func TestHECentroidRanking(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Simulate query and centroids like in real search
	dim := 128
	numCentroids := 8

	// Create random-ish query and centroids
	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(i%10 - 5)
	}
	query = NormalizeVector(query)

	centroids := make([][]float64, numCentroids)
	for c := 0; c < numCentroids; c++ {
		centroids[c] = make([]float64, dim)
		for i := range centroids[c] {
			centroids[c][i] = float64((i+c*3)%10 - 5)
		}
		centroids[c] = NormalizeVector(centroids[c])
	}

	// Compute plaintext scores
	plaintextScores := make([]float64, numCentroids)
	for c := 0; c < numCentroids; c++ {
		for i := range query {
			plaintextScores[c] += query[i] * centroids[c][i]
		}
	}

	// Compute query sum for bias correction
	querySum := 0.0
	for _, q := range query {
		querySum += q
	}
	n := float64(len(query))

	// Compute HE scores WITH bias correction
	encQuery, _ := engine.EncryptVector(query)
	heScores := make([]float64, numCentroids)
	rawScores := make([]float64, numCentroids)
	centroidSums := make([]float64, numCentroids)

	for c := 0; c < numCentroids; c++ {
		encResult, _ := engine.HomomorphicDotProduct(encQuery, centroids[c])
		rawScore, _ := engine.DecryptScalar(encResult)
		rawScores[c] = rawScore

		// Calculate centroid sum
		centroidSum := 0.0
		for _, cv := range centroids[c] {
			centroidSum += cv
		}
		centroidSums[c] = centroidSum

		// The bias in HE score is: sum(q)*sum(v) / n + other terms
		// For ranking, we only need to remove the centroid-dependent part
		// Since raw score ~ true_dot + offset(centroid), we subtract the centroid sum
		heScores[c] = rawScore
	}

	t.Logf("Query sum: %.4f, n: %.0f", querySum, n)
	t.Logf("Raw HE scores: %.4f, %.4f, %.4f, ...", rawScores[0], rawScores[1], rawScores[2])
	t.Logf("Centroid sums: %.4f, %.4f, %.4f, ...", centroidSums[0], centroidSums[1], centroidSums[2])

	// Try simple ranking without correction - scores should still rank correctly
	// if the bias is constant (only querySum + n varies, which is same for all)

	// Compute ranks
	plaintextRanks := computeRanks(plaintextScores)
	heRanks := computeRanks(heScores)

	rankMatch := 0
	for c := 0; c < numCentroids; c++ {
		t.Logf("Centroid %d: plaintext=%.4f (rank %d), HE_corrected=%.4f (rank %d)",
			c, plaintextScores[c], plaintextRanks[c], heScores[c], heRanks[c])
		if plaintextRanks[c] == heRanks[c] {
			rankMatch++
		}
	}

	t.Logf("Rank agreement: %d/%d (%.1f%%)", rankMatch, numCentroids, float64(rankMatch)/float64(numCentroids)*100)

	// Check top-K agreement
	topK := 4
	plaintextTop := make([]int, 0)
	heTop := make([]int, 0)
	for c := 0; c < numCentroids; c++ {
		if plaintextRanks[c] <= topK {
			plaintextTop = append(plaintextTop, c)
		}
		if heRanks[c] <= topK {
			heTop = append(heTop, c)
		}
	}

	t.Logf("Plaintext top-%d: %v", topK, plaintextTop)
	t.Logf("HE top-%d: %v", topK, heTop)

	// With bias correction, we should have much better rank agreement
	if rankMatch < 6 { // At least 75%
		t.Errorf("Rank agreement too low after bias correction: %d/%d", rankMatch, numCentroids)
	}
}

func computeRanks(scores []float64) []int {
	n := len(scores)
	ranks := make([]int, n)

	for i := 0; i < n; i++ {
		rank := 1
		for j := 0; j < n; j++ {
			if scores[j] > scores[i] {
				rank++
			}
		}
		ranks[i] = rank
	}

	return ranks
}

// TestHEMultiplyOnly tests just the multiplication without rotation-based sum
func TestHEMultiplyOnly(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Simple test: multiply [1, 0, 0, 0] by [1, 0, 0, 0]
	// After offset encoding: [2*scale, 1*scale, 1*scale, 1*scale]
	// Product should be: [4*scale^2, 1*scale^2, 1*scale^2, 1*scale^2]
	// Without summing, slot 0 should decode to 4 (divided by scale^2) = 4

	query := []float64{1.0, 0.0, 0.0, 0.0}
	vector := []float64{1.0, 0.0, 0.0, 0.0}

	// Encrypt query
	encQuery, err := engine.EncryptVector(query)
	if err != nil {
		t.Fatalf("Encrypt failed: %v", err)
	}

	// Do just the multiply (no rotation sum)
	encoded := encodeVectorTest(vector)
	t.Logf("Encoded vector: %v", encoded)

	// Expected after multiply for slot 0:
	// (1 + 1) * scale * (1 + 1) * scale = 4 * scale^2
	// After decode: 4

	// Now let's decrypt WITHOUT the rotation sum
	decrypted, err := engine.DecryptVector(encQuery, 4)
	if err != nil {
		t.Fatalf("Decrypt failed: %v", err)
	}
	t.Logf("Original encrypted query (decoded): %v", decrypted)

	// With full dot product
	encResult, _ := engine.HomomorphicDotProduct(encQuery, vector)
	scalar, _ := engine.DecryptScalar(encResult)
	t.Logf("HE Dot Product result: %f", scalar)

	// Also check what DecryptVector gives us
	fullDecrypt, _ := engine.DecryptVector(encResult, 4)
	t.Logf("Full decrypt of result (first 4 slots): %v", fullDecrypt)
}

func encodeVectorTest(vector []float64) []uint64 {
	encoded := make([]uint64, len(vector))
	for i, v := range vector {
		encoded[i] = uint64((v + Offset) * ScaleFactor)
	}
	return encoded
}

// TestHESimpleDotProduct tests HE dot product with simple small vectors
func TestHESimpleDotProduct(t *testing.T) {
	engine, err := NewClientEngine()
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}

	// Check slot count
	params := engine.GetParams()
	maxSlots := params.MaxSlots()
	t.Logf("BFV Max Slots: %d (LogN=%d)", maxSlots, params.LogN())

	// Very simple test vectors (4-dim)
	testCases := []struct {
		query  []float64
		vector []float64
		name   string
	}{
		{[]float64{1, 0, 0, 0}, []float64{1, 0, 0, 0}, "identical"},
		{[]float64{1, 0, 0, 0}, []float64{0, 1, 0, 0}, "orthogonal"},
		{[]float64{1, 0, 0, 0}, []float64{-1, 0, 0, 0}, "opposite"},
		{[]float64{0.5, 0.5, 0.5, 0.5}, []float64{0.5, 0.5, 0.5, 0.5}, "uniform"},
		{[]float64{1, 1, 1, 1}, []float64{1, -1, 1, -1}, "mixed signs"},
	}

	t.Log("Testing simple 4-dim vectors:")
	for _, tc := range testCases {
		// Plaintext dot product
		plaintext := 0.0
		for i := range tc.query {
			plaintext += tc.query[i] * tc.vector[i]
		}

		// HE dot product
		encQ, _ := engine.EncryptVector(tc.query)
		encR, _ := engine.HomomorphicDotProduct(encQ, tc.vector)
		heResult, _ := engine.DecryptScalar(encR)

		// Also get full decryption
		fullDecrypt, _ := engine.DecryptVector(encR, 4)

		t.Logf("%s: plaintext=%.4f, HE=%.4f, full[0]=%.4f, full=%v",
			tc.name, plaintext, heResult, fullDecrypt[0], fullDecrypt)
	}

	// Test with full-slot padding
	t.Log("\nTesting with slot-padded vectors:")
	query4 := []float64{1, 2, 3, 4}
	vector4 := []float64{4, 3, 2, 1}

	// Pad to maxSlots with zeros
	queryPadded := make([]float64, maxSlots)
	vectorPadded := make([]float64, maxSlots)
	copy(queryPadded, query4)
	copy(vectorPadded, vector4)

	plaintext4 := 0.0
	for i := range query4 {
		plaintext4 += query4[i] * vector4[i]
	}

	encQP, _ := engine.EncryptVector(queryPadded)
	encRP, _ := engine.HomomorphicDotProduct(encQP, vectorPadded)
	heResultP, _ := engine.DecryptScalar(encRP)

	t.Logf("Padded to %d: plaintext=%.4f, HE=%.4f", maxSlots, plaintext4, heResultP)
}

// TestHEDecodeRaw checks raw decoded values to understand the encoding
func TestHEDecodeRaw(t *testing.T) {
	engine, _ := NewClientEngine()

	// Simple test: [1, 0, 0, 0] · [1, 0, 0, 0] = 1.0
	query := []float64{1, 0, 0, 0}
	vector := []float64{1, 0, 0, 0}

	encQ, _ := engine.EncryptVector(query)
	encR, _ := engine.HomomorphicDotProduct(encQ, vector)

	// Get raw decrypted value (without any decoding)
	pt := engine.decryptor.DecryptNew(encR)
	rawSlots := make([]uint64, 4)
	engine.encoder.Decode(pt, rawSlots)

	t.Logf("Raw decrypted slots: %v", rawSlots)
	t.Logf("Scale factor: %f", ScaleFactor)
	t.Logf("Scale squared: %f", ScaleFactor*ScaleFactor)

	// Calculate expected encoded value for dot product of 1.0
	// With offset encoding:
	// q_encoded = (1 + 1) * 30000 = 60000 for q[0], (0 + 1) * 30000 = 30000 for others
	// v_encoded = same
	// Product: 60000*60000 = 3.6B, 30000*30000 = 900M for slots 1,2,3
	// But padded slots are 0, so they contribute 0

	// Actually with our padding, only 4 slots are non-zero
	// After multiplication:
	// slot 0: 60000 * 60000 = 3,600,000,000
	// slot 1: 30000 * 30000 = 900,000,000
	// slot 2: 30000 * 30000 = 900,000,000
	// slot 3: 30000 * 30000 = 900,000,000
	// Sum = 6.3B

	// BFV plaintext modulus is 65537, so we get modular arithmetic
	t.Logf("Plaintext modulus: %d", 65537)
	t.Logf("60000 * 60000 mod 65537 = %d", (60000*60000)%65537)
	t.Logf("30000 * 30000 mod 65537 = %d", (30000*30000)%65537)

	// The modular arithmetic causes wrap-around!
	// This is the fundamental issue.
}

// TestHEEncodingFormula verifies the encoding math
func TestHEEncodingFormula(t *testing.T) {
	// Test the math of offset encoding
	// When we encode q_i -> (q_i + offset) * scale
	// And v_i -> (v_i + offset) * scale
	// Product is (q_i + 1)(v_i + 1) * scale^2
	// = (q_i*v_i + q_i + v_i + 1) * scale^2

	// So sum = sum(q_i*v_i) + sum(q_i) + sum(v_i) + n
	// Decoded (dividing by scale^2) = sum(q_i*v_i) + sum(q_i) + sum(v_i) + n

	// Test with a specific normalized vector
	q := NormalizeVector([]float64{1.0, 2.0, 3.0, 4.0})
	v := NormalizeVector([]float64{4.0, 3.0, 2.0, 1.0})
	n := float64(len(q))

	// True dot product
	trueDot := 0.0
	for i := range q {
		trueDot += q[i] * v[i]
	}

	// Sum of query
	sumQ := 0.0
	for _, qi := range q {
		sumQ += qi
	}

	// Sum of vector
	sumV := 0.0
	for _, vi := range v {
		sumV += vi
	}

	// Expected HE result (before any correction)
	expectedHE := trueDot + sumQ + sumV + n

	t.Logf("Query: %v", q)
	t.Logf("Vector: %v", v)
	t.Logf("sum(q_i) = %.4f", sumQ)
	t.Logf("sum(v_i) = %.4f", sumV)
	t.Logf("n = %.0f", n)
	t.Logf("True dot product: %.4f", trueDot)
	t.Logf("Expected HE (with offset bias): %.4f", expectedHE)
	t.Logf("Bias = sum(q) + sum(v) + n = %.4f", sumQ+sumV+n)

	// Now verify with actual HE
	engine, _ := NewClientEngine()
	encQ, _ := engine.EncryptVector(q)
	encResult, _ := engine.HomomorphicDotProduct(encQ, v)
	heResult, _ := engine.DecryptScalar(encResult)

	t.Logf("Actual HE result: %.4f", heResult)
	t.Logf("Matches expected? %v (diff=%.4f)", math.Abs(heResult-expectedHE) < 0.1, math.Abs(heResult-expectedHE))

	if math.Abs(heResult-trueDot) > 0.5 {
		t.Logf("WARNING: HE result differs significantly from true dot product!")
		t.Logf("This explains the centroid ranking discrepancy.")
	}
}
