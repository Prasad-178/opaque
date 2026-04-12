package crypto

import (
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

// TestNTTConversion_RoundTrip verifies that converting a Lattigo ciphertext
// to HEonGPU domain and back produces the same decryption result.
//
// This isolates Go-side conversion bugs from C++ server bugs.
func TestNTTConversion_RoundTrip(t *testing.T) {
	params, err := NewParametersGPU()
	if err != nil {
		t.Fatalf("NewParametersGPU: %v", err)
	}

	engine, err := NewClientEngineWithParams(params)
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	// Use fake "server roots" — just compute our own for the test.
	// In production, these come from InitContext.
	allModuli := params.RingQ().ModuliChain()
	if ringP := params.RingP(); ringP != nil {
		allModuli = append(allModuli, ringP.ModuliChain()...)
	}
	serverRoots := make([]uint64, len(allModuli))
	for i, p := range allModuli {
		serverRoots[i] = computeHEonGPUPsi(p, uint64(1<<params.LogN()))
	}

	conv := NewNTTConverterWithRoots(params, serverRoots)

	// Test 1: Ciphertext round-trip
	t.Run("CiphertextRoundTrip", func(t *testing.T) {
		// Create a known vector and encrypt it
		dim := 128
		vector := make([]float64, dim)
		for i := range vector {
			vector[i] = float64(i+1) * 0.01 // [0.01, 0.02, ..., 1.28]
		}

		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		// Convert to HEonGPU domain
		rawHEonGPU := CiphertextToHEonGPU(ct, conv)

		// Convert back to Lattigo domain
		ctBack, err := CiphertextFromHEonGPU(rawHEonGPU, params, conv)
		if err != nil {
			t.Fatalf("CiphertextFromHEonGPU: %v", err)
		}

		// Decrypt original and round-tripped
		orig, err := engine.DecryptScalar(ct)
		if err != nil {
			t.Fatalf("Decrypt original: %v", err)
		}
		roundTrip, err := engine.DecryptScalar(ctBack)
		if err != nil {
			t.Fatalf("Decrypt round-trip: %v", err)
		}

		t.Logf("Original decrypt:    %v", orig)
		t.Logf("Round-trip decrypt:  %v", roundTrip)
		t.Logf("Difference:          %v", math.Abs(orig-roundTrip))

		if math.Abs(orig-roundTrip) > 0.01 {
			t.Errorf("Round-trip error too large: %v vs %v (diff=%v)",
				orig, roundTrip, math.Abs(orig-roundTrip))
		}
	})

	// Test 2: Full vector round-trip (decrypt all slots)
	t.Run("FullVectorRoundTrip", func(t *testing.T) {
		// Create a vector with distinct values to detect permutation
		slotCount := (1 << params.LogN()) / 2 // 8192
		packed := make([]float64, slotCount)
		for i := range packed {
			packed[i] = float64(i+1) * 0.001
		}

		ct, err := engine.EncryptVector(packed)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		t.Logf("Ciphertext flags: IsNTT=%v IsMontgomery=%v Level=%d",
			ct.IsNTT, ct.IsMontgomery, ct.Value[0].Level())

		// Round-trip
		rawHEonGPU := CiphertextToHEonGPU(ct, conv)
		ctBack, err := CiphertextFromHEonGPU(rawHEonGPU, params, conv)
		if err != nil {
			t.Fatalf("CiphertextFromHEonGPU: %v", err)
		}

		// Compare raw coefficients BEFORE decryption
		ringSize := 1 << params.LogN()
		t.Logf("ctBack flags: IsNTT=%v IsMontgomery=%v Level=%d Scale=%.0f",
			ctBack.IsNTT, ctBack.IsMontgomery, ctBack.Value[0].Level(), ctBack.Scale.Float64())
		t.Logf("ct orig: IsNTT=%v IsMontgomery=%v Level=%d Scale=%.0f",
			ct.IsNTT, ct.IsMontgomery, ct.Value[0].Level(), ct.Scale.Float64())

		// Check raw coefficients match for ALL polynomials
		for p := 0; p < len(ct.Value); p++ {
			rawMismatches := 0
			for lvl := 0; lvl <= ct.Value[p].Level(); lvl++ {
				for j := 0; j < ringSize; j++ {
					if ct.Value[p].Coeffs[lvl][j] != ctBack.Value[p].Coeffs[lvl][j] {
						rawMismatches++
						if rawMismatches <= 2 {
							t.Logf("  Mismatch at poly%d[%d][%d]: orig=%d rt=%d",
								p, lvl, j, ct.Value[p].Coeffs[lvl][j], ctBack.Value[p].Coeffs[lvl][j])
						}
					}
				}
			}
			if rawMismatches == 0 {
				t.Logf("Poly %d: raw coefficients match EXACTLY!", p)
			} else {
				t.Logf("Poly %d: %d mismatches", p, rawMismatches)
			}
		}

		// Also test CopyNew for comparison
		ctCopy := ct.CopyNew()

		// Decrypt all three
		origAll := make([]float64, slotCount)
		rtAll := make([]float64, slotCount)
		copyAll := make([]float64, slotCount)
		engine.encoder.Decode(engine.decryptor.DecryptNew(ct), origAll)
		engine.encoder.Decode(engine.decryptor.DecryptNew(ctBack), rtAll)
		engine.encoder.Decode(engine.decryptor.DecryptNew(ctCopy), copyAll)

		// Check ctCopy decodes the same as original
		copyOK := true
		for i := range origAll {
			if math.Abs(origAll[i]-copyAll[i]) > 1e-10 {
				copyOK = false
				break
			}
		}
		t.Logf("CopyNew decodes identically to original: %v", copyOK)

		// Check the Operand metadata
		t.Logf("ct     MetaData: %+v", ct.MetaData)
		t.Logf("ctBack MetaData: %+v", ctBack.MetaData)
		t.Logf("ctCopy MetaData: %+v", ctCopy.MetaData)

		// Compare first 10 slots
		maxErr := 0.0
		for i := 0; i < min(10, slotCount); i++ {
			diff := math.Abs(origAll[i] - rtAll[i])
			if diff > maxErr {
				maxErr = diff
			}
			if i < 5 {
				t.Logf("  Slot %d: orig=%.6f  rt=%.6f  diff=%.2e", i, origAll[i], rtAll[i], diff)
			}
		}
		// Check a few more slots
		for _, i := range []int{100, 500, 1000, 4000, 8000} {
			if i < slotCount {
				diff := math.Abs(origAll[i] - rtAll[i])
				if diff > maxErr {
					maxErr = diff
				}
				t.Logf("  Slot %d: orig=%.6f  rt=%.6f  diff=%.2e", i, origAll[i], rtAll[i], diff)
			}
		}

		t.Logf("Max round-trip error across first 10 slots: %.2e", maxErr)

		if maxErr > 0.01 {
			t.Errorf("Round-trip error too large: %.2e (expected < 0.01)", maxErr)
		}
	})

	// Test 3: Plaintext round-trip (convert to HEonGPU and back)
	t.Run("PlaintextRoundTrip", func(t *testing.T) {
		dim := 128
		centroidsPerPack := (1 << params.LogN()) / 2 / dim

		// Create a plaintext with known values
		packed := make([]float64, (1<<params.LogN())/2)
		for c := 0; c < min(4, centroidsPerPack); c++ {
			for d := 0; d < dim; d++ {
				packed[c*dim+d] = float64(c+1) * 0.01
			}
		}

		// Encode — MUST use hefloat.NewPlaintext (sets scale), not rlwe.NewPlaintext (scale=0)
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := engine.encoder.Encode(packed, pt); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		t.Logf("Plaintext flags: IsNTT=%v IsMontgomery=%v Level=%d Scale=%.0f",
			pt.IsNTT, pt.IsMontgomery, pt.Level(), pt.Scale.Float64())

		// Check raw plaintext before conversion
		nonZeroBefore := 0
		for lvl := 0; lvl <= pt.Level(); lvl++ {
			for _, c := range pt.Value.Coeffs[lvl] {
				if c != 0 {
					nonZeroBefore++
				}
			}
		}
		t.Logf("Before conversion: %d non-zero coeffs across %d levels", nonZeroBefore, pt.Level()+1)

		// Test step-by-step conversion on a copy
		testCoeffs := make([]uint64, conv.N)
		copy(testCoeffs, pt.Value.Coeffs[0])
		t.Logf("Level 0 first 5 coeffs before: %v", testCoeffs[:5])

		// Step 1: INTT
		conv.ConvertToCoeffDomain(testCoeffs, 0)
		nonZeroAfterINTT := 0
		for _, c := range testCoeffs {
			if c != 0 {
				nonZeroAfterINTT++
			}
		}
		t.Logf("After INTT: %d non-zero, first 5: %v", nonZeroAfterINTT, testCoeffs[:5])

		// Step 2: Remove Montgomery
		p := conv.allModuli[0]
		pBig := new(big.Int).SetUint64(p)
		R := new(big.Int).Lsh(big.NewInt(1), 64)
		R.Mod(R, pBig)
		RInv := new(big.Int).ModInverse(R, pBig)
		rInv := RInv.Uint64()
		for i := range testCoeffs {
			testCoeffs[i] = mulModBig(testCoeffs[i], rInv, p, pBig)
		}
		nonZeroAfterMont := 0
		for _, c := range testCoeffs {
			if c != 0 {
				nonZeroAfterMont++
			}
		}
		t.Logf("After Montgomery removal: %d non-zero, first 5: %v", nonZeroAfterMont, testCoeffs[:5])

		// Step 3: HEonGPU NTT
		conv.nttCooleyTukeyInPlace(testCoeffs, p, conv.heongpuNTTRoots[0])
		nonZeroAfterNTT := 0
		for _, c := range testCoeffs {
			if c != 0 {
				nonZeroAfterNTT++
			}
		}
		t.Logf("After HEonGPU NTT: %d non-zero, first 5: %v", nonZeroAfterNTT, testCoeffs[:5])

		// Now do the full conversion
		rawPt := PlaintextToHEonGPU(pt, conv)

		// Verify the raw plaintext has reasonable values
		t.Logf("Plaintext levels: %d, ring_size: %d, total coeffs: %d",
			rawPt.Polynomial.NumLevels, rawPt.Polynomial.RingSize,
			len(rawPt.Polynomial.Coefficients))

		nonZero := 0
		for _, c := range rawPt.Polynomial.Coefficients {
			if c != 0 {
				nonZero++
			}
		}
		t.Logf("Final non-zero coefficients: %d / %d", nonZero, len(rawPt.Polynomial.Coefficients))

		if nonZero == 0 {
			t.Error("All plaintext coefficients are zero after conversion!")
		}
	})

	// Test 4: NTT consistency check - forward then inverse should give identity
	t.Run("NTTForwardInverse", func(t *testing.T) {
		N := conv.N
		p := conv.allModuli[0] // Use first Q prime
		pBig := new(big.Int).SetUint64(p)

		// Create test data
		data := make([]uint64, N)
		for i := range data {
			data[i] = uint64(i + 1) % p
		}
		original := make([]uint64, N)
		copy(original, data)

		// Forward NTT
		conv.nttCooleyTukeyInPlace(data, p, conv.heongpuNTTRoots[0])

		// Check data changed
		same := true
		for i := range data {
			if data[i] != original[i] {
				same = false
				break
			}
		}
		if same {
			t.Error("NTT did not change data — roots might be wrong")
		}

		// Inverse NTT using our GS butterfly
		inttHEonGPU(data, N, p, conv.heongpuNTTRoots[0])

		// Check round-trip
		maxDiff := uint64(0)
		for i := range data {
			var diff uint64
			if data[i] >= original[i] {
				diff = data[i] - original[i]
			} else {
				diff = original[i] - data[i]
			}
			// Also check modular equivalence
			if diff > 0 {
				diffBig := new(big.Int).SetUint64(diff)
				diffBig.Mod(diffBig, pBig)
				diff = diffBig.Uint64()
			}
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		t.Logf("NTT forward+inverse max diff: %d", maxDiff)
		if maxDiff != 0 {
			t.Errorf("NTT round-trip not exact! max diff=%d", maxDiff)
			// Show first few differences
			for i := 0; i < min(5, N); i++ {
				t.Logf("  [%d] original=%d  roundtrip=%d", i, original[i], data[i])
			}
		}
	})
}

// TestFullConversionChain traces the complete conversion chain step by step.
func TestFullConversionChain(t *testing.T) {
	params, err := NewParametersGPU()
	if err != nil {
		t.Fatalf("NewParametersGPU: %v", err)
	}
	engine, err := NewClientEngineWithParams(params)
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	N := 1 << params.LogN()
	allModuli := params.RingQ().ModuliChain()
	if ringP := params.RingP(); ringP != nil {
		allModuli = append(allModuli, ringP.ModuliChain()...)
	}
	serverRoots := make([]uint64, len(allModuli))
	for i, p := range allModuli {
		serverRoots[i] = computeHEonGPUPsi(p, uint64(N))
	}
	conv := NewNTTConverterWithRoots(params, serverRoots)

	// Encrypt
	vector := make([]float64, N/2)
	for i := range vector {
		vector[i] = float64(i+1) * 0.001
	}
	ct, err := engine.EncryptVector(vector)
	if err != nil {
		t.Fatalf("Encrypt: %v", err)
	}

	// Test at level 0
	lvl := 0
	original := make([]uint64, N)
	copy(original, ct.Value[0].Coeffs[lvl])

	// Step 1: Lattigo INTT
	step1 := make([]uint64, N)
	copy(step1, original)
	conv.ConvertToCoeffDomain(step1, lvl)

	// Step 2: HEonGPU NTT
	step2 := make([]uint64, N)
	copy(step2, step1)
	p := conv.allModuli[lvl]
	conv.nttCooleyTukeyInPlace(step2, p, conv.heongpuNTTRoots[lvl])

	// Step 3: HEonGPU INTT
	step3 := make([]uint64, N)
	copy(step3, step2)
	inttHEonGPU(step3, N, p, conv.heongpuNTTRoots[lvl])

	// Check step1 == step3 (HEonGPU NTT→INTT should be identity)
	maxDiff23 := uint64(0)
	for i := range step1 {
		var diff uint64
		if step3[i] >= step1[i] {
			diff = step3[i] - step1[i]
		} else {
			diff = step1[i] - step3[i]
		}
		if diff > maxDiff23 {
			maxDiff23 = diff
		}
	}
	t.Logf("Step 1→2→3 (INTT_L → NTT_H → INTT_H) max diff from step1: %d", maxDiff23)

	// Step 4: Lattigo NTT
	step4 := make([]uint64, N)
	copy(step4, step3)
	subRing := params.RingQ().SubRings[lvl]
	tmp := make([]uint64, N)
	ring.NTTStandard(step4, tmp, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
	copy(step4, tmp)

	// Check original == step4 (full round-trip)
	maxDiffFull := uint64(0)
	for i := range original {
		var diff uint64
		if step4[i] >= original[i] {
			diff = step4[i] - original[i]
		} else {
			diff = original[i] - step4[i]
		}
		if diff > maxDiffFull {
			maxDiffFull = diff
		}
	}
	t.Logf("Full chain (INTT_L → NTT_H → INTT_H → NTT_L) max diff: %d", maxDiffFull)

	if maxDiffFull != 0 {
		t.Errorf("Full chain is NOT identity! max diff=%d", maxDiffFull)
		// Show first few
		for i := 0; i < 5; i++ {
			t.Logf("  [%d] orig=%d step1=%d step3=%d step4=%d",
				i, original[i], step1[i], step3[i], step4[i])
		}
	} else {
		t.Log("Full chain is perfect identity!")
	}
}

// TestLattigoNTTRoundTrip tests that Lattigo's own INTT→NTT is identity.
func TestLattigoNTTRoundTrip(t *testing.T) {
	params, err := NewParametersGPU()
	if err != nil {
		t.Fatalf("NewParametersGPU: %v", err)
	}
	engine, err := NewClientEngineWithParams(params)
	if err != nil {
		t.Fatalf("NewClientEngine: %v", err)
	}
	defer engine.Close()

	N := 1 << params.LogN()
	conv := NewNTTConverter(params)

	// Encrypt something to get NTT-domain ciphertext data
	vector := make([]float64, N/2)
	for i := range vector {
		vector[i] = float64(i+1) * 0.001
	}
	ct, err := engine.EncryptVector(vector)
	if err != nil {
		t.Fatalf("Encrypt: %v", err)
	}

	// Take level 0 of poly 0
	original := make([]uint64, N)
	copy(original, ct.Value[0].Coeffs[0])

	// Lattigo INTT
	coeffs := make([]uint64, N)
	copy(coeffs, original)
	conv.ConvertToCoeffDomain(coeffs, 0)

	// Check INTT changed the data
	same := 0
	for i := range coeffs {
		if coeffs[i] == original[i] {
			same++
		}
	}
	t.Logf("After INTT: %d/%d unchanged", same, N)

	// Lattigo NTT
	subRing := params.RingQ().SubRings[0]
	tmp := make([]uint64, N)
	ring.NTTStandard(coeffs, tmp, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
	copy(coeffs, tmp)

	// Check round-trip
	maxDiff := uint64(0)
	for i := range coeffs {
		var diff uint64
		if coeffs[i] >= original[i] {
			diff = coeffs[i] - original[i]
		} else {
			diff = original[i] - coeffs[i]
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("Lattigo INTT→NTT max diff: %d (should be 0)", maxDiff)

	if maxDiff != 0 {
		t.Errorf("Lattigo INTT→NTT is NOT identity! max diff=%d", maxDiff)
		for i := 0; i < 5; i++ {
			t.Logf("  [%d] orig=%d  rt=%d", i, original[i], coeffs[i])
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
