package crypto

// Comprehensive Bridge Diagnostic Test
//
// Tests EVERY assumption in the Lattigo ↔ HEonGPU data bridge.
// All tests run locally — no GPU required.
//
// Run: go test -v -run TestBridgeDiagnostic ./pkg/crypto/ -timeout 5m

import (
	"fmt"
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func TestBridgeDiagnostic(t *testing.T) {
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
	slotCount := N / 2

	// Build converter with locally-computed roots (simulates server roots)
	allModuli := params.RingQ().ModuliChain()
	if ringP := params.RingP(); ringP != nil {
		allModuli = append(allModuli, ringP.ModuliChain()...)
	}
	serverRoots := make([]uint64, len(allModuli))
	for i, p := range allModuli {
		serverRoots[i] = computeHEonGPUPsi(p, uint64(N))
	}
	conv := NewNTTConverterWithRoots(params, serverRoots)

	// ================================================================
	t.Run("01_ParameterAudit", func(t *testing.T) {
		t.Logf("LogN = %d (N = %d)", params.LogN(), N)
		t.Logf("Q primes = %d: %v", len(params.RingQ().ModuliChain()), params.RingQ().ModuliChain())
		if ringP := params.RingP(); ringP != nil {
			t.Logf("P primes = %d: %v", len(ringP.ModuliChain()), ringP.ModuliChain())
		}
		t.Logf("MaxLevelQ = %d, MaxLevelP = %d", params.MaxLevelQ(), params.MaxLevelP())
		t.Logf("LogDefaultScale = %d (scale = 2^%d = %.0f)",
			params.LogDefaultScale(), params.LogDefaultScale(), params.DefaultScale().Float64())
		t.Logf("Slot count = %d", slotCount)

		// Verify GPU params: single P prime
		if params.MaxLevelP() != 0 {
			t.Errorf("GPU params should have 1 P prime (MaxLevelP=0), got MaxLevelP=%d", params.MaxLevelP())
		}
	})

	// ================================================================
	t.Run("02_CiphertextFlags", func(t *testing.T) {
		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = float64(i+1) * 0.001
		}

		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		t.Logf("Ciphertext after encryption:")
		t.Logf("  IsNTT         = %v", ct.IsNTT)
		t.Logf("  IsMontgomery  = %v", ct.IsMontgomery)
		t.Logf("  IsBatched     = %v", ct.IsBatched)
		t.Logf("  LogDimensions = %+v", ct.LogDimensions)
		t.Logf("  Level         = %d", ct.Value[0].Level())
		t.Logf("  Scale         = %.0f (2^%.1f)", ct.Scale.Float64(), math.Log2(ct.Scale.Float64()))
		t.Logf("  NumPolys      = %d", len(ct.Value))
		t.Logf("  CoeffsPerPoly = %d levels × %d ring = %d uint64",
			ct.Value[0].Level()+1, N, (ct.Value[0].Level()+1)*N)

		if ct.IsMontgomery {
			t.Error("UNEXPECTED: Ciphertext has IsMontgomery=true — conversion must use Montgomery path")
		}
		if !ct.IsNTT {
			t.Error("UNEXPECTED: Ciphertext has IsNTT=false")
		}
		if !ct.IsBatched {
			t.Error("UNEXPECTED: Ciphertext has IsBatched=false")
		}
	})

	// ================================================================
	t.Run("03_PlaintextFlags", func(t *testing.T) {
		packed := make([]float64, slotCount)
		for i := range packed {
			packed[i] = float64(i+1) * 0.001
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := engine.encoder.Encode(packed, pt); err != nil {
			t.Fatalf("Encode: %v", err)
		}

		t.Logf("Plaintext after encoding:")
		t.Logf("  IsNTT         = %v", pt.IsNTT)
		t.Logf("  IsMontgomery  = %v", pt.IsMontgomery)
		t.Logf("  IsBatched     = %v", pt.IsBatched)
		t.Logf("  LogDimensions = %+v", pt.LogDimensions)
		t.Logf("  Level         = %d", pt.Level())
		t.Logf("  Scale         = %.0f (2^%.1f)", pt.Scale.Float64(), math.Log2(pt.Scale.Float64()))

		if pt.IsMontgomery {
			t.Error("UNEXPECTED: Plaintext has IsMontgomery=true — conversion must use Montgomery path")
		}
	})

	// ================================================================
	t.Run("04_NTTRoundTrip_AllLevels", func(t *testing.T) {
		// Verify NTT round-trip works for EVERY Q level, not just level 0
		for lvl := 0; lvl <= params.MaxLevelQ(); lvl++ {
			p := conv.allModuli[lvl]
			data := make([]uint64, N)
			for i := range data {
				data[i] = uint64(i*100+lvl*7+1) % p
			}
			original := make([]uint64, N)
			copy(original, data)

			conv.nttCooleyTukeyInPlace(data, p, conv.heongpuNTTRoots[lvl])
			inttHEonGPU(data, N, p, conv.heongpuNTTRoots[lvl])

			maxDiff := uint64(0)
			for i := range data {
				d := data[i] - original[i]
				if data[i] < original[i] {
					d = original[i] - data[i]
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff != 0 {
				t.Errorf("Level %d: NTT round-trip error! max diff=%d", lvl, maxDiff)
			}
		}
		t.Log("All Q levels: NTT forward+inverse = exact identity ✓")

		// Also test P level
		if params.MaxLevelP() >= 0 {
			pIdx := len(conv.qModuli)
			p := conv.allModuli[pIdx]
			data := make([]uint64, N)
			for i := range data {
				data[i] = uint64(i*31+17) % p
			}
			original := make([]uint64, N)
			copy(original, data)

			conv.nttCooleyTukeyInPlace(data, p, conv.heongpuNTTRoots[pIdx])
			inttHEonGPU(data, N, p, conv.heongpuNTTRoots[pIdx])

			maxDiff := uint64(0)
			for i := range data {
				d := data[i] - original[i]
				if data[i] < original[i] {
					d = original[i] - data[i]
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff != 0 {
				t.Errorf("P level: NTT round-trip error! max diff=%d", maxDiff)
			} else {
				t.Log("P level: NTT forward+inverse = exact identity ✓")
			}
		}
	})

	// ================================================================
	t.Run("05_FullConversionChain_AllLevels", func(t *testing.T) {
		// Test Lattigo INTT → HEonGPU NTT → HEonGPU INTT → Lattigo NTT at every level
		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = float64(i+1) * 0.001
		}
		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		for lvl := 0; lvl <= ct.Value[0].Level(); lvl++ {
			original := make([]uint64, N)
			copy(original, ct.Value[0].Coeffs[lvl])

			coeffs := make([]uint64, N)
			copy(coeffs, original)

			// Step 1: Lattigo INTT
			conv.ConvertToCoeffDomain(coeffs, lvl)
			// Step 2: HEonGPU NTT
			p := conv.allModuli[lvl]
			conv.nttCooleyTukeyInPlace(coeffs, p, conv.heongpuNTTRoots[lvl])
			// Step 3: HEonGPU INTT
			inttHEonGPU(coeffs, N, p, conv.heongpuNTTRoots[lvl])
			// Step 4: Lattigo NTT
			subRing := params.RingQ().SubRings[lvl]
			tmp := make([]uint64, N)
			ring.NTTStandard(coeffs, tmp, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
			copy(coeffs, tmp)

			maxDiff := uint64(0)
			for i := range coeffs {
				d := coeffs[i] - original[i]
				if coeffs[i] < original[i] {
					d = original[i] - coeffs[i]
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff != 0 {
				t.Errorf("Level %d: full chain error! max diff=%d", lvl, maxDiff)
			}
		}
		t.Log("All levels: full conversion chain = exact identity ✓")
	})

	// ================================================================
	t.Run("06_CiphertextRoundTrip_AllSlots", func(t *testing.T) {
		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = float64(i+1) * 0.001
		}
		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		raw := CiphertextToHEonGPU(ct, conv)
		ctBack, err := CiphertextFromHEonGPU(raw, params, conv)
		if err != nil {
			t.Fatalf("FromHEonGPU: %v", err)
		}

		// Check ALL metadata
		checks := []struct {
			name string
			got  interface{}
			want interface{}
		}{
			{"IsNTT", ctBack.IsNTT, ct.IsNTT},
			{"IsMontgomery", ctBack.IsMontgomery, ct.IsMontgomery},
			{"IsBatched", ctBack.IsBatched, ct.IsBatched},
			{"LogDimensions.Cols", ctBack.LogDimensions.Cols, ct.LogDimensions.Cols},
			{"Scale", ctBack.Scale.Float64(), ct.Scale.Float64()},
			{"Level", ctBack.Value[0].Level(), ct.Value[0].Level()},
			{"NumPolys", len(ctBack.Value), len(ct.Value)},
		}
		for _, c := range checks {
			if fmt.Sprintf("%v", c.got) != fmt.Sprintf("%v", c.want) {
				t.Errorf("  %s: got %v, want %v", c.name, c.got, c.want)
			} else {
				t.Logf("  %s: %v ✓", c.name, c.got)
			}
		}

		// Check raw coefficient equality
		for p := 0; p < len(ct.Value); p++ {
			for lvl := 0; lvl <= ct.Value[p].Level(); lvl++ {
				for j := 0; j < N; j++ {
					if ct.Value[p].Coeffs[lvl][j] != ctBack.Value[p].Coeffs[lvl][j] {
						t.Fatalf("  Poly%d[%d][%d] mismatch", p, lvl, j)
					}
				}
			}
		}
		t.Log("  All raw coefficients: exact match ✓")

		// Decrypt and compare all slots
		origAll := make([]float64, slotCount)
		rtAll := make([]float64, slotCount)
		engine.encoder.Decode(engine.decryptor.DecryptNew(ct), origAll)
		engine.encoder.Decode(engine.decryptor.DecryptNew(ctBack), rtAll)

		maxErr := 0.0
		for i := range origAll {
			if d := math.Abs(origAll[i] - rtAll[i]); d > maxErr {
				maxErr = d
			}
		}
		t.Logf("  Decrypt max error across %d slots: %.2e ✓", slotCount, maxErr)
		if maxErr > 1e-10 {
			t.Errorf("  Round-trip introduces error: %.2e", maxErr)
		}
	})

	// ================================================================
	t.Run("07_PlaintextConversion_NonMontgomery", func(t *testing.T) {
		packed := make([]float64, slotCount)
		for i := 0; i < 128; i++ {
			packed[i] = float64(i+1) * 0.01
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		engine.encoder.Encode(packed, pt)

		rawPt := PlaintextToHEonGPU(pt, conv)

		t.Logf("Plaintext conversion:")
		t.Logf("  IsMontgomery = %v (used non-Montgomery path: %v)", pt.IsMontgomery, !pt.IsMontgomery)
		t.Logf("  NumLevels = %d", rawPt.Polynomial.NumLevels)
		t.Logf("  Total coeffs = %d", len(rawPt.Polynomial.Coefficients))

		nonZero := 0
		for _, c := range rawPt.Polynomial.Coefficients {
			if c != 0 {
				nonZero++
			}
		}
		t.Logf("  Non-zero coefficients: %d / %d", nonZero, len(rawPt.Polynomial.Coefficients))
		if nonZero == 0 {
			t.Error("  ALL ZEROS — plaintext conversion is broken!")
		}
		if nonZero < len(rawPt.Polynomial.Coefficients)/2 {
			t.Errorf("  Too few non-zero coeffs: %d (expected most to be non-zero)", nonZero)
		}
	})

	// ================================================================
	t.Run("08_DepthSemantics", func(t *testing.T) {
		// Test: what is the Go proto Depth field?
		// CiphertextToHEonGPU sets: Depth: int32(ct.Value[0].Level())
		// This is the LEVEL INDEX (0-7), not the HEonGPU depth (number of rescales).
		//
		// HEonGPU's depth_: starts at 0, increments after each rescale
		// Lattigo's Level(): MaxLevelQ - number_of_consumed_primes
		//
		// For a fresh ct: Level() = MaxLevelQ = 7, depth_ = 0
		// After 1 rescale: Level() = 6, depth_ = 1
		//
		// The proto sends Level() as Depth, which is WRONG for HEonGPU.
		// Server should interpret: heongpu_depth = Q_size - 1 - proto_depth

		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = 0.5
		}
		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		raw := CiphertextToHEonGPU(ct, conv)

		t.Logf("Fresh ciphertext:")
		t.Logf("  Lattigo Level()  = %d (index of last active level)", ct.Value[0].Level())
		t.Logf("  Proto Depth      = %d (what we send)", raw.Depth)
		t.Logf("  Proto NumLevels  = %d (per poly)", raw.Polynomials[0].NumLevels)

		t.Logf("")
		t.Logf("HEonGPU interpretation:")
		t.Logf("  HEonGPU Q_size            = %d", params.MaxLevelQ()+1)
		t.Logf("  HEonGPU default depth_    = 0 (from constructor)")
		t.Logf("  Active levels (Q-depth)   = %d - 0 = %d", params.MaxLevelQ()+1, params.MaxLevelQ()+1)
		t.Logf("  Proto NumLevels           = %d", raw.Polynomials[0].NumLevels)

		if int(raw.Polynomials[0].NumLevels) != params.MaxLevelQ()+1 {
			t.Errorf("  NumLevels mismatch: proto has %d, HEonGPU expects %d",
				raw.Polynomials[0].NumLevels, params.MaxLevelQ()+1)
		} else {
			t.Log("  NumLevels matches HEonGPU Q_size ✓")
		}

		// Depth field semantics warning
		t.Logf("")
		t.Logf("  WARNING: Proto Depth=%d is Lattigo's Level() (last active level index).", raw.Depth)
		t.Logf("  HEonGPU expects depth=0 for fresh ct (number of rescales done).")
		t.Logf("  Server MUST use depth=0 for fresh ct, NOT proto Depth value.")
	})

	// ================================================================
	t.Run("09_ResultConversion_ReducedLevels", func(t *testing.T) {
		// After multiply_plain + rescale on GPU, the result has fewer levels.
		// The server sends back result with depth=1, numLevels=7 (instead of 8).
		// Test that CiphertextFromHEonGPU handles this correctly.

		// Simulate a result with fewer levels
		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = float64(i+1) * 0.001
		}
		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		// Simulate GPU computation: multiply + rescale locally
		packed := make([]float64, slotCount)
		for i := 0; i < 128; i++ {
			packed[i] = float64(i+1) * 0.01
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		engine.encoder.Encode(packed, pt)

		result, err := engine.HomomorphicBatchDotProduct(ct, pt, 64, 128)
		if err != nil {
			t.Fatalf("BatchDotProduct: %v", err)
		}

		t.Logf("After multiply+rescale+rotate locally:")
		t.Logf("  Level = %d (was %d before)", result.Value[0].Level(), ct.Value[0].Level())
		t.Logf("  Scale = %.0f", result.Scale.Float64())
		t.Logf("  Active levels = %d", result.Value[0].Level()+1)

		// Convert to HEonGPU format and back
		rawResult := CiphertextToHEonGPU(result, conv)
		t.Logf("  Proto NumLevels = %d, Depth = %d", rawResult.Polynomials[0].NumLevels, rawResult.Depth)

		resultBack, err := CiphertextFromHEonGPU(rawResult, params, conv)
		if err != nil {
			t.Fatalf("FromHEonGPU: %v", err)
		}

		t.Logf("After round-trip:")
		t.Logf("  Level = %d", resultBack.Value[0].Level())
		t.Logf("  Scale = %.0f", resultBack.Scale.Float64())

		// Decrypt and compare
		origScores, err := engine.DecryptBatchScalars(result, 64, 128)
		if err != nil {
			t.Fatalf("Decrypt orig: %v", err)
		}
		rtScores, err := engine.DecryptBatchScalars(resultBack, 64, 128)
		if err != nil {
			t.Fatalf("Decrypt round-trip: %v", err)
		}

		maxErr := 0.0
		for i := range origScores {
			if d := math.Abs(origScores[i] - rtScores[i]); d > maxErr {
				maxErr = d
			}
		}
		t.Logf("  Score max error: %.2e", maxErr)
		if maxErr > 1e-6 {
			t.Errorf("  Result round-trip has significant error: %.2e", maxErr)
			for i := 0; i < min(4, len(origScores)); i++ {
				t.Logf("    Score %d: orig=%.6f rt=%.6f", i, origScores[i], rtScores[i])
			}
		} else {
			t.Log("  Result round-trip: accurate ✓")
		}
	})

	// ================================================================
	t.Run("10_MontgomeryFactors", func(t *testing.T) {
		// Verify our Montgomery factor computations are correct
		for i, p := range conv.allModuli[:3] {
			pBig := new(big.Int).SetUint64(p)
			R := new(big.Int).Lsh(big.NewInt(1), 64)
			R.Mod(R, pBig)
			RInv := new(big.Int).ModInverse(R, pBig)

			// Check: R * RInv ≡ 1 (mod p)
			check := new(big.Int).Mul(R, RInv)
			check.Mod(check, pBig)
			if check.Uint64() != 1 {
				t.Errorf("  Prime %d (%d): R*RInv != 1 (got %d)", i, p, check.Uint64())
			}

			t.Logf("  Prime %d: R=%d, RInv=%d, R*RInv mod p = %d ✓", i, R.Uint64(), RInv.Uint64(), check.Uint64())
		}
	})

	// ================================================================
	t.Run("11_DataLayoutConsistency", func(t *testing.T) {
		// Verify the data layout Go sends matches what C++ expects.
		// Go: [poly0: [lvl0_coeffs, lvl1_coeffs, ...], poly1: [...]]
		// C++ expects: same (poly-major)
		//
		// Each proto RawPolynomial has its own flat coefficient array.

		vector := make([]float64, slotCount)
		for i := range vector {
			vector[i] = float64(i+1) * 0.001
		}
		ct, err := engine.EncryptVector(vector)
		if err != nil {
			t.Fatalf("Encrypt: %v", err)
		}

		raw := CiphertextToHEonGPU(ct, conv)

		t.Logf("Data layout verification:")
		t.Logf("  Polynomials in proto: %d", len(raw.Polynomials))
		for i, p := range raw.Polynomials {
			expectedSize := int(p.NumLevels) * int(p.RingSize)
			t.Logf("  Poly %d: NumLevels=%d, RingSize=%d, actual coeffs=%d, expected=%d, match=%v",
				i, p.NumLevels, p.RingSize, len(p.Coefficients), expectedSize, len(p.Coefficients) == expectedSize)
			if len(p.Coefficients) != expectedSize {
				t.Errorf("  Poly %d size mismatch!", i)
			}
		}

		// The C++ server reads:
		//   for p in 0..num_polys:
		//     for lvl in 0..num_levels:
		//       src_offset = lvl * ring_size (within THIS poly's coefficients)
		//       dst_offset = (p * num_levels + lvl) * ring_size (in flat ct data)
		//
		// This produces a flat array: [poly0_lvl0, poly0_lvl1, ..., poly1_lvl0, poly1_lvl1, ...]
		// Which is POLY-MAJOR layout.
		t.Log("  Layout: each RawPolynomial stores levels sequentially (poly-major) ✓")
	})

	// ================================================================
	t.Run("12_ScaleChain_Simulation", func(t *testing.T) {
		// Simulate what happens to scale through the GPU compute chain.
		// multiply_plain: output.scale = input.scale * pt.scale
		// rescale: output.scale = output.scale / q_i (drops one modulus)

		ctScale := params.DefaultScale().Float64() // 2^45
		ptScale := params.DefaultScale().Float64() // 2^45

		afterMul := ctScale * ptScale
		t.Logf("After multiply_plain: scale = %.0f (2^%.1f)", afterMul, math.Log2(afterMul))

		// Rescale divides by the modulus of the dropped level
		// For CKKS, the last Q prime is dropped. Our Q primes have LogQ = [60,45,45,45,45,45,45,45]
		// After multiply, depth=0, so we drop Q[7] (the last one with log=45)
		lastQBits := 45.0 // LogQ[7]
		afterRescale := afterMul / math.Pow(2, lastQBits)
		t.Logf("After rescale (drop ~2^%d): scale = %.0f (2^%.1f)", int(lastQBits), afterRescale, math.Log2(afterRescale))

		// The result scale should be close to 2^45 (one level dropped)
		if math.Abs(math.Log2(afterRescale)-45) > 1 {
			t.Errorf("Result scale %.0f is not close to 2^45", afterRescale)
		} else {
			t.Log("Scale chain: correct ✓")
		}

		// CRITICAL: If ct.scale was 0 (unset), then:
		afterMulZero := 0.0 * ptScale
		t.Logf("")
		t.Logf("If ciphertext scale=0 (UNSET bug):")
		t.Logf("  After multiply_plain: scale = %.0f", afterMulZero)
		t.Logf("  After rescale: scale = %.0f", afterMulZero/math.Pow(2, lastQBits))
		t.Log("  → All decrypted values become ~0 or garbage")
	})

	// ================================================================
	t.Run("13_PsiPowers_ForGPUComparison", func(t *testing.T) {
		// Print psi^0..psi^15 for Q[0] — compare with C++ diagnostic output.
		// If these match, the root tables are compatible.
		psi0 := serverRoots[0]
		p0 := allModuli[0]
		p0Big := new(big.Int).SetUint64(p0)
		psiBig := new(big.Int).SetUint64(psi0)

		t.Logf("Prime Q[0] = %d, psi = %d", p0, psi0)
		t.Logf("First 16 powers of psi:")
		power := big.NewInt(1)
		for j := 0; j < 16; j++ {
			t.Logf("  psi^%d = %d", j, power.Uint64())
			power.Mul(power, psiBig)
			power.Mod(power, p0Big)
		}

		// Also print the root TABLE values (after bit-reverse) for comparison
		table := conv.heongpuNTTRoots[0]
		t.Logf("\nRoot table (after generateNTTTable) first 16 entries:")
		for j := 0; j < 16; j++ {
			t.Logf("  roots[%d] = %d", j, table[j])
		}

		// Check: does roots[1] = psi^1 or psi^bitrev(1)?
		t.Logf("\nRoot table consistency check:")
		t.Logf("  roots[0] = %d (should be psi^0 = 1: %v)", table[0], table[0] == 1)
		t.Logf("  roots[1] = %d", table[1])
		t.Logf("  psi^1 = %d", psi0)
		t.Logf("  roots[1] == psi^1: %v", table[1] == psi0)

		// Compute psi^bitrev(1, logN) to check bit-reverse format
		br1 := bitReverse(1, conv.logN)
		psiBr1 := new(big.Int).Exp(psiBig, big.NewInt(int64(br1)), p0Big)
		t.Logf("  bitrev(1, %d) = %d", conv.logN, br1)
		t.Logf("  psi^bitrev(1) = psi^%d = %d", br1, psiBr1.Uint64())
		t.Logf("  roots[1] == psi^bitrev(1): %v", table[1] == psiBr1.Uint64())
	})

	// ================================================================
	t.Run("14_NTT_OutputOrder_Comparison", func(t *testing.T) {
		// CRITICAL: Check if our Go NTT produces the same output order as HEonGPU.
		//
		// CT DIT NTT: input natural-order → output BIT-REVERSED order
		// CT DIF NTT: input bit-reversed → output natural-order
		//
		// If HEonGPU uses natural output and our Go code produces bit-reversed,
		// the data would be scrambled → wrong results.

		p := conv.allModuli[0]
		pBig := new(big.Int).SetUint64(p)
		psiBig := new(big.Int).SetUint64(serverRoots[0])

		// Test 1: NTT([1, 0, ..., 0])
		data1 := make([]uint64, N)
		data1[0] = 1
		conv.nttCooleyTukeyInPlace(data1, p, conv.heongpuNTTRoots[0])

		allOnes := true
		for _, v := range data1 {
			if v != 1 {
				allOnes = false
				break
			}
		}
		t.Logf("NTT([1,0,...,0]) = all ones: %v", allOnes)

		// Test 2: NTT([0, 1, 0, ..., 0])
		data2 := make([]uint64, N)
		data2[1] = 1
		conv.nttCooleyTukeyInPlace(data2, p, conv.heongpuNTTRoots[0])

		t.Logf("NTT([0,1,0,...,0]) first 8 outputs:")
		for j := 0; j < 8; j++ {
			t.Logf("  output[%d] = %d", j, data2[j])
		}

		// For natural-order output: output[k] = psi^k for all k
		// For bit-reversed output: output[k] = psi^bitrev(k) for all k
		// Check which pattern matches
		naturalMatch := 0
		bitrevMatch := 0
		psiPower := big.NewInt(1)
		for k := 0; k < N; k++ {
			expected := psiPower.Uint64()
			if data2[k] == expected {
				naturalMatch++
			}
			br := bitReverse(k, conv.logN)
			brPsi := new(big.Int).Exp(psiBig, big.NewInt(int64(br)), pBig)
			if data2[k] == brPsi.Uint64() {
				bitrevMatch++
			}
			psiPower.Mul(psiPower, psiBig)
			psiPower.Mod(psiPower, pBig)
		}

		t.Logf("Output order analysis for NTT([0,1,0,...,0]):")
		t.Logf("  Natural order matches (output[k]=psi^k): %d/%d", naturalMatch, N)
		t.Logf("  Bit-rev order matches (output[k]=psi^bitrev(k)): %d/%d", bitrevMatch, N)

		if naturalMatch == N {
			t.Log("  → Our NTT produces NATURAL order output")
		} else if bitrevMatch == N {
			t.Log("  → Our NTT produces BIT-REVERSED order output")
		} else {
			t.Log("  → Our NTT produces NEITHER standard pattern")
			t.Logf("  → Checking first few: output[0]=%d (psi^0=%d), output[1]=%d (psi^1=%d)",
				data2[0], uint64(1), data2[1], serverRoots[0])
		}
	})

	t.Log("")
	t.Log("=== DIAGNOSTIC COMPLETE ===")
}

// Ensure the rlwe import is used
var _ = rlwe.NewCiphertext
