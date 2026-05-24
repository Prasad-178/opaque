// ciphertext_convert.go converts CKKS ciphertexts between Lattigo and HEonGPU formats.
//
// Lattigo stores ciphertexts in Montgomery-NTT domain.
// HEonGPU stores ciphertexts in standard-NTT domain (different NTT roots, no Montgomery).
//
// The conversion reuses the NTTConverter for the heavy lifting:
//
//	Lattigo → HEonGPU: INTT_L → remove Montgomery → NTT_H
//	HEonGPU → Lattigo: INTT_H → add Montgomery → NTT_L
//
// Privacy: ciphertexts are encrypted data. Converting between formats
// does not decrypt or reveal any information about the plaintext.
package crypto

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"

	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
)

// CiphertextToHEonGPU converts a Lattigo ciphertext to HEonGPU raw format.
// The output coefficients are in HEonGPU's NTT domain (non-Montgomery).
//
// IMPORTANT: Checks IsMontgomery to determine conversion path.
// Lattigo v5 ciphertexts typically have IsMontgomery=false.
func CiphertextToHEonGPU(ct *rlwe.Ciphertext, conv *NTTConverter) *pb.RawCiphertext {
	polys := make([]*pb.RawPolynomial, len(ct.Value))

	for i, poly := range ct.Value {
		numLevels := poly.Level() + 1
		ringSize := poly.N()
		flat := make([]uint64, 0, numLevels*ringSize)

		for lvl := 0; lvl < numLevels; lvl++ {
			coeffs := make([]uint64, ringSize)
			copy(coeffs, poly.Coeffs[lvl])

			if ct.IsMontgomery {
				// Montgomery-NTT → remove Montgomery → HEonGPU NTT
				conv.ConvertToHEonGPUDomain(coeffs, lvl)
			} else {
				// Non-Montgomery NTT → HEonGPU NTT (skip Montgomery removal)
				conv.ConvertToHEonGPUDomainNonMontgomery(coeffs, lvl)
			}

			flat = append(flat, coeffs...)
		}

		polys[i] = &pb.RawPolynomial{
			Coefficients: flat,
			NumLevels:    int32(numLevels),
			RingSize:     int32(ringSize),
		}
	}

	return &pb.RawCiphertext{
		Polynomials: polys,
		Scale:       ct.Scale.Float64(),
		IsNtt:       true, // Now in HEonGPU's NTT domain
		Depth:       int32(ct.Value[0].Level()),
	}
}

// PlaintextToHEonGPU converts a Lattigo plaintext to HEonGPU raw format.
// The output coefficients are in HEonGPU's NTT domain (non-Montgomery).
//
// IMPORTANT: Lattigo plaintexts have IsMontgomery=false (unlike ciphertexts
// which have IsMontgomery=true). This means we must NOT remove a Montgomery
// factor during conversion. Use the non-Montgomery conversion path.
func PlaintextToHEonGPU(pt *rlwe.Plaintext, conv *NTTConverter) *pb.RawPlaintext {
	poly := pt.Value
	numLevels := poly.Level() + 1
	ringSize := poly.N()
	flat := make([]uint64, 0, numLevels*ringSize)

	for lvl := 0; lvl < numLevels; lvl++ {
		coeffs := make([]uint64, ringSize)
		copy(coeffs, poly.Coeffs[lvl])

		if pt.IsMontgomery {
			// Montgomery-NTT → remove Montgomery → HEonGPU NTT
			conv.ConvertToHEonGPUDomain(coeffs, lvl)
		} else {
			// Non-Montgomery NTT → HEonGPU NTT (skip Montgomery removal)
			conv.ConvertToHEonGPUDomainNonMontgomery(coeffs, lvl)
		}

		flat = append(flat, coeffs...)
	}

	return &pb.RawPlaintext{
		Polynomial: &pb.RawPolynomial{
			Coefficients: flat,
			NumLevels:    int32(numLevels),
			RingSize:     int32(ringSize),
		},
		Scale: pt.Scale.Float64(),
		IsNtt: true, // Now in HEonGPU's NTT domain
	}
}

// ciphertextToCoeffDomain converts a Lattigo ciphertext to coefficient domain.
// The output coefficients are standard (non-Montgomery, non-NTT) polynomial coefficients.
// The C++ server will apply HEonGPU's own NTT to put them in the right domain.
// This avoids the cross-library NTT conversion entirely.
func ciphertextToCoeffDomain(ct *rlwe.Ciphertext, conv *NTTConverter) *pb.RawCiphertext {
	polys := make([]*pb.RawPolynomial, len(ct.Value))

	for i, poly := range ct.Value {
		numLevels := poly.Level() + 1
		ringSize := poly.N()
		flat := make([]uint64, 0, numLevels*ringSize)

		for lvl := 0; lvl < numLevels; lvl++ {
			coeffs := make([]uint64, ringSize)
			copy(coeffs, poly.Coeffs[lvl])

			// Step 1: Lattigo INTT → coefficient domain
			conv.ConvertToCoeffDomain(coeffs, lvl)

			// Step 2: Remove Montgomery factor if present
			if ct.IsMontgomery {
				p := conv.allModuli[lvl]
				pBig := new(big.Int).SetUint64(p)
				R := new(big.Int).Lsh(big.NewInt(1), 64)
				R.Mod(R, pBig)
				RInv := new(big.Int).ModInverse(R, pBig)
				rInv := RInv.Uint64()
				for j := range coeffs {
					coeffs[j] = mulModBig(coeffs[j], rInv, p, pBig)
				}
			}

			flat = append(flat, coeffs...)
		}

		polys[i] = &pb.RawPolynomial{
			Coefficients: flat,
			NumLevels:    int32(numLevels),
			RingSize:     int32(ringSize),
		}
	}

	return &pb.RawCiphertext{
		Polynomials: polys,
		Scale:       ct.Scale.Float64(),
		IsNtt:       false, // Coefficient domain, NOT NTT
		Depth:       int32(ct.Value[0].Level()),
	}
}

// CiphertextFromHEonGPU converts a HEonGPU raw ciphertext back to Lattigo format.
// The input coefficients are in HEonGPU's NTT domain (non-Montgomery).
// The output is in Lattigo's non-Montgomery NTT domain (IsMontgomery=false),
// matching Lattigo v5's default ciphertext format.
func CiphertextFromHEonGPU(raw *pb.RawCiphertext, params hefloat.Parameters, conv *NTTConverter) (*rlwe.Ciphertext, error) {
	if len(raw.Polynomials) < 2 {
		return nil, fmt.Errorf("need at least 2 polynomials, got %d", len(raw.Polynomials))
	}

	numLevels := int(raw.Polynomials[0].NumLevels)
	ringSize := int(raw.Polynomials[0].RingSize)
	level := numLevels - 1

	ct := rlwe.NewCiphertext(params, len(raw.Polynomials)-1, level)
	ct.IsNTT = true
	ct.IsMontgomery = false // Lattigo v5 ciphertexts are non-Montgomery
	ct.Scale = rlwe.NewScale(raw.Scale)
	ct.LogDimensions = ring.Dimensions{Rows: 0, Cols: params.LogN() - 1} // CKKS uses N/2 slots
	ct.IsBatched = true

	for i, rawPoly := range raw.Polynomials {
		if len(rawPoly.Coefficients) != numLevels*ringSize {
			return nil, fmt.Errorf("poly %d: expected %d coeffs, got %d",
				i, numLevels*ringSize, len(rawPoly.Coefficients))
		}

		for lvl := 0; lvl < numLevels; lvl++ {
			start := lvl * ringSize
			coeffs := make([]uint64, ringSize)
			copy(coeffs, rawPoly.Coefficients[start:start+ringSize])

			// Convert: HEonGPU NTT → standard coefficients → Lattigo NTT
			// NO Montgomery addition (Lattigo v5 ciphertexts are non-Montgomery)
			convertFromHEonGPUDomainNonMontgomery(coeffs, lvl, params, conv)

			copy(ct.Value[i].Coeffs[lvl], coeffs)
		}
	}

	return ct, nil
}

// coeffDomainToLattigo converts coefficient-domain data to Lattigo's NTT domain.
// The server applied HEonGPU's INTT, giving us standard coefficient-domain polynomials.
// We just need to apply Lattigo's forward NTT to get back to Lattigo's domain.
// This completely avoids the cross-library NTT conversion problem.
func coeffDomainToLattigo(raw *pb.RawCiphertext, params hefloat.Parameters) (*rlwe.Ciphertext, error) {
	if len(raw.Polynomials) < 2 {
		return nil, fmt.Errorf("need at least 2 polynomials, got %d", len(raw.Polynomials))
	}

	numLevels := int(raw.Polynomials[0].NumLevels)
	ringSize := int(raw.Polynomials[0].RingSize)
	level := numLevels - 1

	ct := rlwe.NewCiphertext(params, len(raw.Polynomials)-1, level)
	ct.IsNTT = true
	ct.IsMontgomery = false
	ct.Scale = rlwe.NewScale(raw.Scale)
	ct.LogDimensions = ring.Dimensions{Rows: 0, Cols: params.LogN() - 1}
	ct.IsBatched = true

	for i, rawPoly := range raw.Polynomials {
		if len(rawPoly.Coefficients) != numLevels*ringSize {
			return nil, fmt.Errorf("poly %d: expected %d coeffs, got %d",
				i, numLevels*ringSize, len(rawPoly.Coefficients))
		}

		for lvl := 0; lvl < numLevels; lvl++ {
			start := lvl * ringSize
			coeffs := make([]uint64, ringSize)
			copy(coeffs, rawPoly.Coefficients[start:start+ringSize])

			// Apply Lattigo's forward NTT (coefficient domain → Lattigo NTT domain)
			subRing := params.RingQ().SubRings[lvl]
			tmp := make([]uint64, ringSize)
			ring.NTTStandard(coeffs, tmp, ringSize, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
			copy(ct.Value[i].Coeffs[lvl], tmp)
		}
	}

	return ct, nil
}

// convertFromHEonGPUDomainNonMontgomery converts coefficients from HEonGPU's NTT domain
// to Lattigo's non-Montgomery NTT domain. This is the REVERSE of ConvertToHEonGPUDomainNonMontgomery.
//
// Steps:
// 1. HEonGPU INTT → standard coefficients
// 2. Lattigo NTT → Lattigo NTT domain (non-Montgomery)
func convertFromHEonGPUDomainNonMontgomery(coeffs []uint64, modulusIdx int, params hefloat.Parameters, conv *NTTConverter) {
	N := conv.N
	p := conv.allModuli[modulusIdx]

	// Step 1: HEonGPU INTT → standard coefficients
	inttHEonGPU(coeffs, N, p, conv.heongpuNTTRoots[modulusIdx])

	// Step 3: Lattigo NTT → Lattigo NTT domain
	var subRing *ring.SubRing
	if modulusIdx <= params.MaxLevelQ() {
		subRing = params.RingQ().SubRings[modulusIdx]
	} else {
		pIdx := modulusIdx - params.MaxLevelQ() - 1
		subRing = params.RingP().SubRings[pIdx]
	}
	tmp := make([]uint64, N)
	ring.NTTStandard(coeffs, tmp, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
	copy(coeffs, tmp)
}

// convertFromHEonGPUDomain converts coefficients from HEonGPU's NTT domain
// to Lattigo's Montgomery-NTT domain. This is the REVERSE of ConvertToHEonGPUDomain.
//
// Steps:
// 1. HEonGPU INTT (using HEonGPU's inverse NTT roots) → standard coefficients
// 2. Add Montgomery factor (multiply by R = 2^64 mod Q)
// 3. Lattigo NTT → Lattigo Montgomery-NTT domain
func convertFromHEonGPUDomain(coeffs []uint64, modulusIdx int, params hefloat.Parameters, conv *NTTConverter) {
	N := conv.N
	p := conv.allModuli[modulusIdx]
	pBig := new(big.Int).SetUint64(p)

	// Step 1: HEonGPU INTT → standard coefficients
	// We need HEonGPU's inverse NTT. For CT butterfly INTT:
	// Use inverse roots and divide by N.
	inttHEonGPU(coeffs, N, p, conv.heongpuNTTRoots[modulusIdx])

	// Step 2: Add Montgomery factor (multiply by R)
	R := new(big.Int).Lsh(big.NewInt(1), 64)
	R.Mod(R, pBig)
	Ru := R.Uint64()
	for i := range coeffs {
		coeffs[i] = mulModBig(coeffs[i], Ru, p, pBig)
	}

	// Step 3: Lattigo NTT → Montgomery-NTT domain
	var subRing *ring.SubRing
	if modulusIdx <= params.MaxLevelQ() {
		subRing = params.RingQ().SubRings[modulusIdx]
	} else {
		pIdx := modulusIdx - params.MaxLevelQ() - 1
		subRing = params.RingP().SubRings[pIdx]
	}
	tmp := make([]uint64, N)
	ring.NTTStandard(coeffs, tmp, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
	copy(coeffs, tmp)
}

// inttHEonGPU performs the inverse NTT using HEonGPU's roots.
// Gentleman-Sande (GS) butterfly INTT, which is the inverse of the CT NTT.
func inttHEonGPU(p []uint64, N int, Q uint64, forwardRoots []uint64) {
	QBig := new(big.Int).SetUint64(Q)

	// Generate inverse roots from forward roots
	// For CT NTT with forward roots, the GS INTT uses:
	// invRoots[i] = modInverse(forwardRoots[i], Q) for the butterfly
	// But the standard approach is to compute invRoots separately.

	// Actually, for a CT NTT output, the GS INTT is:
	// Reverse the butterfly stages and use inverse twiddle factors.
	// invRoot[m+i] = modInverse(root[m+i], Q)

	invRoots := make([]uint64, N)
	for i := 0; i < N; i++ {
		if forwardRoots[i] == 0 {
			invRoots[i] = 0
		} else {
			inv := new(big.Int).ModInverse(new(big.Int).SetUint64(forwardRoots[i]), QBig)
			if inv != nil {
				invRoots[i] = inv.Uint64()
			}
		}
	}

	// N inverse
	nInv := new(big.Int).ModInverse(new(big.Int).SetUint64(uint64(N)), QBig)
	nInvU := nInv.Uint64()

	// GS butterfly INTT (reverse of CT NTT)
	t := 1
	for m := N >> 1; m >= 1; m >>= 1 {
		for i := 0; i < m; i++ {
			j1 := (i * t) << 1
			j2 := j1 + t
			W := invRoots[m+i]
			for j := j1; j < j2; j++ {
				u := p[j]
				v := p[j+t]
				p[j] = addModU(u, v, Q)
				p[j+t] = mulModBig(subModU(u, v, Q), W, Q, QBig)
			}
		}
		t <<= 1
	}

	// Divide by N
	for i := range p {
		p[i] = mulModBig(p[i], nInvU, Q, QBig)
	}
}
