// Package crypto provides HEonGPU format serialization for evaluation keys.
//
// This file converts Lattigo's GaloisKey/RelinKey structures to the binary
// format expected by HEonGPU's load() method, enabling cross-library transfer
// of evaluation keys for GPU-accelerated HE computation.
//
// IMPORTANT: This only works with GPU-compatible parameters (NewParametersGPU)
// which use LogP=[61] (single P prime), producing decomposition d=Q_size=8
// matching HEonGPU's KEYSWITCHING_METHOD_I.
package crypto

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// HEonGPU binary format constants (from heongpu/util/schemes.h)
const (
	heongpuSchemeTypeCKKS       int32 = 2 // scheme_type::ckks
	heongpuKeySwitchingMethodI  int32 = 1 // keyswitching_type::KEYSWITCHING_METHOD_I
	heongpuStorageTypeHost      int32 = 1 // storage_type::HOST
)

// SerializeGaloisKeysHEonGPU converts Lattigo Galois keys to HEonGPU's binary format.
// The output can be loaded by HEonGPU's Galoiskey::load() method.
//
// Requires GPU-compatible parameters (NewParametersGPU with LogP=[61]).
func SerializeGaloisKeysHEonGPU(
	params hefloat.Parameters,
	galoisKeys []*rlwe.GaloisKey,
	galoisElements []uint64,
) ([]byte, error) {
	if len(galoisKeys) != len(galoisElements) {
		return nil, fmt.Errorf("galoisKeys length %d != galoisElements length %d",
			len(galoisKeys), len(galoisElements))
	}

	ringSize := 1 << params.LogN()
	qSize := params.MaxLevelQ() + 1
	pSize := params.MaxLevelP() + 1
	qPrimeSize := qSize + pSize

	if pSize != 1 {
		return nil, fmt.Errorf("HEonGPU format requires LogP with 1 prime (got %d P primes); use NewParametersGPU()", pSize)
	}

	// For KEYSWITCHING_METHOD_I: d = Q_size, galoiskey_size = 2 * d * Q_prime_size * ring_size
	d := qSize
	galoisKeySize := int64(2 * d * qPrimeSize * ringSize)

	// Group order (generator of multiplicative group mod 2N)
	// HEonGPU uses the same generator as the NTT. For N=16384, group_order=5.
	groupOrder := int32(findGenerator(2 * ringSize))

	var buf bytes.Buffer

	// Header
	binary.Write(&buf, binary.LittleEndian, heongpuSchemeTypeCKKS)    // scheme_
	binary.Write(&buf, binary.LittleEndian, heongpuKeySwitchingMethodI) // key_type
	binary.Write(&buf, binary.LittleEndian, int32(ringSize))          // ring_size
	binary.Write(&buf, binary.LittleEndian, int32(qPrimeSize))       // Q_prime_size_
	binary.Write(&buf, binary.LittleEndian, int32(qSize))            // Q_size_
	binary.Write(&buf, binary.LittleEndian, int32(d))                // d_
	binary.Write(&buf, binary.LittleEndian, true)                    // customized (use custom galois elements)
	binary.Write(&buf, binary.LittleEndian, groupOrder)              // group_order_
	binary.Write(&buf, binary.LittleEndian, heongpuStorageTypeHost)  // storage_type_
	binary.Write(&buf, binary.LittleEndian, true)                    // galois_key_generated_

	// Custom galois elements (we use the "customized" path)
	customGaloisElts := make([]uint32, len(galoisElements))
	for i, el := range galoisElements {
		customGaloisElts[i] = uint32(el)
	}
	binary.Write(&buf, binary.LittleEndian, uint32(len(customGaloisElts)))
	for _, el := range customGaloisElts {
		binary.Write(&buf, binary.LittleEndian, el)
	}

	// galois_elt_zero (conjugation element = 2N - 1)
	galoisEltZero := int32(2*ringSize - 1)
	binary.Write(&buf, binary.LittleEndian, galoisEltZero)

	// galoiskey_size_
	binary.Write(&buf, binary.LittleEndian, galoisKeySize)

	// Key data: key_count + per-key data + zero_key data
	binary.Write(&buf, binary.LittleEndian, uint32(len(galoisKeys)))

	for i, gk := range galoisKeys {
		galoisElt := int32(galoisElements[i])
		binary.Write(&buf, binary.LittleEndian, galoisElt)

		// Extract raw coefficients from Lattigo's GadgetCiphertext.
		// Layout: for each decomposition level d, for each polynomial (c0, c1),
		//         flatten all Q levels then P levels.
		keyData, err := extractGadgetData(gk.GadgetCiphertext, d, qSize, pSize, ringSize)
		if err != nil {
			return nil, fmt.Errorf("galois key %d (element %d): %w", i, galoisElements[i], err)
		}
		if int64(len(keyData)) != galoisKeySize {
			return nil, fmt.Errorf("galois key %d: expected %d uint64, got %d",
				i, galoisKeySize, len(keyData))
		}

		// Write as raw uint64 little-endian
		for _, v := range keyData {
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}

	// Zero key (conjugation key) — write zeros since we don't use conjugation
	zeroData := make([]uint64, galoisKeySize)
	for _, v := range zeroData {
		binary.Write(&buf, binary.LittleEndian, v)
	}

	return buf.Bytes(), nil
}

// extractGadgetData extracts the raw uint64 coefficients from a Lattigo GadgetCiphertext
// in the order expected by HEonGPU's galoiskey_gen_kernel output.
//
// HEonGPU layout (KEYSWITCHING_METHOD_I):
// For d decomposition levels, each level has 2 polynomials (c0, c1) in ring Q×P.
// Total: d * 2 * Q_prime_size * ring_size uint64 values.
//
// The exact ordering within the flat array:
// [decomp0_poly0_level0[N] | decomp0_poly0_level1[N] | ... | decomp0_poly0_levelQP-1[N] |
//  decomp0_poly1_level0[N] | ... | decomp0_poly1_levelQP-1[N] |
//  decomp1_poly0_level0[N] | ... ]
func extractGadgetData(gc rlwe.GadgetCiphertext, d, qSize, pSize, ringSize int) ([]uint64, error) {
	qPrimeSize := qSize + pSize
	totalSize := 2 * d * qPrimeSize * ringSize
	data := make([]uint64, totalSize)

	if gc.BaseRNSDecompositionVectorSize() != d {
		return nil, fmt.Errorf("decomposition size mismatch: got %d, want %d",
			gc.BaseRNSDecompositionVectorSize(), d)
	}

	offset := 0
	for decomp := 0; decomp < d; decomp++ {
		if len(gc.Value[decomp]) != 1 {
			return nil, fmt.Errorf("expected 1 base2 decomposition, got %d at decomp %d",
				len(gc.Value[decomp]), decomp)
		}

		vqp := gc.Value[decomp][0] // VectorQP: [degree+1]ringqp.Poly
		if len(vqp) != 2 {
			return nil, fmt.Errorf("expected degree 1 (2 polys), got %d at decomp %d",
				len(vqp), decomp)
		}

		for poly := 0; poly < 2; poly++ {
			// Q levels
			for lvl := 0; lvl < qSize; lvl++ {
				if lvl > vqp[poly].Q.Level() {
					// Pad with zeros if this level doesn't exist
					offset += ringSize
					continue
				}
				copy(data[offset:offset+ringSize], vqp[poly].Q.Coeffs[lvl])
				offset += ringSize
			}
			// P levels
			for lvl := 0; lvl < pSize; lvl++ {
				if lvl > vqp[poly].P.Level() {
					offset += ringSize
					continue
				}
				copy(data[offset:offset+ringSize], vqp[poly].P.Coeffs[lvl])
				offset += ringSize
			}
		}
	}

	if offset != totalSize {
		return nil, fmt.Errorf("wrote %d uint64, expected %d", offset, totalSize)
	}

	return data, nil
}

// findGenerator finds the smallest generator of Z/(2N)Z* for NTT.
// Standard choice for power-of-2 N in CKKS is typically 3 or 5.
func findGenerator(m int) int {
	// For m = 2N where N is power of 2, the multiplicative group Z/mZ*
	// has order N. We need a primitive N-th root of unity mod m.
	// Standard generators: try 3, 5, etc.
	n := m / 2
	for _, g := range []int{3, 5, 7, 11, 13} {
		if isPrimitiveRoot(g, n, m) {
			return g
		}
	}
	return 3 // fallback
}

func isPrimitiveRoot(g, order, m int) bool {
	val := 1
	for i := 0; i < order; i++ {
		val = (val * g) % m
		if val == 1 && i < order-1 {
			return false
		}
	}
	return val == 1
}

// SerializeRelinKeyHEonGPU converts a Lattigo RelinearizationKey to HEonGPU binary format.
func SerializeRelinKeyHEonGPU(
	params hefloat.Parameters,
	rlk *rlwe.RelinearizationKey,
) ([]byte, error) {
	ringSize := 1 << params.LogN()
	qSize := params.MaxLevelQ() + 1
	pSize := params.MaxLevelP() + 1
	qPrimeSize := qSize + pSize

	if pSize != 1 {
		return nil, fmt.Errorf("HEonGPU format requires 1 P prime; use NewParametersGPU()")
	}

	d := qSize
	relinKeySize := int64(2 * d * qPrimeSize * ringSize)

	var buf bytes.Buffer

	// Header
	binary.Write(&buf, binary.LittleEndian, heongpuSchemeTypeCKKS)
	binary.Write(&buf, binary.LittleEndian, heongpuKeySwitchingMethodI)
	binary.Write(&buf, binary.LittleEndian, int32(ringSize))
	binary.Write(&buf, binary.LittleEndian, int32(qPrimeSize))
	binary.Write(&buf, binary.LittleEndian, int32(qSize))
	binary.Write(&buf, binary.LittleEndian, int32(d))

	// d_tilda (for method I, d_tilda = 1)
	binary.Write(&buf, binary.LittleEndian, int32(1))
	// r_prime_ (for method I, r_prime_ = Q_prime_size)
	binary.Write(&buf, binary.LittleEndian, int32(qPrimeSize))

	binary.Write(&buf, binary.LittleEndian, heongpuStorageTypeHost)
	binary.Write(&buf, binary.LittleEndian, true) // relin_key_generated_
	binary.Write(&buf, binary.LittleEndian, relinKeySize)

	// Key data
	keyData, err := extractGadgetData(rlk.GadgetCiphertext, d, qSize, pSize, ringSize)
	if err != nil {
		return nil, fmt.Errorf("relin key: %w", err)
	}

	for _, v := range keyData {
		binary.Write(&buf, binary.LittleEndian, v)
	}

	return buf.Bytes(), nil
}

// Suppress unused import warning
var _ = math.Log
