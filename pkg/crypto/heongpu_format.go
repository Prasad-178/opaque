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

// HEonGPU binary format constants.
// These enums are all uint8_t in HEonGPU (1 byte each).
const (
	heongpuSchemeTypeCKKS       uint8 = 2 // scheme_type::ckks
	heongpuKeySwitchingMethodI  uint8 = 1 // keyswitching_type::KEYSWITCHING_METHOD_I
	heongpuStorageTypeHost      uint8 = 1 // storage_type::HOST
)

// SerializeGaloisKeysHEonGPU converts Lattigo Galois keys to HEonGPU's binary format.
// The output can be loaded by HEonGPU's Galoiskey::load() method.
//
// Requires GPU-compatible parameters (NewParametersGPU with LogP=[61]).
// If serverNTTRoots is provided (from InitContext), uses those exact roots
// for NTT domain conversion. If nil, computes roots locally (may not match server).
func SerializeGaloisKeysHEonGPU(
	params hefloat.Parameters,
	galoisKeys []*rlwe.GaloisKey,
	galoisElements []uint64,
	serverNTTRoots []uint64,
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

	// Create NTT domain converter (Lattigo NTT → HEonGPU NTT)
	var converter *NTTConverter
	if len(serverNTTRoots) > 0 {
		converter = NewNTTConverterWithRoots(params, serverNTTRoots)
	} else {
		converter = NewNTTConverter(params)
	}

	// For KEYSWITCHING_METHOD_I: d_ is NOT used (stays 0 in HEonGPU).
	// galoiskey_size = 2 * Q_size * Q_prime_size * ring_size.
	d := 0 // d_ is 0 for METHOD_I (only set for METHOD_II)
	galoisKeySize := int64(2 * qSize * qPrimeSize * ringSize)

	// Group order (generator of multiplicative group mod 2N).
	// HEonGPU uses 5 as the generator for N=16384 (verified via native key dump).
	// This matches Lattigo's GaloisElement(1) = 5.
	groupOrder := int32(5) // Hardcoded to match HEonGPU's default for N=16384

	var buf bytes.Buffer

	// Header — types must EXACTLY match HEonGPU's save() sizeof() calls.
	// Enums are uint8_t (1 byte). Ints are int (4 bytes). Bool is 1 byte.
	// galoiskey_size_ is Data64 = uint64_t (8 bytes).
	binary.Write(&buf, binary.LittleEndian, heongpuSchemeTypeCKKS)      // scheme_type (uint8, 1B)
	binary.Write(&buf, binary.LittleEndian, heongpuKeySwitchingMethodI) // keyswitching_type (uint8, 1B)
	binary.Write(&buf, binary.LittleEndian, int32(ringSize))            // ring_size (int, 4B)
	binary.Write(&buf, binary.LittleEndian, int32(qPrimeSize))         // Q_prime_size_ (int, 4B)
	binary.Write(&buf, binary.LittleEndian, int32(qSize))              // Q_size_ (int, 4B)
	binary.Write(&buf, binary.LittleEndian, int32(d))                  // d_ (int, 4B)
	// customized = false — use the galois_elt map format.
	// (HEonGPU's load() has a bug in the customized=true path: it passes
	// the uninitialized size variable as the read length instead of sizeof.)
	binary.Write(&buf, binary.LittleEndian, uint8(0))                  // customized (bool, 1B) = false
	binary.Write(&buf, binary.LittleEndian, groupOrder)                // group_order_ (int, 4B)
	binary.Write(&buf, binary.LittleEndian, heongpuStorageTypeHost)    // storage_type (uint8, 1B)
	binary.Write(&buf, binary.LittleEndian, uint8(1))                  // galois_key_generated_ (bool, 1B) = true

	// Galois element map: pairs of {shift_step (int) → galois_element (int)}.
	// HEonGPU's load() reads: uint32 count, then count × (int first, int second).
	// The key data map uses galois_element as the key (galois_elt[first] = second
	// in save(), but load() reads first/second and does galois_elt[first] = second).
	// From save(): galois_elt map stores {rotation_step → galois_element}.
	// But wait — save() iterates the map and writes {first, second} = {step, element}.
	// And load() reads them and does galois_elt[first] = second.
	// The device_location_ map uses galois_element (second) as the key.
	// So we write: count, then pairs of {rotation_power, galois_element}.
	//
	// For our rotations by powers of 2: step 1→elt 5, step 2→elt 25, etc.
	// Also need negative steps: step -1→elt X, step -2→elt Y, etc.
	// HEonGPU generates both positive and negative rotation keys.
	//
	// Actually, looking at the save() code more carefully: the key_data in
	// device_location_ is keyed by galois_element (galois.second), not by step.
	// So the map entry {step, element} tells load() to register that galois_element,
	// and the data section is keyed by galois_element.
	type galoisPair struct {
		step    int32
		element int32
	}
	var pairs []galoisPair
	for i, el := range galoisElements {
		step := int32(1 << i)
		pairs = append(pairs, galoisPair{step: step, element: int32(el)})
	}

	binary.Write(&buf, binary.LittleEndian, uint32(len(pairs))) // galois_elt_size
	for _, p := range pairs {
		binary.Write(&buf, binary.LittleEndian, p.step)    // int first (rotation step)
		binary.Write(&buf, binary.LittleEndian, p.element) // int second (galois element)
	}

	// galois_elt_zero (conjugation galois element = 2N - 1)
	galoisEltZero := int32(2*ringSize - 1)
	binary.Write(&buf, binary.LittleEndian, galoisEltZero) // int, 4B

	// galoiskey_size_ — Data64 = uint64_t, 8 bytes
	binary.Write(&buf, binary.LittleEndian, uint64(galoisKeySize)) // uint64, 8B

	// Key data section: key_count + per-key {galois_element (int), Data64[] data} + zero_key {Data64[] data}
	// The map key is galois_element (matching device_location_/host_location_ map keys in HEonGPU).
	binary.Write(&buf, binary.LittleEndian, uint32(len(galoisKeys))) // key_count (uint32)

	for i, gk := range galoisKeys {
		binary.Write(&buf, binary.LittleEndian, int32(galoisElements[i])) // galois_element (int, 4B)

		// Extract raw coefficients from Lattigo's GadgetCiphertext.
		// Layout: for each decomposition level d, for each polynomial (c0, c1),
		//         flatten all Q levels then P levels.
		// Note: d_ in header is 0 for METHOD_I, but data has qSize decomposition levels.
		// Apply NTT domain conversion: Lattigo NTT → HEonGPU NTT
		keyData, err := extractGadgetDataConverted(gk.GadgetCiphertext, qSize, qSize, pSize, ringSize, converter)
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
// extractGadgetDataConverted extracts coefficients and converts them from
// Lattigo's NTT domain directly to HEonGPU's NTT domain.
// The full conversion (Lattigo INTT → coeff → HEonGPU NTT) is done in Go.
// This is verified to produce byte-identical output to HEonGPU's GPU NTT.
func extractGadgetDataConverted(gc rlwe.GadgetCiphertext, d, qSize, pSize, ringSize int, conv *NTTConverter) ([]uint64, error) {
	data, err := extractGadgetData(gc, d, qSize, pSize, ringSize)
	if err != nil {
		return nil, err
	}

	// Convert each polynomial: Lattigo NTT → coeff domain → HEonGPU NTT
	qPrimeSize := qSize + pSize
	for decomp := 0; decomp < d; decomp++ {
		for poly := 0; poly < 2; poly++ {
			for lvl := 0; lvl < qPrimeSize; lvl++ {
				offset := (decomp*2*qPrimeSize + poly*qPrimeSize + lvl) * ringSize
				conv.ConvertToHEonGPUDomain(data[offset:offset+ringSize], lvl)
			}
		}
	}

	return data, nil
}

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
