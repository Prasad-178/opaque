// verify-ntt-go verifies that a Go implementation of HEonGPU's NTT
// produces correct results by doing:
// 1. Take a known polynomial [1, 2, 3, 0, ...]
// 2. NTT with Lattigo → Lattigo NTT domain
// 3. INTT with Lattigo → back to [1, 2, 3, 0, ...] (coefficient domain)
// 4. NTT with our Go implementation of HEonGPU's NTT → should match HEonGPU
//
// HEonGPU's NTT table: table[bitrev(j)] = psi^j for j=0..N-1
// Same butterfly as Lattigo: Cooley-Tukey, roots[m+i] indexing
// But in STANDARD form (not Montgomery)
package main

import (
	"fmt"
	"log"
	"math/big"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func main() {
	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	N := 1 << params.LogN()
	subRing := params.RingQ().SubRings[0]
	Q := subRing.Modulus

	// HEonGPU psi for Q[0] (from GPU server)
	psiH := uint64(62213374832584)

	fmt.Printf("N=%d, Q[0]=%d, psi_H=%d\n", N, Q, psiH)

	// Step 1: Known polynomial [1, 2, 3, 0, ...]
	coeffs := make([]uint64, N)
	coeffs[0] = 1
	coeffs[1] = 2
	coeffs[2] = 3

	// Step 2: NTT with Lattigo
	lattigoNTT := make([]uint64, N)
	ring.NTTStandard(coeffs, lattigoNTT, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)
	fmt.Printf("Lattigo NTT[0..4]: %d %d %d %d %d\n", lattigoNTT[0], lattigoNTT[1], lattigoNTT[2], lattigoNTT[3], lattigoNTT[4])

	// Step 3: INTT with Lattigo → back to coefficient domain
	recovered := make([]uint64, N)
	ring.INTTStandard(lattigoNTT, recovered, N, subRing.NInv, subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)
	fmt.Printf("Lattigo INTT[0..4]: %d %d %d %d %d (should be 1 2 3 0 0)\n",
		recovered[0], recovered[1], recovered[2], recovered[3], recovered[4])

	// Step 4: NTT with HEonGPU's table (Go implementation)
	// Generate HEonGPU's NTT table: table[bitrev(j)] = psi^j
	table := generateHEonGPUTable(psiH, Q, N)

	// Apply CT butterfly with roots[m+i] indexing (same as both libraries)
	heongpuNTT := make([]uint64, N)
	copy(heongpuNTT, recovered) // start from coefficient domain
	nttCooleyTukey(heongpuNTT, N, Q, table)

	fmt.Printf("Go HEonGPU NTT[0..4]: %d %d %d %d %d\n",
		heongpuNTT[0], heongpuNTT[1], heongpuNTT[2], heongpuNTT[3], heongpuNTT[4])

	// Compare with expected HEonGPU output (from GPU run):
	// HEonGPU NTT of [1,2,3,0,...] under Q[0]:
	// 1113637472726183760 1113388619226853424 261474998253597111 970263423613611728 932540745375191153
	expected := []uint64{1113637472726183760, 1113388619226853424, 261474998253597111, 970263423613611728, 932540745375191153}

	fmt.Printf("\nExpected HEonGPU[0..4]: %d %d %d %d %d\n",
		expected[0], expected[1], expected[2], expected[3], expected[4])

	allMatch := true
	for i := 0; i < 5; i++ {
		if heongpuNTT[i] != expected[i] {
			allMatch = false
			fmt.Printf("  MISMATCH at [%d]: got %d, want %d\n", i, heongpuNTT[i], expected[i])
		}
	}
	if allMatch {
		fmt.Println("\n✓ Go HEonGPU NTT MATCHES GPU output!")
		fmt.Println("The coefficient domain IS correct. The full conversion works in Go.")
		fmt.Println("The GPU-side NTT call must have wrong parameters.")
	}
}

// generateHEonGPUTable generates the NTT root table exactly as HEonGPU does:
// powers[j] = psi^j mod p for j=0..N-1, then bit-reverse
func generateHEonGPUTable(psi, p uint64, N int) []uint64 {
	logN := 0
	for n := N; n > 1; n >>= 1 {
		logN++
	}

	pBig := new(big.Int).SetUint64(p)
	psiBig := new(big.Int).SetUint64(psi)

	// powers[j] = psi^j mod p
	powers := make([]uint64, N)
	powers[0] = 1
	cur := big.NewInt(1)
	for j := 1; j < N; j++ {
		cur.Mul(cur, psiBig)
		cur.Mod(cur, pBig)
		powers[j] = cur.Uint64()
	}

	// Bit-reverse: table[bitrev(j)] = powers[j]
	table := make([]uint64, N)
	for j := 0; j < N; j++ {
		table[bitReverse(j, logN)] = powers[j]
	}

	return table
}

// nttCooleyTukey applies the Cooley-Tukey butterfly NTT in-place.
// Uses roots[m+i] indexing, identical to both Lattigo and HEonGPU.
// Roots are in STANDARD form (not Montgomery).
func nttCooleyTukey(p []uint64, N int, Q uint64, roots []uint64) {
	t := N >> 1
	QBig := new(big.Int).SetUint64(Q)

	// First butterfly (m=1)
	W := roots[1]
	for j := 0; j < t; j++ {
		u := p[j]
		v := mulmod(p[j+t], W, Q, QBig)
		p[j] = addmod(u, v, Q)
		p[j+t] = submod(u, v, Q)
	}

	// Remaining butterflies
	for m := 2; m < N; m <<= 1 {
		t >>= 1
		for i := 0; i < m; i++ {
			j1 := (i * t) << 1
			j2 := j1 + t
			W := roots[m+i]
			for j := j1; j < j2; j++ {
				u := p[j]
				v := mulmod(p[j+t], W, Q, QBig)
				p[j] = addmod(u, v, Q)
				p[j+t] = submod(u, v, Q)
			}
		}
	}
}

func mulmod(a, b, m uint64, mBig *big.Int) uint64 {
	r := new(big.Int).Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(b))
	r.Mod(r, mBig)
	return r.Uint64()
}

func addmod(a, b, m uint64) uint64 {
	s := a + b
	if s >= m {
		s -= m
	}
	return s
}

func submod(a, b, m uint64) uint64 {
	if a >= b {
		return a - b
	}
	return m - (b - a)
}

func bitReverse(v, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (v & 1)
		v >>= 1
	}
	return result
}
