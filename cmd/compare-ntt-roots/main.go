// compare-ntt-roots computes NTT roots using HEonGPU's algorithm and
// compares them with Lattigo's roots to find the mismatch.
//
// HEonGPU: psi = smallest_generator^((p-1)/(2N)) mod p
// Lattigo: uses its own root selection via ring.SubRing
//
// If roots differ, we know the conversion factor needed.
package main

import (
	"fmt"
	"log"
	"math/big"

	"github.com/Prasad-178/opaque/pkg/crypto"
)

func main() {
	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	N := uint64(1 << params.LogN()) // 16384
	ringQ := params.RingQ()

	fmt.Println("=== NTT Root Comparison: Lattigo vs HEonGPU ===")
	fmt.Printf("N = %d, 2N = %d\n\n", N, 2*N)

	allPrimes := ringQ.ModuliChain()
	if ringP := params.RingP(); ringP != nil {
		allPrimes = append(allPrimes, ringP.ModuliChain()...)
	}

	for i, p := range allPrimes {
		label := fmt.Sprintf("Q[%d]", i)
		if i >= params.MaxLevelQ()+1 {
			label = fmt.Sprintf("P[%d]", i-params.MaxLevelQ()-1)
		}

		// Lattigo's root
		var lattigoRoot uint64
		if i <= params.MaxLevelQ() {
			lattigoRoot = ringQ.SubRings[i].RootsForward[1]
		} else {
			ringP := params.RingP()
			pIdx := i - params.MaxLevelQ() - 1
			lattigoRoot = ringP.SubRings[pIdx].RootsForward[1]
		}

		// HEonGPU's root: smallest generator g, psi = g^((p-1)/(2N)) mod p
		heongpuRoot := computeHEonGPURoot(p, N)

		match := "✗ MISMATCH"
		if lattigoRoot == heongpuRoot {
			match = "✓ MATCH"
		}

		fmt.Printf("%s (mod %d):\n", label, p)
		fmt.Printf("  Lattigo ψ:  %d\n", lattigoRoot)
		fmt.Printf("  HEonGPU ψ:  %d\n", heongpuRoot)
		fmt.Printf("  %s\n\n", match)
	}
}

// computeHEonGPURoot replicates HEonGPU's NTT root computation:
// Find smallest primitive root g of Z/pZ*, compute psi = g^((p-1)/(2N)) mod p.
func computeHEonGPURoot(p, N uint64) uint64 {
	pBig := new(big.Int).SetUint64(p)
	pm1 := new(big.Int).Sub(pBig, big.NewInt(1))              // p-1
	exp := new(big.Int).Div(pm1, new(big.Int).SetUint64(2*N)) // (p-1)/(2N)

	// Find smallest primitive root of Z/pZ*
	// A generator g has order p-1, meaning g^((p-1)/q) != 1 mod p
	// for all prime factors q of p-1.
	//
	// HEonGPU (GPU-NTT) uses the smallest such g, typically 2, 3, 5, etc.

	// Factor p-1 (we only need to check small prime factors for the primitive root test)
	factors := primeFactors(pm1)

	for g := uint64(2); g < 1000; g++ {
		gBig := new(big.Int).SetUint64(g)

		isPrimRoot := true
		for _, f := range factors {
			// Check g^((p-1)/f) mod p != 1
			subExp := new(big.Int).Div(pm1, f)
			result := new(big.Int).Exp(gBig, subExp, pBig)
			if result.Cmp(big.NewInt(1)) == 0 {
				isPrimRoot = false
				break
			}
		}

		if isPrimRoot {
			// Compute psi = g^((p-1)/(2N)) mod p
			psi := new(big.Int).Exp(gBig, exp, pBig)

			// Verify: psi^N mod p should NOT be 1 (psi is 2N-th root, not N-th)
			psiN := new(big.Int).Exp(psi, new(big.Int).SetUint64(N), pBig)
			if psiN.Cmp(big.NewInt(1)) != 0 {
				return psi.Uint64()
			}
			// If psi^N == 1, this generator gives an N-th root, not 2N-th. Try next.
		}
	}

	return 0
}

// primeFactors returns the distinct prime factors of n.
func primeFactors(n *big.Int) []*big.Int {
	var factors []*big.Int
	d := big.NewInt(2)
	tmp := new(big.Int).Set(n)

	for d.Cmp(new(big.Int).Sqrt(tmp)) <= 0 {
		if new(big.Int).Mod(tmp, d).Sign() == 0 {
			factors = append(factors, new(big.Int).Set(d))
			for new(big.Int).Mod(tmp, d).Sign() == 0 {
				tmp.Div(tmp, d)
			}
		}
		d.Add(d, big.NewInt(1))
		// Optimization: after checking 2, only check odd numbers
		if d.Cmp(big.NewInt(3)) == 0 {
			continue
		}
		if d.Cmp(big.NewInt(4)) >= 0 && new(big.Int).Mod(d, big.NewInt(2)).Sign() == 0 {
			d.Add(d, big.NewInt(1))
		}
	}
	if tmp.Cmp(big.NewInt(1)) > 0 {
		factors = append(factors, tmp)
	}
	return factors
}
