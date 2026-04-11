// dump-ntt-roots extracts the NTT primitive roots of unity from Lattigo
// for comparison with HEonGPU's roots.
//
// If the roots differ between libraries, we need to convert between
// NTT domains when transferring evaluation key coefficients.
//
// Run: go run ./cmd/dump-ntt-roots/
package main

import (
	"fmt"
	"log"

	"github.com/Prasad-178/opaque/pkg/crypto"
)

func main() {
	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	ringQ := params.RingQ()

	fmt.Println("=== Lattigo NTT Roots for GPU-Compatible Parameters ===")
	fmt.Printf("LogN: %d, N: %d\n", params.LogN(), 1<<params.LogN())
	fmt.Printf("Q primes: %d, P primes: %d\n", params.MaxLevelQ()+1, params.MaxLevelP()+1)
	fmt.Println()

	// For each Q prime, print the primitive root and first few NTT twiddle factors
	for lvl := 0; lvl <= params.MaxLevelQ(); lvl++ {
		subRing := ringQ.SubRings[lvl]
		modulus := subRing.Modulus
		nttRoots := subRing.RootsForward

		fmt.Printf("--- Q[%d]: modulus = %d ---\n", lvl, modulus)
		fmt.Printf("  BRedConstant: %v\n", subRing.BRedConstant)
		fmt.Printf("  MRedConstant: %d\n", subRing.MRedConstant)
		fmt.Printf("  NTT roots (first 8): ")
		for i := 0; i < 8 && i < len(nttRoots); i++ {
			fmt.Printf("%d ", nttRoots[i])
		}
		fmt.Println()
		fmt.Printf("  NTT roots count: %d\n", len(nttRoots))
		fmt.Printf("  roots[1] (ψ or primitive 2N-th root): %d\n", nttRoots[1])
		fmt.Println()
	}

	// Also print P prime roots
	ringP := params.RingP()
	if ringP != nil {
		for lvl := 0; lvl <= params.MaxLevelP(); lvl++ {
			subRing := ringP.SubRings[lvl]
			modulus := subRing.Modulus
			nttRoots := subRing.RootsForward

			fmt.Printf("--- P[%d]: modulus = %d ---\n", lvl, modulus)
			fmt.Printf("  NTT roots (first 8): ")
			for i := 0; i < 8 && i < len(nttRoots); i++ {
				fmt.Printf("%d ", nttRoots[i])
			}
			fmt.Println()
			fmt.Printf("  roots[1] (ψ): %d\n", nttRoots[1])
			fmt.Println()
		}
	}
}
