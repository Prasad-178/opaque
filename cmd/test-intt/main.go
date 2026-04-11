// test-intt verifies that our Lattigo INTT → coefficient domain conversion
// produces correct results by round-tripping: NTT → INTT → NTT should equal identity.
package main

import (
	"fmt"
	"log"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func main() {
	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	N := 1 << params.LogN()
	ringQ := params.RingQ()

	// Create a simple polynomial in coefficient domain
	coeffDomain := make([]uint64, N)
	coeffDomain[0] = 1
	coeffDomain[1] = 2
	coeffDomain[2] = 3

	fmt.Printf("Original coefficients (first 5): %v\n", coeffDomain[:5])

	// Apply Lattigo's NTT
	subRing := ringQ.SubRings[0] // First Q prime
	nttDomain := make([]uint64, N)
	ring.NTTStandard(coeffDomain, nttDomain, N, subRing.Modulus, subRing.MRedConstant, subRing.BRedConstant, subRing.RootsForward)

	fmt.Printf("After NTT (first 5): %v\n", nttDomain[:5])

	// Apply Lattigo's INTT — should recover original
	recovered := make([]uint64, N)
	ring.INTTStandard(nttDomain, recovered, N, subRing.NInv, subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)

	fmt.Printf("After INTT (first 5): %v\n", recovered[:5])

	// Check round-trip
	match := true
	for i := 0; i < N; i++ {
		if recovered[i] != coeffDomain[i] {
			fmt.Printf("MISMATCH at %d: expected %d, got %d\n", i, coeffDomain[i], recovered[i])
			match = false
			break
		}
	}
	if match {
		fmt.Println("✓ Round-trip NTT → INTT is correct!")
	}

	// Now test our NTTConverter.ConvertToCoeffDomain
	conv := crypto.NewNTTConverter(params)

	nttCopy := make([]uint64, N)
	copy(nttCopy, nttDomain)

	conv.ConvertToCoeffDomain(nttCopy, 0) // level 0

	fmt.Printf("After ConvertToCoeffDomain (first 5): %v\n", nttCopy[:5])

	match2 := true
	for i := 0; i < N; i++ {
		if nttCopy[i] != coeffDomain[i] {
			fmt.Printf("ConvertToCoeffDomain MISMATCH at %d: expected %d, got %d\n", i, coeffDomain[i], nttCopy[i])
			match2 = false
			break
		}
	}
	if match2 {
		fmt.Println("✓ ConvertToCoeffDomain matches Lattigo INTT!")
	} else {
		fmt.Println("✗ ConvertToCoeffDomain does NOT match!")
	}
}
