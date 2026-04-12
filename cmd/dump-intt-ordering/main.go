// dump-intt-ordering produces Lattigo's INTT output for a known NTT-domain input.
// Compare these values with HEonGPU's INTT output to find the index permutation.
//
// Input: NTT-domain polynomial where index i has value (i+1) mod Q
// This makes each position unique and easy to trace through the permutation.
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

	N := 1 << params.LogN() // 16384
	subRing := params.RingQ().SubRings[0]
	Q := subRing.Modulus

	fmt.Printf("N=%d, Q[0]=%d\n\n", N, Q)

	// Create known NTT-domain input: value at position i = (i+1) mod Q
	nttInput := make([]uint64, N)
	for i := 0; i < N; i++ {
		nttInput[i] = uint64(i+1) % Q
	}

	fmt.Printf("NTT input[0..9]: ")
	for i := 0; i < 10; i++ {
		fmt.Printf("%d ", nttInput[i])
	}
	fmt.Println()

	// Apply Lattigo INTT
	coeffOutput := make([]uint64, N)
	ring.INTTStandard(nttInput, coeffOutput, N, subRing.NInv, subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)

	fmt.Printf("\nLattigo INTT output (first 20):\n")
	for i := 0; i < 20; i++ {
		fmt.Printf("  [%5d] = %d\n", i, coeffOutput[i])
	}

	// Also print at specific positions that would reveal bit-reversal
	fmt.Printf("\nLattigo INTT at key positions:\n")
	positions := []int{0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
	for _, p := range positions {
		if p < N {
			fmt.Printf("  [%5d] = %d\n", p, coeffOutput[p])
		}
	}

	// Print the values formatted for C++ comparison
	fmt.Printf("\n// For C++ comparison - Lattigo INTT first 20 values:\n")
	fmt.Printf("// uint64_t lattigo_intt[] = {")
	for i := 0; i < 20; i++ {
		if i > 0 {
			fmt.Printf(", ")
		}
		fmt.Printf("%dULL", coeffOutput[i])
	}
	fmt.Printf("};\n")
}
