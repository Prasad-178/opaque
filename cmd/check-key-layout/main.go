// check-key-layout prints the internal structure of Lattigo's evaluation keys
// for comparison with HEonGPU's expected layout.
//
// Run: go run ./cmd/check-key-layout/
package main

import (
	"fmt"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════╗")
	fmt.Println("║  COMPARING CPU PATH [61,61] vs GPU PATH [61]    ║")
	fmt.Println("╚══════════════════════════════════════════════════╝")

	fmt.Println("\n========== CPU PATH (LogP = [61, 61]) ==========")
	inspectParams(crypto.NewParameters())

	fmt.Println("\n========== GPU PATH (LogP = [61]) ==========")
	inspectParams(crypto.NewParametersGPU())
}

func inspectParams(params hefloat.Parameters, err error) {
	if err != nil {
		panic(err)
	}

	fmt.Println("=== CKKS Parameters ===")
	fmt.Printf("LogN:          %d (N=%d)\n", params.LogN(), 1<<params.LogN())
	fmt.Printf("MaxLevelQ:     %d (Q has %d primes)\n", params.MaxLevelQ(), params.MaxLevelQ()+1)
	fmt.Printf("MaxLevelP:     %d (P has %d primes)\n", params.MaxLevelP(), params.MaxLevelP()+1)
	fmt.Printf("LogDefaultScale: %d\n", params.LogDefaultScale())

	fmt.Println("\n=== Q Moduli (ciphertext chain) ===")
	qModuli := params.RingQ().ModuliChain()
	for i, q := range qModuli {
		fmt.Printf("  Q[%d]: %d (bits: %d)\n", i, q, bitLen(q))
	}

	fmt.Println("\n=== P Moduli (key-switching chain) ===")
	pModuli := params.RingP().ModuliChain()
	for i, p := range pModuli {
		fmt.Printf("  P[%d]: %d (bits: %d)\n", i, p, bitLen(p))
	}

	fmt.Println("\n=== Galois Key Structure ===")

	// Create engine from THESE params (not default)
	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		panic(err)
	}

	// Access the eval key set
	if engine.GetEvalKeys() == nil {
		fmt.Println("ERROR: no eval keys")
		return
	}

	// Check GaloisKey structure
	galoisElems := galoisElements(params)
	fmt.Printf("Galois elements needed: %d\n", len(galoisElems))
	for i, el := range galoisElems {
		fmt.Printf("  Element[%d]: %d (rotation by %d)\n", i, el, 1<<i)
	}

	// Get one Galois key and inspect its structure
	gk, err := engine.GetEvalKeys().GetGaloisKey(galoisElems[0])
	if err != nil {
		fmt.Printf("ERROR getting Galois key: %v\n", err)
		return
	}

	fmt.Println("\n=== GaloisKey Internal Structure ===")
	fmt.Printf("GaloisElement:   %d\n", gk.GaloisElement)
	fmt.Printf("NthRoot:         %d\n", gk.NthRoot)

	gc := gk.GadgetCiphertext
	fmt.Printf("BaseTwoDecomposition: %d\n", gc.BaseTwoDecomposition)
	fmt.Printf("BaseRNSDecompositionVectorSize: %d\n", gc.BaseRNSDecompositionVectorSize())
	fmt.Printf("BaseTwoDecompositionVectorSize: %v\n", gc.BaseTwoDecompositionVectorSize())

	fmt.Println("\n=== Per-entry structure ===")
	for i := 0; i < gc.BaseRNSDecompositionVectorSize(); i++ {
		for j := 0; j < len(gc.Value[i]); j++ {
			vqp := gc.Value[i][j]
			fmt.Printf("  Value[%d][%d]: %d polynomials (degree)\n", i, j, len(vqp))
			if len(vqp) > 0 {
				fmt.Printf("    Poly[0].Q: %d levels × %d coeffs\n",
					vqp[0].Q.Level()+1, vqp[0].Q.N())
				fmt.Printf("    Poly[0].P: %d levels × %d coeffs\n",
					vqp[0].P.Level()+1, vqp[0].P.N())
			}
		}
	}

	// Dump actual P level count from the key
	fmt.Printf("\n=== Actual P levels in Galois key polynomials ===\n")
	for i := 0; i < gc.BaseRNSDecompositionVectorSize() && i < 1; i++ {
		for j := 0; j < len(gc.Value[i]) && j < 1; j++ {
			for k := 0; k < len(gc.Value[i][j]); k++ {
				pLevel := gc.Value[i][j][k].P.Level()
				pN := gc.Value[i][j][k].P.N()
				qLevel := gc.Value[i][j][k].Q.Level()
				fmt.Printf("  Value[%d][%d][%d]: Q.Level=%d Q.N=%d  P.Level=%d P.N=%d\n",
					i, j, k, qLevel, gc.Value[i][j][k].Q.N(), pLevel, pN)
			}
		}
	}

	// Calculate total raw data size
	totalUint64 := 0
	for i := 0; i < gc.BaseRNSDecompositionVectorSize(); i++ {
		for j := 0; j < len(gc.Value[i]); j++ {
			for k := 0; k < len(gc.Value[i][j]); k++ {
				qLevels := gc.Value[i][j][k].Q.Level() + 1
				pLevels := gc.Value[i][j][k].P.Level() + 1
				N := gc.Value[i][j][k].Q.N()
				totalUint64 += (qLevels + pLevels) * N
			}
		}
	}
	fmt.Printf("\nTotal raw data per Galois key: %d uint64 = %d bytes = %.1f MB\n",
		totalUint64, totalUint64*8, float64(totalUint64*8)/(1024*1024))
	fmt.Printf("Total for all %d Galois keys: %.1f MB\n",
		len(galoisElems), float64(totalUint64*8*len(galoisElems))/(1024*1024))

	// HEonGPU comparison values
	Q_size := params.MaxLevelQ() + 1                // number of Q primes
	Q_prime_size := Q_size + params.MaxLevelP() + 1 // Q + P primes
	d := Q_size                                     // RNS decomposition (one per Q prime, standard approach)
	galoiskey_size_heongpu := 2 * d * Q_prime_size * (1 << params.LogN())

	fmt.Printf("\n=== HEonGPU Expected Layout (estimated) ===\n")
	fmt.Printf("Q_size (num Q primes):    %d\n", Q_size)
	fmt.Printf("Q_prime_size (Q+P):       %d\n", Q_prime_size)
	fmt.Printf("d (decomposition):        %d\n", d)
	fmt.Printf("ring_size:                %d\n", 1<<params.LogN())
	fmt.Printf("galoiskey_size estimate:  %d uint64 = %.1f MB\n",
		galoiskey_size_heongpu, float64(galoiskey_size_heongpu*8)/(1024*1024))
}

func galoisElements(params hefloat.Parameters) []uint64 {
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}

func bitLen(v uint64) int {
	bits := 0
	for v > 0 {
		bits++
		v >>= 1
	}
	return bits
}
