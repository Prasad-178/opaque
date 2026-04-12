// test-ct-convert verifies ciphertext conversion round-trip:
// Lattigo encrypt → convert to HEonGPU → convert back → Lattigo decrypt
package main

import (
	"fmt"
	"log"
	"math"

	"github.com/Prasad-178/opaque/pkg/crypto"
	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func main() {
	serverRoots := []uint64{62213374832584, 5039896293, 2384729787, 790617983,
		3307163079, 3070413789, 2513895442, 3151733578, 70072284713359}

	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		log.Fatal(err)
	}

	conv := crypto.NewNTTConverterWithRoots(params, serverRoots)

	// Encrypt a test vector
	dim := 128
	query := make([]float64, params.MaxSlots())
	for i := 0; i < dim; i++ {
		query[i] = float64(i + 1) / 10.0 // [0.1, 0.2, 0.3, ...]
	}

	ct, err := engine.EncryptVector(query)
	if err != nil {
		log.Fatal(err)
	}

	// Decrypt before conversion (sanity check)
	original, err := engine.DecryptScalar(ct)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Before conversion: slot[0] = %.4f (expect 0.1000)\n", original)

	// Convert to HEonGPU format
	heongpuCT := crypto.CiphertextToHEonGPU(ct, conv)
	fmt.Printf("Converted to HEonGPU: %d polys, %d levels, scale=%.0f\n",
		len(heongpuCT.Polynomials), heongpuCT.Polynomials[0].NumLevels, heongpuCT.Scale)

	// Convert back to Lattigo format
	recovered, err := crypto.CiphertextFromHEonGPU(heongpuCT, params, conv)
	if err != nil {
		log.Fatalf("CiphertextFromHEonGPU: %v", err)
	}

	// Decrypt after round-trip
	afterRT, err := engine.DecryptScalar(recovered)
	if err != nil {
		log.Fatalf("Decrypt after round-trip: %v", err)
	}
	fmt.Printf("After round-trip: slot[0] = %.4f (expect 0.1000)\n", afterRT)

	diff := math.Abs(original - afterRT)
	fmt.Printf("Difference: %.10f\n", diff)

	if diff < 0.001 {
		fmt.Println("\n✓ Ciphertext round-trip CORRECT!")
	} else {
		fmt.Println("\n✗ Ciphertext round-trip FAILED!")
	}

	_ = pb.BatchDotProductRequest{} // suppress unused import
	_ = hefloat.Parameters{}
}
