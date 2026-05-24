// export-gpu-test-keys exports Galois keys AND secret key for bridge testing.
//
// The secret key export is FOR TESTING ONLY — in production, the secret key
// NEVER leaves the client. This is needed because the GPU-side test must
// encrypt and decrypt with the SAME secret key that generated the Galois keys.
//
// Outputs:
//
//	galois_keys.bin — Galois keys in HEonGPU format (NTT-domain converted)
//	secret_key.bin  — Raw secret key coefficients (for testing only)
//	public_key.bin  — Raw public key coefficients
package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func main() {
	// Server NTT roots (from GPU instance — these are deterministic for same primes)
	serverRoots := []uint64{
		62213374832584, 5039896293, 2384729787, 790617983,
		3307163079, 3070413789, 2513895442, 3151733578, 70072284713359,
	}

	log.Println("Generating GPU-compatible keys...")
	start := time.Now()

	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}

	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		log.Fatal(err)
	}

	N := 1 << params.LogN()
	evk := engine.GetEvalKeys()
	elements := galoisElements(params)
	keys := make([]*rlwe.GaloisKey, len(elements))
	for i, el := range elements {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			log.Fatalf("GetGaloisKey(%d): %v", el, err)
		}
		keys[i] = gk
	}

	log.Printf("Key gen: %v", time.Since(start))

	// Export Galois keys in HEonGPU format
	gkData, err := crypto.SerializeGaloisKeysHEonGPU(params, keys, elements, serverRoots)
	if err != nil {
		log.Fatal(err)
	}
	os.WriteFile("/tmp/galois_keys.bin", gkData, 0600)
	log.Printf("Galois keys: %d bytes (%.1f MB)", len(gkData), float64(len(gkData))/(1024*1024))

	// Export secret key raw coefficients
	// SK is stored as polynomial in ring Q×P
	// For testing: export Q coefficients (NTT domain, HEonGPU-converted)
	sk := engine.GetSecretKey()
	qLevels := sk.Value.Q.Level() + 1
	pLevels := 0
	if sk.Value.P.N() > 0 {
		pLevels = sk.Value.P.Level() + 1
	}

	conv := crypto.NewNTTConverterWithRoots(params, serverRoots)

	// Secret key: write all Q + P levels, each N coefficients
	// Convert from Lattigo NTT to HEonGPU NTT
	totalLevels := qLevels + pLevels
	skBuf := make([]byte, 0, totalLevels*N*8)

	for lvl := 0; lvl < qLevels; lvl++ {
		coeffs := make([]uint64, N)
		copy(coeffs, sk.Value.Q.Coeffs[lvl])
		conv.ConvertToHEonGPUDomain(coeffs, lvl)
		for _, c := range coeffs {
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, c)
			skBuf = append(skBuf, b...)
		}
	}
	for lvl := 0; lvl < pLevels; lvl++ {
		coeffs := make([]uint64, N)
		copy(coeffs, sk.Value.P.Coeffs[lvl])
		conv.ConvertToHEonGPUDomain(coeffs, qLevels+lvl)
		for _, c := range coeffs {
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, c)
			skBuf = append(skBuf, b...)
		}
	}
	os.WriteFile("/tmp/secret_key.bin", skBuf, 0600)
	log.Printf("Secret key: %d bytes (%d levels × %d coeffs)", len(skBuf), totalLevels, N)

	fmt.Printf("\nFiles written:\n")
	fmt.Printf("  /tmp/galois_keys.bin (%d bytes)\n", len(gkData))
	fmt.Printf("  /tmp/secret_key.bin (%d bytes)\n", len(skBuf))
	fmt.Printf("\nSecret key levels: Q=%d P=%d\n", qLevels, pLevels)
}

func galoisElements(params interface {
	LogN() int
	GaloisElement(int) uint64
}) []uint64 {
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}
