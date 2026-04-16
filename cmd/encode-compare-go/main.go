// encode-compare-go dumps polynomial coefficient representations of
// known CKKS-encoded vectors in Lattigo's convention. The binary twin
// in deploy/gpu/gpu-he-server/encode_compare.cpp dumps the same vectors
// in HEonGPU's convention. If the two dumps disagree, the canonical
// embedding (SpecialIFFT + rotGroup rewrite) differs between libraries.
//
// Usage: go run ./cmd/encode-compare-go -out lattigo_dump.bin
//
// Dump format (little-endian):
//   magic      u32   = 0x4F504151 ("OPAQ")
//   version    u32   = 1
//   logN       u32
//   numPrimes  u32   = Q_size (does not include P)
//   numTests   u32
//   for each test:
//     nameLen u32, name bytes
//     for each prime:
//       N × u64 polynomial coefficients (coefficient domain, non-Montgomery)

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

const (
	magic   uint32 = 0x4F504151 // "OPAQ"
	version uint32 = 1
)

type testVec struct {
	name   string
	values []float64
}

func main() {
	out := flag.String("out", "lattigo_dump.bin", "output dump file")
	flag.Parse()

	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatalf("NewParametersGPU: %v", err)
	}

	N := 1 << params.LogN()
	slotCount := N / 2
	qSize := params.MaxLevelQ() + 1

	fmt.Printf("Lattigo encode dump\n")
	fmt.Printf("  LogN=%d N=%d slots=%d qSize=%d scale=%.0f\n",
		params.LogN(), N, slotCount, qSize, params.DefaultScale().Float64())
	fmt.Printf("  Q primes: %v\n", params.RingQ().ModuliChain())

	encoder := hefloat.NewEncoder(params)

	// Test vectors — chosen so coefficient differences are obvious.
	tests := []testVec{
		{name: "slot0_eq_1", values: withSlot(slotCount, 0, 1.0)},
		{name: "slot1_eq_1", values: withSlot(slotCount, 1, 1.0)},
		{name: "slot2_eq_1", values: withSlot(slotCount, 2, 1.0)},
		{name: "slot_last_eq_1", values: withSlot(slotCount, slotCount-1, 1.0)},
		{name: "ramp_0_to_15", values: rampTo(slotCount, 16)},
		{name: "constant_0.5", values: constant(slotCount, 0.5)},
		{name: "sine_wave", values: sineWave(slotCount)},
	}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, magic)
	binary.Write(f, binary.LittleEndian, version)
	binary.Write(f, binary.LittleEndian, uint32(params.LogN()))
	binary.Write(f, binary.LittleEndian, uint32(qSize))
	binary.Write(f, binary.LittleEndian, uint32(len(tests)))

	for _, t := range tests {
		// Encode to plaintext at max level
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(t.values, pt); err != nil {
			log.Fatalf("Encode(%s): %v", t.name, err)
		}

		// Apply INTT to get coefficient domain
		// Lattigo plaintext is stored NTT-domain (IsNTT=true), non-Montgomery.
		coeffsPerPrime := make([][]uint64, qSize)
		for lvl := 0; lvl < qSize; lvl++ {
			subRing := params.RingQ().SubRings[lvl]
			tmp := make([]uint64, N)
			in := pt.Value.Coeffs[lvl]
			ring.INTTStandard(in, tmp, N, subRing.NInv, subRing.Modulus,
				subRing.MRedConstant, subRing.RootsBackward)
			coeffsPerPrime[lvl] = tmp
		}

		// Sanity print
		fmt.Printf("\nTest %q: first 8 coeffs of prime Q[0]: %d %d %d %d %d %d %d %d\n",
			t.name,
			coeffsPerPrime[0][0], coeffsPerPrime[0][1], coeffsPerPrime[0][2], coeffsPerPrime[0][3],
			coeffsPerPrime[0][4], coeffsPerPrime[0][5], coeffsPerPrime[0][6], coeffsPerPrime[0][7])

		// Write to file
		name := []byte(t.name)
		binary.Write(f, binary.LittleEndian, uint32(len(name)))
		f.Write(name)
		for lvl := 0; lvl < qSize; lvl++ {
			for _, c := range coeffsPerPrime[lvl] {
				binary.Write(f, binary.LittleEndian, c)
			}
		}
	}

	fmt.Printf("\nWrote %s (%d tests, %d primes, N=%d)\n", *out, len(tests), qSize, N)
}

func withSlot(n, idx int, v float64) []float64 {
	out := make([]float64, n)
	out[idx] = v
	return out
}

func rampTo(n, k int) []float64 {
	out := make([]float64, n)
	for i := 0; i < k && i < n; i++ {
		out[i] = float64(i)
	}
	return out
}

func constant(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func sineWave(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = 0.5 * math.Sin(2*math.Pi*float64(i)/float64(n))
	}
	return out
}
