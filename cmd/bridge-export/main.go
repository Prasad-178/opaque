// bridge-export dumps all Lattigo state needed by the GPU-side bridge_suite
// to reproduce every step of the search pipeline in C++ and catch the layer
// where the bridged ciphertext diverges.
//
// Outputs (all under /tmp by default, override with -out):
//   bridge_sk.bin       — secret key in HEonGPU NTT domain (Go-converted)
//   bridge_galois.bin   — Galois keys in HEonGPU binary format
//   bridge_test.bin     — packed tests: query vector, plaintext centroids,
//                         Lattigo-encrypted c0/c1 (coefficient domain),
//                         Lattigo-encoded plaintext polynomial (coefficient
//                         domain), and the expected batched dot-product scores.
//
// The secret key export is for diagnostic use only; in production the secret
// key never leaves the client.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

const (
	magic    uint32 = 0x42524745 // "BRGE"
	version  uint32 = 1
	dim      int    = 128 // vector dimension per centroid
	centPack int    = 64  // centroids per pack (N/2 / dim)
)

// Deterministic server NTT roots (match setup.sh output on Tesla T4).
var serverRoots = []uint64{
	62213374832584, 5039896293, 2384729787, 790617983,
	3307163079, 3070413789, 2513895442, 3151733578, 70072284713359,
}

func main() {
	out := flag.String("out", "/tmp", "output directory")
	numTests := flag.Int("n", 3, "number of test cases")
	flag.Parse()

	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatal(err)
	}
	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		log.Fatal(err)
	}

	N := 1 << params.LogN()
	slotCount := N / 2
	qSize := params.MaxLevelQ() + 1
	pSize := params.MaxLevelP() + 1

	log.Printf("LogN=%d N=%d slots=%d Q=%d P=%d", params.LogN(), N, slotCount, qSize, pSize)

	// --- Galois keys -------------------------------------------------------
	galoisElems := galoisElements(params)
	evk := engine.GetEvalKeys()
	gkeys := make([]*rlwe.GaloisKey, len(galoisElems))
	for i, el := range galoisElems {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			log.Fatal(err)
		}
		gkeys[i] = gk
	}
	gkData, err := crypto.SerializeGaloisKeysHEonGPU(params, gkeys, galoisElems, serverRoots)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out+"/bridge_galois.bin", gkData, 0600); err != nil {
		log.Fatal(err)
	}
	log.Printf("galois keys: %d bytes", len(gkData))

	// --- Secret key (Q + P, HEonGPU NTT domain) ----------------------------
	conv := crypto.NewNTTConverterWithRoots(params, serverRoots)
	sk := engine.GetSecretKey()

	skBuf := make([]byte, 0, (qSize+pSize)*N*8)
	for lvl := 0; lvl < qSize; lvl++ {
		coeffs := make([]uint64, N)
		copy(coeffs, sk.Value.Q.Coeffs[lvl])
		conv.ConvertToHEonGPUDomain(coeffs, lvl)
		for _, c := range coeffs {
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, c)
			skBuf = append(skBuf, b...)
		}
	}
	for lvl := 0; lvl < pSize; lvl++ {
		coeffs := make([]uint64, N)
		copy(coeffs, sk.Value.P.Coeffs[lvl])
		conv.ConvertToHEonGPUDomain(coeffs, qSize+lvl)
		for _, c := range coeffs {
			b := make([]byte, 8)
			binary.LittleEndian.PutUint64(b, c)
			skBuf = append(skBuf, b...)
		}
	}
	if err := os.WriteFile(*out+"/bridge_sk.bin", skBuf, 0600); err != nil {
		log.Fatal(err)
	}
	log.Printf("secret key: %d bytes (%d levels)", len(skBuf), qSize+pSize)

	// --- Test cases --------------------------------------------------------
	f, err := os.Create(*out + "/bridge_test.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	writeU32(f, magic)
	writeU32(f, version)
	writeU32(f, uint32(params.LogN()))
	writeU32(f, uint32(qSize))
	writeU32(f, uint32(slotCount))
	writeU32(f, uint32(dim))
	writeU32(f, uint32(centPack))
	writeU32(f, uint32(*numTests))

	rng := rand.New(rand.NewSource(42))
	encoder := engine.GetEncoder()
	for t := 0; t < *numTests; t++ {
		q := randomVector(slotCount, rng)
		c := randomCentroidPack(slotCount, dim, centPack, rng)
		// Pre-normalize so dot products land in [-1,1]-ish range.
		normalizeBlocks(q, dim)
		normalizeBlocks(c, dim)

		// Expected batched dot products: for each centroid, sum slot-wise
		// product over dim. After the HE pipeline this is what should land in
		// slot 0 of each dim-sized segment.
		expected := batchedDotProduct(q, c, dim, centPack)

		// Encrypt query.
		ct, err := engine.EncryptVector(q)
		if err != nil {
			log.Fatal(err)
		}
		// Encode centroid pack as plaintext.
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(c, pt); err != nil {
			log.Fatal(err)
		}

		// Dump vectors.
		writeF64Slice(f, q)
		writeF64Slice(f, c)
		writeF64Slice(f, expected)
		writeU64(f, uint64Bits(ct.Scale.Float64()))
		writeU64(f, uint64Bits(pt.Scale.Float64()))

		// c0, c1 coefficient-domain (Lattigo INTT-ed).
		for _, poly := range ct.Value {
			for lvl := 0; lvl < qSize; lvl++ {
				tmp := make([]uint64, N)
				subRing := params.RingQ().SubRings[lvl]
				ring.INTTStandard(poly.Coeffs[lvl], tmp, N, subRing.NInv,
					subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)
				writeU64s(f, tmp)
			}
		}
		// Plaintext coefficient-domain.
		for lvl := 0; lvl < qSize; lvl++ {
			tmp := make([]uint64, N)
			subRing := params.RingQ().SubRings[lvl]
			ring.INTTStandard(pt.Value.Coeffs[lvl], tmp, N, subRing.NInv,
				subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)
			writeU64s(f, tmp)
		}

		log.Printf("test %d: q[0..3]=%.3f %.3f %.3f %.3f  expected[0..3]=%.4f %.4f %.4f %.4f",
			t, q[0], q[1], q[2], q[3],
			expected[0], expected[1], expected[2], expected[3])
	}

	fmt.Println("Exported to", *out)
}

func galoisElements(p hefloat.Parameters) []uint64 {
	logN := p.LogN()
	out := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		out[i] = p.GaloisElement(1 << i)
	}
	return out
}

func randomVector(n int, r *rand.Rand) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = r.Float64()*2 - 1
	}
	return out
}

// randomCentroidPack lays out centPack centroids of dim dimensions into the
// slot_count vector. Any leftover slots are zero.
func randomCentroidPack(slots, dim, packs int, r *rand.Rand) []float64 {
	out := make([]float64, slots)
	for c := 0; c < packs; c++ {
		for d := 0; d < dim; d++ {
			out[c*dim+d] = r.Float64()*2 - 1
		}
	}
	return out
}

func normalizeBlocks(v []float64, dim int) {
	for base := 0; base < len(v); base += dim {
		if base+dim > len(v) {
			break
		}
		var norm float64
		for j := 0; j < dim; j++ {
			norm += v[base+j] * v[base+j]
		}
		if norm == 0 {
			continue
		}
		inv := 1.0 / sqrt(norm)
		for j := 0; j < dim; j++ {
			v[base+j] *= inv
		}
	}
}

func sqrt(x float64) float64 { return math.Sqrt(x) }

func batchedDotProduct(q, c []float64, dim, packs int) []float64 {
	out := make([]float64, packs)
	for p := 0; p < packs; p++ {
		var sum float64
		for d := 0; d < dim; d++ {
			sum += q[p*dim+d] * c[p*dim+d]
		}
		out[p] = sum
	}
	return out
}

func writeU32(w io.Writer, v uint32) {
	if err := binary.Write(w, binary.LittleEndian, v); err != nil {
		log.Fatal(err)
	}
}
func writeU64(w io.Writer, v uint64) {
	if err := binary.Write(w, binary.LittleEndian, v); err != nil {
		log.Fatal(err)
	}
}
func writeU64s(w io.Writer, v []uint64) {
	if err := binary.Write(w, binary.LittleEndian, v); err != nil {
		log.Fatal(err)
	}
}
func writeF64Slice(w io.Writer, v []float64) {
	if err := binary.Write(w, binary.LittleEndian, v); err != nil {
		log.Fatal(err)
	}
}
func uint64Bits(f float64) uint64 { return math.Float64bits(f) }
