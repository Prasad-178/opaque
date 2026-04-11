// ntt_convert.go converts polynomial coefficients between Lattigo's NTT domain
// and HEonGPU's NTT domain.
//
// Both libraries use the same NTT algorithm (Cooley-Tukey butterfly) but with
// different primitive roots of unity (ψ). This converter applies:
//   Lattigo INTT → coefficient domain → HEonGPU NTT
//
// This is needed for evaluation key transfer: the keys are generated in Lattigo's
// NTT domain but must be in HEonGPU's NTT domain to work correctly on the GPU.
//
// The conversion is O(N log N) per polynomial and is done once during setup.
package crypto

import (
	"math/big"

	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

// NTTConverter converts polynomial coefficients between Lattigo and HEonGPU NTT domains.
type NTTConverter struct {
	params    hefloat.Parameters
	N         int
	logN      int
	qModuli   []uint64 // Q primes
	pModuli   []uint64 // P primes
	allModuli []uint64 // Q + P combined

	// Per-modulus conversion data
	lattigoINTTRoots [][]uint64 // Lattigo's inverse NTT roots (from RootsBackward)
	lattigoNInv      []uint64   // Lattigo's N^{-1} mod p
	heongpuNTTRoots  [][]uint64 // HEonGPU's forward NTT roots (computed)
}

// NewNTTConverter creates a converter using locally-computed HEonGPU roots.
// NOTE: This may not match the actual GPU server's roots (which are non-deterministic).
// Prefer NewNTTConverterWithRoots when server-provided roots are available.
func NewNTTConverter(params hefloat.Parameters) *NTTConverter {
	return newNTTConverter(params, nil)
}

// NewNTTConverterWithRoots creates a converter using server-provided NTT roots.
// The serverRoots are the ψ values returned by InitContext — one per prime in Q∪P order.
// This guarantees the conversion matches the GPU server's NTT domain exactly.
func NewNTTConverterWithRoots(params hefloat.Parameters, serverRoots []uint64) *NTTConverter {
	return newNTTConverter(params, serverRoots)
}

func newNTTConverter(params hefloat.Parameters, serverRoots []uint64) *NTTConverter {
	N := 1 << params.LogN()
	logN := params.LogN()

	qModuli := params.RingQ().ModuliChain()
	var pModuli []uint64
	if ringP := params.RingP(); ringP != nil {
		pModuli = ringP.ModuliChain()
	}

	allModuli := make([]uint64, 0, len(qModuli)+len(pModuli))
	allModuli = append(allModuli, qModuli...)
	allModuli = append(allModuli, pModuli...)

	conv := &NTTConverter{
		params:    params,
		N:         N,
		logN:      logN,
		qModuli:   qModuli,
		pModuli:   pModuli,
		allModuli: allModuli,
	}

	// Extract Lattigo's INTT roots
	conv.lattigoINTTRoots = make([][]uint64, len(allModuli))
	conv.lattigoNInv = make([]uint64, len(allModuli))

	ringQ := params.RingQ()
	for i := 0; i <= params.MaxLevelQ(); i++ {
		conv.lattigoINTTRoots[i] = make([]uint64, N)
		copy(conv.lattigoINTTRoots[i], ringQ.SubRings[i].RootsBackward)
		conv.lattigoNInv[i] = ringQ.SubRings[i].NInv
	}
	if ringP := params.RingP(); ringP != nil {
		for i := 0; i <= params.MaxLevelP(); i++ {
			idx := len(qModuli) + i
			conv.lattigoINTTRoots[idx] = make([]uint64, N)
			copy(conv.lattigoINTTRoots[idx], ringP.SubRings[i].RootsBackward)
			conv.lattigoNInv[idx] = ringP.SubRings[i].NInv
		}
	}

	// Compute HEonGPU's NTT roots
	conv.heongpuNTTRoots = make([][]uint64, len(allModuli))
	for i, p := range allModuli {
		var psi uint64
		if serverRoots != nil && i < len(serverRoots) {
			// Use server-provided root (guaranteed to match GPU server's NTT)
			psi = serverRoots[i]
		} else {
			// Compute locally (may not match GPU server due to non-deterministic root selection)
			psi = computeHEonGPUPsi(p, uint64(N))
		}
		conv.heongpuNTTRoots[i] = generateNTTTable(psi, p, N)
	}

	return conv
}

// ConvertToCoeffDomain converts coefficients from Lattigo's NTT domain
// to standard coefficient domain using Lattigo's own INTT.
//
// The GPU server can then apply its own NTT to get to HEonGPU's NTT domain.
// This avoids reimplementing HEonGPU's NTT algorithm in Go.
func (c *NTTConverter) ConvertToCoeffDomain(coeffs []uint64, modulusIdx int) {
	N := c.N

	var subRing *ring.SubRing
	if modulusIdx <= c.params.MaxLevelQ() {
		subRing = c.params.RingQ().SubRings[modulusIdx]
	} else {
		pIdx := modulusIdx - c.params.MaxLevelQ() - 1
		subRing = c.params.RingP().SubRings[pIdx]
	}

	// Lattigo's INTTStandard correctly handles Montgomery-form roots
	tmp := make([]uint64, N)
	ring.INTTStandard(coeffs, tmp, N, subRing.NInv, subRing.Modulus, subRing.MRedConstant, subRing.RootsBackward)
	copy(coeffs, tmp)
}

// computeHEonGPUPsi finds HEonGPU's primitive 2N-th root of unity for prime p.
//
// HEonGPU's algorithm (find_minimal_primitive_root):
// 1. Find ANY primitive (2N)-th root by: random^((p-1)/(2N)) mod p,
//    checking that the result has exact order 2N (is_primitive_root).
// 2. Then find the SMALLEST such root by iterating through all (2N)-th
//    roots of unity (there are phi(2N) = N of them) and keeping the minimum.
//
// This is deterministic because the minimal root is unique.
func computeHEonGPUPsi(p, N uint64) uint64 {
	degree := 2 * N
	pBig := new(big.Int).SetUint64(p)
	pm1 := new(big.Int).Sub(pBig, big.NewInt(1))
	quotient := new(big.Int).Div(pm1, new(big.Int).SetUint64(degree))

	// Find a primitive (2N)-th root of unity.
	// Try small candidates: candidate = i^((p-1)/(2N)) mod p
	var root *big.Int
	for i := uint64(2); i < 10000; i++ {
		candidate := new(big.Int).Exp(new(big.Int).SetUint64(i), quotient, pBig)
		if isPrimitive2NthRoot(candidate, degree, pBig) {
			root = candidate
			break
		}
	}

	if root == nil {
		return 0
	}

	// Find the MINIMAL root by squaring through all equivalent roots.
	// root^2 is also a generator of the group of (2N)-th roots.
	// By iterating root, root^3, root^5, ... we visit all primitive roots.
	minRoot := new(big.Int).Set(root)
	generatorSq := new(big.Int).Mul(root, root)
	generatorSq.Mod(generatorSq, pBig)

	current := new(big.Int).Set(root)
	for i := uint64(0); i < degree; i += 2 {
		if current.Cmp(minRoot) < 0 {
			minRoot.Set(current)
		}
		current.Mul(current, generatorSq)
		current.Mod(current, pBig)
	}

	return minRoot.Uint64()
}

// isPrimitive2NthRoot checks if candidate is a primitive (degree)-th root of unity mod p.
// It must satisfy: candidate^degree = 1 mod p AND candidate^(degree/2) != 1 mod p.
func isPrimitive2NthRoot(candidate *big.Int, degree uint64, p *big.Int) bool {
	if candidate.Sign() == 0 || candidate.Cmp(big.NewInt(1)) == 0 {
		return false
	}

	// Check candidate^degree = 1 mod p
	check := new(big.Int).Exp(candidate, new(big.Int).SetUint64(degree), p)
	if check.Cmp(big.NewInt(1)) != 0 {
		return false
	}

	// Check candidate^(degree/2) != 1 mod p (ensures primitive, not just order dividing degree)
	halfCheck := new(big.Int).Exp(candidate, new(big.Int).SetUint64(degree/2), p)
	return halfCheck.Cmp(big.NewInt(1)) != 0
}

// generateNTTTable generates the forward NTT root table from psi.
// Uses bit-reversed ordering (standard Cooley-Tukey).
func generateNTTTable(psi, p uint64, N int) []uint64 {
	table := make([]uint64, N)
	pBig := new(big.Int).SetUint64(p)
	psiBig := new(big.Int).SetUint64(psi)

	// Compute powers of psi in bit-reversed order
	// table[bitrev(i)] = psi^i mod p for i = 0..N-1
	// Actually, the standard CT NTT table stores:
	// table[0] = 1 (not used in some implementations)
	// table[1] = psi
	// table[i] = psi^(bitreverse(i, logN)) mod p (for butterfly lookup)
	//
	// Different libraries have slightly different table layouts.
	// HEonGPU's GPU-NTT generates the table in its NTTParameters constructor.
	// The key function: forward_root_of_unity_table_generator()
	//
	// From GPU-NTT source, the forward table is:
	//   for m = N/2 down to 1 by halving:
	//     for i = 0 to m-1:
	//       table[m + i] = psi^(bitrev(i, logN-log2(m))) mod p
	//
	// This is the standard CT butterfly ordering.

	logN := 0
	for n := N; n > 1; n >>= 1 {
		logN++
	}

	// Generate powers of psi: psi^0, psi^1, psi^2, ...
	powers := make([]*big.Int, 2*N)
	powers[0] = big.NewInt(1)
	for i := 1; i < 2*N; i++ {
		powers[i] = new(big.Int).Mul(powers[i-1], psiBig)
		powers[i].Mod(powers[i], pBig)
	}

	// Fill table in bit-reversed butterfly order
	// table[1] = psi (used for the first butterfly layer)
	// table[m+i] = psi^(bitrev(i, log2(m))) for butterfly at size m
	for m := 1; m < N; m <<= 1 {
		logM := 0
		for x := m; x > 1; x >>= 1 {
			logM++
		}
		step := N / m // = 2^(logN - logM)
		for i := 0; i < m; i++ {
			idx := bitReverse(i, logM)
			table[m+i] = powers[idx*step].Uint64()
		}
	}

	return table
}

// bitReverse reverses the bottom `bits` bits of v.
func bitReverse(v, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (v & 1)
		v >>= 1
	}
	return result
}

// nttInPlace performs forward NTT using the given root table.
// Standard Cooley-Tukey butterfly, in-place.
func nttInPlace(p []uint64, N int, Q uint64, roots []uint64) {
	t := N >> 1
	for m := 1; m < N; m <<= 1 {
		for i := 0; i < m; i++ {
			j1 := 2 * i * t
			j2 := j1 + t
			W := roots[m+i]
			for j := j1; j < j2; j++ {
				// Butterfly: (a, b) -> (a + W*b, a - W*b) mod Q
				u := p[j]
				v := mulMod(p[j+t], W, Q)
				p[j] = addMod(u, v, Q)
				p[j+t] = subMod(u, v, Q)
			}
		}
		t >>= 1
	}
}

// inttInPlace performs inverse NTT using the given inverse root table.
func inttInPlace(p []uint64, N int, Q uint64, roots []uint64, NInv uint64) {
	t := 1
	for m := N >> 1; m >= 1; m >>= 1 {
		for i := 0; i < m; i++ {
			j1 := 2 * i * t
			j2 := j1 + t
			W := roots[m+i]
			for j := j1; j < j2; j++ {
				u := p[j]
				v := p[j+t]
				p[j] = addMod(u, v, Q)
				p[j+t] = mulMod(subMod(u, v, Q), W, Q)
			}
		}
		t <<= 1
	}
	// Multiply by N^{-1}
	for i := range p {
		p[i] = mulMod(p[i], NInv, Q)
	}
}

// Modular arithmetic helpers
func mulMod(a, b, m uint64) uint64 {
	// Use big.Int for correctness (avoid overflow)
	r := new(big.Int).Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(b))
	r.Mod(r, new(big.Int).SetUint64(m))
	return r.Uint64()
}

func addMod(a, b, m uint64) uint64 {
	s := a + b
	if s >= m {
		s -= m
	}
	return s
}

func subMod(a, b, m uint64) uint64 {
	if a >= b {
		return a - b
	}
	return m - (b - a)
}

func primeFactorsOf(n *big.Int) []*big.Int {
	var factors []*big.Int
	d := big.NewInt(2)
	tmp := new(big.Int).Set(n)

	limit := new(big.Int).Sqrt(tmp)
	for d.Cmp(limit) <= 0 {
		if new(big.Int).Mod(tmp, d).Sign() == 0 {
			factors = append(factors, new(big.Int).Set(d))
			for new(big.Int).Mod(tmp, d).Sign() == 0 {
				tmp.Div(tmp, d)
			}
			limit.Sqrt(tmp)
		}
		d.Add(d, big.NewInt(1))
	}
	if tmp.Cmp(big.NewInt(1)) > 0 {
		factors = append(factors, tmp)
	}
	return factors
}
