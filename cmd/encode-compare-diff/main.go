// encode-compare-diff reads a Lattigo dump and a HEonGPU dump produced by
// encode-compare-go and encode_compare.cpp and reports coefficient-level
// differences. Matching dumps prove the canonical embedding (IFFT +
// slot-to-coefficient mapping) is identical between the two libraries.
// Any mismatch reveals the bug at the heart of the Lattigo→HEonGPU bridge.
//
// Usage: go run ./cmd/encode-compare-diff lattigo_dump.bin heongpu_dump.bin

package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

const magic uint32 = 0x4F504151

type dump struct {
	logN      int
	numPrimes int
	N         int
	tests     []testEntry
}
type testEntry struct {
	name  string
	polys [][]uint64 // per-prime coefficient arrays
}

func main() {
	if len(os.Args) < 3 {
		log.Fatalf("usage: %s lattigo_dump.bin heongpu_dump.bin", os.Args[0])
	}
	a, err := loadDump(os.Args[1])
	if err != nil {
		log.Fatalf("load %s: %v", os.Args[1], err)
	}
	b, err := loadDump(os.Args[2])
	if err != nil {
		log.Fatalf("load %s: %v", os.Args[2], err)
	}

	if a.logN != b.logN || a.numPrimes != b.numPrimes {
		log.Fatalf("header mismatch: A(logN=%d primes=%d) B(logN=%d primes=%d)",
			a.logN, a.numPrimes, b.logN, b.numPrimes)
	}
	fmt.Printf("Both dumps: logN=%d N=%d numPrimes=%d\n", a.logN, a.N, a.numPrimes)

	if len(a.tests) != len(b.tests) {
		log.Fatalf("different number of tests: A=%d B=%d", len(a.tests), len(b.tests))
	}

	overallMismatch := 0
	for i, ta := range a.tests {
		tb := b.tests[i]
		if ta.name != tb.name {
			log.Fatalf("test %d: name mismatch %q vs %q", i, ta.name, tb.name)
		}
		fmt.Printf("\n=== Test %q ===\n", ta.name)
		for p := 0; p < a.numPrimes; p++ {
			pa := ta.polys[p]
			pb := tb.polys[p]
			mismatches := 0
			firstMismatch := -1
			for j := range pa {
				if pa[j] != pb[j] {
					if firstMismatch < 0 {
						firstMismatch = j
					}
					mismatches++
				}
			}
			if mismatches == 0 {
				fmt.Printf("  Prime %d: MATCH (all %d coeffs)\n", p, len(pa))
				continue
			}
			overallMismatch++
			fmt.Printf("  Prime %d: MISMATCH %d / %d  first_diff_idx=%d  L=%d  H=%d\n",
				p, mismatches, len(pa), firstMismatch,
				pa[firstMismatch], pb[firstMismatch])

			// Show first few mismatches with positional context
			shown := 0
			for j := range pa {
				if pa[j] == pb[j] || shown >= 8 {
					continue
				}
				fmt.Printf("    [%d]  L=%d  H=%d  diff=%d\n",
					j, pa[j], pb[j], int64(pa[j])-int64(pb[j]))
				shown++
			}

			// Permutation check: does L[i] == H[perm(i)] for common perms?
			checkPermutation(pa, pb, a.logN)
		}
	}

	if overallMismatch == 0 {
		fmt.Println("\n✓ All polynomials match — canonical embeddings are compatible.")
	} else {
		fmt.Printf("\n✗ %d (test × prime) pairs differ — embeddings DIFFER.\n", overallMismatch)
	}
}

func loadDump(path string) (*dump, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var m, ver, logN, numPrimes, numTests uint32
	if err := binary.Read(f, binary.LittleEndian, &m); err != nil {
		return nil, err
	}
	if m != magic {
		return nil, fmt.Errorf("bad magic 0x%x", m)
	}
	binary.Read(f, binary.LittleEndian, &ver)
	binary.Read(f, binary.LittleEndian, &logN)
	binary.Read(f, binary.LittleEndian, &numPrimes)
	binary.Read(f, binary.LittleEndian, &numTests)
	N := 1 << logN

	d := &dump{
		logN:      int(logN),
		numPrimes: int(numPrimes),
		N:         N,
	}
	for i := uint32(0); i < numTests; i++ {
		var nl uint32
		binary.Read(f, binary.LittleEndian, &nl)
		nameBuf := make([]byte, nl)
		if _, err := io.ReadFull(f, nameBuf); err != nil {
			return nil, err
		}
		te := testEntry{name: string(nameBuf)}
		for p := uint32(0); p < numPrimes; p++ {
			poly := make([]uint64, N)
			if err := binary.Read(f, binary.LittleEndian, poly); err != nil {
				return nil, fmt.Errorf("test %d prime %d: %w", i, p, err)
			}
			te.polys = append(te.polys, poly)
		}
		d.tests = append(d.tests, te)
	}
	return d, nil
}

// checkPermutation tests whether B is a known permutation of A: bit-reversal,
// negation (X → -X, coeff_i → -coeff_{N-i}), slot-rotGroup, or pairwise swap.
func checkPermutation(a, b []uint64, logN int) {
	N := len(a)

	// Bit-reverse?
	brMatch := 0
	for i := 0; i < N; i++ {
		j := bitrev(i, logN)
		if a[j] == b[i] {
			brMatch++
		}
	}
	if brMatch == N {
		fmt.Println("    >>> B == bitrev(A)  (Lattigo↔HEonGPU bit-reversed index)")
	} else if brMatch > N*3/4 {
		fmt.Printf("    ~ partial bit-reverse match: %d/%d\n", brMatch, N)
	}

	// Negacyclic reverse: b[i] should equal p - a[N-i] for some prime. Skip.

	// Slot swap (halves): first half↔second half
	swapMatch := 0
	half := N / 2
	for i := 0; i < half; i++ {
		if a[i] == b[i+half] && a[i+half] == b[i] {
			swapMatch++
		}
	}
	if swapMatch == half {
		fmt.Println("    >>> B is A with halves swapped")
	}

	// Same first coeff but different rest — suggests similar but not identical structure
	if a[0] == b[0] && a[1] != b[1] {
		fmt.Printf("    note: coeff[0] matches (%d) but coeff[1] differs (L=%d H=%d)\n",
			a[0], a[1], b[1])
	}
}

func bitrev(v, bits int) int {
	r := 0
	for i := 0; i < bits; i++ {
		r = (r << 1) | (v & 1)
		v >>= 1
	}
	return r
}
