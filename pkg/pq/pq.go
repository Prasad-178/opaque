// Package pq implements Product Quantization (PQ) for fast approximate
// nearest neighbor search.
//
// PQ splits D-dimensional vectors into M subspaces of D/M dimensions each,
// then quantizes each subspace independently to Ks centroids (default 256).
// This compresses each vector from D×8 bytes (float64) to M bytes (uint8 codes).
//
// At search time, Asymmetric Distance Computation (ADC) precomputes a lookup
// table of query-to-centroid distances for each subspace, then scores each
// database vector by summing M table lookups — O(M) per vector instead of O(D).
//
// PQ is applied client-side before AES encryption, so it has zero privacy impact.
// The server never sees original vectors, PQ codes, or codebooks.
package pq

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

const (
	// DefaultKs is the number of centroids per subspace (256 = 1 byte per code).
	DefaultKs = 256

	// DefaultMaxIter is the maximum k-means iterations per subspace.
	DefaultMaxIter = 25

	// DefaultSeed is the default random seed for reproducible training.
	DefaultSeed = 42
)

// Config controls PQ training behavior.
type Config struct {
	// M is the number of subspaces. D must be divisible by M.
	// Common values: 8, 16, 32. Higher M = better recall, larger codes.
	M int

	// Ks is the number of centroids per subspace. Default: 256 (1 byte per code).
	// Must be <= 256 for uint8 encoding.
	Ks int

	// MaxIter is the maximum k-means iterations per subspace. Default: 25.
	MaxIter int

	// Seed is the random seed for reproducible training. Default: 42.
	Seed int64
}

// PQ holds a trained product quantizer: M codebooks, each with Ks centroids
// of dimension Dsub = D/M.
type PQ struct {
	M         int           // Number of subspaces
	Ks        int           // Centroids per subspace
	D         int           // Original vector dimension
	Dsub      int           // Subspace dimension (D/M)
	Codebooks [][][]float64 // [M][Ks][Dsub] — centroids per subspace
}

// Train builds a product quantizer from training vectors.
// All vectors must have dimension D, and D must be divisible by cfg.M.
func Train(vectors [][]float64, D int, cfg Config) (*PQ, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("pq: no training vectors")
	}
	if cfg.M <= 0 {
		return nil, fmt.Errorf("pq: M must be positive, got %d", cfg.M)
	}
	if D%cfg.M != 0 {
		return nil, fmt.Errorf("pq: dimension %d not divisible by M=%d", D, cfg.M)
	}

	ks := cfg.Ks
	if ks <= 0 {
		ks = DefaultKs
	}
	if ks > 256 {
		return nil, fmt.Errorf("pq: Ks=%d exceeds uint8 max (256)", ks)
	}

	maxIter := cfg.MaxIter
	if maxIter <= 0 {
		maxIter = DefaultMaxIter
	}

	seed := cfg.Seed
	if seed == 0 {
		seed = DefaultSeed
	}

	dsub := D / cfg.M

	// Limit Ks to number of training vectors if needed.
	if ks > len(vectors) {
		ks = len(vectors)
	}

	pq := &PQ{
		M:         cfg.M,
		Ks:        ks,
		D:         D,
		Dsub:      dsub,
		Codebooks: make([][][]float64, cfg.M),
	}

	// Train each subspace in parallel.
	numWorkers := runtime.NumCPU()
	if numWorkers > cfg.M {
		numWorkers = cfg.M
	}

	var mu sync.Mutex
	errs := make([]error, cfg.M)

	var wg sync.WaitGroup
	work := make(chan int, cfg.M)

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for m := range work {
				subStart := m * dsub
				subEnd := subStart + dsub

				// Extract subvectors for this subspace.
				subVecs := make([][]float64, len(vectors))
				for i, v := range vectors {
					subVecs[i] = v[subStart:subEnd]
				}

				// Run k-means on this subspace.
				centroids, err := kmeansSubspace(subVecs, ks, dsub, maxIter, seed+int64(m))
				mu.Lock()
				if err != nil {
					errs[m] = fmt.Errorf("pq: subspace %d k-means failed: %w", m, err)
				} else {
					pq.Codebooks[m] = centroids
				}
				mu.Unlock()
			}
		}()
	}

	for m := 0; m < cfg.M; m++ {
		work <- m
	}
	close(work)
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}

	return pq, nil
}

// Encode quantizes a vector to M uint8 codes using the trained codebooks.
// Each code is the index of the nearest centroid in the corresponding subspace.
func (pq *PQ) Encode(vector []float64) []byte {
	codes := make([]byte, pq.M)
	for m := 0; m < pq.M; m++ {
		subStart := m * pq.Dsub
		sub := vector[subStart : subStart+pq.Dsub]
		codes[m] = pq.nearestCentroid(m, sub)
	}
	return codes
}

// EncodeBatch quantizes multiple vectors in parallel.
func (pq *PQ) EncodeBatch(vectors [][]float64) [][]byte {
	codes := make([][]byte, len(vectors))

	numWorkers := runtime.NumCPU()
	if numWorkers > len(vectors) {
		numWorkers = len(vectors)
	}
	chunkSize := (len(vectors) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				codes[i] = pq.Encode(vectors[i])
			}
		}(start, end)
	}
	wg.Wait()
	return codes
}

// ADCTable is a precomputed lookup table for asymmetric distance computation.
// table[m][k] = partial inner product between query subvector m and centroid k.
type ADCTable [][][]float64 // unused — we use a flat representation below

// BuildADCTable precomputes the inner product between each query subvector
// and each centroid in each subspace.
//
// Returns table[m][k] = dot(query[m*Dsub:(m+1)*Dsub], codebook[m][k]).
// This allows scoring a PQ-encoded vector in O(M) via table lookups.
func (pq *PQ) BuildADCTable(query []float64) [][]float64 {
	table := make([][]float64, pq.M)
	for m := 0; m < pq.M; m++ {
		subStart := m * pq.Dsub
		querySub := query[subStart : subStart+pq.Dsub]
		table[m] = make([]float64, pq.Ks)
		for k := 0; k < pq.Ks; k++ {
			table[m][k] = dot(querySub, pq.Codebooks[m][k])
		}
	}
	return table
}

// ADCScore computes the approximate inner product (cosine similarity for
// normalized vectors) between a query and a PQ-encoded vector using a
// precomputed ADC table.
//
// This is O(M) — just M table lookups and additions.
func ADCScore(table [][]float64, codes []byte) float64 {
	var score float64
	for m, code := range codes {
		score += table[m][code]
	}
	return score
}

// nearestCentroid finds the closest centroid in subspace m to the given subvector.
func (pq *PQ) nearestCentroid(m int, sub []float64) byte {
	bestIdx := 0
	bestDist := math.MaxFloat64
	centroids := pq.Codebooks[m]
	for k, c := range centroids {
		dist := sqDist(sub, c)
		if dist < bestDist {
			bestDist = dist
			bestIdx = k
		}
	}
	return byte(bestIdx)
}

// --- k-means for subspace training ---

func kmeansSubspace(vectors [][]float64, k, dsub, maxIter int, seed int64) ([][]float64, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors")
	}
	if k > len(vectors) {
		k = len(vectors)
	}

	rng := rand.New(rand.NewSource(seed))

	// k-means++ initialization.
	centroids := make([][]float64, k)
	centroids[0] = copyVec(vectors[rng.Intn(len(vectors))], dsub)

	dists := make([]float64, len(vectors))
	for i := 1; i < k; i++ {
		totalDist := 0.0
		for j, v := range vectors {
			d := sqDist(v, centroids[i-1])
			if i == 1 || d < dists[j] {
				dists[j] = d
			}
			totalDist += dists[j]
		}
		// Weighted random selection.
		target := rng.Float64() * totalDist
		cum := 0.0
		selected := len(vectors) - 1
		for j, d := range dists {
			cum += d
			if cum >= target {
				selected = j
				break
			}
		}
		centroids[i] = copyVec(vectors[selected], dsub)
	}

	// Run Lloyd's iterations.
	labels := make([]int, len(vectors))
	counts := make([]int, k)
	sums := make([][]float64, k)
	for i := range sums {
		sums[i] = make([]float64, dsub)
	}

	for iter := 0; iter < maxIter; iter++ {
		// Assign.
		for i, v := range vectors {
			bestK := 0
			bestD := math.MaxFloat64
			for c := 0; c < k; c++ {
				d := sqDist(v, centroids[c])
				if d < bestD {
					bestD = d
					bestK = c
				}
			}
			labels[i] = bestK
		}

		// Update centroids.
		for c := 0; c < k; c++ {
			counts[c] = 0
			for d := 0; d < dsub; d++ {
				sums[c][d] = 0
			}
		}
		for i, v := range vectors {
			c := labels[i]
			counts[c]++
			for d := 0; d < dsub; d++ {
				sums[c][d] += v[d]
			}
		}

		converged := true
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue
			}
			for d := 0; d < dsub; d++ {
				newVal := sums[c][d] / float64(counts[c])
				if math.Abs(newVal-centroids[c][d]) > 1e-6 {
					converged = false
				}
				centroids[c][d] = newVal
			}
		}

		if converged {
			break
		}
	}

	return centroids, nil
}

// --- math helpers ---

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sqDist(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func copyVec(v []float64, n int) []float64 {
	c := make([]float64, n)
	copy(c, v)
	return c
}

// Serialize returns the PQ codebooks as a flat byte slice for storage.
// Format: [M(4B)][Ks(4B)][D(4B)][Dsub(4B)] then M×Ks×Dsub float64s (8B each).
func (pq *PQ) Serialize() []byte {
	headerSize := 16
	dataSize := pq.M * pq.Ks * pq.Dsub * 8
	buf := make([]byte, headerSize+dataSize)

	writeU32(buf[0:], uint32(pq.M))
	writeU32(buf[4:], uint32(pq.Ks))
	writeU32(buf[8:], uint32(pq.D))
	writeU32(buf[12:], uint32(pq.Dsub))

	offset := headerSize
	for m := 0; m < pq.M; m++ {
		for k := 0; k < pq.Ks; k++ {
			for d := 0; d < pq.Dsub; d++ {
				writeF64(buf[offset:], pq.Codebooks[m][k][d])
				offset += 8
			}
		}
	}
	return buf
}

// Deserialize reconstructs a PQ from a serialized byte slice.
func Deserialize(data []byte) (*PQ, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("pq: data too short for header")
	}

	M := int(readU32(data[0:]))
	Ks := int(readU32(data[4:]))
	D := int(readU32(data[8:]))
	Dsub := int(readU32(data[12:]))

	expected := 16 + M*Ks*Dsub*8
	if len(data) < expected {
		return nil, fmt.Errorf("pq: data too short: need %d, got %d", expected, len(data))
	}

	pq := &PQ{
		M:         M,
		Ks:        Ks,
		D:         D,
		Dsub:      Dsub,
		Codebooks: make([][][]float64, M),
	}

	offset := 16
	for m := 0; m < M; m++ {
		pq.Codebooks[m] = make([][]float64, Ks)
		for k := 0; k < Ks; k++ {
			pq.Codebooks[m][k] = make([]float64, Dsub)
			for d := 0; d < Dsub; d++ {
				pq.Codebooks[m][k][d] = readF64(data[offset:])
				offset += 8
			}
		}
	}
	return pq, nil
}

func writeU32(b []byte, v uint32) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func readU32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func writeF64(b []byte, v float64) {
	bits := math.Float64bits(v)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
	b[2] = byte(bits >> 16)
	b[3] = byte(bits >> 24)
	b[4] = byte(bits >> 32)
	b[5] = byte(bits >> 40)
	b[6] = byte(bits >> 48)
	b[7] = byte(bits >> 56)
}

func readF64(b []byte) float64 {
	bits := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	return math.Float64frombits(bits)
}
