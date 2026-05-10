//go:build sift1m || dbpedia1m

package test

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// bruteForceTopK computes exact top-K nearest neighbors by cosine similarity
// across one or more queries. Parallelized across queries (each query is an
// independent O(N×D) scan), giving ~M-fold speedup at M cores for the
// 50-query × 1M × 1536-dim ground-truth pass that runs once per bench test.
//
// Per-vector database norms are computed once and reused across queries so the
// inner loop only does one accumulator per (q, v) pair instead of two.
func bruteForceTopK(queries, vectors [][]float64, dim, topK int) [][]int {
	type scored struct {
		idx   int
		score float64
	}

	// Pre-compute database norms once (shared across all queries).
	dbNorm := make([]float64, len(vectors))
	for i, v := range vectors {
		var n float64
		for d := 0; d < dim; d++ {
			n += v[d] * v[d]
		}
		dbNorm[i] = math.Sqrt(n)
	}

	result := make([][]int, len(queries))

	numWorkers := runtime.NumCPU()
	if numWorkers > len(queries) {
		numWorkers = len(queries)
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	chunkSize := (len(queries) + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(queries) {
			end = len(queries)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			scores := make([]scored, len(vectors))

			for q := start; q < end; q++ {
				query := queries[q]
				var qNorm float64
				for d := 0; d < dim; d++ {
					qNorm += query[d] * query[d]
				}
				qNorm = math.Sqrt(qNorm)

				for i, v := range vectors {
					var dot float64
					for d := 0; d < dim; d++ {
						dot += query[d] * v[d]
					}
					var sim float64
					if qNorm > 0 && dbNorm[i] > 0 {
						sim = dot / (qNorm * dbNorm[i])
					}
					scores[i] = scored{idx: i, score: sim}
				}
				sort.Slice(scores, func(i, j int) bool {
					return scores[i].score > scores[j].score
				})

				out := make([]int, topK)
				for i := 0; i < topK && i < len(scores); i++ {
					out[i] = scores[i].idx
				}
				result[q] = out
			}
		}(start, end)
	}
	wg.Wait()
	return result
}
