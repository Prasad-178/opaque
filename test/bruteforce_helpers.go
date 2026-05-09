//go:build sift1m || dbpedia1m

package test

import (
	"math"
	"sort"
)

// bruteForceTopK computes exact top-K nearest neighbors by cosine similarity.
// Shared across the bench-tagged tests (sift1m, dbpedia1m, ...).
func bruteForceTopK(queries, vectors [][]float64, dim, topK int) [][]int {
	type scored struct {
		idx   int
		score float64
	}

	result := make([][]int, len(queries))
	for q := range queries {
		scores := make([]scored, len(vectors))
		for i := range vectors {
			var dot, normA, normB float64
			for d := 0; d < dim; d++ {
				dot += queries[q][d] * vectors[i][d]
				normA += queries[q][d] * queries[q][d]
				normB += vectors[i][d] * vectors[i][d]
			}
			scores[i] = scored{idx: i, score: dot / (math.Sqrt(normA) * math.Sqrt(normB))}
		}
		sort.Slice(scores, func(i, j int) bool {
			return scores[i].score > scores[j].score
		})
		result[q] = make([]int, topK)
		for i := 0; i < topK && i < len(scores); i++ {
			result[q][i] = scores[i].idx
		}
	}
	return result
}
