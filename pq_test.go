package opaque

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

// TestPQEndToEnd verifies the full pipeline with PQ enabled:
// Build with PQ → Search → compare recall against non-PQ baseline.
func TestPQEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping PQ integration test in short mode")
	}

	rng := rand.New(rand.NewSource(42))
	D := 128
	N := 5000
	numQueries := 20
	topK := 10

	// Generate clustered data (20 cluster centers + perturbation).
	nClusters := 20
	centers := make([][]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		v := make([]float64, D)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		centers[c] = v
	}

	ids := make([]string, N)
	vectors := make([][]float64, N)
	for i := 0; i < N; i++ {
		center := centers[rng.Intn(nClusters)]
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = center[j] + rng.NormFloat64()*0.3
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		ids[i] = randomID(rng)
		vectors[i] = v
	}

	// Generate queries.
	queries := make([][]float64, numQueries)
	for q := 0; q < numQueries; q++ {
		center := centers[rng.Intn(nClusters)]
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = center[j] + rng.NormFloat64()*0.2
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		queries[q] = v
	}

	// Compute brute-force ground truth.
	groundTruth := make([][]string, numQueries)
	for q := 0; q < numQueries; q++ {
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, N)
		for i := 0; i < N; i++ {
			var dot float64
			for j := 0; j < D; j++ {
				dot += queries[q][j] * vectors[i][j]
			}
			scores[i] = scored{id: ids[i], score: dot}
		}
		// Partial sort for top-K.
		for i := 0; i < topK; i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}
		groundTruth[q] = make([]string, topK)
		for i := 0; i < topK; i++ {
			groundTruth[q][i] = scores[i].id
		}
	}

	ctx := context.Background()

	// Build and search WITHOUT PQ (baseline).
	dbBaseline, err := NewDB(Config{
		Dimension:            D,
		NumClusters:          32,
		TopClusters:          8,
		RedundantAssignments: 2,
	})
	if err != nil {
		t.Fatalf("NewDB baseline: %v", err)
	}
	defer dbBaseline.Close()

	if err := dbBaseline.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch baseline: %v", err)
	}
	if err := dbBaseline.Build(ctx); err != nil {
		t.Fatalf("Build baseline: %v", err)
	}

	var baselineRecall float64
	for q := 0; q < numQueries; q++ {
		results, err := dbBaseline.Search(ctx, queries[q], topK)
		if err != nil {
			t.Fatalf("Search baseline q=%d: %v", q, err)
		}
		gt := make(map[string]bool, topK)
		for _, id := range groundTruth[q] {
			gt[id] = true
		}
		hits := 0
		for _, r := range results {
			if gt[r.ID] {
				hits++
			}
		}
		baselineRecall += float64(hits) / float64(topK)
	}
	baselineRecall /= float64(numQueries)

	// Build and search WITH PQ.
	dbPQ, err := NewDB(Config{
		Dimension:            D,
		NumClusters:          32,
		TopClusters:          8,
		RedundantAssignments: 2,
		PQSubspaces:          16,
	})
	if err != nil {
		t.Fatalf("NewDB PQ: %v", err)
	}
	defer dbPQ.Close()

	if err := dbPQ.AddBatch(ctx, ids, vectors); err != nil {
		t.Fatalf("AddBatch PQ: %v", err)
	}
	if err := dbPQ.Build(ctx); err != nil {
		t.Fatalf("Build PQ: %v", err)
	}

	var pqRecall float64
	for q := 0; q < numQueries; q++ {
		results, err := dbPQ.Search(ctx, queries[q], topK)
		if err != nil {
			t.Fatalf("Search PQ q=%d: %v", q, err)
		}
		gt := make(map[string]bool, topK)
		for _, id := range groundTruth[q] {
			gt[id] = true
		}
		hits := 0
		for _, r := range results {
			if gt[r.ID] {
				hits++
			}
		}
		pqRecall += float64(hits) / float64(topK)
	}
	pqRecall /= float64(numQueries)

	t.Logf("Baseline Recall@%d: %.1f%%", topK, baselineRecall*100)
	t.Logf("PQ Recall@%d:      %.1f%%", topK, pqRecall*100)
	t.Logf("Recall delta:      %.1f pp", (baselineRecall-pqRecall)*100)

	// PQ with re-ranking should be within 10 percentage points of baseline.
	if baselineRecall-pqRecall > 0.10 {
		t.Errorf("PQ recall too much lower than baseline: %.1f%% vs %.1f%% (delta > 10pp)",
			pqRecall*100, baselineRecall*100)
	}
}

func randomID(rng *rand.Rand) string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 12)
	for i := range b {
		b[i] = chars[rng.Intn(len(chars))]
	}
	return "vec-" + string(b)
}
