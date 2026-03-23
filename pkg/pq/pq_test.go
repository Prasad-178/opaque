package pq

import (
	"math"
	"math/rand"
	"testing"
)

func TestTrainAndEncode(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	D := 16
	M := 4
	N := 500

	// Generate random normalized vectors.
	vectors := make([][]float64, N)
	for i := range vectors {
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = rng.NormFloat64()
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		vectors[i] = v
	}

	pq, err := Train(vectors, D, Config{M: M, Ks: 32, Seed: 42})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	if pq.M != M {
		t.Errorf("M: got %d, want %d", pq.M, M)
	}
	if pq.Dsub != D/M {
		t.Errorf("Dsub: got %d, want %d", pq.Dsub, D/M)
	}
	if len(pq.Codebooks) != M {
		t.Errorf("Codebooks: got %d subspaces, want %d", len(pq.Codebooks), M)
	}
	for m, cb := range pq.Codebooks {
		if len(cb) != 32 {
			t.Errorf("Codebook[%d]: got %d centroids, want 32", m, len(cb))
		}
	}

	// Encode a vector.
	codes := pq.Encode(vectors[0])
	if len(codes) != M {
		t.Fatalf("codes length: got %d, want %d", len(codes), M)
	}
	for _, c := range codes {
		if int(c) >= 32 {
			t.Errorf("code %d exceeds Ks=32", c)
		}
	}
}

func TestADCApproximatesDotProduct(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	D := 128
	M := 8
	N := 2000

	// Generate normalized vectors.
	vectors := make([][]float64, N)
	for i := range vectors {
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = rng.NormFloat64()
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		vectors[i] = v
	}

	pq, err := Train(vectors, D, Config{M: M, Seed: 42})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Encode all vectors.
	allCodes := pq.EncodeBatch(vectors)

	// Pick a query and build ADC table.
	query := vectors[0]
	table := pq.BuildADCTable(query)

	// Compare ADC scores to exact dot products.
	var totalError float64
	for i := 1; i < N; i++ {
		exact := dot(query, vectors[i])
		approx := ADCScore(table, allCodes[i])
		totalError += math.Abs(exact - approx)
	}
	avgError := totalError / float64(N-1)

	t.Logf("Average ADC error vs exact dot product: %.6f", avgError)

	// ADC with 256 centroids per subspace on normalized vectors should be quite accurate.
	if avgError > 0.05 {
		t.Errorf("Average ADC error too high: %.6f (want < 0.05)", avgError)
	}
}

func TestADCPreservesRanking(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	D := 128
	M := 8
	N := 5000
	topK := 10

	// Generate clustered data (more realistic than pure random).
	// Create 20 cluster centers, then perturb.
	nClusters := 20
	centers := make([][]float64, nClusters)
	for c := 0; c < nClusters; c++ {
		v := make([]float64, D)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		centers[c] = v
	}

	vectors := make([][]float64, N)
	for i := range vectors {
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
		vectors[i] = v
	}

	pq, err := Train(vectors, D, Config{M: M, Seed: 42})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	allCodes := pq.EncodeBatch(vectors)
	query := vectors[0]
	table := pq.BuildADCTable(query)

	// Get exact top-K.
	type scored struct {
		idx   int
		score float64
	}
	exactScores := make([]scored, N-1)
	for i := 1; i < N; i++ {
		exactScores[i-1] = scored{idx: i, score: dot(query, vectors[i])}
	}
	// Sort descending.
	for i := 0; i < topK; i++ {
		maxIdx := i
		for j := i + 1; j < len(exactScores); j++ {
			if exactScores[j].score > exactScores[maxIdx].score {
				maxIdx = j
			}
		}
		exactScores[i], exactScores[maxIdx] = exactScores[maxIdx], exactScores[i]
	}

	exactTopK := make(map[int]bool, topK)
	for i := 0; i < topK; i++ {
		exactTopK[exactScores[i].idx] = true
	}

	// Get ADC top-K (with 10x re-rank).
	rerankK := topK * 10
	adcScores := make([]scored, N-1)
	for i := 1; i < N; i++ {
		adcScores[i-1] = scored{idx: i, score: ADCScore(table, allCodes[i])}
	}
	for i := 0; i < rerankK && i < len(adcScores); i++ {
		maxIdx := i
		for j := i + 1; j < len(adcScores); j++ {
			if adcScores[j].score > adcScores[maxIdx].score {
				maxIdx = j
			}
		}
		adcScores[i], adcScores[maxIdx] = adcScores[maxIdx], adcScores[i]
	}

	// Re-rank top candidates with exact scores.
	rerankScores := make([]scored, 0, rerankK)
	for i := 0; i < rerankK && i < len(adcScores); i++ {
		idx := adcScores[i].idx
		rerankScores = append(rerankScores, scored{idx: idx, score: dot(query, vectors[idx])})
	}
	for i := 0; i < topK && i < len(rerankScores); i++ {
		maxIdx := i
		for j := i + 1; j < len(rerankScores); j++ {
			if rerankScores[j].score > rerankScores[maxIdx].score {
				maxIdx = j
			}
		}
		rerankScores[i], rerankScores[maxIdx] = rerankScores[maxIdx], rerankScores[i]
	}

	// Measure recall.
	hits := 0
	for i := 0; i < topK && i < len(rerankScores); i++ {
		if exactTopK[rerankScores[i].idx] {
			hits++
		}
	}
	recall := float64(hits) / float64(topK)
	t.Logf("PQ+rerank Recall@%d: %.1f%% (%d/%d)", topK, recall*100, hits, topK)

	// With synthetic clustered data and M=8, recall can vary. Real embeddings
	// (SIFT, GloVe) achieve much higher recall due to structured variance.
	// The integration benchmark validates real-world recall.
	if recall < 0.5 {
		t.Errorf("PQ+rerank recall too low: %.1f%% (want >= 50%%)", recall*100)
	}
}

func TestSerializeDeserialize(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	D := 32
	M := 4
	N := 200

	vectors := make([][]float64, N)
	for i := range vectors {
		v := make([]float64, D)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		vectors[i] = v
	}

	original, err := Train(vectors, D, Config{M: M, Ks: 16, Seed: 42})
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	data := original.Serialize()
	restored, err := Deserialize(data)
	if err != nil {
		t.Fatalf("Deserialize: %v", err)
	}

	if restored.M != original.M || restored.Ks != original.Ks ||
		restored.D != original.D || restored.Dsub != original.Dsub {
		t.Fatalf("metadata mismatch: got M=%d Ks=%d D=%d Dsub=%d, want M=%d Ks=%d D=%d Dsub=%d",
			restored.M, restored.Ks, restored.D, restored.Dsub,
			original.M, original.Ks, original.D, original.Dsub)
	}

	// Verify codebooks match.
	for m := 0; m < M; m++ {
		for k := 0; k < original.Ks; k++ {
			for d := 0; d < original.Dsub; d++ {
				if restored.Codebooks[m][k][d] != original.Codebooks[m][k][d] {
					t.Fatalf("codebook mismatch at [%d][%d][%d]", m, k, d)
				}
			}
		}
	}

	// Verify encoding produces same results.
	for i := 0; i < 10; i++ {
		origCodes := original.Encode(vectors[i])
		resCodes := restored.Encode(vectors[i])
		for j := range origCodes {
			if origCodes[j] != resCodes[j] {
				t.Fatalf("encode mismatch for vector %d at subspace %d", i, j)
			}
		}
	}
}

func TestTrainErrors(t *testing.T) {
	_, err := Train(nil, 16, Config{M: 4})
	if err == nil {
		t.Error("expected error for empty vectors")
	}

	_, err = Train([][]float64{{1, 2, 3}}, 3, Config{M: 0})
	if err == nil {
		t.Error("expected error for M=0")
	}

	_, err = Train([][]float64{{1, 2, 3}}, 3, Config{M: 2})
	if err == nil {
		t.Error("expected error for D not divisible by M")
	}
}

func BenchmarkADCScore(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	D := 128
	M := 8
	N := 10000

	vectors := make([][]float64, N)
	for i := range vectors {
		v := make([]float64, D)
		var norm float64
		for j := range v {
			v[j] = rng.NormFloat64()
			norm += v[j] * v[j]
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= norm
		}
		vectors[i] = v
	}

	pq, _ := Train(vectors, D, Config{M: M, Seed: 42})
	allCodes := pq.EncodeBatch(vectors)
	query := vectors[0]
	table := pq.BuildADCTable(query)

	b.Run("ADC_M8", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ADCScore(table, allCodes[i%N])
		}
	})

	b.Run("ExactDot_128D", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			dot(query, vectors[i%N])
		}
	})
}
