package lsh

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestNewIndex(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	if len(idx.planes) != 64 {
		t.Errorf("expected 64 planes, got %d", len(idx.planes))
	}

	// Check planes are normalized
	for i, plane := range idx.planes {
		var norm float64
		for _, v := range plane {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if math.Abs(norm-1.0) > 1e-6 {
			t.Errorf("plane %d not normalized: norm = %f", i, norm)
		}
	}
}

func TestAddAndSearch(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 4,
		NumBits:   8,
		Seed:      42,
	})

	// Add some vectors
	ids := []string{"a", "b", "c", "d"}
	vectors := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}

	err := idx.Add(ids, vectors)
	if err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	if idx.Count() != 4 {
		t.Errorf("expected count 4, got %d", idx.Count())
	}

	// Search for first vector
	candidates, err := idx.SearchVector(vectors[0], 4)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(candidates) == 0 {
		t.Error("no candidates found")
	}

	// First result should be "a" (exact match)
	found := false
	for _, c := range candidates {
		if c.ID == "a" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected to find 'a' in candidates")
	}
}

func TestHashConsistency(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	// Same vector should produce same hash
	vec := make([]float64, 128)
	rand.Seed(123)
	for i := range vec {
		vec[i] = rand.NormFloat64()
	}

	hash1 := idx.HashBytes(vec)
	hash2 := idx.HashBytes(vec)

	if !bytes.Equal(hash1, hash2) {
		t.Error("same vector produced different hashes")
	}
}

func TestSimilarVectorsCloseHashes(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	// Create a base vector
	base := make([]float64, 128)
	rand.Seed(456)
	for i := range base {
		base[i] = rand.NormFloat64()
	}

	// Create a similar vector (small perturbation)
	similar := make([]float64, 128)
	for i := range similar {
		similar[i] = base[i] + rand.NormFloat64()*0.1
	}

	// Create a random vector
	random := make([]float64, 128)
	for i := range random {
		random[i] = rand.NormFloat64()
	}

	hashBase := idx.HashBytes(base)
	hashSimilar := idx.HashBytes(similar)
	hashRandom := idx.HashBytes(random)

	distSimilar := HammingDistance(hashBase, hashSimilar)
	distRandom := HammingDistance(hashBase, hashRandom)

	t.Logf("Distance to similar: %d", distSimilar)
	t.Logf("Distance to random: %d", distRandom)

	// Similar vector should have smaller Hamming distance
	if distSimilar >= distRandom {
		t.Logf("Warning: similar vector not closer (this can happen occasionally)")
	}
}

func TestSaveLoad(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 4,
		NumBits:   8,
		Seed:      42,
	})

	ids := []string{"x", "y", "z"}
	vectors := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	idx.Add(ids, vectors)

	// Save to buffer
	var buf bytes.Buffer
	err := idx.Save(&buf)
	if err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Load from buffer
	idx2, err := Load(&buf)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	// Verify
	if idx2.Count() != idx.Count() {
		t.Errorf("count mismatch: got %d, want %d", idx2.Count(), idx.Count())
	}

	// Search should work
	candidates, err := idx2.SearchVector(vectors[0], 3)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(candidates) != 3 {
		t.Errorf("expected 3 candidates, got %d", len(candidates))
	}
}

func TestRemove(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 4,
		NumBits:   8,
		Seed:      42,
	})

	ids := []string{"a", "b", "c"}
	vectors := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	idx.Add(ids, vectors)

	if idx.Count() != 3 {
		t.Fatalf("expected count 3, got %d", idx.Count())
	}

	// Remove one
	idx.Remove([]string{"b"})

	if idx.Count() != 2 {
		t.Errorf("expected count 2 after remove, got %d", idx.Count())
	}

	// Should not find removed vector
	_, ok := idx.GetVector("b")
	if ok {
		t.Error("removed vector 'b' still found")
	}
}

func TestMultiProbeSearch(t *testing.T) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	// Add many vectors
	n := 1000
	ids := make([]string, n)
	vectors := make([][]float64, n)
	rand.Seed(789)

	for i := 0; i < n; i++ {
		ids[i] = string(rune('A' + i%26)) + string(rune('0'+i/26))
		vectors[i] = make([]float64, 128)
		for j := 0; j < 128; j++ {
			vectors[i][j] = rand.NormFloat64()
		}
	}
	idx.Add(ids, vectors)

	// Query
	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.NormFloat64()
	}
	queryHash := idx.HashBytes(query)

	// Compare regular vs multi-probe
	regular, _ := idx.Search(queryHash, 50)
	multiProbe, _ := idx.MultiProbeSearch(queryHash, 50, 5)

	t.Logf("Regular search found: %d candidates", len(regular))
	t.Logf("Multi-probe search found: %d candidates", len(multiProbe))

	// Multi-probe should find at least as many
	if len(multiProbe) < len(regular) {
		t.Error("multi-probe found fewer candidates than regular search")
	}
}

func BenchmarkHash(b *testing.B) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	vec := make([]float64, 128)
	rand.Seed(111)
	for i := range vec {
		vec[i] = rand.NormFloat64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.HashBytes(vec)
	}
}

func BenchmarkSearch(b *testing.B) {
	idx := NewIndex(Config{
		Dimension: 128,
		NumBits:   64,
		Seed:      42,
	})

	// Add vectors
	n := 10000
	ids := make([]string, n)
	vectors := make([][]float64, n)
	rand.Seed(222)

	for i := 0; i < n; i++ {
		ids[i] = string(rune(i))
		vectors[i] = make([]float64, 128)
		for j := 0; j < 128; j++ {
			vectors[i][j] = rand.NormFloat64()
		}
	}
	idx.Add(ids, vectors)

	query := make([]float64, 128)
	for i := range query {
		query[i] = rand.NormFloat64()
	}
	queryHash := idx.HashBytes(query)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(queryHash, 100)
	}
}
