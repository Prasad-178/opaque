package embeddings

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestReadFvecs(t *testing.T) {
	// Create a test buffer with 3 vectors of dimension 4
	var buf bytes.Buffer

	vectors := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}

	for _, vec := range vectors {
		dim := int32(len(vec))
		binary.Write(&buf, binary.LittleEndian, dim)
		binary.Write(&buf, binary.LittleEndian, vec)
	}

	// Read back
	result, err := ReadFvecs(&buf)
	if err != nil {
		t.Fatalf("ReadFvecs failed: %v", err)
	}

	if len(result) != 3 {
		t.Errorf("expected 3 vectors, got %d", len(result))
	}

	for i, vec := range result {
		if len(vec) != 4 {
			t.Errorf("vector %d: expected dimension 4, got %d", i, len(vec))
		}
		for j, v := range vec {
			expected := float64(vectors[i][j])
			if v != expected {
				t.Errorf("vector[%d][%d]: expected %f, got %f", i, j, expected, v)
			}
		}
	}
}

func TestReadIvecs(t *testing.T) {
	// Create a test buffer with 2 vectors of dimension 5 (ground truth format)
	var buf bytes.Buffer

	vectors := [][]int32{
		{0, 1, 2, 3, 4},
		{5, 6, 7, 8, 9},
	}

	for _, vec := range vectors {
		dim := int32(len(vec))
		binary.Write(&buf, binary.LittleEndian, dim)
		binary.Write(&buf, binary.LittleEndian, vec)
	}

	// Read back
	result, err := ReadIvecs(&buf)
	if err != nil {
		t.Fatalf("ReadIvecs failed: %v", err)
	}

	if len(result) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(result))
	}

	for i, vec := range result {
		if len(vec) != 5 {
			t.Errorf("vector %d: expected dimension 5, got %d", i, len(vec))
		}
		for j, v := range vec {
			expected := int(vectors[i][j])
			if v != expected {
				t.Errorf("vector[%d][%d]: expected %d, got %d", i, j, expected, v)
			}
		}
	}
}

func TestWriteAndReadFvecs(t *testing.T) {
	// Create test vectors
	original := [][]float64{
		{1.5, 2.5, 3.5},
		{4.5, 5.5, 6.5},
	}

	// Write to buffer
	var buf bytes.Buffer
	err := WriteFvecs(&buf, original)
	if err != nil {
		t.Fatalf("WriteFvecs failed: %v", err)
	}

	// Read back
	result, err := ReadFvecs(&buf)
	if err != nil {
		t.Fatalf("ReadFvecs failed: %v", err)
	}

	if len(result) != len(original) {
		t.Fatalf("expected %d vectors, got %d", len(original), len(result))
	}

	for i, vec := range result {
		for j, v := range vec {
			// Allow small floating point difference due to float32 conversion
			diff := v - original[i][j]
			if diff > 0.0001 || diff < -0.0001 {
				t.Errorf("vector[%d][%d]: expected ~%f, got %f", i, j, original[i][j], v)
			}
		}
	}
}

func TestGenerate(t *testing.T) {
	dataset := Generate(100, 64, 42)

	if dataset.Name != "random" {
		t.Errorf("expected name 'random', got '%s'", dataset.Name)
	}

	if len(dataset.Vectors) != 100 {
		t.Errorf("expected 100 vectors, got %d", len(dataset.Vectors))
	}

	if dataset.Dimension != 64 {
		t.Errorf("expected dimension 64, got %d", dataset.Dimension)
	}

	if len(dataset.IDs) != 100 {
		t.Errorf("expected 100 IDs, got %d", len(dataset.IDs))
	}

	// Test reproducibility
	dataset2 := Generate(100, 64, 42)
	for i := range dataset.Vectors {
		for j := range dataset.Vectors[i] {
			if dataset.Vectors[i][j] != dataset2.Vectors[i][j] {
				t.Errorf("Generate not reproducible at [%d][%d]", i, j)
			}
		}
	}
}

func TestDatasetSubset(t *testing.T) {
	dataset := Generate(1000, 64, 42)
	subset := dataset.Subset(100)

	if len(subset.Vectors) != 100 {
		t.Errorf("expected 100 vectors in subset, got %d", len(subset.Vectors))
	}

	if len(subset.IDs) != 100 {
		t.Errorf("expected 100 IDs in subset, got %d", len(subset.IDs))
	}

	// Verify subset is the same as first N of original
	for i := 0; i < 100; i++ {
		for j := range dataset.Vectors[i] {
			if subset.Vectors[i][j] != dataset.Vectors[i][j] {
				t.Errorf("subset mismatch at [%d][%d]", i, j)
			}
		}
	}
}

func TestDatasetStats(t *testing.T) {
	dataset := Generate(500, 128, 42)
	dataset.Queries = [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	dataset.GroundTruth = [][]int{{0, 1, 2}, {3, 4, 5}}

	stats := dataset.Stats()

	if stats.NumVectors != 500 {
		t.Errorf("expected 500 vectors, got %d", stats.NumVectors)
	}

	if stats.NumQueries != 2 {
		t.Errorf("expected 2 queries, got %d", stats.NumQueries)
	}

	if stats.Dimension != 128 {
		t.Errorf("expected dimension 128, got %d", stats.Dimension)
	}

	if !stats.HasGroundTruth {
		t.Error("expected HasGroundTruth to be true")
	}

	if stats.GroundTruthDepth != 3 {
		t.Errorf("expected ground truth depth 3, got %d", stats.GroundTruthDepth)
	}
}
