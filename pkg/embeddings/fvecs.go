package embeddings

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// LoadFvecs loads vectors from a .fvecs file (used by SIFT1M, etc.)
//
// FVECS format:
// For each vector:
//   - 4 bytes: dimension (int32, little-endian)
//   - dimension * 4 bytes: float32 values (little-endian)
//
// All vectors must have the same dimension.
func LoadFvecs(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open fvecs file: %w", err)
	}
	defer f.Close()

	return ReadFvecs(f)
}

// ReadFvecs reads vectors from an io.Reader in FVECS format.
func ReadFvecs(r io.Reader) ([][]float64, error) {
	var vectors [][]float64
	var expectedDim int32 = -1

	for {
		// Read dimension
		var dim int32
		err := binary.Read(r, binary.LittleEndian, &dim)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read dimension: %w", err)
		}

		// Validate dimension consistency
		if expectedDim == -1 {
			expectedDim = dim
		} else if dim != expectedDim {
			return nil, fmt.Errorf("inconsistent dimensions: expected %d, got %d", expectedDim, dim)
		}

		// Read vector values
		floats := make([]float32, dim)
		err = binary.Read(r, binary.LittleEndian, floats)
		if err != nil {
			return nil, fmt.Errorf("failed to read vector values: %w", err)
		}

		// Convert to float64
		vec := make([]float64, dim)
		for i, v := range floats {
			vec[i] = float64(v)
		}

		vectors = append(vectors, vec)
	}

	return vectors, nil
}

// LoadIvecs loads integer vectors from a .ivecs file (used for ground truth).
//
// IVECS format (same structure as FVECS but with int32 values):
// For each vector:
//   - 4 bytes: dimension (int32, little-endian)
//   - dimension * 4 bytes: int32 values (little-endian)
func LoadIvecs(path string) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open ivecs file: %w", err)
	}
	defer f.Close()

	return ReadIvecs(f)
}

// ReadIvecs reads integer vectors from an io.Reader in IVECS format.
func ReadIvecs(r io.Reader) ([][]int, error) {
	var vectors [][]int
	var expectedDim int32 = -1

	for {
		// Read dimension
		var dim int32
		err := binary.Read(r, binary.LittleEndian, &dim)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read dimension: %w", err)
		}

		// Validate dimension consistency
		if expectedDim == -1 {
			expectedDim = dim
		} else if dim != expectedDim {
			return nil, fmt.Errorf("inconsistent dimensions: expected %d, got %d", expectedDim, dim)
		}

		// Read vector values
		ints := make([]int32, dim)
		err = binary.Read(r, binary.LittleEndian, ints)
		if err != nil {
			return nil, fmt.Errorf("failed to read vector values: %w", err)
		}

		// Convert to int
		vec := make([]int, dim)
		for i, v := range ints {
			vec[i] = int(v)
		}

		vectors = append(vectors, vec)
	}

	return vectors, nil
}

// LoadBvecs loads byte vectors from a .bvecs file (used by some datasets).
//
// BVECS format:
// For each vector:
//   - 4 bytes: dimension (int32, little-endian)
//   - dimension bytes: uint8 values
func LoadBvecs(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open bvecs file: %w", err)
	}
	defer f.Close()

	return ReadBvecs(f)
}

// ReadBvecs reads byte vectors from an io.Reader in BVECS format.
func ReadBvecs(r io.Reader) ([][]float64, error) {
	var vectors [][]float64
	var expectedDim int32 = -1

	for {
		// Read dimension
		var dim int32
		err := binary.Read(r, binary.LittleEndian, &dim)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read dimension: %w", err)
		}

		// Validate dimension consistency
		if expectedDim == -1 {
			expectedDim = dim
		} else if dim != expectedDim {
			return nil, fmt.Errorf("inconsistent dimensions: expected %d, got %d", expectedDim, dim)
		}

		// Read vector values
		bytes := make([]uint8, dim)
		_, err = io.ReadFull(r, bytes)
		if err != nil {
			return nil, fmt.Errorf("failed to read vector values: %w", err)
		}

		// Convert to float64
		vec := make([]float64, dim)
		for i, v := range bytes {
			vec[i] = float64(v)
		}

		vectors = append(vectors, vec)
	}

	return vectors, nil
}

// SaveFvecs saves vectors to a .fvecs file.
func SaveFvecs(path string, vectors [][]float64) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create fvecs file: %w", err)
	}
	defer f.Close()

	return WriteFvecs(f, vectors)
}

// WriteFvecs writes vectors to an io.Writer in FVECS format.
func WriteFvecs(w io.Writer, vectors [][]float64) error {
	for _, vec := range vectors {
		dim := int32(len(vec))

		// Write dimension
		if err := binary.Write(w, binary.LittleEndian, dim); err != nil {
			return fmt.Errorf("failed to write dimension: %w", err)
		}

		// Convert to float32 and write
		floats := make([]float32, dim)
		for i, v := range vec {
			floats[i] = float32(v)
		}
		if err := binary.Write(w, binary.LittleEndian, floats); err != nil {
			return fmt.Errorf("failed to write vector values: %w", err)
		}
	}

	return nil
}
