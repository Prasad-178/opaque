package encrypt

import (
	"encoding/binary"
	"math"
)

// VectorToBytes converts a float64 slice to bytes using little-endian encoding.
func VectorToBytes(vector []float64) []byte {
	buf := make([]byte, len(vector)*8)
	for i, v := range vector {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return buf
}

// BytesToVector converts bytes back to a float64 slice.
func BytesToVector(data []byte) []float64 {
	if len(data)%8 != 0 {
		return nil
	}

	vector := make([]float64, len(data)/8)
	for i := range vector {
		vector[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return vector
}

// VectorDimension returns the dimension of a vector stored in bytes.
func VectorDimension(data []byte) int {
	return len(data) / 8
}
