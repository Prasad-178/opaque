package encrypt

import (
	"encoding/binary"
	"math"
)

// Vectors are encoded on the wire (and at rest after AES-GCM) as little-endian
// float32 — 4 bytes per element instead of float64's 8 bytes. This halves the
// AES ciphertext size and the worker accumulator memory peak during the build
// phase, which is the dominant memory cost at high dimensionality
// (e.g. 1M × 1536-dim ada-002 → ~6 GB ciphertext savings vs the prior float64
// encoding). Callers continue to work with []float64; the upcast on decode is
// loss-free in the sense that ada-002 is float32-native and SIFT precision is
// well above the float32 floor (~1.2e-7 per element).
//
// Once the broader float64 → float32 refactor lands across pkg/cluster,
// pkg/hierarchical, etc., these helpers should switch to native float32 input/
// output. Until then this is the cheapest place to capture the ciphertext-size
// win without changing the public API or the internal type system.
const bytesPerElement = 4

// VectorToBytes converts a float64 slice to bytes by downcasting each element
// to float32, then writing little-endian. 4 bytes per element.
func VectorToBytes(vector []float64) []byte {
	buf := make([]byte, len(vector)*bytesPerElement)
	for i, v := range vector {
		binary.LittleEndian.PutUint32(buf[i*bytesPerElement:], math.Float32bits(float32(v)))
	}
	return buf
}

// BytesToVector converts bytes back to a float64 slice. Each 4-byte chunk is
// decoded as float32 and upcast to float64 for callers' downstream math.
func BytesToVector(data []byte) []float64 {
	if len(data)%bytesPerElement != 0 {
		return nil
	}

	vector := make([]float64, len(data)/bytesPerElement)
	for i := range vector {
		bits := binary.LittleEndian.Uint32(data[i*bytesPerElement:])
		vector[i] = float64(math.Float32frombits(bits))
	}
	return vector
}

// VectorDimension returns the dimension of a vector stored in bytes.
func VectorDimension(data []byte) int {
	return len(data) / bytesPerElement
}
