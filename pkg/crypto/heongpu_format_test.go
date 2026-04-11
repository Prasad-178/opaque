package crypto

import (
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func TestSerializeGaloisKeysHEonGPU(t *testing.T) {
	params, err := NewParametersGPU()
	if err != nil {
		t.Fatalf("NewParametersGPU: %v", err)
	}

	engine, err := NewClientEngineWithParams(params)
	if err != nil {
		t.Fatalf("NewClientEngineWithParams: %v", err)
	}

	evk := engine.GetEvalKeys()
	if evk == nil {
		t.Fatal("no eval keys")
	}

	elements := galoisElements(params)
	keys := make([]*rlwe.GaloisKey, len(elements))
	for i, el := range elements {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			t.Fatalf("GetGaloisKey(%d): %v", el, err)
		}
		keys[i] = gk
	}

	data, err := SerializeGaloisKeysHEonGPU(params, keys, elements)
	if err != nil {
		t.Fatalf("SerializeGaloisKeysHEonGPU: %v", err)
	}

	ringSize := 1 << params.LogN()
	qSize := params.MaxLevelQ() + 1
	pSize := params.MaxLevelP() + 1
	qPrimeSize := qSize + pSize
	d := qSize
	galoisKeySize := 2 * d * qPrimeSize * ringSize

	t.Logf("Serialized Galois keys: %d bytes (%.1f MB)", len(data), float64(len(data))/(1024*1024))
	t.Logf("Parameters: LogN=%d, Q=%d, P=%d, d=%d, galoisKeySize=%d uint64",
		params.LogN(), qSize, pSize, d, galoisKeySize)
	t.Logf("Galois elements: %d keys", len(elements))

	// Basic sanity: size should be header + key_count * (4 + galoisKeySize*8) + zero_key
	expectedDataSize := len(elements) * galoisKeySize * 8 // key data
	if len(data) < expectedDataSize {
		t.Errorf("serialized data too small: %d bytes, expected at least %d bytes of key data",
			len(data), expectedDataSize)
	}
}

func TestSerializeRelinKeyHEonGPU(t *testing.T) {
	params, err := NewParametersGPU()
	if err != nil {
		t.Fatalf("NewParametersGPU: %v", err)
	}

	engine, err := NewClientEngineWithParams(params)
	if err != nil {
		t.Fatalf("NewClientEngineWithParams: %v", err)
	}

	evk := engine.GetEvalKeys()
	rlkKey, err := evk.GetRelinearizationKey()
	if err != nil {
		t.Skipf("No relin key available: %v", err)
	}

	data, err := SerializeRelinKeyHEonGPU(params, rlkKey)
	if err != nil {
		t.Fatalf("SerializeRelinKeyHEonGPU: %v", err)
	}

	t.Logf("Serialized relin key: %d bytes (%.1f MB)", len(data), float64(len(data))/(1024*1024))
}
