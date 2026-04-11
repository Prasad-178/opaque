// export-gpu-keys generates CKKS evaluation keys in HEonGPU format.
//
// This program creates keys using GPU-compatible parameters (LogP=[61]),
// serializes them in HEonGPU's binary format, and writes to a file.
// The file can be loaded by HEonGPU's Galoiskey::load() on the GPU server.
//
// Usage:
//
//	go run ./cmd/export-gpu-keys/ -out galois_keys.bin
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func main() {
	outPath := flag.String("out", "galois_keys.bin", "Output file path")
	flag.Parse()

	log.Println("Generating GPU-compatible CKKS keys...")
	start := time.Now()

	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatalf("NewParametersGPU: %v", err)
	}

	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		log.Fatalf("NewClientEngineWithParams: %v", err)
	}

	evk := engine.GetEvalKeys()
	if evk == nil {
		log.Fatal("No eval keys generated")
	}

	elements := galoisElements(params)
	keys := make([]*rlwe.GaloisKey, len(elements))
	for i, el := range elements {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			log.Fatalf("GetGaloisKey(%d): %v", el, err)
		}
		keys[i] = gk
	}

	log.Printf("Key generation: %v", time.Since(start))
	log.Printf("Parameters: LogN=%d, Q=%d primes, P=%d primes",
		params.LogN(), params.MaxLevelQ()+1, params.MaxLevelP()+1)
	log.Printf("Galois elements: %d keys", len(elements))

	log.Println("Serializing to HEonGPU format...")
	serStart := time.Now()

	data, err := crypto.SerializeGaloisKeysHEonGPU(params, keys, elements)
	if err != nil {
		log.Fatalf("SerializeGaloisKeysHEonGPU: %v", err)
	}

	log.Printf("Serialization: %v", time.Since(serStart))
	log.Printf("Output size: %d bytes (%.1f MB)", len(data), float64(len(data))/(1024*1024))

	if err := os.WriteFile(*outPath, data, 0600); err != nil {
		log.Fatalf("WriteFile: %v", err)
	}

	log.Printf("Written to: %s", *outPath)

	// Also print galois elements for the C++ side
	fmt.Println("\nGalois elements (for HEonGPU):")
	for i, el := range elements {
		fmt.Printf("  Element[%d]: %d (rotation by %d)\n", i, el, 1<<i)
	}
}

func galoisElements(params interface{ LogN() int; GaloisElement(int) uint64 }) []uint64 {
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}
