// dump-key-format dumps the exact byte layout of our HEonGPU-format Galois key file.
// This helps debug format mismatches by showing exact offsets and values.
//
// Run: go run ./cmd/dump-key-format/ /tmp/galois_keys.bin
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: dump-key-format <galois_keys.bin>")
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	offset := 0

	// Read fields in exact HEonGPU load() order
	fmt.Println("=== HEonGPU Galois Key File Header ===")
	fmt.Println("(Comparing against HEonGPU's Galoiskey::load() read order)")
	fmt.Println()

	scheme := readU8(f, &offset, "scheme_type (uint8)")
	keyType := readU8(f, &offset, "keyswitching_type (uint8)")
	ringSize := readI32(f, &offset, "ring_size (int32)")
	qPrimeSize := readI32(f, &offset, "Q_prime_size_ (int32)")
	qSize := readI32(f, &offset, "Q_size_ (int32)")
	d := readI32(f, &offset, "d_ (int32)")
	customized := readBool(f, &offset, "customized (bool)")
	groupOrder := readI32(f, &offset, "group_order_ (int32)")
	storageType := readU8(f, &offset, "storage_type_ (uint8)")
	generated := readBool(f, &offset, "galois_key_generated_ (bool)")

	fmt.Println()
	fmt.Printf("Summary: scheme=%d keyType=%d ring=%d Q'=%d Q=%d d=%d custom=%v group=%d storage=%d gen=%v\n",
		scheme, keyType, ringSize, qPrimeSize, qSize, d, customized, groupOrder, storageType, generated)

	if customized {
		fmt.Println("\n=== Custom Galois Elements ===")
		count := readU32(f, &offset, "custom_galois_elt_size (uint32)")
		fmt.Printf("  Count: %d\n", count)
		for i := 0; i < int(count); i++ {
			el := readU32(f, &offset, fmt.Sprintf("  galois_elt[%d] (uint32)", i))
			_ = el
		}
	} else {
		fmt.Println("\n=== Galois Element Map ===")
		count := readU32(f, &offset, "galois_elt_size (uint32)")
		fmt.Printf("  Count: %d pairs\n", count)
		for i := 0; i < int(count); i++ {
			first := readI32(f, &offset, fmt.Sprintf("  pair[%d].first (int)", i))
			second := readI32(f, &offset, fmt.Sprintf("  pair[%d].second (int)", i))
			_, _ = first, second
		}
	}

	galoisEltZero := readI32(f, &offset, "galois_elt_zero (int)")
	fmt.Printf("\n  galois_elt_zero: %d\n", galoisEltZero)

	// galoiskey_size_ is Data64 = uint64
	galoisKeySize := readU64(f, &offset, "galoiskey_size_ (uint64)")
	fmt.Printf("  galoiskey_size: %d uint64 = %d bytes = %.1f MB\n",
		galoisKeySize, galoisKeySize*8, float64(galoisKeySize*8)/(1024*1024))

	fmt.Println("\n=== Key Data Section ===")
	keyCount := readU32(f, &offset, "key_count (uint32)")
	fmt.Printf("  key_count: %d\n", keyCount)

	for i := 0; i < int(keyCount); i++ {
		keyId := readI32(f, &offset, fmt.Sprintf("  key[%d].galois_element (int)", i))
		fmt.Printf("  key[%d]: element=%d, data=%d uint64 (%d bytes)\n",
			i, keyId, galoisKeySize, galoisKeySize*8)

		// Skip key data
		skip := int64(galoisKeySize) * 8
		n, err := io.CopyN(io.Discard, f, skip)
		if err != nil {
			fmt.Printf("  ERROR: could only read %d of %d bytes: %v\n", n, skip, err)
			return
		}
		offset += int(skip)
	}

	// Zero key (conjugation)
	fmt.Printf("\n  zero_key: %d uint64 (%d bytes)\n", galoisKeySize, galoisKeySize*8)
	skip := int64(galoisKeySize) * 8
	n, err := io.CopyN(io.Discard, f, skip)
	if err != nil {
		fmt.Printf("  ERROR reading zero key: only %d of %d bytes: %v\n", n, skip, err)
	} else {
		offset += int(skip)
	}

	// Check remaining
	remaining, _ := io.Copy(io.Discard, f)
	fmt.Printf("\n=== File Summary ===\n")
	fmt.Printf("  Total read: %d bytes\n", offset)
	fmt.Printf("  Remaining:  %d bytes\n", remaining)
	fmt.Printf("  File total: %d bytes\n", int64(offset)+remaining)

	if remaining == 0 {
		fmt.Println("\n  ✓ File size matches expected layout perfectly!")
	} else {
		fmt.Printf("\n  ✗ %d extra bytes — format mismatch somewhere\n", remaining)
	}
}

func readU8(r io.Reader, offset *int, label string) uint8 {
	var v uint8
	binary.Read(r, binary.LittleEndian, &v)
	fmt.Printf("  [%4d] %s = %d\n", *offset, label, v)
	*offset += 1
	return v
}

func readBool(r io.Reader, offset *int, label string) bool {
	var v uint8
	binary.Read(r, binary.LittleEndian, &v)
	b := v != 0
	fmt.Printf("  [%4d] %s = %v (0x%02x)\n", *offset, label, b, v)
	*offset += 1
	return b
}

func readI32(r io.Reader, offset *int, label string) int32 {
	var v int32
	binary.Read(r, binary.LittleEndian, &v)
	fmt.Printf("  [%4d] %s = %d\n", *offset, label, v)
	*offset += 4
	return v
}

func readU32(r io.Reader, offset *int, label string) uint32 {
	var v uint32
	binary.Read(r, binary.LittleEndian, &v)
	fmt.Printf("  [%4d] %s = %d\n", *offset, label, v)
	*offset += 4
	return v
}

func readU64(r io.Reader, offset *int, label string) uint64 {
	var v uint64
	binary.Read(r, binary.LittleEndian, &v)
	fmt.Printf("  [%4d] %s = %d\n", *offset, label, v)
	*offset += 8
	return v
}
