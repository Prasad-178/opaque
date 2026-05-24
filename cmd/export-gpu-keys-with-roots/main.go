// export-gpu-keys-with-roots connects to a GPU HE server, gets its NTT roots,
// and exports Galois keys converted to the server's NTT domain.
//
// Usage:
//
//	go run ./cmd/export-gpu-keys-with-roots/ -server localhost:50052 -out galois_keys.bin
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	pb "github.com/Prasad-178/opaque/api/proto/gpuhe"
	"github.com/Prasad-178/opaque/pkg/crypto"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	serverAddr := flag.String("server", "localhost:50052", "GPU HE server address")
	outPath := flag.String("out", "galois_keys.bin", "Output file path")
	sessionID := flag.String("session", "export-test", "Session ID")
	flag.Parse()

	log.Printf("Connecting to GPU server at %s...", *serverAddr)

	conn, err := grpc.NewClient(*serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(256*1024*1024)),
	)
	if err != nil {
		log.Fatalf("Connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewGPUHEServiceClient(conn)

	// Generate GPU-compatible keys
	params, err := crypto.NewParametersGPU()
	if err != nil {
		log.Fatalf("NewParametersGPU: %v", err)
	}

	engine, err := crypto.NewClientEngineWithParams(params)
	if err != nil {
		log.Fatalf("NewClientEngineWithParams: %v", err)
	}

	// Step 1: InitContext — get server's NTT roots
	log.Println("Step 1: InitContext...")
	ringQ := params.RingQ()
	qModuli := ringQ.ModuliChain()
	var pModuli []uint64
	if ringP := params.RingP(); ringP != nil {
		pModuli = ringP.ModuliChain()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	initResp, err := client.InitContext(ctx, &pb.InitContextRequest{
		SessionId: *sessionID,
		Params: &pb.CKKSParams{
			LogN:            int32(params.LogN()),
			QModuli:         qModuli,
			PModuli:         pModuli,
			LogDefaultScale: 45,
		},
	})
	if err != nil {
		log.Fatalf("InitContext: %v", err)
	}
	if !initResp.Success {
		log.Fatalf("InitContext failed: %s", initResp.Error)
	}

	log.Printf("Got %d NTT roots from server", len(initResp.NttRoots))
	for i, r := range initResp.NttRoots {
		fmt.Printf("  Root[%d]: %d\n", i, r)
	}

	// Step 2: Export keys with server's NTT roots
	log.Println("Step 2: Serializing with server's NTT roots...")

	elements := galoisElements(params)
	evk := engine.GetEvalKeys()
	keys := make([]*rlwe.GaloisKey, len(elements))
	for i, el := range elements {
		gk, err := evk.GetGaloisKey(el)
		if err != nil {
			log.Fatalf("GetGaloisKey(%d): %v", el, err)
		}
		keys[i] = gk
	}

	data, err := crypto.SerializeGaloisKeysHEonGPU(params, keys, elements, initResp.NttRoots)
	if err != nil {
		log.Fatalf("SerializeGaloisKeysHEonGPU: %v", err)
	}

	log.Printf("Serialized: %d bytes (%.1f MB)", len(data), float64(len(data))/(1024*1024))

	if err := os.WriteFile(*outPath, data, 0600); err != nil {
		log.Fatalf("WriteFile: %v", err)
	}

	log.Printf("Written to: %s", *outPath)
}

func galoisElements(params interface {
	LogN() int
	GaloisElement(int) uint64
}) []uint64 {
	logN := params.LogN()
	elements := make([]uint64, logN)
	for i := 0; i < logN; i++ {
		elements[i] = params.GaloisElement(1 << i)
	}
	return elements
}
