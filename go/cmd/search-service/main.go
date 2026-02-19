// Command search-service runs the Opaque search gRPC server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	pb "github.com/opaque/opaque/go/api/proto"
	"github.com/opaque/opaque/go/internal/service"
	"github.com/opaque/opaque/go/internal/store"
	"github.com/opaque/opaque/go/pkg/grpcserver"
)

var (
	grpcPort    = flag.Int("grpc-port", 50051, "gRPC server port")
	httpPort    = flag.Int("http-port", 8080, "HTTP metrics/health port")
	dimension   = flag.Int("dimension", 128, "Vector dimension")
	lshBits     = flag.Int("lsh-bits", 128, "Number of LSH bits")
	lshSeed     = flag.Int64("lsh-seed", 42, "LSH random seed")
	demoVectors = flag.Int("demo-vectors", 1000, "Number of demo vectors to generate")
	tlsCert     = flag.String("tls-cert", "", "TLS certificate file (optional)")
	tlsKey      = flag.String("tls-key", "", "TLS key file (optional)")
)

func main() {
	flag.Parse()

	log.Println("Starting Opaque Search Service...")

	// Create in-memory vector store (replace with Milvus for production)
	vectorStore := store.NewMemoryStore()

	// Create search service
	cfg := service.Config{
		LSHNumBits:          *lshBits,
		LSHDimension:        *dimension,
		LSHSeed:             *lshSeed,
		MaxSessionTTL:       24 * time.Hour,
		MaxConcurrentScores: 16,
	}

	svc, err := service.NewSearchService(cfg, vectorStore)
	if err != nil {
		log.Fatalf("Failed to create search service: %v", err)
	}

	// Generate demo vectors
	if *demoVectors > 0 {
		log.Printf("Generating %d demo vectors...", *demoVectors)
		generateDemoVectors(svc, *demoVectors, *dimension)
		log.Printf("Generated %d vectors", *demoVectors)
	}

	// Create gRPC server
	serverOpts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(50 * 1024 * 1024), // 50MB for large ciphertexts
		grpc.MaxSendMsgSize(50 * 1024 * 1024),
		grpc.ChainUnaryInterceptor(
			grpcserver.RecoveryUnaryInterceptor(),
			grpcserver.LoggingUnaryInterceptor(),
		),
		grpc.ChainStreamInterceptor(
			grpcserver.RecoveryStreamInterceptor(),
			grpcserver.LoggingStreamInterceptor(),
		),
	}
	if *tlsCert != "" && *tlsKey != "" {
		creds, err := grpcserver.LoadTLSCredentials(*tlsCert, *tlsKey)
		if err != nil {
			log.Fatalf("Failed to load TLS credentials: %v", err)
		}
		serverOpts = append(serverOpts, grpc.Creds(creds))
		log.Println("TLS enabled")
	}
	grpcServer := grpc.NewServer(serverOpts...)

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Register reflection for grpcurl/grpcui
	reflection.Register(grpcServer)

	// Register Opaque search service
	pb.RegisterOpaqueSearchServer(grpcServer, grpcserver.New(svc))

	// Start gRPC server
	grpcLis, err := net.Listen("tcp", fmt.Sprintf(":%d", *grpcPort))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", *grpcPort, err)
	}

	go func() {
		log.Printf("gRPC server listening on :%d", *grpcPort)
		if err := grpcServer.Serve(grpcLis); err != nil {
			log.Fatalf("gRPC server failed: %v", err)
		}
	}()

	// Start HTTP server for metrics/health
	httpMux := http.NewServeMux()
	httpMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		healthy, msg, sessions, vectors := svc.HealthCheck(r.Context())
		if healthy {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "OK\n")
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, "ERROR: %s\n", msg)
		}
		fmt.Fprintf(w, "Sessions: %d\n", sessions)
		fmt.Fprintf(w, "Vectors: %d\n", vectors)
	})

	httpMux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Ready\n")
	})

	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", *httpPort),
		Handler: httpMux,
	}

	go func() {
		log.Printf("HTTP server listening on :%d", *httpPort)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")

	// Graceful shutdown
	grpcServer.GracefulStop()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	httpServer.Shutdown(ctx)

	log.Println("Shutdown complete")
}

// generateDemoVectors creates synthetic vectors for testing
func generateDemoVectors(svc *service.SearchService, n, dim int) {
	rand.Seed(time.Now().UnixNano())

	ids := make([]string, n)
	vectors := make([][]float64, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)

		// Generate random normalized vector
		vec := make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			vec[j] = rand.NormFloat64()
			norm += vec[j] * vec[j]
		}
		norm = float64(1.0 / float64(norm))
		for j := range vec {
			vec[j] *= norm
		}

		vectors[i] = vec
	}

	if err := svc.AddVectors(context.Background(), ids, vectors, nil); err != nil {
		log.Printf("Failed to add demo vectors: %v", err)
	}
}
