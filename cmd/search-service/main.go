// Command search-service runs the Opaque search gRPC server.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	pb "github.com/Prasad-178/opaque/api/proto"
	"github.com/Prasad-178/opaque/internal/service"
	"github.com/Prasad-178/opaque/internal/store"
	"github.com/Prasad-178/opaque/pkg/grpcserver"
)

var (
	grpcPort            = flag.Int("grpc-port", 50051, "gRPC server port")
	httpPort            = flag.Int("http-port", 8080, "HTTP metrics/health port")
	dimension           = flag.Int("dimension", 128, "Vector dimension")
	lshBits             = flag.Int("lsh-bits", 128, "Number of LSH bits")
	lshSeed             = flag.Int64("lsh-seed", 42, "LSH random seed")
	storageBackend      = flag.String("storage-backend", "file", "Vector storage backend: file|memory")
	storagePath         = flag.String("storage-path", "./data/vectors.json", "Path for file storage backend")
	bootstrapVectors    = flag.String("bootstrap-vectors", "", "Optional JSON file containing vectors to preload")
	demoVectors         = flag.Int("demo-vectors", 0, "Number of generated demo vectors to seed (disabled by default)")
	allowUnsafeDemoData = flag.Bool("allow-unsafe-demo-data", false, "Allow startup with generated demo vectors")
	tlsCert             = flag.String("tls-cert", "", "TLS certificate file (optional)")
	tlsKey              = flag.String("tls-key", "", "TLS key file (optional)")
)

type bootstrapVector struct {
	ID       string         `json:"id"`
	Values   []float64      `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func main() {
	flag.Parse()

	log.Println("Starting Opaque Search Service...")

	// Create vector store.
	vectorStore, err := createStore()
	if err != nil {
		log.Fatalf("Failed to create vector store: %v", err)
	}
	defer vectorStore.Close()

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

	if err := seedVectors(svc); err != nil {
		log.Fatalf("Failed to seed vectors: %v", err)
	}

	serverOpts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(50 * 1024 * 1024),
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

	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	reflection.Register(grpcServer)
	pb.RegisterOpaqueSearchServer(grpcServer, grpcserver.New(svc))

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
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	grpcServer.GracefulStop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(ctx)

	log.Println("Shutdown complete")
}

func createStore() (store.VectorStore, error) {
	switch strings.ToLower(*storageBackend) {
	case "memory":
		log.Println("Using in-memory vector store (non-durable)")
		return store.NewMemoryStore(), nil
	case "file":
		if *storagePath == "" {
			return nil, fmt.Errorf("storage-path is required when storage-backend=file")
		}
		absPath, err := filepath.Abs(*storagePath)
		if err != nil {
			return nil, fmt.Errorf("resolve storage path: %w", err)
		}
		fs, err := store.NewFileStore(absPath)
		if err != nil {
			return nil, err
		}
		log.Printf("Using file vector store at %s", absPath)
		return fs, nil
	default:
		return nil, fmt.Errorf("unknown storage backend %q (expected file|memory)", *storageBackend)
	}
}

func seedVectors(svc *service.SearchService) error {
	if *bootstrapVectors != "" {
		if err := loadBootstrapVectors(svc, *bootstrapVectors); err != nil {
			return err
		}
	}

	if *demoVectors > 0 {
		if !*allowUnsafeDemoData {
			return fmt.Errorf("refusing to generate demo vectors without --allow-unsafe-demo-data")
		}
		log.Printf("Generating %d demo vectors (unsafe demo mode enabled)", *demoVectors)
		generateDemoVectors(svc, *demoVectors, *dimension)
	}

	return nil
}

func loadBootstrapVectors(svc *service.SearchService, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read bootstrap file: %w", err)
	}

	var records []bootstrapVector
	if err := json.Unmarshal(data, &records); err != nil {
		return fmt.Errorf("parse bootstrap file: %w", err)
	}
	if len(records) == 0 {
		return nil
	}

	ids := make([]string, 0, len(records))
	vectors := make([][]float64, 0, len(records))
	metas := make([]map[string]any, 0, len(records))

	for i, rec := range records {
		if rec.ID == "" {
			return fmt.Errorf("bootstrap vector %d has empty id", i)
		}
		if len(rec.Values) != *dimension {
			return fmt.Errorf("bootstrap vector %q has dimension %d, expected %d", rec.ID, len(rec.Values), *dimension)
		}
		ids = append(ids, rec.ID)
		vectors = append(vectors, rec.Values)
		metas = append(metas, rec.Metadata)
	}

	if err := svc.AddVectors(context.Background(), ids, vectors, metas); err != nil {
		return fmt.Errorf("add bootstrap vectors: %w", err)
	}
	log.Printf("Loaded %d vectors from %s", len(records), path)
	return nil
}

// generateDemoVectors creates synthetic vectors for testing.
func generateDemoVectors(svc *service.SearchService, n, dim int) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	ids := make([]string, n)
	vectors := make([][]float64, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)

		vec := make([]float64, dim)
		var normSq float64
		for j := 0; j < dim; j++ {
			vec[j] = rng.NormFloat64()
			normSq += vec[j] * vec[j]
		}
		norm := math.Sqrt(normSq)
		if norm == 0 {
			norm = 1
		}
		for j := range vec {
			vec[j] /= norm
		}

		vectors[i] = vec
	}

	if err := svc.AddVectors(context.Background(), ids, vectors, nil); err != nil {
		log.Printf("Failed to add demo vectors: %v", err)
	}
}
