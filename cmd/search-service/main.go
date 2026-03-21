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
	"strconv"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	pb "github.com/Prasad-178/opaque/api/proto"
	"github.com/Prasad-178/opaque/pkg/grpcserver"

	opaque "github.com/Prasad-178/opaque"
)

var (
	grpcPort            = flag.Int("grpc-port", envInt("OPAQUE_GRPC_PORT", 50051), "gRPC server port")
	httpPort            = flag.Int("http-port", envInt("OPAQUE_HTTP_PORT", 8080), "HTTP metrics/health port")
	dimension           = flag.Int("dimension", envInt("OPAQUE_DIMENSION", 128), "Vector dimension")
	numClusters         = flag.Int("num-clusters", envInt("OPAQUE_NUM_CLUSTERS", 64), "Number of k-means clusters")
	storageBackend      = flag.String("storage-backend", envString("OPAQUE_STORAGE_BACKEND", "memory"), "Vector storage backend: file|memory")
	storagePath         = flag.String("storage-path", envString("OPAQUE_STORAGE_PATH", ""), "Path for file storage backend")
	bootstrapVectors    = flag.String("bootstrap-vectors", envString("OPAQUE_BOOTSTRAP_VECTORS", ""), "Optional JSON file containing vectors to preload")
	demoVectors         = flag.Int("demo-vectors", envInt("OPAQUE_DEMO_VECTORS", 0), "Number of generated demo vectors to seed (disabled by default)")
	allowUnsafeDemoData = flag.Bool("allow-unsafe-demo-data", envBool("OPAQUE_ALLOW_UNSAFE_DEMO_DATA", false), "Allow startup with generated demo vectors")
	tlsCert             = flag.String("tls-cert", envString("OPAQUE_TLS_CERT", ""), "TLS certificate file (optional)")
	tlsKey              = flag.String("tls-key", envString("OPAQUE_TLS_KEY", ""), "TLS key file (optional)")
)

type bootstrapVector struct {
	ID       string         `json:"id"`
	Values   []float64      `json:"values"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func main() {
	flag.Parse()

	log.Println("Starting Opaque Search Service...")

	// Configure the opaque DB.
	cfg := opaque.Config{
		Dimension:   *dimension,
		NumClusters: *numClusters,
	}

	switch strings.ToLower(*storageBackend) {
	case "file":
		if *storagePath == "" {
			log.Fatal("storage-path is required when storage-backend=file")
		}
		cfg.Storage = opaque.File
		cfg.StoragePath = *storagePath
		log.Printf("Using file storage at %s", *storagePath)
	case "memory":
		cfg.Storage = opaque.Memory
		log.Println("Using in-memory storage (non-durable)")
	default:
		log.Fatalf("Unknown storage backend %q (expected file|memory)", *storageBackend)
	}

	db, err := opaque.NewDB(cfg)
	if err != nil {
		log.Fatalf("Failed to create DB: %v", err)
	}
	defer db.Close()

	if err := seedVectors(db); err != nil {
		log.Fatalf("Failed to seed vectors: %v", err)
	}

	// Build the index if vectors were seeded.
	if db.Size() > 0 {
		log.Printf("Building index for %d vectors...", db.Size())
		if err := db.Build(context.Background()); err != nil {
			log.Fatalf("Failed to build index: %v", err)
		}
		log.Println("Index built successfully")
	}

	// Start gRPC server.
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

	// Only enable gRPC reflection in development (aids reconnaissance in production).
	if envBool("OPAQUE_GRPC_REFLECTION", false) {
		reflection.Register(grpcServer)
		log.Println("gRPC reflection enabled (development mode)")
	}
	pb.RegisterOpaqueSearchServer(grpcServer, grpcserver.New(db))

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

	// Start HTTP health/readiness server.
	httpMux := http.NewServeMux()
	httpMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if db.IsReady() {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "OK\nVectors: %d\n", db.Size())
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, "NOT READY\nVectors: %d\n", db.Size())
		}
	})

	httpMux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if db.IsReady() {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "Ready\n")
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, "Not Ready\n")
		}
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

func seedVectors(db *opaque.DB) error {
	if *bootstrapVectors != "" {
		if err := loadBootstrapVectors(db, *bootstrapVectors); err != nil {
			return err
		}
	}

	if *demoVectors > 0 {
		if !*allowUnsafeDemoData {
			return fmt.Errorf("refusing to generate demo vectors without --allow-unsafe-demo-data")
		}
		log.Printf("Generating %d demo vectors (unsafe demo mode enabled)", *demoVectors)
		generateDemoVectors(db, *demoVectors, *dimension)
	}

	return nil
}

func loadBootstrapVectors(db *opaque.DB, path string) error {
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

	for i, rec := range records {
		if rec.ID == "" {
			return fmt.Errorf("bootstrap vector %d has empty id", i)
		}
		if len(rec.Values) != *dimension {
			return fmt.Errorf("bootstrap vector %q has dimension %d, expected %d", rec.ID, len(rec.Values), *dimension)
		}
		ids = append(ids, rec.ID)
		vectors = append(vectors, rec.Values)
	}

	ctx := context.Background()
	if err := db.AddBatch(ctx, ids, vectors); err != nil {
		return fmt.Errorf("add bootstrap vectors: %w", err)
	}
	log.Printf("Loaded %d vectors from %s", len(records), path)
	return nil
}

// generateDemoVectors creates synthetic vectors for testing.
func generateDemoVectors(db *opaque.DB, n, dim int) {
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

	if err := db.AddBatch(context.Background(), ids, vectors); err != nil {
		log.Printf("Failed to add demo vectors: %v", err)
	}
}

func envString(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}

func envInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(v)
	if err != nil {
		log.Printf("Invalid %s=%q, using default %d", key, v, fallback)
		return fallback
	}
	return parsed
}

func envBool(key string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(v)
	if err != nil {
		log.Printf("Invalid %s=%q, using default %t", key, v, fallback)
		return fallback
	}
	return parsed
}
