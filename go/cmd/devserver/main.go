// Development server for local testing of the privacy-preserving vector search.
//
// Usage:
//
//	go run ./cmd/devserver
//
// This starts a local server on :8080 with:
//   - In-memory blob storage
//   - File-based enterprise config
//   - Test user authentication
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/opaque/opaque/go/pkg/auth"
	"github.com/opaque/opaque/go/pkg/blob"
	"github.com/opaque/opaque/go/pkg/enterprise"
	"github.com/opaque/opaque/go/pkg/server"
)

func main() {
	addr := flag.String("addr", ":8080", "Server address")
	dataDir := flag.String("data", "", "Data directory for file-based storage (default: in-memory)")
	flag.Parse()

	log.Println("Starting development server...")

	// Setup storage
	var blobStore blob.Store
	if *dataDir != "" {
		var err error
		blobStore, err = blob.NewFileStore(*dataDir)
		if err != nil {
			log.Fatalf("Failed to create file store: %v", err)
		}
		log.Printf("Using file-based storage at: %s", *dataDir)
	} else {
		blobStore = blob.NewMemoryStore()
		log.Println("Using in-memory storage")
	}

	// Setup enterprise store
	enterpriseStore := enterprise.NewMemoryStore()

	// Setup auth service
	authCfg := auth.DefaultServiceConfig()
	authCfg.TokenTTL = 24 * time.Hour // Long tokens for development
	authService := auth.NewService(authCfg, enterpriseStore)

	// Create a test enterprise and user
	ctx := context.Background()
	testEnterpriseID := "test-enterprise"
	testDimension := 128
	testNumClusters := 32

	// Create test enterprise config
	enterpriseCfg, err := enterprise.NewConfig(testEnterpriseID, testDimension, testNumClusters)
	if err != nil {
		log.Fatalf("Failed to create enterprise config: %v", err)
	}
	if err := enterpriseStore.Put(ctx, enterpriseCfg); err != nil {
		log.Fatalf("Failed to store enterprise config: %v", err)
	}
	log.Printf("Created test enterprise: %s (dim=%d, clusters=%d)", testEnterpriseID, testDimension, testNumClusters)

	// Register test user
	if err := authService.RegisterUser(ctx, "testuser", testEnterpriseID, []byte("testpass"), []string{auth.ScopeSearch}); err != nil {
		log.Fatalf("Failed to register test user: %v", err)
	}
	log.Println("Created test user: testuser (password: testpass)")

	// Create and start server
	serverCfg := server.DefaultConfig()
	serverCfg.Address = *addr

	srv := server.New(serverCfg, blobStore, authService, enterpriseStore)

	// Handle graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("Shutdown error: %v", err)
		}
	}()

	log.Printf("Server listening on %s", *addr)
	log.Println("")
	log.Println("Test endpoints:")
	log.Println("  GET  /health                          - Health check")
	log.Println("  POST /api/v1/auth/login               - Login")
	log.Println("  POST /api/v1/auth/refresh             - Refresh token")
	log.Println("  POST /api/v1/score-centroids          - HE scoring")
	log.Println("  GET  /api/v1/buckets/{eid}/{superID}  - Get blobs")
	log.Println("")
	log.Println("Test credentials:")
	log.Printf("  Enterprise: %s", testEnterpriseID)
	log.Println("  User:       testuser")
	log.Println("  Password:   testpass")
	log.Println("")

	if err := srv.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
