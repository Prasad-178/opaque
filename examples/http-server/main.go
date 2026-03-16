// Example http-server wraps Opaque in a lightweight HTTP API,
// demonstrating a realistic self-hosted deployment pattern.
//
//	POST /vectors       — add vectors (with optional metadata)
//	POST /search        — search with optional metadata filter
//	POST /admin/build   — trigger index build
//	POST /admin/save    — persist database to disk
//
// Run: go run ./examples/http-server/
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"

	"github.com/Prasad-178/opaque"
)

var db *opaque.DB

func main() {
	var err error
	db, err = opaque.NewDB(opaque.Config{
		Dimension:   128,
		NumClusters: 16,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /vectors", handleAddVectors)
	mux.HandleFunc("POST /search", handleSearch)
	mux.HandleFunc("POST /admin/build", handleBuild)
	mux.HandleFunc("POST /admin/save", handleSave)

	addr := ":8080"
	if p := os.Getenv("PORT"); p != "" {
		addr = ":" + p
	}

	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt)
		<-sigCh
		srv.Close()
	}()

	fmt.Printf("Opaque HTTP server listening on %s\n", addr)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

// --- request/response types ---

type addRequest struct {
	Vectors []vectorEntry `json:"vectors"`
}

type vectorEntry struct {
	ID       string            `json:"id"`
	Values   []float64         `json:"values"`
	Metadata map[string]any    `json:"metadata,omitempty"`
}

type searchRequest struct {
	Vector []float64      `json:"vector"`
	TopK   int            `json:"top_k"`
	Filter *opaque.Filter `json:"filter,omitempty"`
}

type saveRequest struct {
	Path string `json:"path"`
}

// --- handlers ---

func handleAddVectors(w http.ResponseWriter, r *http.Request) {
	var req addRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	for _, v := range req.Vectors {
		if v.Metadata != nil {
			if err := db.AddWithMetadata(ctx, v.ID, v.Values, v.Metadata); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			if err := db.Add(ctx, v.ID, v.Values); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}

	writeJSON(w, map[string]any{"added": len(req.Vectors)})
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	ctx := r.Context()
	var results []opaque.Result
	var err error

	if req.Filter != nil {
		results, err = db.SearchWithFilter(ctx, req.Vector, req.TopK, *req.Filter)
	} else {
		results, err = db.Search(ctx, req.Vector, req.TopK)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, results)
}

func handleBuild(w http.ResponseWriter, r *http.Request) {
	if err := db.Build(r.Context()); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]string{"status": "built"})
}

func handleSave(w http.ResponseWriter, r *http.Request) {
	var req saveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Path == "" {
		req.Path = "opaque-data"
	}
	if err := db.Save(req.Path); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]string{"status": "saved", "path": req.Path})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
