package embeddings

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.BaseURL == "" {
		t.Fatal("DefaultConfig.BaseURL empty")
	}
	if cfg.Timeout <= 0 {
		t.Fatal("DefaultConfig.Timeout must be > 0")
	}
}

func newTestServer(t *testing.T, embed http.HandlerFunc, health http.HandlerFunc) *Client {
	t.Helper()
	mux := http.NewServeMux()
	if embed != nil {
		mux.HandleFunc("/embed", embed)
	}
	if health != nil {
		mux.HandleFunc("/health", health)
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return NewClient(Config{BaseURL: srv.URL, Timeout: 2 * time.Second})
}

func TestEmbed_HappyPath(t *testing.T) {
	wantTexts := []string{"hello", "world"}
	wantNormalize := true

	client := newTestServer(t,
		func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				t.Errorf("method=%s want POST", r.Method)
			}
			if got := r.Header.Get("Content-Type"); got != "application/json" {
				t.Errorf("content-type=%q want application/json", got)
			}
			body, _ := io.ReadAll(r.Body)
			var req EmbedRequest
			if err := json.Unmarshal(body, &req); err != nil {
				t.Errorf("bad request body: %v", err)
			}
			if len(req.Texts) != len(wantTexts) {
				t.Errorf("texts len=%d want=%d", len(req.Texts), len(wantTexts))
			}
			if req.Normalize != wantNormalize {
				t.Errorf("normalize=%v want=%v", req.Normalize, wantNormalize)
			}
			resp := EmbedResponse{
				Embeddings: [][]float64{{0.1, 0.2}, {0.3, 0.4}},
				Dimension:  2,
				Model:      "test-model",
				LatencyMs:  1.23,
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
		nil,
	)

	resp, err := client.Embed(wantTexts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Dimension != 2 || resp.Model != "test-model" {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if len(resp.Embeddings) != 2 {
		t.Fatalf("len=%d want 2", len(resp.Embeddings))
	}
}

func TestEmbedSingle(t *testing.T) {
	client := newTestServer(t,
		func(w http.ResponseWriter, r *http.Request) {
			resp := EmbedResponse{
				Embeddings: [][]float64{{0.5, 0.5}},
				Dimension:  2,
			}
			_ = json.NewEncoder(w).Encode(resp)
		},
		nil,
	)

	vec, err := client.EmbedSingle("hi")
	if err != nil {
		t.Fatalf("EmbedSingle: %v", err)
	}
	if len(vec) != 2 || vec[0] != 0.5 {
		t.Fatalf("unexpected vec: %v", vec)
	}
}

func TestEmbedSingle_EmptyResponse(t *testing.T) {
	client := newTestServer(t,
		func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(EmbedResponse{Embeddings: nil})
		},
		nil,
	)
	if _, err := client.EmbedSingle("x"); err == nil {
		t.Fatal("expected error on empty embeddings")
	} else if !strings.Contains(err.Error(), "no embeddings") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbed_HTTPError(t *testing.T) {
	client := newTestServer(t,
		func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		},
		nil,
	)
	if _, err := client.Embed([]string{"x"}); err == nil {
		t.Fatal("expected error on 500")
	}
}

func TestEmbed_TransportError(t *testing.T) {
	client := NewClient(Config{
		// Unrouted address; should yield a network error fast.
		BaseURL: "http://127.0.0.1:1",
		Timeout: 200 * time.Millisecond,
	})
	if _, err := client.Embed([]string{"x"}); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestHealth_OK(t *testing.T) {
	client := newTestServer(t, nil, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	if err := client.Health(); err != nil {
		t.Fatalf("Health: %v", err)
	}
}

func TestHealth_Unhealthy(t *testing.T) {
	client := newTestServer(t, nil, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	})
	if err := client.Health(); err == nil {
		t.Fatal("expected unhealthy")
	}
}

func TestHealth_TransportError(t *testing.T) {
	client := NewClient(Config{BaseURL: "http://127.0.0.1:1", Timeout: 200 * time.Millisecond})
	if err := client.Health(); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestEmbedWithOptions_NormalizeFalse(t *testing.T) {
	var got EmbedRequest
	client := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_ = json.NewEncoder(w).Encode(EmbedResponse{
			Embeddings: [][]float64{{0}},
			Dimension:  1,
		})
	}, nil)

	if _, err := client.EmbedWithOptions([]string{"x"}, false); err != nil {
		t.Fatalf("EmbedWithOptions: %v", err)
	}
	if got.Normalize {
		t.Fatal("Normalize=true sent; want false")
	}
}

func TestEmbed_MalformedResponse(t *testing.T) {
	client := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}, nil)
	if _, err := client.Embed([]string{"x"}); err == nil {
		t.Fatal("expected decode error")
	}
}
