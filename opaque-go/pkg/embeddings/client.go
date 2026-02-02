// Package embeddings provides a client for the local embedding service.
package embeddings

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Client is a client for the embedding service.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// EmbedRequest is the request to the embedding service.
type EmbedRequest struct {
	Texts     []string `json:"texts"`
	Normalize bool     `json:"normalize"`
}

// EmbedResponse is the response from the embedding service.
type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	Dimension  int         `json:"dimension"`
	Model      string      `json:"model"`
	LatencyMs  float64     `json:"latency_ms"`
}

// Config holds the client configuration.
type Config struct {
	BaseURL string
	Timeout time.Duration
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		BaseURL: "http://localhost:8090",
		Timeout: 30 * time.Second,
	}
}

// NewClient creates a new embedding client.
func NewClient(cfg Config) *Client {
	return &Client{
		baseURL: cfg.BaseURL,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// Embed generates embeddings for the given texts.
func (c *Client) Embed(texts []string) (*EmbedResponse, error) {
	return c.EmbedWithOptions(texts, true)
}

// EmbedWithOptions generates embeddings with custom options.
func (c *Client) EmbedWithOptions(texts []string, normalize bool) (*EmbedResponse, error) {
	req := EmbedRequest{
		Texts:     texts,
		Normalize: normalize,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/embed",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call embedding service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned status %d", resp.StatusCode)
	}

	var result EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// EmbedSingle generates an embedding for a single text.
func (c *Client) EmbedSingle(text string) ([]float64, error) {
	resp, err := c.Embed([]string{text})
	if err != nil {
		return nil, err
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return resp.Embeddings[0], nil
}

// Health checks if the embedding service is healthy.
func (c *Client) Health() error {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("service unhealthy: status %d", resp.StatusCode)
	}

	return nil
}
