package enterprise

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestNewConfig(t *testing.T) {
	cfg, err := NewConfig("test-enterprise", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	// Check that secrets were generated
	if len(cfg.AESKey) != 32 {
		t.Errorf("AES key should be 32 bytes, got %d", len(cfg.AESKey))
	}
	if len(cfg.LSHSeed) != 32 {
		t.Errorf("LSH seed should be 32 bytes, got %d", len(cfg.LSHSeed))
	}

	// Check that AES key is random (not all zeros)
	allZeros := true
	for _, b := range cfg.AESKey {
		if b != 0 {
			allZeros = false
			break
		}
	}
	if allZeros {
		t.Error("AES key should not be all zeros")
	}

	// Check LSH seed is random
	allZeros = true
	for _, b := range cfg.LSHSeed {
		if b != 0 {
			allZeros = false
			break
		}
	}
	if allZeros {
		t.Error("LSH seed should not be all zeros")
	}

	// Check validation passes
	if err := cfg.Validate(); err != nil {
		t.Errorf("validation should pass: %v", err)
	}
}

func TestConfigSeedConversion(t *testing.T) {
	cfg, err := NewConfig("test", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	seed := cfg.GetLSHSeedAsInt64()
	subSeed := cfg.GetSubLSHSeedAsInt64()

	// Seeds should be different
	if seed == subSeed {
		t.Error("super and sub seeds should be different")
	}

	// Seed should be deterministic
	seed2 := cfg.GetLSHSeedAsInt64()
	if seed != seed2 {
		t.Error("seed should be deterministic")
	}
}

func TestConfigClone(t *testing.T) {
	cfg, err := NewConfig("test", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	clone := cfg.Clone()

	// Check that clone is equal
	if clone.EnterpriseID != cfg.EnterpriseID {
		t.Error("clone should have same enterprise ID")
	}
	if !bytes.Equal(clone.AESKey, cfg.AESKey) {
		t.Error("clone should have same AES key")
	}
	if !bytes.Equal(clone.LSHSeed, cfg.LSHSeed) {
		t.Error("clone should have same LSH seed")
	}

	// Modify clone, original should not change
	clone.AESKey[0] = 0xFF
	if cfg.AESKey[0] == 0xFF {
		t.Error("modifying clone should not affect original")
	}
}

func TestConfigSerialization(t *testing.T) {
	cfg, err := NewConfig("test-enterprise", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	// Serialize
	var buf bytes.Buffer
	if err := cfg.Save(&buf); err != nil {
		t.Fatalf("failed to save config: %v", err)
	}

	// Deserialize
	loaded, err := LoadConfig(&buf)
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	// Compare
	if loaded.EnterpriseID != cfg.EnterpriseID {
		t.Error("enterprise ID mismatch")
	}
	if !bytes.Equal(loaded.AESKey, cfg.AESKey) {
		t.Error("AES key mismatch")
	}
	if !bytes.Equal(loaded.LSHSeed, cfg.LSHSeed) {
		t.Error("LSH seed mismatch")
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*Config)
		wantErr bool
	}{
		{
			name:    "valid config",
			modify:  func(c *Config) {},
			wantErr: false,
		},
		{
			name:    "empty enterprise ID",
			modify:  func(c *Config) { c.EnterpriseID = "" },
			wantErr: true,
		},
		{
			name:    "short AES key",
			modify:  func(c *Config) { c.AESKey = make([]byte, 16) },
			wantErr: true,
		},
		{
			name:    "short LSH seed",
			modify:  func(c *Config) { c.LSHSeed = make([]byte, 4) },
			wantErr: true,
		},
		{
			name:    "zero dimension",
			modify:  func(c *Config) { c.Dimension = 0 },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, _ := NewConfig("test", 128, 64)
			tt.modify(cfg)
			err := cfg.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMemoryStore(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	defer store.Close()

	// Create config
	cfg, err := NewConfig("test-enterprise", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	// Test Put
	if err := store.Put(ctx, cfg); err != nil {
		t.Fatalf("failed to put config: %v", err)
	}

	// Test Exists
	if !store.Exists(ctx, "test-enterprise") {
		t.Error("enterprise should exist")
	}
	if store.Exists(ctx, "non-existent") {
		t.Error("non-existent enterprise should not exist")
	}

	// Test Get
	loaded, err := store.Get(ctx, "test-enterprise")
	if err != nil {
		t.Fatalf("failed to get config: %v", err)
	}
	if loaded.EnterpriseID != cfg.EnterpriseID {
		t.Error("enterprise ID mismatch")
	}

	// Test Get returns clone (mutation safety)
	loaded.AESKey[0] = 0xFF
	loaded2, _ := store.Get(ctx, "test-enterprise")
	if loaded2.AESKey[0] == 0xFF {
		t.Error("store should return clones")
	}

	// Test List
	ids, err := store.List(ctx)
	if err != nil {
		t.Fatalf("failed to list: %v", err)
	}
	if len(ids) != 1 || ids[0] != "test-enterprise" {
		t.Errorf("unexpected list: %v", ids)
	}

	// Test Delete
	if err := store.Delete(ctx, "test-enterprise"); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}
	if store.Exists(ctx, "test-enterprise") {
		t.Error("enterprise should not exist after delete")
	}

	// Test Get non-existent
	_, err = store.Get(ctx, "non-existent")
	if err != ErrEnterpriseNotFound {
		t.Errorf("expected ErrEnterpriseNotFound, got %v", err)
	}
}

func TestFileStore(t *testing.T) {
	ctx := context.Background()

	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "enterprise-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	store, err := NewFileStore(tmpDir)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	defer store.Close()

	// Create config
	cfg, err := NewConfig("test-enterprise", 128, 64)
	if err != nil {
		t.Fatalf("failed to create config: %v", err)
	}

	// Test Put
	if err := store.Put(ctx, cfg); err != nil {
		t.Fatalf("failed to put config: %v", err)
	}

	// Check file was created with correct permissions
	path := filepath.Join(tmpDir, "test-enterprise.gob")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("config file should exist: %v", err)
	}
	if info.Mode().Perm() != 0600 {
		t.Errorf("config file should have 0600 permissions, got %o", info.Mode().Perm())
	}

	// Test Get
	loaded, err := store.Get(ctx, "test-enterprise")
	if err != nil {
		t.Fatalf("failed to get config: %v", err)
	}
	if loaded.EnterpriseID != cfg.EnterpriseID {
		t.Error("enterprise ID mismatch")
	}
	if !bytes.Equal(loaded.AESKey, cfg.AESKey) {
		t.Error("AES key mismatch")
	}

	// Test List
	ids, err := store.List(ctx)
	if err != nil {
		t.Fatalf("failed to list: %v", err)
	}
	if len(ids) != 1 || ids[0] != "test-enterprise" {
		t.Errorf("unexpected list: %v", ids)
	}

	// Test Delete
	if err := store.Delete(ctx, "test-enterprise"); err != nil {
		t.Fatalf("failed to delete: %v", err)
	}
	if store.Exists(ctx, "test-enterprise") {
		t.Error("enterprise should not exist after delete")
	}
}
