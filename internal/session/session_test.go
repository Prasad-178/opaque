package session

import (
	"errors"
	"testing"
	"time"
)

func TestManager_CreateAndGet(t *testing.T) {
	m := NewManager(time.Minute)

	s, err := m.Create([]byte("pk"), time.Second)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if s.ID == "" {
		t.Fatal("expected non-empty session ID")
	}

	got, err := m.Get(s.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got.PublicKey) != "pk" {
		t.Fatalf("public key = %q, want pk", string(got.PublicKey))
	}
}

func TestManager_ExpiredSession(t *testing.T) {
	m := NewManager(100 * time.Millisecond)

	s, err := m.Create([]byte("pk"), 50*time.Millisecond)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	time.Sleep(80 * time.Millisecond)
	_, err = m.Get(s.ID)
	if !errors.Is(err, ErrSessionExpired) {
		t.Fatalf("expected ErrSessionExpired, got %v", err)
	}
}

func TestManager_Refresh(t *testing.T) {
	m := NewManager(time.Second)

	s, err := m.Create([]byte("pk"), 200*time.Millisecond)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	if err := m.Refresh(s.ID, time.Second); err != nil {
		t.Fatalf("Refresh: %v", err)
	}

	if _, err := m.Get(s.ID); err != nil {
		t.Fatalf("Get after refresh: %v", err)
	}
}
