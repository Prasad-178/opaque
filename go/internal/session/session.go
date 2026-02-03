// Package session provides session management for client keys.
package session

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

var (
	ErrSessionNotFound = errors.New("session not found")
	ErrSessionExpired  = errors.New("session expired")
)

// Session holds client session data
type Session struct {
	ID           string
	PublicKey    []byte
	CreatedAt    time.Time
	ExpiresAt    time.Time
	LastAccessAt time.Time
}

// Manager manages client sessions in memory.
// For production, replace with Redis-backed implementation.
type Manager struct {
	sessions map[string]*Session
	maxTTL   time.Duration
	mu       sync.RWMutex
}

// NewManager creates a new session manager.
func NewManager(maxTTL time.Duration) *Manager {
	m := &Manager{
		sessions: make(map[string]*Session),
		maxTTL:   maxTTL,
	}

	// Start cleanup goroutine
	go m.cleanupLoop()

	return m
}

// Create creates a new session with the given public key.
func (m *Manager) Create(publicKey []byte, requestedTTL time.Duration) (*Session, error) {
	// Generate session ID
	id, err := generateSessionID()
	if err != nil {
		return nil, err
	}

	// Cap TTL to max
	ttl := requestedTTL
	if ttl > m.maxTTL || ttl <= 0 {
		ttl = m.maxTTL
	}

	now := time.Now()
	session := &Session{
		ID:           id,
		PublicKey:    publicKey,
		CreatedAt:    now,
		ExpiresAt:    now.Add(ttl),
		LastAccessAt: now,
	}

	m.mu.Lock()
	m.sessions[id] = session
	m.mu.Unlock()

	return session, nil
}

// Get retrieves a session by ID.
func (m *Manager) Get(id string) (*Session, error) {
	m.mu.RLock()
	session, ok := m.sessions[id]
	m.mu.RUnlock()

	if !ok {
		return nil, ErrSessionNotFound
	}

	if time.Now().After(session.ExpiresAt) {
		m.Delete(id)
		return nil, ErrSessionExpired
	}

	// Update last access time
	m.mu.Lock()
	session.LastAccessAt = time.Now()
	m.mu.Unlock()

	return session, nil
}

// GetPublicKey retrieves only the public key for a session.
func (m *Manager) GetPublicKey(id string) ([]byte, error) {
	session, err := m.Get(id)
	if err != nil {
		return nil, err
	}
	return session.PublicKey, nil
}

// Delete removes a session.
func (m *Manager) Delete(id string) {
	m.mu.Lock()
	delete(m.sessions, id)
	m.mu.Unlock()
}

// Count returns the number of active sessions.
func (m *Manager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.sessions)
}

// Refresh extends a session's TTL.
func (m *Manager) Refresh(id string, ttl time.Duration) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	session, ok := m.sessions[id]
	if !ok {
		return ErrSessionNotFound
	}

	if ttl > m.maxTTL {
		ttl = m.maxTTL
	}

	session.ExpiresAt = time.Now().Add(ttl)
	session.LastAccessAt = time.Now()

	return nil
}

// cleanupLoop periodically removes expired sessions.
func (m *Manager) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		m.cleanup()
	}
}

// cleanup removes expired sessions.
func (m *Manager) cleanup() {
	now := time.Now()

	m.mu.Lock()
	defer m.mu.Unlock()

	for id, session := range m.sessions {
		if now.After(session.ExpiresAt) {
			delete(m.sessions, id)
		}
	}
}

// generateSessionID generates a cryptographically secure session ID.
func generateSessionID() (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes), nil
}
