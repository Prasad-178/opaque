package opaque

import (
	"context"
	"errors"
	"time"
)

// lifecycleManager runs background Build/Rebuild based on data mutations.
type lifecycleManager struct {
	db         *DB
	interval   time.Duration
	minChanges uint64
	timeout    time.Duration
	stopCh     chan struct{}
	doneCh     chan struct{}
}

func newLifecycleManager(db *DB) *lifecycleManager {
	return &lifecycleManager{
		db:         db,
		interval:   db.cfg.AutoIndexInterval,
		minChanges: uint64(db.cfg.AutoIndexMinChanges),
		timeout:    db.cfg.AutoIndexBuildTimeout,
		stopCh:     make(chan struct{}),
		doneCh:     make(chan struct{}),
	}
}

func (m *lifecycleManager) Start() {
	go m.loop()
}

func (m *lifecycleManager) Stop() {
	close(m.stopCh)
	<-m.doneCh
}

func (m *lifecycleManager) loop() {
	defer close(m.doneCh)

	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.reconcile()
		case <-m.stopCh:
			return
		}
	}
}

func (m *lifecycleManager) reconcile() {
	m.db.mu.RLock()
	changes := m.db.dataVersion - m.db.builtVersion
	if changes < m.minChanges {
		m.db.mu.RUnlock()
		return
	}
	state := m.db.state
	m.db.mu.RUnlock()

	ctx, cancel := context.WithTimeout(context.Background(), m.timeout)
	defer cancel()

	var err error
	switch state {
	case stateBuffered:
		err = m.db.Build(ctx)
	case stateReady:
		err = m.db.Rebuild(ctx)
	default:
		return
	}

	if err != nil {
		if errors.Is(err, ErrNoVectors) {
			return
		}
		if cb := m.db.cfg.OnAutoIndexError; cb != nil {
			cb(err)
		}
	}
}

func (db *DB) startLifecycleManager() {
	if db.lifecycle != nil {
		return
	}
	m := newLifecycleManager(db)
	db.lifecycle = m
	m.Start()
}
