package opaque

import "time"

// ConfigureAutoIndex updates background index lifecycle behavior at runtime.
//
// This is useful for loaded databases, where persisted config may differ from
// deployment-time requirements.
func (db *DB) ConfigureAutoIndex(enabled bool, interval time.Duration, minChanges int, timeout time.Duration, onErr func(error)) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	if minChanges <= 0 {
		minChanges = 1
	}
	if timeout <= 0 {
		timeout = 15 * time.Minute
	}

	db.mu.Lock()
	db.cfg.AutoIndexEnabled = enabled
	db.cfg.AutoIndexInterval = interval
	db.cfg.AutoIndexMinChanges = minChanges
	db.cfg.AutoIndexBuildTimeout = timeout
	db.cfg.OnAutoIndexError = onErr

	lifecycle := db.lifecycle
	shouldStart := enabled && lifecycle == nil
	shouldStop := !enabled && lifecycle != nil
	if shouldStart {
		db.startLifecycleManager()
	}
	if shouldStop {
		db.lifecycle = nil
	}
	db.mu.Unlock()

	if shouldStop {
		lifecycle.Stop()
	}
}
