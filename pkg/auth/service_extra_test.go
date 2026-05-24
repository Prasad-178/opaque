package auth

import (
	"context"
	"testing"
)

func TestEnableDisableUser_Roundtrip(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)
	pw := []byte("pass-xyz")

	if err := service.RegisterUser(ctx, "u", "test-enterprise", pw, []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser: %v", err)
	}

	// Sanity: starts enabled.
	if _, err := service.Authenticate(ctx, "u", pw); err != nil {
		t.Fatalf("Authenticate before disable: %v", err)
	}

	if err := service.DisableUser(ctx, "u"); err != nil {
		t.Fatalf("DisableUser: %v", err)
	}
	if _, err := service.Authenticate(ctx, "u", pw); err == nil {
		t.Fatal("Authenticate succeeded after disable")
	}

	if err := service.EnableUser(ctx, "u"); err != nil {
		t.Fatalf("EnableUser: %v", err)
	}
	if _, err := service.Authenticate(ctx, "u", pw); err != nil {
		t.Fatalf("Authenticate after re-enable: %v", err)
	}
}

func TestEnableUser_NotFound(t *testing.T) {
	service, _ := setupTestService(t)
	if err := service.EnableUser(context.Background(), "missing"); err == nil {
		t.Fatal("EnableUser missing: expected error")
	}
}

func TestDisableUser_NotFound(t *testing.T) {
	service, _ := setupTestService(t)
	if err := service.DisableUser(context.Background(), "missing"); err == nil {
		t.Fatal("DisableUser missing: expected error")
	}
}

func TestGetUser_RoundtripAndMissing(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)
	if err := service.RegisterUser(ctx, "u", "test-enterprise", []byte("p"), []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser: %v", err)
	}
	u, err := service.GetUser(ctx, "u")
	if err != nil {
		t.Fatalf("GetUser: %v", err)
	}
	if u.UserID != "u" || u.EnterpriseID != "test-enterprise" {
		t.Fatalf("unexpected user: %+v", u)
	}
	if _, err := service.GetUser(ctx, "missing"); err == nil {
		t.Fatal("GetUser missing: expected error")
	}
}

func TestListUsers_FiltersByEnterprise(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)
	if err := service.RegisterUser(ctx, "u1", "test-enterprise", []byte("p"), []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser u1: %v", err)
	}
	if err := service.RegisterUser(ctx, "u2", "test-enterprise", []byte("p"), []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser u2: %v", err)
	}
	users, err := service.ListUsers(ctx, "test-enterprise")
	if err != nil {
		t.Fatalf("ListUsers: %v", err)
	}
	if len(users) != 2 {
		t.Fatalf("len=%d want 2", len(users))
	}
	for _, u := range users {
		if u.EnterpriseID != "test-enterprise" {
			t.Fatalf("EnterpriseID=%s want test-enterprise", u.EnterpriseID)
		}
	}

	// ListUsers for an unknown enterprise should not include any users from
	// other enterprises.
	other, err := service.ListUsers(ctx, "no-such-enterprise")
	if err != nil {
		t.Fatalf("ListUsers other: %v", err)
	}
	if len(other) != 0 {
		t.Fatalf("expected 0 users for unknown enterprise; got %d", len(other))
	}
}

func TestRevokeAllUserTokens(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)
	pw := []byte("p")
	if err := service.RegisterUser(ctx, "u", "test-enterprise", pw, []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser: %v", err)
	}

	t1, err := service.Authenticate(ctx, "u", pw)
	if err != nil {
		t.Fatalf("Authenticate 1: %v", err)
	}
	t2, err := service.Authenticate(ctx, "u", pw)
	if err != nil {
		t.Fatalf("Authenticate 2: %v", err)
	}

	before := service.ActiveTokenCount()
	if before < 2 {
		t.Fatalf("ActiveTokenCount=%d want >= 2", before)
	}

	if err := service.RevokeAllUserTokens(ctx, "u"); err != nil {
		t.Fatalf("RevokeAllUserTokens: %v", err)
	}

	if _, err := service.ValidateToken(ctx, t1.Token); err == nil {
		t.Fatal("ValidateToken(t1): expected error after revoke")
	}
	if _, err := service.ValidateToken(ctx, t2.Token); err == nil {
		t.Fatal("ValidateToken(t2): expected error after revoke")
	}
}

func TestActiveTokenCount_GrowsAndShrinks(t *testing.T) {
	ctx := context.Background()
	service, _ := setupTestService(t)
	pw := []byte("p")
	if err := service.RegisterUser(ctx, "u", "test-enterprise", pw, []string{ScopeSearch}); err != nil {
		t.Fatalf("RegisterUser: %v", err)
	}
	start := service.ActiveTokenCount()
	t1, err := service.Authenticate(ctx, "u", pw)
	if err != nil {
		t.Fatalf("Authenticate: %v", err)
	}
	if service.ActiveTokenCount() != start+1 {
		t.Fatalf("ActiveTokenCount after auth=%d want %d", service.ActiveTokenCount(), start+1)
	}
	if err := service.RevokeToken(ctx, t1.Token); err != nil {
		t.Fatalf("RevokeToken: %v", err)
	}
	if service.ActiveTokenCount() != start {
		t.Fatalf("ActiveTokenCount after revoke=%d want %d", service.ActiveTokenCount(), start)
	}
}
