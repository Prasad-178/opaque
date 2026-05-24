package grpcserver

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io"
	"log"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func captureLogs(t *testing.T) (restore func() string) {
	t.Helper()
	var buf bytes.Buffer
	prev := log.Writer()
	log.SetOutput(&buf)
	return func() string {
		log.SetOutput(prev)
		return buf.String()
	}
}

func TestRecoveryUnaryInterceptor_Recovers(t *testing.T) {
	flush := captureLogs(t)
	defer flush()

	interceptor := RecoveryUnaryInterceptor()
	handler := func(ctx context.Context, req any) (any, error) {
		panic("kaboom")
	}
	info := &grpc.UnaryServerInfo{FullMethod: "/test/Method"}

	resp, err := interceptor(context.Background(), nil, info, handler)
	if resp != nil {
		t.Fatalf("resp=%v want nil", resp)
	}
	if err == nil {
		t.Fatal("expected error from panic recovery")
	}
	if c := status.Code(err); c != codes.Internal {
		t.Fatalf("code=%v want Internal", c)
	}
}

func TestRecoveryUnaryInterceptor_PassThrough(t *testing.T) {
	flush := captureLogs(t)
	defer flush()

	interceptor := RecoveryUnaryInterceptor()
	want := "ok"
	handler := func(ctx context.Context, req any) (any, error) {
		return want, nil
	}
	info := &grpc.UnaryServerInfo{FullMethod: "/test/OK"}
	resp, err := interceptor(context.Background(), nil, info, handler)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if resp.(string) != want {
		t.Fatalf("resp=%v want %v", resp, want)
	}
}

func TestRecoveryUnaryInterceptor_HandlerError(t *testing.T) {
	flush := captureLogs(t)
	defer flush()
	interceptor := RecoveryUnaryInterceptor()
	want := errors.New("boom")
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, want
	}
	_, err := interceptor(context.Background(), nil, &grpc.UnaryServerInfo{FullMethod: "/x"}, handler)
	if !errors.Is(err, want) {
		t.Fatalf("err=%v want %v", err, want)
	}
}

type fakeStream struct{ ctx context.Context }

func (f *fakeStream) SetHeader(_ any) error    { return nil }
func (f *fakeStream) SendHeader(_ any) error   { return nil }
func (f *fakeStream) SetTrailer(_ any)         {}
func (f *fakeStream) Context() context.Context { return f.ctx }
func (f *fakeStream) SendMsg(_ any) error      { return nil }
func (f *fakeStream) RecvMsg(_ any) error      { return io.EOF }

// satisfy grpc.ServerStream
func (f *fakeStream) SetHeaderMD(md any) error { return nil }

func TestRecoveryStreamInterceptor_Recovers(t *testing.T) {
	flush := captureLogs(t)
	defer flush()

	interceptor := RecoveryStreamInterceptor()
	handler := func(srv any, ss grpc.ServerStream) error {
		panic("kaboom-stream")
	}
	info := &grpc.StreamServerInfo{FullMethod: "/test/Stream"}
	err := interceptor(nil, nil, info, handler)
	if err == nil {
		t.Fatal("expected recovered error")
	}
	if c := status.Code(err); c != codes.Internal {
		t.Fatalf("code=%v want Internal", c)
	}
}

func TestRecoveryStreamInterceptor_PassThrough(t *testing.T) {
	flush := captureLogs(t)
	defer flush()
	interceptor := RecoveryStreamInterceptor()
	handler := func(srv any, ss grpc.ServerStream) error { return nil }
	if err := interceptor(nil, nil, &grpc.StreamServerInfo{FullMethod: "/x"}, handler); err != nil {
		t.Fatalf("err=%v", err)
	}
}

func TestLoggingUnaryInterceptor_LogsAndPassesThrough(t *testing.T) {
	flush := captureLogs(t)
	defer func() {
		out := flush()
		if !bytes.Contains([]byte(out), []byte("/test/Log")) {
			t.Errorf("log missing method name: %q", out)
		}
	}()

	interceptor := LoggingUnaryInterceptor()
	handler := func(ctx context.Context, req any) (any, error) {
		time.Sleep(time.Millisecond)
		return "ok", nil
	}
	resp, err := interceptor(context.Background(), nil, &grpc.UnaryServerInfo{FullMethod: "/test/Log"}, handler)
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if resp.(string) != "ok" {
		t.Fatalf("resp=%v", resp)
	}
}

func TestLoggingStreamInterceptor_LogsAndPassesThrough(t *testing.T) {
	flush := captureLogs(t)
	defer func() {
		out := flush()
		if !bytes.Contains([]byte(out), []byte("/test/LogStream")) {
			t.Errorf("log missing method name: %q", out)
		}
	}()

	interceptor := LoggingStreamInterceptor()
	handler := func(srv any, ss grpc.ServerStream) error { return nil }
	if err := interceptor(nil, nil, &grpc.StreamServerInfo{FullMethod: "/test/LogStream"}, handler); err != nil {
		t.Fatalf("err=%v", err)
	}
}

func writeTestCert(t *testing.T, dir string) (certPath, keyPath string) {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("genkey: %v", err)
	}
	tmpl := x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "test"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IsCA:                  true,
		BasicConstraintsValid: true,
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &priv.PublicKey, priv)
	if err != nil {
		t.Fatalf("cert: %v", err)
	}
	keyBytes, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		t.Fatalf("key marshal: %v", err)
	}
	certPath = filepath.Join(dir, "cert.pem")
	keyPath = filepath.Join(dir, "key.pem")
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o644); err != nil {
		t.Fatalf("write cert: %v", err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}), 0o600); err != nil {
		t.Fatalf("write key: %v", err)
	}
	return certPath, keyPath
}

func TestLoadTLSCredentials_OK(t *testing.T) {
	dir := t.TempDir()
	certPath, keyPath := writeTestCert(t, dir)

	creds, err := LoadTLSCredentials(certPath, keyPath)
	if err != nil {
		t.Fatalf("LoadTLSCredentials: %v", err)
	}
	if creds == nil {
		t.Fatal("creds nil")
	}
}

func TestLoadTLSCredentials_BadPath(t *testing.T) {
	if _, err := LoadTLSCredentials("/no/such/cert", "/no/such/key"); err == nil {
		t.Fatal("expected error")
	}
}

func TestLoadTLSCredentials_BadPair(t *testing.T) {
	dir := t.TempDir()
	cert := filepath.Join(dir, "cert.pem")
	key := filepath.Join(dir, "key.pem")
	if err := os.WriteFile(cert, []byte("not a cert"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.WriteFile(key, []byte("not a key"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadTLSCredentials(cert, key); err == nil {
		t.Fatal("expected error from malformed PEM")
	}
}
