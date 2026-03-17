# AWS Deployment (Search Service)

This service is now configured via environment variables, which maps cleanly to ECS task definitions.

## Required env vars

- `OPAQUE_STORAGE_BACKEND=file`
- `OPAQUE_STORAGE_PATH=/var/lib/opaque/vectors.json`
- `OPAQUE_DIMENSION=128` (must match your vectors)

## Optional env vars

- `OPAQUE_GRPC_PORT` (default `50051`)
- `OPAQUE_HTTP_PORT` (default `8080`)
- `OPAQUE_LSH_BITS` (default `128`)
- `OPAQUE_LSH_SEED` (default `42`)
- `OPAQUE_BOOTSTRAP_VECTORS` (JSON file path mounted into container)
- `OPAQUE_DEMO_VECTORS` (default `0`; keep disabled in production)
- `OPAQUE_ALLOW_UNSAFE_DEMO_DATA` (default `false`)
- `OPAQUE_TLS_CERT` / `OPAQUE_TLS_KEY`

## Health checks

- Liveness: `GET /healthz`
- Readiness: `GET /readyz`

## Storage recommendations

- Use EFS or an attached persistent volume for `/var/lib/opaque`.
- Avoid memory backend in production.
