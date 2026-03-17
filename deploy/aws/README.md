# AWS Deployment (SDK Search Service)

`cmd/search-service` now runs the `opaque.DB` SDK internally, including autonomous background index lifecycle management.

## Required env vars

- `OPAQUE_DB_PATH=/var/lib/opaque/db`
- `OPAQUE_DIMENSION=128`

## Recommended env vars

- `OPAQUE_NUM_CLUSTERS=64`
- `OPAQUE_AUTO_INDEX_ENABLED=true`
- `OPAQUE_AUTO_INDEX_INTERVAL=5s`
- `OPAQUE_AUTO_INDEX_MIN_CHANGES=1`
- `OPAQUE_AUTO_INDEX_TIMEOUT=15m`

## Optional env vars

- `OPAQUE_HTTP_PORT` (default `8080`)
- `OPAQUE_TOP_CLUSTERS` (default `0`, SDK defaulting)
- `OPAQUE_NUM_DECOYS` (default `8`)
- `OPAQUE_WORKER_POOL_SIZE` (default `0`)
- `OPAQUE_BOOTSTRAP_VECTORS` (JSON path)

## API endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /v1/vectors/batch`
- `PUT /v1/vectors/{id}`
- `DELETE /v1/vectors/{id}`
- `POST /v1/search`
- `GET /v1/stats`
- `POST /v1/admin/build`
- `POST /v1/admin/save`

## Storage recommendations

- Use EFS or an attached persistent volume for `/var/lib/opaque`.
- Keep `OPAQUE_DB_PATH` on persistent storage so snapshots survive restarts.
