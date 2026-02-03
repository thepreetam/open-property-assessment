# API Specification (OpenAPI)

The API is documented via FastAPI’s OpenAPI schema at **`/docs`** (Swagger UI) and **`/openapi.json`**.

## Base URL

- Local: `http://localhost:8000`
- Set `API_BASE_URL` for webhooks and status URLs.

## Authentication

- **API key** (optional): Set `API_KEY` in the environment; then send **`X-API-Key: <key>`** on protected routes (properties, strategies, execute, workspaces, analytics, contractor, export).
- **Workspace** (Phase 3): Send **`X-Workspace-Id`** and **`X-User-Id`** where RBAC is enforced.

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/v1/health | Liveness |
| GET | /api/v1/ready | Readiness |
| POST | /api/v1/jobs | Create analysis job (multipart: files, home_value, zip_code, property_id) |
| GET | /api/v1/jobs/{job_id} | Job status and result |
| WS | /api/v1/jobs/{job_id}/stream | WebSocket: job progress (status pushed every ~1.5 s until completed/failed) |
| POST | /api/v1/jobs/bulk | Bulk job creation (JSON: jobs with address, zip_code, home_value, photo_urls) |
| GET | /metrics | Prometheus metrics (jobs started/completed/failed, pipeline duration) |
| POST | /api/v1/properties | Create property |
| GET | /api/v1/properties | List properties (optional workspace_id) |
| GET | /api/v1/properties/{id} | Get property |
| GET | /api/v1/strategies | Get strategies (job_id or property_id; optional strategy_key for timeline) |
| POST | /api/v1/execute | Execute strategy (property_id, strategy_key) |
| POST | /api/v1/workspaces/teams | Create team |
| POST | /api/v1/workspaces | Create workspace |
| GET | /api/v1/analytics/metrics | Aggregate metrics |
| GET | /api/v1/contractor/repairs | Contractor: get repairs (property_id or job_id) |
| POST | /api/v1/contractor/completion | Contractor: submit work completion |
| GET | /api/v1/export/job/{job_id}/csv | Export job result as CSV |
| GET | /api/v1/export/job/{job_id}/summary | Export job summary as text |

## OpenAPI

Export schema: `GET /openapi.json`. Use for codegen, Postman, or API gateways.
