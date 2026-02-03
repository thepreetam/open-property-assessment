# Security and Compliance (Phase 3)

## API Key

- Set **`API_KEY`** in the environment to require **`X-API-Key`** header on protected routes (properties, strategies, execute, workspaces, analytics, contractor).
- Do not commit API keys; use secrets management in production.

## Data Protection

- **Encryption at rest**: Use PostgreSQL and Redis with encryption-at-rest (e.g. cloud provider defaults). Sensitive fields (e.g. PII) should be tokenized or encrypted before storage in production.
- **In transit**: Serve the API over HTTPS (TLS). Use `API_BASE_URL` with `https://` for webhooks and callbacks.

## GDPR / CCPA

- Audit log (`audit_logs` table) records actions for accountability. Implement data export and deletion endpoints for user requests (right to access, right to erasure).
- Minimize retention of raw photos and PII; define retention policies per workspace.

## Security Audit

- Production deployment should include: dependency scanning, SAST, penetration testing, and regular access review for workspace members and API keys.
