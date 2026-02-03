# Database Schema

PostgreSQL (or compatible). Tables created by SQLAlchemy `create_all` on first use.

## Tables

- **teams** – id (PK), name, created_at  
- **workspaces** – id (PK), team_id (FK teams), name, created_at  
- **workspace_members** – id (PK), workspace_id (FK workspaces), user_id, role (admin|member|viewer), created_at  
- **audit_logs** – id (PK), workspace_id, user_id, action, resource_type, resource_id, payload (JSONB), created_at  
- **properties** – id (PK), workspace_id (FK workspaces), address, zip_code, home_value, created_at, updated_at  
- **jobs** – id (PK), property_id (FK properties), workspace_id (FK workspaces), status, created_at, updated_at, result (JSONB), error  
- **executions** – id (PK), property_id (FK properties), job_id (FK jobs), strategy_key, status, created_at, updated_at, payload (JSONB)  

## Relationships

- Team → Workspaces (one-to-many)  
- Workspace → Properties, Jobs, WorkspaceMembers, AuditLogs  
- Property → Jobs, Executions  
- Job → Executions (optional)  

## Migrations

Phase 1–2 use `Base.metadata.create_all(engine)`. For production, add Alembic (or similar) and versioned migrations under `db/migrations/`.
