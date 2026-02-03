# Deployment

Deployment automation and production runbooks live in **[docs/RUNBOOK.md](../docs/RUNBOOK.md)**.

- **Render**: Blueprint or separate Web Service (API) + Background Worker (Celery) + Redis + PostgreSQL. See RUNBOOK “Deployment automation.”
- **Docker**: `docker-compose up` for full stack; override for production (API + worker only).
- **Terraform/Ansible**: Use RUNBOOK as reference; provision Postgres + Redis and deploy API + worker with the same env vars.
