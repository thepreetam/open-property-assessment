# Opendoor Integration Outline (Acquisition Package)

## Fit

- **Repair Co-Pilot / RiskAI**: This platform automates condition flagging from photos and turns findings into repair strategies with cost, uplift, and ROI—aligned with Opendoor’s repair and risk workflows.
- **Instant-offer refinement**: CV + multimodal signals (room, condition, objects) can feed into pricing and repair scoping.

## Integration Points

1. **Photo intake**: Accept uploads from Opendoor’s assessment flow (or link to existing property/photos).  
2. **Job API**: POST /api/v1/jobs with property_id and photos; poll or webhook for completion.  
3. **Strategies**: GET /api/v1/strategies by property_id; consume budget/value/premium options and recommendation.  
4. **Execute**: POST /api/v1/execute to “dispatch” a chosen strategy; integrate with contractor dispatch or internal repair workflow.  
5. **Export**: CSV/PDF for reports and handoff to Key Agents or partners.  

## Data and Security

- Multi-tenant workspaces map to Opendoor teams or org units.  
- API key or OAuth for server-to-server calls; PII and photos handled per Opendoor policy (encryption, retention).  
- Audit logs support compliance and traceability.  

## Technical Handoff

- **API spec**: `/openapi.json` and docs/API-SPEC.md.  
- **Architecture**: docs/ARCHITECTURE.md.  
- **Schema**: docs/DATABASE-SCHEMA.md.  
- **Runbook**: docs/RUNBOOK.md.  
- **Security**: docs/SECURITY.md.  

Production deployment would add: fine-tuned models, real comps/cost data, SSO, and SLA monitoring.
