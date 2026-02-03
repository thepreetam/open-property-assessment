"""
SQLAlchemy models for jobs, properties (Phase 2), teams/workspaces/audit (Phase 3).
"""
from datetime import datetime
from typing import Any, Optional
from sqlalchemy import Column, DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Team(Base):
    """Team (Phase 3 multi-tenant)."""

    __tablename__ = "teams"

    id = Column(String(36), primary_key=True)
    name = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Workspace(Base):
    """Workspace: project container (Phase 3)."""

    __tablename__ = "workspaces"

    id = Column(String(36), primary_key=True)
    team_id = Column(String(36), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class WorkspaceMember(Base):
    """Workspace membership and role (Phase 3 RBAC)."""

    __tablename__ = "workspace_members"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(128), nullable=False, index=True)  # external id or email
    role = Column(String(32), nullable=False, default="member")  # admin | member | viewer
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = ({"sqlite_autoincrement": False})


class AuditLog(Base):
    """Audit log for decisions (Phase 3)."""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), nullable=True, index=True)
    user_id = Column(String(128), nullable=True)
    action = Column(String(64), nullable=False)  # job.create, strategy.execute, etc.
    resource_type = Column(String(64), nullable=True)  # job, property, execution
    resource_id = Column(String(36), nullable=True)
    payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Property(Base):
    """Property: first-class entity for portfolio (Phase 2)."""

    __tablename__ = "properties"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    address = Column(String(512), nullable=True)
    zip_code = Column(String(20), nullable=True)
    home_value = Column(String(24), nullable=True)  # store as string to avoid precision issues
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Job(Base):
    """Analysis job: photos + params → status + result. Optional property_id (Phase 2)."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    property_id = Column(String(36), ForeignKey("properties.id", ondelete="SET NULL"), nullable=True, index=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    status = Column(String(32), nullable=False, default="pending")  # pending | processing | completed | failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    result = Column(JSONB, nullable=True)  # all_data, strategies, recommendation
    error = Column(Text, nullable=True)


class Execution(Base):
    """Execute strategy stub: property + strategy_key → dispatched (Phase 2)."""

    __tablename__ = "executions"

    id = Column(String(36), primary_key=True)
    property_id = Column(String(36), ForeignKey("properties.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    strategy_key = Column(String(64), nullable=False)  # budget_flip | value_add | premium_reno
    status = Column(String(32), nullable=False, default="dispatched")  # dispatched | in_progress | completed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    payload = Column(JSONB, nullable=True)


def get_engine(database_url: str):
    """Create SQLAlchemy engine. Use postgresql+psycopg2 for async later if needed."""
    url = database_url
    if url.startswith("postgresql://") and "postgresql+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, pool_pre_ping=True)


def get_session_factory(database_url: str):
    """Session factory for the app."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables(engine):
    """Create all tables."""
    Base.metadata.create_all(engine)
