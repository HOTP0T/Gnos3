"""Add data_connector and data_connector_document tables

Revision ID: b3d914a6e0b4
Revises: b2c3d4e5f6a7
Create Date: 2026-04-02 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b3d914a6e0b4"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_connector",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("connector_type", sa.String(50), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("knowledge_id", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("sync_interval", sa.Integer(), nullable=False, server_default=sa.text("3600")),
        sa.Column("last_sync_at", sa.BigInteger(), nullable=True),
        sa.Column("last_sync_status", sa.String(20), nullable=True),
        sa.Column("last_sync_error", sa.Text(), nullable=True),
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
    )

    op.create_table(
        "data_connector_document",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("connector_id", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("file_id", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("external_url", sa.Text(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("last_synced_at", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.Index("ix_data_connector_doc_connector", "connector_id"),
        sa.Index("ix_data_connector_doc_external", "external_id"),
        sa.UniqueConstraint("connector_id", "external_id", name="uq_connector_external_id"),
    )


def downgrade() -> None:
    op.drop_table("data_connector_document")
    op.drop_table("data_connector")
