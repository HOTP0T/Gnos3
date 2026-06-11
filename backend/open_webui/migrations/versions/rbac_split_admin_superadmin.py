"""Split admin into admin (data-admin) + superadmin (platform-admin)

Phase 3.7 RBAC: existing `role='admin'` users get promoted to `'superadmin'`
(keeping the access they had today). The string `'admin'` becomes a new lesser
tier reserved for users who get data-access bypass but NO platform-config
access (LDAP / OAuth / RAG / model providers / functions / etc.).

Revision ID: a1b2c3d4e5f6
Revises: 9f039da644b7
Create Date: 2026-05-28
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'rbac_split'
down_revision: Union[str, None] = '9f039da644b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE \"user\" SET role = 'superadmin' WHERE role = 'admin'")


def downgrade() -> None:
    op.execute("UPDATE \"user\" SET role = 'admin' WHERE role = 'superadmin'")
