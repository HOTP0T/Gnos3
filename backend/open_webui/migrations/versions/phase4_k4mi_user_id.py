"""Add k4mi_user_id column to user table (Phase 4 SSO bridge).

Maps each Gnos3 user to a K4mi (Paperless-NGX) user.id so the SSO bridge
can mint a JWT that K4mi's GnosJWTBackend can resolve to the right Django
user. NULL is the normal initial state — the bridge resolves the mapping
lazily on first sign-in and caches it.

Revision ID: phase4_k4mi_uid
Revises: rbac_split
Create Date: 2026-06-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'phase4_k4mi_uid'
down_revision: Union[str, None] = 'rbac_split'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user',
        sa.Column('k4mi_user_id', sa.Integer(), nullable=True),
    )
    op.create_index(
        'ix_user_k4mi_user_id',
        'user',
        ['k4mi_user_id'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_user_k4mi_user_id', table_name='user')
    op.drop_column('user', 'k4mi_user_id')
