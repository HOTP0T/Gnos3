"""merge data_connector with v0.9.2 chain

Revision ID: 9f039da644b7
Revises: 56359461a091, b3d914a6e0b4
Create Date: 2026-04-30 15:28:53.538978

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = '9f039da644b7'
down_revision: Union[str, None] = ('56359461a091', 'b3d914a6e0b4')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
