"""OCR submission: corrections and review items

Revision ID: ocr_review_items
Revises: ocr_tables
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ocr_review_items'
down_revision: Union[str, None] = 'ocr_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('ocr_submission', sa.Column('llm_used', sa.Boolean(), nullable=True))
    op.add_column('ocr_submission', sa.Column('corrections', postgresql.JSONB(), nullable=True))
    op.add_column('ocr_submission', sa.Column('review_items', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ocr_submission', 'review_items')
    op.drop_column('ocr_submission', 'corrections')
    op.drop_column('ocr_submission', 'llm_used')
