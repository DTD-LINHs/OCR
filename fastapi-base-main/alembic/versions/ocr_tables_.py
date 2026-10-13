"""OCR handwritten code tables

Revision ID: ocr_tables
Revises: initial
Create Date: 2026-10-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ocr_tables'
down_revision: Union[str, None] = 'initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamps() -> list:
    return [
        sa.Column('created_at', sa.Float(), nullable=True),
        sa.Column('updated_at', sa.Float(), nullable=True),
    ]


def _submission_fk() -> sa.Column:
    return sa.Column(
        'submission_id',
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey('ocr_submission.id', ondelete='CASCADE'),
        nullable=False,
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'ocr_submission',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('image_path', sa.String(), nullable=False),
        sa.Column('original_filename', sa.String(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('run', sa.Integer(), nullable=False),
        sa.Column('language', sa.String(length=32), nullable=True),
        sa.Column('final_code', sa.Text(), nullable=True),
        sa.Column('syntax_ok', sa.Boolean(), nullable=True),
        sa.Column('syntax_errors', postgresql.JSONB(), nullable=True),
        sa.Column('need_review', sa.Boolean(), nullable=False),
        sa.Column('review_reasons', postgresql.JSONB(), nullable=True),
        sa.Column('uncertain_lines', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('raw_ocr', postgresql.JSONB(), nullable=True),
        sa.Column('timings', postgresql.JSONB(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        *_timestamps(),
    )
    op.create_index('ix_ocr_submission_user_id', 'ocr_submission', ['user_id'])
    op.create_index('ix_ocr_submission_status', 'ocr_submission', ['status'])
    op.create_index('ix_ocr_submission_need_review', 'ocr_submission', ['need_review'])

    op.create_table(
        'ocr_code_version',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        _submission_fk(),
        sa.Column('run', sa.Integer(), nullable=False),
        sa.Column('stage', sa.String(length=32), nullable=False),
        sa.Column('fix_round', sa.Integer(), nullable=False),
        sa.Column('language', sa.String(length=32), nullable=True),
        sa.Column('code', sa.Text(), nullable=False),
        sa.Column('corrections', postgresql.JSONB(), nullable=True),
        *_timestamps(),
    )
    op.create_index('ix_ocr_code_version_submission_id', 'ocr_code_version', ['submission_id'])

    op.create_table(
        'ocr_sandbox_check',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        _submission_fk(),
        sa.Column(
            'code_version_id',
            sa.Integer(),
            sa.ForeignKey('ocr_code_version.id', ondelete='SET NULL'),
            nullable=True,
        ),
        sa.Column('run', sa.Integer(), nullable=False),
        sa.Column('language', sa.String(length=32), nullable=False),
        sa.Column('image', sa.String(), nullable=True),
        sa.Column('command', sa.Text(), nullable=True),
        sa.Column('exit_code', sa.Integer(), nullable=True),
        sa.Column('ok', sa.Boolean(), nullable=False),
        sa.Column('timed_out', sa.Boolean(), nullable=False),
        sa.Column('errors', postgresql.JSONB(), nullable=True),
        sa.Column('output', sa.Text(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        *_timestamps(),
    )
    op.create_index('ix_ocr_sandbox_check_submission_id', 'ocr_sandbox_check', ['submission_id'])

    op.create_table(
        'ocr_llm_call_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        _submission_fk(),
        sa.Column('run', sa.Integer(), nullable=False),
        sa.Column('purpose', sa.String(length=32), nullable=False),
        sa.Column('model', sa.String(length=64), nullable=True),
        sa.Column('prompt', sa.Text(), nullable=True),
        sa.Column('response', sa.Text(), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('parse_ok', sa.Boolean(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        *_timestamps(),
    )
    op.create_index('ix_ocr_llm_call_log_submission_id', 'ocr_llm_call_log', ['submission_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('ocr_llm_call_log')
    op.drop_table('ocr_sandbox_check')
    op.drop_table('ocr_code_version')
    op.drop_table('ocr_submission')
