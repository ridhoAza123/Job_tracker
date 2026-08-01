"""Database helpers: URL surgery, bootstrap and lightweight auto-migration."""

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def build_fallback_sqlite_uri(base_dir: Path) -> str:
    """Local SQLite file used when the configured server is unreachable."""
    target = Path(base_dir) / 'database'
    target.mkdir(parents=True, exist_ok=True)
    return f'sqlite:///{(target / "speed_tracker.db").as_posix()}'


def is_server_unavailable(error: Exception) -> bool:
    """True when the error looks like 'server down / db missing / auth'."""
    text_error = str(error).lower()
    markers = (
        'can\'t connect', 'cannot connect', 'connection refused', 'timed out',
        'unknown database', 'access denied', 'no such host', 'lost connection',
        'server has gone away', 'operationalerror', 'connection reset',
    )
    return any(marker in text_error for marker in markers)


def ensure_database_exists(database_uri: str, timeout: int = 8) -> bool:
    """Create the target schema when it does not exist yet.

    Returns True when the database is present (or was just created), False
    when the server could not be reached at all. Never raises.
    """
    try:
        url = make_url(database_uri)
    except Exception as error:  # malformed URL
        logger.error('Invalid DATABASE_URL: %s', error)
        return False

    if url.get_backend_name() != 'mysql' or not url.database:
        # SQLite and friends create their store on demand.
        return True

    server_url = url.set(database=None)
    try:
        from sqlalchemy import create_engine

        engine = create_engine(
            server_url, connect_args={'connect_timeout': timeout}, poolclass=None
        )
        with engine.connect() as connection:
            connection.execute(
                text(
                    f'CREATE DATABASE IF NOT EXISTS `{url.database}` '
                    'CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci'
                )
            )
            connection.commit()
        engine.dispose()
        logger.info('Database "%s" is ready.', url.database)
        return True
    except SQLAlchemyError as error:
        logger.warning('Could not ensure database "%s": %s', url.database, error)
        return False
    except Exception as error:  # driver-level failures
        logger.warning('Unexpected error preparing database: %s', error)
        return False


# Columns this build expects on ``detections``, with the DDL used to add them
# to a table created by an older version of the app.
_EXPECTED_COLUMNS = {
    'track_id': 'INTEGER NULL',
    'camera_name': 'VARCHAR(128) NULL',
    'stream_url': 'VARCHAR(512) NULL',
    'confidence': 'FLOAT NULL',
}


def migrate_detections_table(engine, table_name: str = 'detections') -> None:
    """Add any missing columns in place so existing rows are preserved."""
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            return
        existing = {column['name'] for column in inspector.get_columns(table_name)}
        missing = {
            name: ddl for name, ddl in _EXPECTED_COLUMNS.items() if name not in existing
        }
        if not missing:
            return
        with engine.begin() as connection:
            for name, ddl in missing.items():
                connection.execute(
                    text(f'ALTER TABLE {table_name} ADD COLUMN {name} {ddl}')
                )
                logger.info('Migrated %s: added column %s', table_name, name)
    except SQLAlchemyError as error:
        logger.error('Auto-migration of %s failed: %s', table_name, error)
    except Exception as error:
        logger.error('Unexpected auto-migration error: %s', error)


def relax_legacy_not_null(engine, table_name: str = 'detections') -> None:
    """Make legacy NOT NULL columns nullable when we no longer populate them.

    Older schemas declared ``camera_id`` / ``vehicle_id`` NOT NULL. We still
    write ``vehicle_id``, but ``camera_id`` belongs to a table we do not own.
    """
    try:
        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            return
        if engine.dialect.name != 'mysql':
            return
        for column in inspector.get_columns(table_name):
            if column['name'] == 'camera_id' and not column.get('nullable', True):
                with engine.begin() as connection:
                    connection.execute(
                        text(f'ALTER TABLE {table_name} MODIFY camera_id INT NULL')
                    )
                    logger.info('Relaxed %s.camera_id to NULL', table_name)
    except Exception as error:
        logger.debug('Could not relax legacy columns: %s', error)


def describe_uri(database_uri: str) -> str:
    """Password-free rendering of a database URI, safe for logs and the UI."""
    try:
        url = make_url(database_uri)
        return url.render_as_string(hide_password=True)
    except Exception:
        return 'unknown'


def sqlite_path_from_uri(database_uri: str) -> Optional[str]:
    try:
        url = make_url(database_uri)
        if url.get_backend_name() == 'sqlite':
            return url.database
    except Exception:
        return None
    return None
