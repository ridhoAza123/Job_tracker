"""SQLAlchemy bootstrap shared by the whole application."""

import logging
from pathlib import Path

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError

from database.utils import (
    build_fallback_sqlite_uri,
    describe_uri,
    ensure_database_exists,
    is_server_unavailable,
    migrate_detections_table,
    relax_legacy_not_null,
)

logger = logging.getLogger(__name__)

# SQLAlchemy instance shared across modules.
db = SQLAlchemy()

BASE_DIR = Path(__file__).resolve().parent


def init_db(app) -> bool:
    """Initialise the database, creating and migrating it as needed.

    Falls back to SQLite when the configured server is unreachable so the
    stream keeps running even without a database. Returns True when a
    working connection was established.
    """
    configured_uri = app.config['SQLALCHEMY_DATABASE_URI']
    ensure_database_exists(configured_uri)

    db.init_app(app)
    if _try_bootstrap(app, configured_uri):
        return True

    fallback_uri = build_fallback_sqlite_uri(BASE_DIR)
    logger.warning(
        'Falling back to SQLite (%s) because %s is unavailable.',
        describe_uri(fallback_uri), describe_uri(configured_uri),
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = fallback_uri
    # Engine options tuned for MySQL pooling are invalid for SQLite.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {}
    _dispose_engine(app)
    return _try_bootstrap(app, fallback_uri)


def _try_bootstrap(app, uri: str) -> bool:
    """Create tables + migrate for the currently configured URI."""
    try:
        with app.app_context():
            # Importing here keeps db.py free of model-level imports and
            # avoids a circular import at module load time.
            import models.detection  # noqa: F401

            engine = db.engine
            relax_legacy_not_null(engine)
            migrate_detections_table(engine)
            db.create_all()
            # create_all skips existing tables, so migrate again in case the
            # table was only just created by another process.
            migrate_detections_table(engine)
        app.config['DB_ACTIVE_URI'] = describe_uri(uri)
        logger.info('Database ready: %s', describe_uri(uri))
        return True
    except SQLAlchemyError as error:
        if is_server_unavailable(error):
            logger.warning('Database unavailable at %s: %s', describe_uri(uri), error)
        else:
            logger.error('Database bootstrap failed for %s: %s', describe_uri(uri), error)
        return False
    except Exception as error:
        logger.error('Unexpected database bootstrap error: %s', error)
        return False


def _dispose_engine(app) -> None:
    # flask_sqlalchemy's ``db.engines`` is itself a property that resolves
    # the current app from context, so clearing it must happen inside the
    # same app context as the dispose() call, not after it exits.
    try:
        with app.app_context():
            db.engine.dispose()
            for attr in ('engines', '_engines'):
                holder = getattr(db, attr, None)
                if isinstance(holder, dict):
                    holder.clear()
    except Exception:
        pass
