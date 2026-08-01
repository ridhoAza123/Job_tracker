"""Application factory and entry point."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask  # noqa: E402  (import order: .env must load first)

from config import Config  # noqa: E402
from db import init_db  # noqa: E402

logger = logging.getLogger(__name__)


def configure_logging(app: Flask) -> None:
    log_dir = Path(app.config['LOG_DIR'])
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(app.config.get('LOG_LEVEL', 'INFO'))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-7s [%(name)s] %(message)s', '%Y-%m-%d %H:%M:%S'
    )
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # These libraries are extremely chatty at INFO level.
    for noisy in ('werkzeug', 'urllib3', 'ultralytics', 'PIL'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(start_pipeline: bool = True) -> Flask:
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    configure_logging(app)
    init_db(app)

    from routes.api import api_bp
    from routes.web import web_bp

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    app.pipeline = None
    if start_pipeline:
        # Imported here so `create_app(start_pipeline=False)` stays light and
        # test/CLI users do not pay for loading torch.
        from services.pipeline import DetectionPipeline

        app.pipeline = DetectionPipeline(app)
        app.pipeline.start()

    logger.info('Speed Tracker ready on http://%s:%s', Config.HOST, Config.PORT)
    return app


app = None

if __name__ == '__main__':
    app = create_app()
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.FLASK_DEBUG,
            threaded=True,
            # The reloader would start the capture threads and the model twice.
            use_reloader=False,
        )
    finally:
        if app.pipeline is not None:
            app.pipeline.stop()
