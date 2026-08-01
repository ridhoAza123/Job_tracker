"""HTML pages and the MJPEG video endpoint."""

from flask import Blueprint, Response, current_app, render_template

web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def dashboard():
    return render_template('dashboard.html')


@web_bp.route('/history')
def history():
    return render_template('history.html')


@web_bp.route('/stream')
def stream():
    pipeline = getattr(current_app, 'pipeline', None)
    if pipeline is None:
        return 'Stream not available', 503
    response = Response(
        pipeline.mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    # Proxies and browsers must not buffer a live stream.
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@web_bp.route('/healthz')
def healthz():
    pipeline = getattr(current_app, 'pipeline', None)
    return {
        'status': 'ok',
        'pipeline': bool(pipeline and pipeline.is_running),
        'camera': pipeline.cctv.current_camera_name if pipeline else None,
    }
