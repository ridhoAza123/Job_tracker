"""JSON API consumed by the dashboard."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from db import db
from models.detection import Detection

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)


def _pipeline():
    return getattr(current_app, 'pipeline', None)


def _float_arg(payload: dict, name: str):
    value = payload.get(name)
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# Detection history
# ----------------------------------------------------------------------
@api_bp.route('/detections', methods=['GET'])
def get_detections():
    try:
        page = max(1, int(request.args.get('page', 1)))
        limit = min(200, max(1, int(request.args.get('limit', 20))))
    except (TypeError, ValueError):
        page, limit = 1, 20

    vehicle_type = request.args.get('vehicle_type', '').strip()
    status = request.args.get('status', '').strip()
    search = request.args.get('search', '').strip()

    try:
        query = Detection.query.order_by(Detection.created_at.desc())
        if vehicle_type:
            query = query.filter(Detection.vehicle_type == vehicle_type)
        if status:
            query = query.filter(Detection.status == status)
        if search:
            pattern = f'%{search}%'
            query = query.filter(
                or_(
                    Detection.vehicle_type.ilike(pattern),
                    Detection.vehicle_id.ilike(pattern),
                    Detection.status.ilike(pattern),
                    Detection.camera_name.ilike(pattern),
                )
            )
        paginated = query.paginate(page=page, per_page=limit, error_out=False)
        return jsonify({
            'items': [item.to_dict() for item in paginated.items],
            'page': paginated.page,
            'pages': paginated.pages,
            'total': paginated.total,
        })
    except SQLAlchemyError as error:
        logger.error('Detection query failed: %s', error)
        db.session.rollback()
        return jsonify({'items': [], 'page': 1, 'pages': 0, 'total': 0,
                        'error': 'database unavailable'}), 200


@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Aggregates from the database, merged with live pipeline counters."""
    payload = {
        'total': 0, 'overspeed': 0, 'average_speed': 0.0, 'by_type': [],
    }
    try:
        payload['total'] = Detection.query.count()
        payload['overspeed'] = Detection.query.filter(
            Detection.status == 'Overspeed'
        ).count()
        average = Detection.query.with_entities(db.func.avg(Detection.speed)).scalar()
        payload['average_speed'] = round(float(average or 0.0), 2)
        payload['by_type'] = [
            {'vehicle_type': row[0], 'count': row[1]}
            for row in Detection.query.with_entities(
                Detection.vehicle_type, db.func.count(Detection.id)
            ).group_by(Detection.vehicle_type).all()
        ]
    except SQLAlchemyError as error:
        logger.error('Statistics query failed: %s', error)
        db.session.rollback()
        payload['error'] = 'database unavailable'

    pipeline = _pipeline()
    payload['live'] = pipeline.live_stats() if pipeline else {}
    return jsonify(payload)


@api_bp.route('/live', methods=['GET'])
def get_live():
    """High-frequency endpoint: realtime counters only, no database access."""
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    return jsonify(pipeline.live_stats())


# ----------------------------------------------------------------------
# Camera control
# ----------------------------------------------------------------------
@api_bp.route('/status', methods=['GET'])
def get_status():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'cctv_status': 'OFFLINE', 'reason': 'pipeline unavailable'}), 503
    payload = pipeline.cctv.status_payload()
    payload.update({
        'model': pipeline.detector.model_name,
        'tracker': pipeline.tracker.tracker_name,
        'fps': round(pipeline.live_stats().get('fps', 0.0), 1),
        'inference_ms': pipeline.tracker.last_inference_ms,
        'database': current_app.config.get('DB_ACTIVE_URI'),
    })
    return jsonify(payload)


@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'cameras': [], 'count': 0, 'selected_url': None}), 503
    if request.args.get('refresh') == '1':
        pipeline.cctv.refresh_cameras(probe=True)
    return jsonify(pipeline.cctv.cameras_payload())


@api_bp.route('/cameras/refresh', methods=['POST'])
def refresh_cameras():
    """Re-crawl the portal in the background."""
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    pipeline.cctv.refresh_cameras_async(probe=True)
    return jsonify({'started': True})


@api_bp.route('/camera', methods=['POST'])
def select_camera():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    data = request.get_json(silent=True) or {}
    camera_url = (data.get('url') or '').strip()
    if not camera_url:
        return jsonify({'error': 'camera URL is required'}), 400
    if not pipeline.select_camera(camera_url):
        return jsonify({'error': 'camera could not be selected', 'url': camera_url}), 400
    return jsonify({
        'success': True,
        'selected_url': pipeline.cctv.current_camera_url,
        'camera_name': pipeline.cctv.current_camera_name,
    })


# ----------------------------------------------------------------------
# Runtime settings
# ----------------------------------------------------------------------
@api_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    if request.method == 'GET':
        return jsonify(pipeline.settings())

    data = request.get_json(silent=True) or {}
    preprocess = data.get('preprocess')
    return jsonify(pipeline.update_settings(
        confidence=_float_arg(data, 'confidence'),
        iou=_float_arg(data, 'iou'),
        speed_limit=_float_arg(data, 'speed_limit'),
        pixels_per_meter=_float_arg(data, 'pixels_per_meter'),
        preprocess=None if preprocess is None else bool(preprocess),
    ))


@api_bp.route('/roi', methods=['GET', 'POST', 'DELETE'])
def roi():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    if request.method == 'GET':
        return jsonify(pipeline.roi.to_dict())
    if request.method == 'DELETE':
        return jsonify(pipeline.roi.clear())

    data = request.get_json(silent=True) or {}
    return jsonify(pipeline.roi.update(
        polygon=data.get('polygon'),
        entry_line=data.get('entry_line'),
        exit_line=data.get('exit_line'),
    ))


@api_bp.route('/capture', methods=['POST'])
def manual_capture():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    path = pipeline.manual_capture()
    if path is None:
        return jsonify({'error': 'no frame available yet'}), 409
    return jsonify({'snapshot': path})


@api_bp.route('/stats/reset', methods=['POST'])
def reset_stats():
    pipeline = _pipeline()
    if pipeline is None:
        return jsonify({'error': 'pipeline unavailable'}), 503
    pipeline.reset_stats()
    return jsonify({'reset': True})
