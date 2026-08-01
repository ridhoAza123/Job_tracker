"""Standalone command-line speed tracker.

Runs the same services the web dashboard uses, without Flask or a database.
Useful for calibrating PIXELS_PER_METER against a known stretch of road.

    python speed_tracker.py                       # first online CCTV camera
    python speed_tracker.py --source video.mp4    # a local file
    python speed_tracker.py --list                # list discovered cameras
    python speed_tracker.py --no-window           # headless, log only
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2

from config import Config
from services.cctv_fetcher import STATUS_ONLINE, CCTVCrawler, is_direct_stream_url
from services.detector import Detector
from services.preprocessor import FramePreprocessor
from services.roi import RegionOfInterest
from services.speed_calculator import SpeedEstimator
from services.streamer import Streamer
from services.tracker import VehicleTracker

logger = logging.getLogger('speed_tracker')

_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CCTV vehicle speed tracker (CLI).')
    parser.add_argument('--source', help='Stream URL or video file. Defaults to CCTV discovery.')
    parser.add_argument('--weights', help='Override the YOLO weights path.')
    parser.add_argument('--conf', type=float, default=Config.DETECTION_CONFIDENCE)
    parser.add_argument('--iou', type=float, default=Config.DETECTION_IOU)
    parser.add_argument('--list', action='store_true', help='List discovered cameras and exit.')
    parser.add_argument('--no-window', action='store_true', help='Do not open a preview window.')
    parser.add_argument('--no-preprocess', action='store_true', help='Disable preprocessing.')
    parser.add_argument('--seconds', type=float, default=0.0, help='Stop after N seconds.')
    return parser.parse_args()


def discover_source() -> str | None:
    """First online camera from the portal, else the raw entry point."""
    crawler = CCTVCrawler(Config.CCTV_STREAM_URL, logger)
    cameras = crawler.discover(probe=True)
    for camera in cameras:
        if camera.status == STATUS_ONLINE:
            logger.info('Using camera: %s', camera.name)
            return camera.url
    if cameras:
        logger.warning('No camera probed as online; trying the first one found.')
        return cameras[0].url
    if is_direct_stream_url(Config.CCTV_STREAM_URL):
        return Config.CCTV_STREAM_URL
    logger.error('No usable camera found.')
    return None


def list_cameras() -> int:
    crawler = CCTVCrawler(Config.CCTV_STREAM_URL, logger)
    cameras = crawler.discover(probe=True)
    if not cameras:
        print('No cameras found.')
        return 1
    print(f'{len(cameras)} camera(s) found:\n')
    print(f'{"#":>3}  {"NAME":34} {"STATUS":14} {"RES":11} {"TYPE":6} URL')
    for index, camera in enumerate(cameras, 1):
        print(
            f'{index:>3}  {camera.name[:34]:34} {camera.status:14} '
            f'{(camera.resolution or "-"):11} {camera.stream_type:6} {camera.url}'
        )
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s', datefmt='%H:%M:%S'
    )
    signal.signal(signal.SIGINT, _handle_signal)

    if args.list:
        return list_cameras()

    source = args.source or discover_source()
    if not source:
        return 1

    detector = Detector(args.weights)
    if not detector.is_ready:
        logger.error('No usable YOLO weights; aborting.')
        return 1

    tracker = VehicleTracker(detector=detector, confidence=args.conf, iou=args.iou)
    preprocessor = FramePreprocessor(enabled=not args.no_preprocess)
    roi = RegionOfInterest()
    estimator = SpeedEstimator(roi=roi)

    streamer = Streamer(logger)
    streamer.start(source)
    logger.info('Model=%s tracker=%s source=%s', detector.model_name, tracker.tracker_name, source)

    show_window = not args.no_window
    started = time.time()
    last_report = 0.0
    last_frame_id = -1
    processed = 0

    try:
        while not _STOP:
            if args.seconds and time.time() - started > args.seconds:
                break
            frame, last_frame_id = streamer.read_latest(last_frame_id)
            if frame is None:
                time.sleep(0.02)
                continue

            detections = tracker.update(preprocessor.process(frame))
            estimator.update(detections, frame.shape)
            processed += 1

            for detection in detections:
                x1, y1, x2, y2 = detection.box
                colour = (0, 0, 255) if detection.status == 'Overspeed' else (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                speed = float(detection.speed or 0.0)
                caption = (
                    f'#{detection.track_id} {detection.class_name} {speed:.0f} km/h'
                    if speed >= estimator.min_speed
                    else f'#{detection.track_id} {detection.class_name} ...'
                )
                cv2.putText(
                    frame, caption, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

            now = time.time()
            if now - last_report >= 2.0:
                last_report = now
                moving = [d for d in detections if (d.speed or 0) >= estimator.min_speed]
                logger.info(
                    '%d vehicles (%d measured) | %.0f ms | %s',
                    len(detections), len(moving), tracker.last_inference_ms,
                    ', '.join(f'#{d.track_id}:{d.speed:.0f}km/h' for d in moving[:6]) or '-',
                )

            if show_window:
                cv2.imshow('Speed Tracker', frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    finally:
        streamer.stop()
        if show_window:
            cv2.destroyAllWindows()
        logger.info('Processed %d frames in %.1fs', processed, time.time() - started)
    return 0


if __name__ == '__main__':
    sys.exit(main())
