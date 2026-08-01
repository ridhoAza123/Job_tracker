-- Reference schema. The application creates and migrates this automatically
-- on startup; this file is kept for manual provisioning.

CREATE DATABASE IF NOT EXISTS speed_tracker
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE speed_tracker;

CREATE TABLE IF NOT EXISTS detections (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    vehicle_id    VARCHAR(64)  NOT NULL,
    track_id      INT          NULL,
    vehicle_type  VARCHAR(32)  NOT NULL,
    speed         FLOAT        NOT NULL,
    status        VARCHAR(32)  NOT NULL,
    camera_name   VARCHAR(128) NULL,
    stream_url    VARCHAR(512) NULL,
    snapshot      VARCHAR(255) NULL,
    confidence    FLOAT        NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_track_id (track_id),
    INDEX idx_vehicle_type (vehicle_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Upgrading an installation created before stream_url/camera_name existed:
-- ALTER TABLE detections ADD COLUMN track_id    INTEGER      NULL;
-- ALTER TABLE detections ADD COLUMN camera_name VARCHAR(128) NULL;
-- ALTER TABLE detections ADD COLUMN stream_url  VARCHAR(512) NULL;
-- ALTER TABLE detections ADD COLUMN confidence  FLOAT        NULL;
