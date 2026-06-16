CREATE DATABASE IF NOT EXISTS crack_inspection
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci;

USE crack_inspection;

CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'operator',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_users_role CHECK (role IN ('admin', 'operator', 'inspector'))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS workorders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NULL,
    inspector_name VARCHAR(100) NULL,
    location VARCHAR(255) NOT NULL,
    description TEXT NULL,
    source VARCHAR(20) NOT NULL DEFAULT 'device',
    status VARCHAR(50) NOT NULL DEFAULT '待接收',
    capture_count INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    start_time DATETIME NULL,
    end_time DATETIME NULL,
    completed_at DATETIME NULL,
    ai_conclusion TEXT NULL,
    ai_model VARCHAR(100) NULL,
    ai_analyzed_at DATETIME NULL,
    INDEX idx_workorders_user_created (user_id, created_at),
    CONSTRAINT chk_workorders_source CHECK (source IN ('manual', 'device')),
    CONSTRAINT chk_workorders_capture_count CHECK (capture_count >= 0),
    CONSTRAINT fk_workorders_user FOREIGN KEY (user_id) REFERENCES users(id)
        ON UPDATE CASCADE ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    workorder_id INT NOT NULL,
    capture_id VARCHAR(100) NOT NULL,
    captured_at DATETIME NULL,
    image_path VARCHAR(500) NOT NULL,
    original_image_path VARCHAR(500) NULL,
    image_filename VARCHAR(255) NOT NULL,
    upload_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_images_workorder_capture (workorder_id, capture_id),
    INDEX idx_images_upload_time (upload_time),
    CONSTRAINT fk_images_workorder FOREIGN KEY (workorder_id) REFERENCES workorders(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS crack_results (
    image_id INT PRIMARY KEY,
    crack_count INT NOT NULL DEFAULT 0,
    crack_area DOUBLE NOT NULL DEFAULT 0,
    max_width DOUBLE NOT NULL DEFAULT 0,
    total_length DOUBLE NOT NULL DEFAULT 0,
    severity VARCHAR(50) NOT NULL DEFAULT '无',
    suggestion VARCHAR(500) NULL,
    confidence DOUBLE NOT NULL DEFAULT 0,
    result_json JSON NULL,
    detected_at DATETIME NULL,
    CONSTRAINT chk_crack_count CHECK (crack_count >= 0),
    CONSTRAINT chk_crack_area CHECK (crack_area >= 0),
    CONSTRAINT chk_max_width CHECK (max_width >= 0),
    CONSTRAINT chk_total_length CHECK (total_length >= 0),
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT fk_results_image FOREIGN KEY (image_id) REFERENCES images(id)
        ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

INSERT INTO users (username, password, role)
VALUES ('admin', '123456', 'admin')
ON DUPLICATE KEY UPDATE username = VALUES(username);
