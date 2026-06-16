USE crack_inspection;

DELIMITER //

DROP PROCEDURE IF EXISTS add_column_if_missing//
CREATE PROCEDURE add_column_if_missing(
    IN table_name_value VARCHAR(64),
    IN column_name_value VARCHAR(64),
    IN definition_value TEXT
)
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = table_name_value
          AND column_name = column_name_value
    ) THEN
        SET @sql_value = CONCAT(
            'ALTER TABLE `', table_name_value, '` ADD COLUMN `',
            column_name_value, '` ', definition_value
        );
        PREPARE statement_value FROM @sql_value;
        EXECUTE statement_value;
        DEALLOCATE PREPARE statement_value;
    END IF;
END//

DROP PROCEDURE IF EXISTS add_index_if_missing//
CREATE PROCEDURE add_index_if_missing(
    IN table_name_value VARCHAR(64),
    IN index_name_value VARCHAR(64),
    IN definition_value TEXT
)
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = table_name_value
          AND index_name = index_name_value
    ) THEN
        SET @sql_value = CONCAT(
            'ALTER TABLE `', table_name_value, '` ADD ', definition_value
        );
        PREPARE statement_value FROM @sql_value;
        EXECUTE statement_value;
        DEALLOCATE PREPARE statement_value;
    END IF;
END//

DELIMITER ;

CALL add_column_if_missing('workorders', 'inspector_name', 'VARCHAR(100) NULL');
CALL add_column_if_missing('workorders', 'description', 'TEXT NULL');
CALL add_column_if_missing('workorders', 'source', 'VARCHAR(20) NULL');
CALL add_column_if_missing('workorders', 'capture_count', 'INT NOT NULL DEFAULT 0');
CALL add_column_if_missing('workorders', 'start_time', 'DATETIME NULL');
CALL add_column_if_missing('workorders', 'end_time', 'DATETIME NULL');
CALL add_column_if_missing('workorders', 'ai_conclusion', 'TEXT NULL');
CALL add_column_if_missing('workorders', 'ai_model', 'VARCHAR(100) NULL');
CALL add_column_if_missing('workorders', 'ai_analyzed_at', 'DATETIME NULL');

CALL add_column_if_missing('images', 'capture_id', 'VARCHAR(100) NULL');
CALL add_column_if_missing('images', 'captured_at', 'DATETIME NULL');
CALL add_column_if_missing('images', 'original_image_path', 'VARCHAR(500) NULL');
CALL add_column_if_missing('images', 'image_filename', 'VARCHAR(255) NULL');

CALL add_column_if_missing('crack_results', 'confidence', 'DOUBLE NOT NULL DEFAULT 0');

UPDATE workorders
SET source = 'manual'
WHERE source IS NULL
  AND (
      BINARY status = BINARY '待开始'
      OR BINARY status = BINARY '待接收'
      OR BINARY status = BINARY '巡检中'
      OR description IS NOT NULL
      OR EXISTS (
          SELECT 1
          FROM images img
          JOIN crack_results result ON result.image_id = img.id
          WHERE img.workorder_id = workorders.id
            AND BINARY JSON_UNQUOTE(JSON_EXTRACT(result.result_json, '$.source')) = BINARY 'manual'
      )
  );

UPDATE workorders
SET source = 'device'
WHERE source IS NULL;

ALTER TABLE workorders
    MODIFY source VARCHAR(20) NOT NULL DEFAULT 'device';

ALTER TABLE images
    MODIFY capture_id VARCHAR(100) NULL,
    MODIFY image_filename VARCHAR(255) NULL;

UPDATE images
SET capture_id = CONCAT('legacy-', id)
WHERE capture_id IS NULL OR capture_id = '';

UPDATE images
SET image_filename = SUBSTRING_INDEX(REPLACE(image_path, '\\', '/'), '/', -1)
WHERE image_filename IS NULL OR image_filename = '';

UPDATE workorders w
SET capture_count = (
    SELECT COUNT(*) FROM images img WHERE img.workorder_id = w.id
);

ALTER TABLE images
    MODIFY capture_id VARCHAR(100) NOT NULL,
    MODIFY image_filename VARCHAR(255) NOT NULL,
    MODIFY image_path VARCHAR(500) NOT NULL,
    MODIFY original_image_path VARCHAR(500) NULL;

CALL add_index_if_missing(
    'images',
    'uq_images_workorder_capture',
    'UNIQUE KEY `uq_images_workorder_capture` (`workorder_id`, `capture_id`)'
);
CALL add_index_if_missing(
    'workorders',
    'idx_workorders_user_created',
    'INDEX `idx_workorders_user_created` (`user_id`, `created_at`)'
);
CALL add_index_if_missing(
    'images',
    'idx_images_upload_time',
    'INDEX `idx_images_upload_time` (`upload_time`)'
);

DROP PROCEDURE add_column_if_missing;
DROP PROCEDURE add_index_if_missing;
