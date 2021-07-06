CREATE INDEX ix_rawdata ON rawdata(timestamp, basecur_id, quotecur_id, type_id);

WITH cte AS (
    SELECT rowid AS id,
        ROW_NUMBER() OVER (
            PARTITION BY timestamp,
            basecur_id,
            quotecur_id,
            type_id
        ) AS rn
    FROM rawdata
)
DELETE FROM rawdata
WHERE rowid IN (
        SELECT id
        FROM cte
        WHERE rn > 1
    );

DROP INDEX ix_rawdata;

CREATE UNIQUE INDEX ix_rawdata ON rawdata(
    cast(timestamp * 48 AS INTEGER) / 48.0,
    basecur_id,
    quotecur_id,
    type_id,
    timestamp
);

CREATE TABLE processed_data(
    timestamp REAL NOT NULL,
    basecur_id INTEGER NOT NULL,
    quotecur_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    FOREIGN KEY(basecur_id) REFERENCES currencies(id),
    FOREIGN KEY(quotecur_id) REFERENCES currencies(id),
    FOREIGN KEY(type_id) REFERENCES types(id)
);

INSERT INTO processed_data(
        timestamp,
        basecur_id,
        quotecur_id,
        type_id,
        open,
        high,
        low,
        close
    )
SELECT DISTINCT cast(timestamp * 48 AS INTEGER) / 48.0 AS resampled_timestamp,
    basecur_id,
    quotecur_id,
    type_id,
    FIRST_VALUE(value) OVER win AS open,
    MAX(value) OVER win AS high,
    MIN(value) OVER win AS low,
    LAST_VALUE(value) OVER win AS close
FROM rawdata WINDOW win AS (
        PARTITION BY cast(timestamp * 48 AS INTEGER) / 48.0,
        basecur_id,
        quotecur_id,
        type_id
        ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    );

-- WITH cte AS(
-- SELECT
--     cast(timestamp * 48 AS INTEGER) / 48.0 AS resampled_timestamp,
--     basecur_id,
--     quotecur_id,
--     type_id,
--     ROW_NUMBER() OVER (win ORDER BY timestamp) AS rn_asc,
--     ROW_NUMBER() OVER (win ORDER BY timestamp DESC) AS rn_desc,
--     value
-- FROM rawdata
-- WINDOW win AS (PARTITION BY cast(timestamp * 48 AS INTEGER) / 48.0, basecur_id, quotecur_id, type_id)
-- )
-- INSERT INTO processed_data(
--     timestamp,
--     basecur_id,
--     quotecur_id,
--     type_id,
--     open,
--     high,
--     low,
--     close
-- )
-- SELECT
--     cte.resampled_timestamp,
--     cte.basecur_id,
--     cte.quotecur_id,
--     cte.type_id,
--     cte_first.value AS open,
--     MAX(cte.value) AS high,
--     MIN(cte.value) AS low,
--     cte_last.value AS close
-- FROM cte
-- INNER JOIN cte cte_first ON
--     cte.resampled_timestamp = cte_first.resampled_timestamp AND
--     cte.basecur_id = cte_first.basecur_id AND
--     cte.quotecur_id = cte_first.quotecur_id AND
--     cte.type_id = cte_first.type_id AND
--     cte_first.rn_asc = 1
-- INNER JOIN cte cte_last ON
--     cte.resampled_timestamp = cte_last.resampled_timestamp AND
--     cte.basecur_id = cte_last.basecur_id AND
--     cte.quotecur_id = cte_last.quotecur_id AND
--     cte.type_id = cte_last.type_id AND
--     cte_last.rn_desc = 1
-- GROUP BY cte.resampled_timestamp, cte.basecur_id, cte.quotecur_id, cte.type_id;
-- WITH groups AS (
-- SELECT DISTINCT
--     cast(timestamp * 48 AS INTEGER) / 48.0 AS resampled_timestamp,
--     basecur_id,
--     quotecur_id,
--     type_id
-- FROM rawdata
-- )
--
-- INSERT INTO processed_data(
--     timestamp,
--     basecur_id,
--     quotecur_id,
--     type_id,
--     open,
--     high,
--     low,
--     close
-- )
-- SELECT
--     g.resampled_timestamp,
--     g.basecur_id,
--     g.quotecur_id,
--     g.type_id,
--     o.value AS open,
--     hl.high,
--     hl.low,
--     c.value as close
-- FROM groups g,
-- (
--     SELECT value
--     FROM rawdata
--     WHERE
--         cast(timestamp * 48 AS INTEGER) / 48.0 = g.resampled_timestamp AND
--         basecur_id = g.basecur_id AND
--         quotecur_id = g.quotecur_id AND
--         type_id = g.type_id
--     ORDER BY timestamp
--     LIMIT 1
-- ) o,
-- (
--     SELECT
--         MAX(value) AS high,
--         MIN(value) AS low
--     FROM rawdata
--     WHERE
--         cast(timestamp * 48 AS INTEGER) / 48.0 = g.resampled_timestamp AND
--         basecur_id = g.basecur_id AND
--         quotecur_id = g.quotecur_id AND
--         type_id = g.type_id
-- ) hl,
-- (
--     SELECT value
--     FROM rawdata
--     WHERE
--         cast(timestamp * 48 AS INTEGER) / 48.0 = g.resampled_timestamp AND
--         basecur_id = g.basecur_id AND
--         quotecur_id = g.quotecur_id AND
--         type_id = g.type_id
--     ORDER BY timestamp DESC
--     LIMIT 1
-- ) c;
CREATE UNIQUE INDEX ix_processed_data on processed_data(timestamp, basecur_id, quotecur_id, type_id);

-- CREATE UNIQUE INDEX ix_processed_data on processed_data(basecur_id, quotecur_id, type_id, timestamp);
WITH group_count(value) AS (
    SELECT COUNT(*)
    FROM (
            SELECT DISTINCT basecur_id,
                quotecur_id,
                type_id
            FROM processed_data
        )
)
DELETE FROM processed_data AS pd
WHERE (
        SELECT COUNT(*)
        FROM (
                SELECT DISTINCT basecur_id,
                    quotecur_id,
                    type_id
                FROM processed_data
                WHERE timestamp = pd.timestamp
            )
    ) < (
        SELECT value
        FROM group_count
        LIMIT 1
    );

-- SELECT COUNT(DISTINCT timestamp)
-- FROM processed_data
-- GROUP BY basecur_id, quotecur_id, type_id;
CREATE TABLE types2(
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL UNIQUE
);

INSERT INTO types2(value)
VALUES ('open'),
    ('high'),
    ('low'),
    ('close');

CREATE TABLE long_format_processed_data(
    timestamp REAL NOT NULL,
    basecur_id INTEGER NOT NULL,
    quotecur_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    type2_id INTEGER NOT NULL,
    value REAL,
    FOREIGN KEY(basecur_id) REFERENCES currencies(id),
    FOREIGN KEY(quotecur_id) REFERENCES currencies(id),
    FOREIGN KEY(type_id) REFERENCES types(id),
    FOREIGN KEY(type2_id) REFERENCES types2(id)
);

INSERT INTO long_format_processed_data(
        timestamp,
        basecur_id,
        quotecur_id,
        type_id,
        type2_id,
        value
    )
SELECT pd.timestamp,
    pd.basecur_id,
    pd.quotecur_id,
    pd.type_id,
    t2.id,
    (
        CASE
            t2.value
            WHEN 'open' THEN pd.open
            WHEN 'high' THEN pd.high
            WHEN 'low' THEN pd.low
            WHEN 'close' THEN pd.close
        END
    )
FROM processed_data pd,
    types2 t2;

CREATE UNIQUE INDEX ix_long_format_processed_data ON long_format_processed_data(
    timestamp,
    basecur_id,
    quotecur_id,
    type2_id,
    type_id
);

CREATE TABLE euro_only_processed_data(
    timestamp REAL NOT NULL,
    quotecur_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    type2_id INTEGER NOT NULL,
    value REAL,
    FOREIGN KEY(quotecur_id) REFERENCES currencies(id),
    FOREIGN KEY(type_id) REFERENCES types(id),
    FOREIGN KEY(type2_id) REFERENCES types2(id)
);

-- TODO: should we make this a view?
INSERT INTO euro_only_processed_data(
        timestamp,
        quotecur_id,
        type_id,
        type2_id,
        value
    )
SELECT pd1.timestamp,
    pd1.quotecur_id,
    pd1.type_id,
    pd1.type2_id,
    AVG(pd2.value)
FROM long_format_processed_data pd1
    INNER JOIN long_format_processed_data pd2 ON pd1.timestamp = pd2.timestamp
    AND pd1.basecur_id = pd2.basecur_id
    AND pd1.quotecur_id = pd2.quotecur_id
    AND pd1.type2_id = pd2.type2_id
    INNER JOIN currencies c ON pd1.basecur_id = c.id
WHERE c.value = 'eur'
GROUP BY pd1.timestamp,
    pd1.basecur_id,
    pd1.quotecur_id,
    pd1.type2_id,
    pd1.type_id;

CREATE UNIQUE INDEX ix_euro_only_processed_data ON euro_only_processed_data(
    timestamp,
    quotecur_id,
    type_id,
    type2_id
);

CREATE TABLE final_processed_data(
    timestamp REAL NOT NULL,
    basecur_id INTEGER NOT NULL,
    quotecur_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    type2_id INTEGER NOT NULL,
    value REAL NOT NULL,
    value2 REAL NOT NULL,
    FOREIGN KEY(basecur_id) REFERENCES currencies(id),
    FOREIGN KEY(quotecur_id) REFERENCES currencies(id),
    FOREIGN KEY(type_id) REFERENCES types(id),
    FOREIGN KEY(type2_id) REFERENCES types2(id)
);

INSERT INTO final_processed_data(
        timestamp,
        basecur_id,
        quotecur_id,
        type_id,
        type2_id,
        value,
        value2
    )
SELECT pd.timestamp,
    pd.basecur_id,
    pd.quotecur_id,
    pd.type_id,
    pd.type2_id,
    pd.value,
    (
        CASE
            WHEN epd.value IS NOT NULL THEN epd.value
            WHEN c.value IS 'eur' THEN 1.0
        END
    ) AS value2
FROM long_format_processed_data pd
    INNER JOIN currencies c ON c.id = pd.basecur_id
    LEFT JOIN euro_only_processed_data epd ON pd.timestamp = epd.timestamp
    AND pd.basecur_id = epd.quotecur_id
    AND pd.type2_id = epd.type2_id
    AND pd.type_id = epd.type_id;

SELECT COUNT(*)
FROM final_processed_data
WHERE value2 IS NULL;

/*

 strftime('%Y-%m-%dT%H:%M:%f', ...)

 SELECT
 strftime('%Y-%m-%dT%H:%M:%f', cast(r.timestamp * 48 AS INTEGER) / 48.0) AS ts,
 r.basecur_id,
 r.quotecur_id,
 r.type_id,
 first_value(r.value) OVER (ORDER BY r.timestamp) AS open,
 first_value(r.value) OVER (ORDER BY r.timestamp DESC) AS close,
 -- last_value(r.value) OVER (ORDER BY r.timestamp) AS close,
 max(r.value) AS high,
 min(r.value) AS low
 FROM (SELECT * FROM long_format_rawdata LIMIT 100000) r
 GROUP BY r.basecur_id, r.quotecur_id, ts, r.type_id
 */