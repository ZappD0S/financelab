CREATE TABLE flat_rawdata(timestamp, sym_id, price_type_id, value) AS WITH cte(id, timestamp, sym_id, bid, ask) AS (
    SELECT id,
        to_timestamp(datetime, 'YYYYMMDD HH24MISSMS')::TIMESTAMP,
        sym_id,
        bid,
        ask
    FROM rawdata
)
SELECT timestamp,
    sym_id,
    0::smallint,
    bid
FROM cte
UNION ALL
SELECT timestamp,
    sym_id,
    1::smallint,
    ask
FROM cte;

-- CREATE INDEX timestamp_sym_id_idx ON flat_rawdata (price_type_id, sym_id, timestamp);
-- this is the one with price and spread, find a better name..
CREATE TABLE rawdata AS(
    SELECT bid.timestamp,
        bid.sym_id,
        price_type.id AS price_type_id,
        CASE
            WHEN price_type.id = 1 THEN (bid.value + ask.value) / 2.0
            WHEN price_type.id = 2 THEN ask.value - bid.value
        END AS value
    FROM flat_rawdata bid
        JOIN flat_rawdata ask ON bid.timestamp = ask.timestamp
        AND bid.sym_id = ask.sym_id
        CROSS JOIN (
            VALUES (1),
                (2)
        ) price_type(id)
    WHERE bid.price_type_id = 0
        AND ask.price_type_id = 1
);

-- CREATE SEQUENCE flat_rawdata_id_seq;
-- CREATE TABLE flat_rawdata2(
--     id,
--     timestamp,
--     sym_id,
--     price_type_id,
--     value
-- ) AS (
--     SELECT nextval('flat_rawdata_id_seq')::int,
--         timestamp,
--         sym_id,
--         price_type_id,
--         value
--     FROM flat_rawdata
-- );
-- DROP TABLE flat_rawdata;
-- ALTER TABLE flat_rawdata2
--     RENAME TO flat_rawdata;
-- ALTER TABLE flat_rawdata
-- ALTER COLUMN id
-- SET DEFAULT nextval('flat_rawdata_id_seq');
-- ALTER TABLE flat_rawdata
-- ADD PRIMARY KEY (id);
CREATE TEMP VIEW count_duplicates AS
SELECT "timestamp",
    sym_id,
    price_type_id,
    row_number() OVER (PARTITION BY "timestamp", sym_id, price_type_id) AS rn
FROM flat_rawdata;

-- DELETE FROM flat_rawdata
-- WHERE id IN (
--         SELECT id
--         FROM count_duplicates
--         WHERE rn > 1
--     );
DELETE FROM flat_rawdata USING count_duplicates
WHERE flat_rawdata.timestamp = count_duplicates.timestamp
    AND flat_rawdata.sym_id = count_duplicates.sym_id
    AND flat_rawdata.price_type_id = count_duplicates.price_type_id
    AND count_duplicates.rn > 1;

-- CREATE UNIQUE INDEX ts_sym_type_idx ON flat_rawdata ("timestamp", sym_id, price_type_id) include(value);
CREATE INDEX aggregation_idx ON flat_rawdata (
    get_next_trading_ts("timestamp"),
    sym_id,
    price_type_id,
    "timestamp"
);

CREATE TABLE processed_data(
    timestamp,
    sym_id,
    price_type_id,
    open,
    high,
    low,
    close
) AS (
    SELECT ag.next_trading_ts,
        ag.sym_id,
        ag.price_type_id,
        o.value,
        hl.high,
        hl.low,
        c.value
    FROM aggreg_groups ag,
        LATERAL (
            SELECT value
            FROM flat_rawdata
            WHERE get_next_trading_ts(timestamp) = ag.next_trading_ts
                AND sym_id = ag.sym_id
                AND price_type_id = ag.price_type_id
            ORDER BY timestamp
            LIMIT 1
        ) o, LATERAL (
            SELECT MAX(value) AS high,
                MIN(value) AS low
            FROM flat_rawdata
            WHERE get_next_trading_ts(timestamp) = ag.next_trading_ts
                AND sym_id = ag.sym_id
                AND price_type_id = ag.price_type_id
        ) hl,
        LATERAL (
            SELECT value
            FROM flat_rawdata
            WHERE get_next_trading_ts(timestamp) = ag.next_trading_ts
                AND sym_id = ag.sym_id
                AND price_type_id = ag.price_type_id
            ORDER BY timestamp DESC
            LIMIT 1
        ) c
);

-- CREATE SEQUENCE trendbar_parts_id_seq;
-- DROP TABLE IF EXISTS trendbar_parts;
-- CREATE TABLE trendbar_parts(id, name) AS (
--     SELECT nextval('trendbar_parts_id_seq')::smallint,
--         name
--     FROM (
--             VALUES ('open'),
--                 ('high'),
--                 ('low'),
--                 ('close')
--         ) AS names(name)
-- );
-- ALTER TABLE trendbar_parts ALTER id
-- SET DEFAULT nextval('trendbar_parts_id_seq');
-- CREATE TABLE flat_processed_data(
--         timestamp,
--         sym_id,
--         price_type_id,
--         tb_part_id,
--         value
--     ) AS
-- SELECT pd.timestamp,
--     pd.sym_id,
--     pd.price_type_id,
--     tbp.id,
--     (
--         CASE
--             tbp.name
--             WHEN 'open' THEN pd.open
--             WHEN 'high' THEN pd.high
--             WHEN 'low' THEN pd.low
--             WHEN 'close' THEN pd.close
--         END
--     )
-- FROM processed_data pd,
--     trendbar_parts tbp;
SELECT price.open AS price_open,
    price.high AS price_high,
    price.low AS price_low,
    price.close AS price_close,
    spread.open AS spread_open,
    spread.high AS spread_high,
    spread.low AS spread_low,
    spread.close AS spread_close,
    CASE
        WHEN tcs.base_conv_id = -1 THEN 1.0
        WHEN tcs.base_conv_inverted THEN 1 / base_conv.open
        ELSE base_conv.open
    END AS base_conv_open,
    CASE
        WHEN tcs.base_conv_id = -1 THEN 1.0
        WHEN tcs.base_conv_inverted THEN 1 / base_conv.high
        ELSE base_conv.high
    END AS base_conv_high,
    CASE
        WHEN tcs.base_conv_id = -1 THEN 1.0
        WHEN tcs.base_conv_inverted THEN 1 / base_conv.low
        ELSE base_conv.low
    END AS base_conv_low,
    CASE
        WHEN tcs.base_conv_id = -1 THEN 1.0
        WHEN tcs.base_conv_inverted THEN 1 / base_conv.close
        ELSE base_conv.close
    END AS base_conv_close,
    CASE
        WHEN tcs.quote_conv_id = -1 THEN 1.0
        WHEN tcs.quote_conv_inverted THEN 1 / quote_conv.open
        ELSE quote_conv.open
    END AS quote_conv_open,
    CASE
        WHEN tcs.quote_conv_id = -1 THEN 1.0
        WHEN tcs.quote_conv_inverted THEN 1 / quote_conv.high
        ELSE quote_conv.high
    END AS quote_conv_high,
    CASE
        WHEN tcs.quote_conv_id = -1 THEN 1.0
        WHEN tcs.quote_conv_inverted THEN 1 / quote_conv.low
        ELSE quote_conv.low
    END AS quote_conv_low,
    CASE
        WHEN tcs.quote_conv_id = -1 THEN 1.0
        WHEN tcs.quote_conv_inverted THEN 1 / quote_conv.close
        ELSE quote_conv.close
    END AS quote_conv_close,
    EXTRACT(
        EPOCH
        FROM price.timestamp - date_trunc('day', price.timestamp)
    ) / EXTRACT(
        EPOCH
        FROM INTERVAL '1 day'
    ) AS time_of_day,
    EXTRACT(
        EPOCH
        FROM price.timestamp - prev.timestamp
    ) / EXTRACT(
        EPOCH
        FROM INTERVAL '1 day'
    ) AS delta_to_last
FROM processed_data price
    JOIN processed_data spread ON price.sym_id = spread.sym_id
    AND price.timestamp = spread.timestamp
    JOIN traded_and_conv_symbols tcs ON tcs.traded_id = price.sym_id
    LEFT JOIN processed_data base_conv ON base_conv.sym_id = tcs.base_conv_id
    AND price."timestamp" = base_conv."timestamp"
    AND price.price_type_id = base_conv.price_type_id
    LEFT JOIN processed_data quote_conv ON quote_conv.sym_id = tcs.quote_conv_id
    AND price."timestamp" = quote_conv."timestamp"
    AND price.price_type_id = quote_conv.price_type_id
    CROSS JOIN LATERAL (
        SELECT timestamp
        FROM processed_data
        WHERE sym_id = price.sym_id
            AND price_type_id = price.price_type_id
            AND "timestamp" < price."timestamp"
        ORDER BY "timestamp" DESC
        LIMIT 1
    ) prev
WHERE price.price_type_id = 1
    AND spread.price_type_id = 2;