DROP TABLE IF EXISTS rawdata;

DROP TABLE IF EXISTS currencies;

CREATE TABLE currencies (
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL UNIQUE
);

-- CREATE TABLE timestamps (
--     id INTEGER PRIMARY KEY,
--     value TEXT NOT NULL UNIQUE
-- );
-- CREATE TABLE rawdata (
-- basecur_id INTEGER NOT NULL,
-- quotecur_id INTEGER NOT NULL,
-- timestamp TEXT NOT NULL,
-- sell REAL,
-- buy REAL,
-- FOREIGN KEY(basecur_id, quotecur_id) REFERENCES currencies(id, id) ON DELETE CASCADE,
-- PRIMARY KEY(basecur_id, quotecur_id, timestamp) ON CONFLICT IGNORE
-- ) WITHOUT ROWID;
CREATE TABLE types(
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL UNIQUE
);

INSERT INTO types(value)
VALUES ('sell'),
    ('buy');

-- CREATE TABLE rawdata (
--     timestamp_id INTEGER NOT NULL,
--     basecur_id INTEGER NOT NULL,
--     quotecur_id INTEGER NOT NULL,
--     sell REAL,
--     buy REAL
-- );
CREATE TABLE rawdata (
    timestamp REAL NOT NULL,
    basecur_id INTEGER NOT NULL,
    quotecur_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY(basecur_id) REFERENCES currencies(id),
    FOREIGN KEY(quotecur_id) REFERENCES currencies(id),
    FOREIGN KEY(type_id) REFERENCES types(id)
);