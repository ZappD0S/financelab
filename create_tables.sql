DROP TABLE IF EXISTS rawdata;
DROP TABLE IF EXISTS basecurs;
DROP TABLE IF EXISTS quotecurs;
DROP TABLE IF EXISTS tmpdata;

CREATE TABLE basecurs (
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE quotecurs (
    id INTEGER PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE rawdata (
timestamp TEXT NOT NULL,
basecur_id INTEGER NOT NULL,
quotecur_id INTEGER NOT NULL,
sell REAL,
buy REAL,
FOREIGN KEY(basecur_id) REFERENCES basecurs(id),
FOREIGN KEY(quotecur_id) REFERENCES quotecurs(id),
PRIMARY KEY(timestamp, basecur_id, quotecur_id)
);