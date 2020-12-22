import datetime
import os
import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

SAVE_DIR = "./tick_data"
FILE_PATTERN = re.compile(r"^(?P<basecur>\w{3})(?P<quotecur>\w{3})_201801-202001\.csv$")

sqlite3.register_adapter(pd.Timestamp, lambda x: x.isoformat())


@dataclass
class CsvFile:
    path: str
    basecur: str
    quotecur: str


csv_files = [
    CsvFile(os.path.join(SAVE_DIR, fname), m["basecur"], m["quotecur"])
    for fname in os.listdir(SAVE_DIR)
    if (m := FILE_PATTERN.match(fname))
]

basecurs = sorted((cur,) for cur in set(f.basecur for f in csv_files))
quotecurs = sorted((cur,) for cur in set(f.quotecur for f in csv_files))

epoch = datetime.datetime.fromtimestamp(0)

con = sqlite3.connect(os.path.join(SAVE_DIR, "201801-202001.db"))
cur = con.cursor()

# cur.executescript(
#     """
# DROP TABLE IF EXISTS basecurs;
# DROP TABLE IF EXISTS quotecurs;
# DROP TABLE IF EXISTS tmpdata;

# CREATE TABLE basecurs (
#     id INTEGER PRIMARY KEY,
#     value TEXT NOT NULL
# );

# CREATE TABLE quotecurs (
#     id INTEGER PRIMARY KEY,
#     value TEXT NOT NULL
# );

# CREATE TEMP TABLE tmpdata (
#     timestamp TEXT NOT NULL,
#     basecur TEXT NOT NULL,
#     quotecur TEXT NOT NULL,
#     sell REAL NOT NULL,
#     buy REAL NOT NULL
# );
# """
# )

# cur.executemany("INSERT INTO basecurs (value) VALUES (?)", basecurs)
# cur.executemany("INSERT INTO quotecurs (value) VALUES (?)", quotecurs)

cur.executescript(
    """
DROP TABLE IF EXISTS tmpdata;

CREATE TEMP TABLE tmpdata (
    timestamp TEXT NOT NULL,
    basecur TEXT NOT NULL,
    quotecur TEXT NOT NULL,
    sell REAL NOT NULL,
    buy REAL NOT NULL
);
"""
)

insert = """
INSERT INTO tmpdata (timestamp, basecur, quotecur, sell, buy)
VALUES (:timestamp, :basecur, :quotecur, :sell, :buy)
"""

for file in csv_files:
    reader = pd.read_csv(file.path, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2], chunksize=1e7)
    for df in reader:
        df["timestamp"] = pd.to_datetime(df.timestamp, format="%Y%m%d %H%M%S%f")
        df["basecur"] = file.basecur
        df["quotecur"] = file.quotecur

        cur.executemany(insert, df.to_dict(orient="records"))


cur.executescript(
    """
INSERT INTO rawdata (timestamp, basecur_id, quotecur_id, sell, buy)
SELECT t.timestamp, b.id, q.id, t.sell, t.buy
FROM tmpdata t
INNER JOIN basecurs b ON t.basecur = b.value
INNER JOIN quotecurs q ON t.quotecur = q.value;

DROP TABLE tmpdata;
"""
)

con.commit()
con.close()
