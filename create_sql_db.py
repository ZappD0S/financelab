import itertools
import os
import re
import sqlite3
from dataclasses import dataclass
import ciso8601

import pandas as pd

SAVE_DIR = "./tick_data"
FILE_PATTERN = re.compile(r"^(?P<basecur>\w{3})(?P<quotecur>\w{3})_201801-202001\.csv$")

sqlite3.register_adapter(pd.Timestamp, lambda x: x.isoformat())


@dataclass
class CsvInfo:
    path: str
    basecur: str
    quotecur: str


csv_infos = [
    CsvInfo(os.path.join(SAVE_DIR, fname), m["basecur"], m["quotecur"])
    for fname in os.listdir(SAVE_DIR)
    if (m := FILE_PATTERN.match(fname))
]

currencies = sorted(set(itertools.chain.from_iterable((info.basecur, info.quotecur) for info in csv_infos)))

con = sqlite3.connect(os.path.join(SAVE_DIR, "201801-202001.db"), isolation_level=None)

con.executescript(
    """
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = OFF;
PRAGMA temp_store = MEMORY;
"""
)

cur = con.execute("BEGIN")

cur.execute("DELETE FROM currencies")
cur.executemany("INSERT INTO currencies (value) VALUES (?)", ((c,) for c in currencies))

# con.executescript(
#     """
# DROP TABLE IF EXISTS tmpdata;

# CREATE TEMP TABLE tmpdata (
#     timestamp TEXT NOT NULL,
#     basecur TEXT NOT NULL,
#     quotecur TEXT NOT NULL,
#     sell REAL NOT NULL,
#     buy REAL NOT NULL
# );
# """
# )

# insert = """
# INSERT INTO rawdata (timestamp, basecur_id, quotecur_id, sell, buy)
# SELECT v.timestamp, c1.id, c2.id, v.sell, v.buy FROM
# (SELECT :timestamp as timestamp, :basecur as basecur, :quotecur as quotecur, :sell as sell, :buy as buy) v
# INNER JOIN currencies c1 ON c1.value = v.basecur
# INNER JOIN currencies c2 ON c2.value = v.quotecur
# """

# dt_format = "%Y%m%d %H%M%S%f"
# for i, info in enumerate(csv_infos):
#     print(i)
#     with open(info.path, newline="") as csvfile:
#         csvreader = csv.DictReader(csvfile, ("timestamp", "sell", "buy"))
#         for j, chunk in enumerate(chunked(csvreader, int(1e6))):
#             print(f"\t{j}")
#             records = [
#                 {
#                     "timestamp": datetime.datetime.strptime(row["timestamp"], dt_format),
#                     "basecur": info.basecur,
#                     "quotecur": info.quotecur,
#                     "sell": row["sell"],
#                     "buy": row["buy"],
#                 }
#                 for row in chunk
#             ]

#             print("\tdata ready!")
#             con.executemany(insert, records)

for i, info in enumerate(csv_infos):
    reader = pd.read_csv(info.path, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2], chunksize=1e6)
    print(i)
    for j, df in enumerate(reader):
        print(f"\t{j}")
        df["timestamp"] = df.timestamp.map(lambda dt: ciso8601.parse_datetime(f"{dt[:15]}.{dt[15:]}"))
        df["basecur"] = info.basecur
        df["quotecur"] = info.quotecur
        df = df.melt(id_vars=["timestamp", "basecur", "quotecur"], value_vars=["sell", "buy"], var_name="type")
        df = df.reindex(columns=["timestamp", "basecur", "quotecur", "type", "value"])

        cur.executemany(
            """
            INSERT INTO rawdata(timestamp_id, basecur_id, quotecur_id, type_id, value)
            SELECT v.timestamp, b.id, q.id, t.id, v.value
            FROM (
                SELECT
                julianday(?) AS timestamp,
                ? AS basecur,
                ? AS quotecur,
                ? AS type,
                ? AS value
            ) v
            INNER JOIN currencies b ON b.value = v.basecur
            INNER JOIN currencies q ON q.value = v.quotecur
            INNER JOIN types t ON t.value = v.type
            """,
            df.itertuples(index=False, name=None),
        )

        # cur.executemany(
        #     "INSERT INTO timestamps(value) VALUES (?) ON CONFLICT DO NOTHING", ((dt,) for dt in df.timestamp)
        # )
        # cur.executemany(
        #     """
        #     INSERT INTO rawdata(timestamp_id, basecur_id, quotecur_id, sell, buy)
        #     SELECT t.id, b.id, q.id, v.sell, v.buy
        #     FROM (
        #         SELECT ? AS timestamp,
        #         ? AS basecur,
        #         ? AS quotecur,
        #         ? AS sell,
        #         ? AS buy
        #     ) v
        #     INNER JOIN currencies b ON b.value = v.basecur
        #     INNER JOIN currencies q ON q.value = v.quotecur
        #     INNER JOIN timestamps t ON t.value = v.timestamp
        #     """,
        #     df.itertuples(index=False, name=None),
        # )

cur.execute("COMMIT")

con.close()
