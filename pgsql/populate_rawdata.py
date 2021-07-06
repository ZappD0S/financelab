import io
import os
import re

import psycopg2

CHUNK_SIZE = 10 ** 6

SAVE_DIR = "./tick_data"
FILE_PATTERN = re.compile(r"(?P<symbol>\w{6})_201801-202001\.csv")


conn = psycopg2.connect(database="marketdata", user="postgres", host="localhost")
cur = conn.cursor()


def copy_chunk(stream):
    stream.seek(0)
    cur.copy_from(stream, "rawdata", sep=",", columns=("sym_id", "datetime", "bid", "ask", "ignore"))


def copy_file(file: io.TextIOWrapper, sym_id: str) -> None:
    count = 0
    i = 0
    stream = io.StringIO()
    stream_write = stream.write

    for line in file:
        stream_write(sym_id + "," + line)

        if i % CHUNK_SIZE == 0:
            print(count)
            count += 1

            copy_chunk(stream)
            stream = io.StringIO()
            stream_write = stream.write

        i += 1

    copy_chunk(stream)


sym_name_to_path: dict[str, str] = {
    m["symbol"]: os.path.join(SAVE_DIR, fname)
    for fname in os.listdir(SAVE_DIR)
    if (m := FILE_PATTERN.match(fname)) is not None
}


with conn:
    cur.execute("SELECT id, name FROM symbol_names")

    for sym_id, sym_name in list(cur):
        path = sym_name_to_path[sym_name]

        with open(path) as f:
            copy_file(f, str(sym_id))
