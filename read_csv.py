import os
import re
from math import ceil

import dask
import dask.dataframe as dd
import IPython.core.debugger as ipdb
import pandas as pd

SAVE_DIR = "./tick_data"
FILE_PATTERN = re.compile(r"^(?P<cur1>\w{3})(?P<cur2>\w{3})_201801-202001\.csv$")
dask.config.set({"temporary_directory": "/var/tmp"})
dask.config.set({"dataframe.shuffle-compression": "Blosc"})

pair_to_path = {
    (m["cur1"], m["cur2"]): os.path.join(SAVE_DIR, path)
    for path in os.listdir(SAVE_DIR)
    if (m := FILE_PATTERN.match(path))
}


dfs = []
cur1_values = set()
cur2_values = set()

for (cur1, cur2), path in pair_to_path.items():
    df = dd.read_csv(path, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2])
    # TODO: to remove
    # df = df.head(n=10_000)

    df["cur1"], df["cur2"] = cur1, cur2

    dfs.append(df)
    cur1_values.add(cur1)
    cur2_values.add(cur2)

df = dd.concat(dfs)

df = df.repartition(npartitions=2400)

df["timestamp"] = dd.to_datetime(df.timestamp, format="%Y%m%d %H%M%S%f")
df["cur1"] = df.cur1.astype(pd.CategoricalDtype(cur1_values))
df["cur2"] = df.cur2.astype(pd.CategoricalDtype(cur2_values))

df = df.melt(id_vars=["timestamp", "cur1", "cur2"], value_vars=["sell", "buy"], var_name="type")

# df = df.categorize(["cur1", "cur2", "type"], index=False)
df["type"] = df.type.astype(pd.CategoricalDtype(["sell", "buy"]))

df = df.repartition(npartitions=1200)

df = df.set_index("timestamp")

# df.to_parquet(os.path.join(SAVE_DIR, "201801-202001"), engine="fastparquet", compute=True)
# raise Exception

def f(x):
    # return x.set_index("timestamp").squeeze().resample("30min").ohlc()
    return x.resample("30min").ohlc()


meta = df._meta_nonempty
meta = (
    meta.assign(**{col: meta.value for col in ["open", "high", "low", "close"]})
    .drop(columns="value")
    .reset_index()
    .set_index(["cur1", "cur2", "type", "timestamp"])
)

# df = df.groupby(["cur1", "cur2", "type"])[["timestamp", "value"]].apply(f, meta=meta).reset_index()
df = df.groupby(["cur1", "cur2", "type"]).value.apply(f, meta=meta).reset_index()


df = df.repartition(npartitions=3)

ipdb.set_trace()

min_last_t = df.groupby(["cur1", "cur2", "type"]).timestamp.agg("max").min()

df = df[df.timestamp <= min_last_t]


# df = (
#     df.groupby([pd.Grouper(key="timestamp", freq="30min"), "cur1", "cur2", "type"])
#     .value.agg(["first", "max", "min", "last"])
#     .rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})
#     .reset_index()
# )

# df["allnan"] = df[["open", "high", "low", "close"]].isna().all(axis=1)
# df["allnan"] = df.groupby(["cur1", "cur2"]).allnan.transform("all", meta=df.allnan)

# df = df[~df.allnan].drop(columns="allnan")


def f(x):
    # x["close"] = x.close.ffill().bfill()
    # return x.fillna({col: x.close for col in ["open", "high", "low"]})

    x["open"] = x.open.bfill()
    x["close"] = x.close.ffill()
    x = x.fillna({col: x.open for col in ["high", "low", "close"]})
    x = x.fillna({col: x.close for col in ["open", "high", "low"]})
    return x


df = df.groupby(["cur1", "cur2", "type"]).apply(f, meta=df).reset_index()

df = df.melt(
    id_vars=["timestamp", "cur1", "cur2", "type"], value_vars=["open", "high", "low", "close"], var_name="type2"
)
df = df.repartition(npartitions=df.npartitions * 2)

# df = df.reset_index().set_index("index")
df = df.reset_index(drop=True)
# df = df.categorize(columns="type2", index=False)
df["type2"] = df.type2.astype(pd.CategoricalDtype(["open", "high", "low", "close"]))


# value2 = (
#     df.loc[df.cur1 == "eur", ["type", "value"]]
#     .reset_index()
#     .pivot_table(index="index", columns="type", values="value")
#     .mean(axis=1)
#     .rename("value2")
# )

# res = df.join(value2)

df2 = df[df.cur1 == "eur"].drop(columns="cur1").rename(columns={"value": "value2", "cur2": "cur1"})

df2 = df2.repartition(npartitions=ceil(5 / 13 * df2.npartitions))

df2 = df2.drop(columns="value2").join(
    df2.groupby(["timestamp", "cur1", "type2"]).value2.agg("mean"), on=["timestamp", "cur1", "type2"]
)

# df2["value2"] = (
#     df2[["type", "value2"]]
#     .reset_index()
#     .pivot_table(index="index", columns="type", values="value2")
#     .mean(axis=1)
#     .reset_index()
#     .set_index("index")
#     .squeeze()
# )

cur1_dtype = df.cur1.dtype
df["cur1"] = df.cur1.cat.as_unknown()
df2["cur1"] = df2.cur1.cat.as_unknown()

res = df.merge(df2, on=["timestamp", "cur1", "type", "type2"], how="left")
# res = res.categorize(columns="cur1", index=False)
res["cur1"] = res.cur1.astype(cur1_dtype)

# res = res.set_index("timestamp")

res["value2"] = res.value2.mask(res.cur1 == "eur", 1.0)

# res.to_hdf(os.path.join(SAVE_DIR, "201801-202001.h5"), "df", complib="blosc:blosclz", compute=True)
res.to_hdf(os.path.join(SAVE_DIR, "201801-202001.h5"), "df", compute=True)
