import numpy as np
import pandas as pd
from datetime import datetime
import os
import re
import dask.dataframe as dd


SAVE_DIR = "./tick_data"
FILE_PATTERN = re.compile(r"(?P<pair>\w{6})_201801-202001\.csv")
PAIR_PATTERN = re.compile(r"^(?P<cur1>\w{3})(?P<cur2>\w{3})$")

# https://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex
# https://stackoverflow.com/questions/19798229/how-to-do-group-by-on-a-multiindex-in-pandas


pair_to_path = {
    m.group("pair"): os.path.join(SAVE_DIR, path) for path in os.listdir(SAVE_DIR) if (m := FILE_PATTERN.match(path))
}

dfs = []
pairs = []

for pair, path in pair_to_path.items():
    if not (m := PAIR_PATTERN.match(pair)):
        raise ValueError

    df = dd.read_csv(path, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2])
    df = df.head(n=10000)

    df["cur1"], df["cur2"] = m["cur1"], m["cur2"]

    dfs.append(df)
    pairs.append(pair)

df = dd.concat(dfs)
del dfs
df["timestamp"] = dd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S%f")

df = df.melt(id_vars=["timestamp", "cur1", "cur2"], value_vars=["sell", "buy"], var_name="type")
df = df.categorize(["cur1", "cur2", "type"])

df = (
    df.groupby([pd.Grouper(key="timestamp", freq="27min"), "cur1", "cur2", "type"])
    .value.agg(["first", "max", "min", "last"])
    .rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})
    .reset_index()
)

df["allnan"] = df[["open", "high", "low", "close"]].isna().all(axis=1)
df["allnan"] = df.groupby(["cur1", "cur2"]).allnan.transform("all", meta=df.allnan)

df = df[~df.allnan].drop("allnan", axis=1)


def f(x):
    x["close"] = x.close.ffill().bfill()
    x = x.fillna({col: x["close"] for col in ["open", "high", "low"]})
    return x


df = df.groupby(["cur1", "cur2", "type"]).apply(f, meta=df).reset_index()

df = df.melt(
    id_vars=["timestamp", "cur1", "cur2", "type"], value_vars=["open", "high", "low", "close"], var_name="type2"
)
df = df.categorize(columns="type2", index=False)
df = df.reset_index().set_index("index")


df2 = df.copy()

df3 = df2[df2["cur1"] == "eur"].drop(columns="cur1").rename(columns={"value": "value2", "cur2": "cur1"})

df3["value2"] = (
    df3.reset_index()[["index", "type", "value2"]]
    .pivot_table(index="index", columns="type", values="value2")
    .mean(axis=1)
    .reset_index()
    .set_index("index")
    .squeeze()
)

df2["cur1"] = df2.cur1.cat.as_unknown()
df3["cur1"] = df3.cur1.cat.as_unknown()

res = df2.merge(df3, on=["timestamp", "cur1", "type", "type2"], how="left")
res = res.categorize(columns="cur1", index=False)

res = res.set_index("timestamp")

res["value2"] = res["value2"].mask(res["cur1"] == "eur", 1.0)
