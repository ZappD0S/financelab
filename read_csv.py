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

    # df["pair"] = pair
    df["cur1"], df["cur2"] = m["cur1"], m["cur2"]
    # df = df.rename(columns={"sell": f"{pair}_sell", "buy": f"{pair}_buy"})

    dfs.append(df)
    pairs.append(pair)

df = dd.concat(dfs)
del dfs
df["timestamp"] = dd.to_datetime(df["timestamp"], format="%Y%m%d %H%M%S%f")
# df["pair"] = df.pair.astype(dd.categorical.pd.CategoricalDtype(pairs))
# df = df.drop_duplicates(subset=["pair", "timestamp"])

# df = df.melt(id_vars=["timestamp", "pair"], value_vars=["sell", "buy"], var_name="type")
# df = df.categorize(["pair", "type"])
df = df.melt(id_vars=["timestamp", "cur1", "cur2"], value_vars=["sell", "buy"], var_name="type")
df = df.categorize(["cur1", "cur2", "type"])

df = df.set_index("timestamp")

df = (
    df.groupby([pd.Grouper(freq="27min"), "cur1", "cur2", "type"])
    .value.agg(["first", "max", "min", "last"])
    .rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})
    .reset_index()
)

# df = df.groupby([pd.Grouper(freq="27min"), "cur1", "cur2", "type"]).apply({"open"})
# df = df.groupby(["cur1", "cur2", "type"]).value.apply(lambda x: x.resample("27min").ohlc(), meta=meta).reset_index()

df = df.melt(
    id_vars=["timestamp", "cur1", "cur2", "type"], value_vars=["open", "high", "low", "close"], var_name="type2"
)
df = df.set_index("timestamp")
df = df.categorize(columns="type2", index=False)

meta = (
    pd.DataFrame(columns=["timestamp", "value"])
    .astype({"timestamp": "datetime64[ns]", "value": "bool"})
    .set_index("timestamp")
    .squeeze()
)

mask = df.groupby(["cur1", "cur2"]).value.transform(lambda x: not x.isna().all(), meta=meta)
# NOTE: this is necessary to have known_partitions == True in mask
mask = mask.reset_index().set_index("timestamp").squeeze()
df = df[mask]

# df2 = df.compute()
df2 = df.copy()


def f(x):
    x = x.pivot(columns="type2", values="value")
    x["close"] = x.close.ffill().bfill()
    x = x.fillna({col: x["close"] for col in ["high", "low", "open"]})

    return x.melt(value_vars=["open", "high", "low", "close"], ignore_index=False)


meta = df2.head(n=1).reset_index().set_index(["cur1", "cur2", "type", "timestamp"])

df2 = df2.groupby(["cur1", "cur2", "type"])[["type2", "value"]].apply(f, meta=meta).reset_index()
# df2 = df2.groupby(["cur1", "cur2", "type"]).apply(f).reset_index()

df3 = df2[df2["cur1"] == "eur"].drop(columns="cur1").rename(columns={"value": "value2", "cur2": "cur1"})
df3 = df3.reset_index().set_index("index")

# meta = df3.head(n=1).set_index(["cur1", "type2"])
# df3.groupby(["cur1", "type2"])[["timestamp", "type", "value2"]].apply(f2, meta=meta)

midpoint = (
    df3.reset_index()[["index", "type", "value2"]]
    .pivot_table(index="index", columns="type", values="value2")
    .mean(axis=1)
    .reset_index()
    .set_index("index")
    .squeeze()
)


# midpoint = (
#     df3[["timestamp", "type", "value2"]]
#     .pivot_table(index="timestamp", columns="type", values="value2")
#     .mean(axis=1)
#     .reset_index()
#     .set_index("timestamp")
#     .squeeze()
# )
# df3 = df3.set_index("timestamp")
df3["value2"] = midpoint

res = df2.merge(df3, on=["timestamp", "cur1", "type", "type2"], how="left")
res["cur1"] = res["cur1"].astype("category")

# res = res.set_index("timestamp").sort_index()
res = res.set_index("timestamp")

res["value2"] = res["value2"].mask(res["cur1"] == "eur", 1.0)

# # or reorder_levels
# df.columns = df.columns.swaplevel(0, 1)
# _, indexer = df.columns.sort_values(return_indexer=True)
# df = df.iloc[:, indexer]

# df.drop_duplicates(subset=["pair", "timestamp"])


# df = df[~df.index.duplicated()]
# df.to_pickle(f"tick_data/{PAIR}_{YEAR}.pkl")
