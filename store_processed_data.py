import sqlite3
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def create_map_with_holes(n: int, holes: List[Union[int, Tuple[int, int]]], return_holes_map: bool = False):
    hole_lengths = {}

    for hole in holes:
        if isinstance(hole, int):
            pos, length = hole, 1
        elif isinstance(hole, tuple) and len(hole) == 2:
            pos, length = hole
        else:
            raise Exception

        if pos + length >= n:
            raise Exception

        if pos in hole_lengths:
            raise Exception

        hole_lengths[pos] = length

    inds = []
    hole_inds = []

    i = 0
    while i < n:
        if (length := hole_lengths.get(i)) :
            if return_holes_map:
                hole_inds += [i + j for j in range(length)]

            i += length
            continue

        inds.append(i)
        i += 1

    result = {i: val for i, val in enumerate(inds)}

    if return_holes_map:
        return result, {i: val for i, val in enumerate(hole_inds)}
    else:
        return result


con = sqlite3.connect("tick_data/201801-202001.db")
con.row_factory = sqlite3.Row

types_map = {row["value"]: row["id"] - 1 for row in con.execute("SELECT * FROM types")}
types2_map = {row["value"]: row["id"] - 1 for row in con.execute("SELECT * FROM types2")}
currencies_map = {row["id"] - 1: row["value"] for row in con.execute("SELECT * FROM currencies")}

query = """
SELECT
    strftime('%Y-%m-%dT%H:%M:%f', timestamp) as timestamp,
    basecur_id,
    quotecur_id,
    type_id,
    type2_id,
    value,
    value2
FROM final_processed_data;
"""

df = pd.read_sql_query(query, con)
df[["basecur_id", "quotecur_id", "type_id", "type2_id"]] -= 1
df["timestamp"] = pd.to_datetime(df.timestamp)
df["timestamp_id"] = df.timestamp.factorize(sort=True)[0]

df["cur_id"], unique_sorted_ids = pd.factorize(df[["basecur_id", "quotecur_id"]].to_records(index=False), sort=True)
sorted_pairs = np.array(
    [currencies_map[basecur_id] + currencies_map[quotecur_id] for basecur_id, quotecur_id in unique_sorted_ids]
)

df = df.drop(columns=["basecur_id", "quotecur_id"])

df2 = df[df.type2_id == types2_map["close"]].drop(columns=["timestamp", "type2_id", "value2"])

shape = tuple(df2[["timestamp_id", "cur_id", "type_id"]].max() + 1)

inds = tuple(df2[["timestamp_id", "cur_id", "type_id"]].values.T)
values = df2["value"].values

arr = np.full(shape, np.nan)
arr[inds] = values
assert not np.isnan(arr).any()


df2 = df[(df.type2_id == types2_map["close"]) & (df.type_id == types_map["buy"])]
df2 = df2.drop(columns=["timestamp", "type2_id", "type_id", "value"])

shape = tuple(df2[["timestamp_id", "cur_id"]].max() + 1)

inds = tuple(df2[["timestamp_id", "cur_id"]].values.T)
values = df2["value2"].values
arr2 = np.full(shape, np.nan)
arr2[inds] = values
assert not np.isnan(arr2).any()


df_delta = df[["timestamp_id", "timestamp"]].drop_duplicates("timestamp_id").sort_values("timestamp")
df_delta["value"] = df_delta.timestamp.diff()
del df_delta["timestamp"]
default_value = df_delta.value.mode().item()
df_delta["value"] = df_delta.value.fillna(default_value)
df_delta["value"] = (df_delta.value - df_delta.value.mean()) / df_delta.value.std()

del df["timestamp"]

df = df.rename(columns={"value": "value1"}).melt(
    id_vars=["timestamp_id", "cur_id", "type_id", "type2_id"],
    value_vars=["value1", "value2"],
    value_name="value",
    var_name="value_id",
)

# df["value_id"] = df.value_id.factorize(sort=True)[0]
df["value_id"] = df.value_id.map({"value1": 0, "value2": 1})
assert not df.value_id.isna().any().item()

df = (
    df.pivot_table(values="value", index=["timestamp_id", "cur_id", "type2_id", "value_id"], columns="type_id")
    .rename(columns={v: k for k, v in types_map.items()})
    .reset_index()
)

new_id_map = {types2_map[key]: i for i, key in enumerate(["high", "low", "close"])}

# mask = df.type2_id.isin([types2_map[key] for key in ["high", "low", "close"]])
# mask = df.type2_id.isin(new_id_map.keys())
# df_avg = df[mask]
df_avg = df.assign(type2_id=df.type2_id.map(new_id_map)).dropna(subset=["type2_id"]).astype({"type2_id": int})
df_avg["value"] = df_avg[["buy", "sell"]].mean(axis=1)
df_avg = df_avg.drop(columns=["buy", "sell"])
df_avg["flat_id"] = pd.factorize(df_avg[["value_id", "type2_id"]].to_records(index=False), sort=True)[0]
df_avg = df_avg.drop(columns=["type2_id", "value_id"])

df_spread = df[df.type2_id == types2_map["close"]].drop(columns="type2_id")
df_spread["value"] = df_spread.buy - df_spread.sell
df_spread = df_spread.drop(columns=["buy", "sell"])
df_spread["flat_id"] = df_spread.value_id + df_avg.flat_id.max() + 1
del df_spread["value_id"]

df_delta = df[["timestamp_id", "cur_id"]].drop_duplicates().merge(df_delta, on="timestamp_id")
df_delta["flat_id"] = df_spread.flat_id.max() + 1

df = pd.concat([df_avg, df_spread, df_delta], ignore_index=True)

shape = tuple(df[["timestamp_id", "cur_id", "flat_id"]].max() + 1)

inds = tuple(df[["timestamp_id", "cur_id", "flat_id"]].values.T)
values = df["value"].values

arr3 = np.full(shape, np.nan)
arr3[inds] = values

assert not np.isnan(arr3).any()
assert np.all((arr3[..., 1] <= arr3[..., 2]) & (arr3[..., 2] <= arr3[..., 0]))
assert np.all((arr3[..., 4] <= arr3[..., 5]) & (arr3[..., 5] <= arr3[..., 3]))


np.savez_compressed("train_data/train_data.npz", arr=arr, arr2=arr2, arr3=arr3, sorted_pairs=sorted_pairs)

# TODO: we should add a column that is normally 0 and increseases gradually to 1
# in the last n timestemps of trading days.
