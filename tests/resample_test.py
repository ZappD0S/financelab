import pandas as pd
import dask.dataframe as dd
import random

index = pd.date_range(start="2020-01-01 3:14", freq="5min", periods=500, name="timestamp")

columns = ["type", "value"]

random.seed(1234)

data = {
    "type1": random.choices("abc", k=500),
    "type2": random.choices("def", k=500),
    "value": [random.random() for _ in range(500)]
}

df = pd.DataFrame(data, index=index)
# df["type"] = df.type.astype("category")
df["type1"] = df.type1.astype("category")
df["type2"] = df.type2.astype("category")

raise Exception
df = dd.from_pandas(df, npartitions=2)

def chunk(grouped):
    return grouped.max(), grouped.min()

def agg(chunk_maxes, chunk_mins):
    return chunk_maxes.max(), chunk_mins.min()

def finalize(maxima, minima):
    finalize.maxima, finalize.minima = maxima, minima
    # return maxima - minima
    return pd.DataFrame({"max": maxima, "min": minima})

def f(x):
    # x = x.reset_index().pivot(index="timestamp", values="value", columns=["type1", "type2"])
    # x = x.resample("27min").ohlc()
    # x.columns.names = ["type1", "type2", "type3"]
    # return x.stack(level=[0, 1]).reset_index(level=[1, 2])
    return x.value.resample("27min").ohlc()

# extent = dd.Aggregation('extent', chunk, agg, finalize=finalize)

# df.pivot_table()

# df2 = df.groupby("type").agg(extent).compute()
# df2 = df.groupby(["type1", "type2"]).apply(f).compute()
df2 = df.groupby(["type1", "type2"]).apply(f).reset_index().set_index("timestamp").compute()
# df2 = df.groupby("type").resample("27min").mean()
# df2 = df.groupby("type").apply(lambda x: x.resample("27min").mean())

# df2 = df2.swaplevel(0, 1).sort_index(level=0)

# df2 = df2.reset_index(level=0)
# df2.index = df2.index.rename("timestamp")
# df2 = df2.sort_values(["timestamp", "type"])
# df2 = df2.compute()
