from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Iterator

import psycopg2
from dateutil.parser import parse
from waterstart.client.trader import TraderClient
from waterstart.schedule import ExecutionSchedule
from waterstart.symbols import SymbolInfo, SymbolsList, TradedSymbolInfo

START_DATE = parse("01/01/2018 17:00")
END_DATE = parse("31/01/2020 17:00")


async def main():
    client = await TraderClient.create(
        "demo.ctraderapi.com",
        5035,
        "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB",
        "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC",
        20783271,
        "rIZS2yjH62WzdY7DFg8t1uXdRRYZwJTRcwK6Q8s0ENc",
        "mfsUNsVxqreRgy9-0TIIrJqfeZnIyYCVLglv8U-c2qM",
    )

    try:
        conn = psycopg2.connect(database="marketdata", user="postgres", host="localhost")
        cur = conn.cursor()

        query = """
            SELECT name
            FROM traded_symbols
            JOIN symbol_names ON traded_symbols.sym_id = symbol_names.id
        """

        cur.execute(query)
        names = {name for [name] in cur}

        symlist = SymbolsList(client, 5)
        traded_syminfos = [syminfo async for syminfo in symlist.get_traded_sym_infos(names)]

        def get_syminfos(traded_syminfo: TradedSymbolInfo) -> Iterator[SymbolInfo]:
            yield traded_syminfo

            conv_chains = traded_syminfo.conv_chains
            for chain in (conv_chains.base_asset, conv_chains.quote_asset):
                yield from chain

        id_to_syminfo = {
            syminfo.id: syminfo for traded_syminfo in traded_syminfos for syminfo in get_syminfos(traded_syminfo)
        }

        trading_interval = timedelta(minutes=30)
        exec_shed = ExecutionSchedule(id_to_syminfo.values(), trading_interval, min_delta_to_close=timedelta(minutes=2))

        trading_time = exec_shed.next_valid_time(START_DATE)
        trading_times = []

        while trading_time < END_DATE:
            trading_times.append(trading_time)
            trading_time = exec_shed.next_valid_time(trading_time + trading_interval)

        cur.executemany(
            "INSERT INTO trading_times(timestamp) VALUES (TIMESTAMP %s)", ((str(t),) for t in trading_times)
        )
        conn.commit()
    finally:
        await client.close()


asyncio.run(main(), debug=True)
