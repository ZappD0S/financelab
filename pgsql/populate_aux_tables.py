from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterator

import psycopg2
from waterstart.client.trader import TraderClient
from waterstart.openapi import ProtoOAAssetListReq
from waterstart.openapi.OpenApiMessages_pb2 import ProtoOAAssetListRes
from waterstart.symbols import SymbolInfo, SymbolsList, TradedSymbolInfo


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

    conn = psycopg2.connect(database="marketdata", user="postgres", host="localhost")

    try:
        asset_list_res = await client.send_request_from_trader(
            lambda trader_id: ProtoOAAssetListReq(ctidTraderAccountId=trader_id), ProtoOAAssetListRes
        )

        asset_id_and_name = [(asset.assetId, asset.name) for asset in asset_list_res.asset]
        asset_id_to_name = dict(asset_id_and_name)
        asset_name_to_id = {name: id for id, name in asset_id_and_name}

        dep_asset_id = asset_name_to_id["EUR"]

        symlist = SymbolsList(client, dep_asset_id)
        with open("traded_symbols.json") as f:
            traded_symbols = json.load(f)

        traded_syminfos = [syminfo async for syminfo in symlist.get_traded_sym_infos(set(traded_symbols))]

        # print(
        #     [
        #         base_asset[0].reciprocal
        #         for traded_syminfo in traded_syminfos
        #         if (base_asset := traded_syminfo.conv_chains.base_asset)
        #     ]
        # )
        # print(
        #     [
        #         quote_asset[0].reciprocal
        #         for traded_syminfo in traded_syminfos
        #         if (quote_asset := traded_syminfo.conv_chains.quote_asset)
        #     ]
        # )

        def get_syminfos(traded_syminfo: TradedSymbolInfo) -> Iterator[SymbolInfo]:
            yield traded_syminfo

            conv_chains = traded_syminfo.conv_chains
            for chain in (conv_chains.base_asset, conv_chains.quote_asset):
                yield from chain

        id_to_syminfo = {
            syminfo.id: syminfo for traded_syminfo in traded_syminfos for syminfo in get_syminfos(traded_syminfo)
        }

        traded_asset_id_to_name = {
            asset_id: asset_id_to_name[asset_id]
            for syminfo in id_to_syminfo.values()
            for asset_id in (syminfo.light_symbol.baseAssetId, syminfo.light_symbol.quoteAssetId)
        }

        cur = conn.cursor()
        cur.executemany("INSERT INTO assets VALUES (%s, LOWER(%s))", traded_asset_id_to_name.items())

        cur.executemany(
            "INSERT INTO symbols VALUES (%s, %s, %s)",
            (
                (id, syminfo.light_symbol.baseAssetId, syminfo.light_symbol.quoteAssetId)
                for id, syminfo in id_to_syminfo.items()
            ),
        )

        cur.executemany(
            "INSERT INTO traded_symbols(sym_id) VALUES (%s)", ((syminfo.id,) for syminfo in traded_syminfos)
        )
        conn.commit()

    finally:
        await client.close()


logging.getLogger("asyncio").setLevel(logging.DEBUG)
asyncio.run(main(), debug=True)
