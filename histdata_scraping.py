import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator, List

import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from dateutil.rrule import MONTHLY, rrule

SAVE_PATH = "./tick_data"

START_DATE = parse("01/2018")
END_DATE = parse("01/2020")

HISTDATA_URL = "https://www.histdata.com"
url_template = HISTDATA_URL + "/download-free-forex-historical-data/?/ascii/tick-data-quotes/{sym}/{dt.year}/{dt.month}"
POST_URL = HISTDATA_URL + "/get.php"


@contextmanager
def get_month_data(url: str) -> Iterator[io.BufferedIOBase]:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    form_params = soup.find("form", id="file_down").find_all("input")

    payload = {param["name"]: param["value"] for param in form_params}
    r = requests.post(POST_URL, payload, headers={"Referer": url})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = next(name for name in z.namelist() if name.endswith(".csv"))
    with z.open(fname) as f:
        yield f


def download_symbol(save_dir: str, sym: str, dts: List[datetime]) -> None:
    urls = (url_template.format(sym=sym, dt=dt) for dt in dts)
    date_range = f"{dts[0]:%Y%m}-{dts[-1]:%Y%m}"
    fname = os.path.join(save_dir, f"{sym}_{date_range}.csv")
    with open(fname, "xb") as outfile:
        for url in urls:
            with get_month_data(url) as infile:
                for line in infile:
                    if not line.isspace():
                        outfile.write(line)


def symbols_unique(symbols: Iterable[str]) -> bool:
    seen = set()
    for sym in symbols:
        if sym in seen:
            return False

        seen.add(sym)

    return True


def download_symbols(symbols: Iterable[str], save_dir: str) -> None:
    dts = list(rrule(MONTHLY, dtstart=START_DATE, until=END_DATE))
    assert symbols_unique(symbols)

    with ThreadPoolExecutor() as executor:
        future_to_sym = {executor.submit(download_symbol, save_dir, sym, dts): sym for sym in symbols}

        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]

            try:
                future.result()
                print(f"{sym} donwload finished")
            except FileExistsError as e:
                print(e)


if __name__ == "__main__":
    with open("pairs.json") as f:
        symbols = json.load(f)

    download_symbols(symbols)
