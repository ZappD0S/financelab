import concurrent.futures
import contextlib
import io
import os
import zipfile
from datetime import datetime
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse
from dateutil.rrule import MONTHLY, rrule

PAIRS = [
    "eurusd",  # 1
    "usdjpy",  # eurusd
    "gbpusd",  # eurgbp
    "audusd",  # euraud
    "usdcad",  # eurusd
    "usdchf",  # eurusd
    "nzdusd",  # eurnzd
    "eurjpy",  # 1
    "gbpjpy",  # eurgbp
    "eurgbp",  # 1
    "audjpy",  # euraud
    "euraud",  # 1
    "eurnzd",  # 1
]

SAVE_PATH = "./tick_data"

START_DATE = parse("01/2018")
END_DATE = parse("01/2020")

HISTDATA_URL = "https://www.histdata.com"
url_template = (
    HISTDATA_URL + "/download-free-forex-historical-data/?/ascii/tick-data-quotes/{pair}/{dt.year}/{dt.month}"
)
POST_URL = HISTDATA_URL + "/get.php"


@contextlib.contextmanager
def get_month_data(url: str):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    form_params = soup.find("form", id="file_down").find_all("input")

    payload = {param["name"]: param["value"] for param in form_params}
    r = requests.post(POST_URL, payload, headers={"Referer": url})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = next(name for name in z.namelist() if name.endswith(".csv"))
    with z.open(fname) as f:
        yield f


def download_pair_data(path: str, pair: str, dts: List[datetime]):
    urls = (url_template.format(pair=pair, dt=dt) for dt in dts)
    date_range = "{:%Y%m}-{:%Y%m}".format(dts[0], dts[-1])
    # with open(f"{path}/{pair}_{date_range}.csv", "wb") as outfile:
    fname = os.path.join(path, f"{pair}_{date_range}.csv")
    with open(fname, "xb") as outfile:
        for url in urls:
            with get_month_data(url) as infile:
                for line in infile:
                    if not line.isspace():
                        outfile.write(line)


def pairs_unique(pairs: Iterable[str]) -> bool:
    seen = set()
    for pair in pairs:
        if pair in seen:
            return False

        seen.add(pair)

    return True


dts = list(rrule(MONTHLY, dtstart=START_DATE, until=END_DATE))
assert pairs_unique(PAIRS)

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_pair = {executor.submit(download_pair_data, SAVE_PATH, pair, dts): pair for pair in PAIRS}

    for future in concurrent.futures.as_completed(future_to_pair):
        pair = future_to_pair[future]
        try:
            future.result()
            print(f"donwloaded {pair} pair")
        except FileExistsError as e:
            print(e)

# with open(f"tick_data/{PAIR}_{YEAR}.csv", "ab") as outfile:
#     for month in range(1, 13):
#         with get_month_data(month) as infile:
#             for line in infile:
#                 if not line.isspace():
#                     outfile.write(line)
