import io
import zipfile
import requests
import contextlib
from bs4 import BeautifulSoup
import urllib.parse as urlparse


PAIR = "eurusd"
YEAR = 2019

histdata_url = "https://www.histdata.com"
url = f"{histdata_url}/download-free-forex-historical-data/?/ascii/tick-data-quotes/{PAIR}/{YEAR}/"
post_url = f"{histdata_url}/get.php"
# csv_file = re.compile(r".*\.csv")


@contextlib.contextmanager
def get_month_data(month):
    # current_url = urlparse.urljoin(url, str(month))
    current_url = url + str(month)
    page = requests.get(current_url)
    soup = BeautifulSoup(page.content, "lxml")
    # form_params = soup.select("form#file_down input")
    form_params = soup.find("form", id="file_down").find_all("input")

    payload = {param["name"]: param["value"] for param in form_params}
    r = requests.post(post_url, payload, headers={"Referer": current_url})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = next(name for name in z.namelist() if name.endswith(".csv"))
    with z.open(fname) as f:
        yield f


with open(f"tick_data/{PAIR}_{YEAR}.csv", "ab") as outfile:
    for month in range(1, 13):
        with get_month_data(month) as infile:
            for line in infile:
                if not line.isspace():
                    outfile.write(line)


# for month in range(1, 13):
#     current_url = urlparse.urljoin(url, str(month))
#     # current_url = url + f"/{month}"
#     page = requests.get(current_url)
#     soup = BeautifulSoup(page.content, "html.parser")
#     form_params = soup.select("form#file_down input")

#     payload = {param["name"]: param["value"] for param in form_params}
#     r = requests.post(post_url, payload, headers={"Referer": current_url})
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     fname = next(name for name in z.namelist() if name.endswith(".csv"))
#     with open("eurusd_2018.csv", "ab") as outfile, z.open(fname) as infile:
#         for line in infile:
#             if line.strip():
#                 outfile.write(line)
#     print(f"completato mese {month}")
