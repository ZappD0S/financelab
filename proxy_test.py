import os
import csv
import requests
from requests.exceptions import SSLError, ProxyError, ConnectTimeout
import urllib.parse as urlparse
from distutils.util import strtobool

SYMBOL = "AAPL"

with open("alphavantage_api_key.txt", "r") as keys_file:
    apikey = next(keys_file).strip()

payload = {
    "function": "TIME_SERIES_DAILY",
    "symbol": SYMBOL,
    "apikey": apikey,
    "datatype": "csv",
    "outputsize": "full",
}

# possible exception: SSLError

with open("proxies.csv", "r", newline='') as proxies_file:
    proxies_reader = csv.reader(proxies_file, skipinitialspace=True)
    proxy_data = [(proxy_addr, strtobool(is_https)) for proxy_addr, is_https in proxies_reader]


# proxies = {"https": f"https://217.113.122.142:3128"}
# for proxy_addr, is_https in proxy_data:

scheme = "http"

for proxy_addr, _ in proxy_data:
    # scheme = "https" if is_https else "http"
    proxies = {scheme: f"{scheme}://{proxy_addr}"}

    try:
        r = requests.get("https://www.alphavantage.co/query", params=payload, proxies=proxies, timeout=10)
    except (SSLError, ProxyError, ConnectTimeout) as e:
        print(f"proxy {proxies[scheme]} failed. error: {e}")
        continue

    if not r.ok:
        print(f"proxy {proxies[scheme]} failed. status_code: {r.status_code}")
        continue

    try:
        json = r.json()
    except ValueError:
        print(f"proxy {proxies[scheme]} succeded.")
        continue

    if "Note" in json:
        print("api limit reached")
