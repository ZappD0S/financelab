import os
import csv
import requests
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

with open("proxies.csv", "r", newline='') as proxies_file:
    proxies_reader = csv.reader(proxies_file, skipinitialspace=True)
    proxy_data = [(proxy_addr, strtobool(is_https)) for proxy_addr, is_https in proxies_reader]

proxy_addr, is_https = proxy_data[0]
scheme = "https" if is_https else "http"
proxy = {scheme: f"{scheme}://{proxy_addr}"}


r = requests.get("https://www.alphavantage.co/query", params=payload, proxies=proxy, timeout=10)
