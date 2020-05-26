import csv
from bs4 import BeautifulSoup, ResultSet

with open("free_proxy_list.html", 'r') as f:
    soup = BeautifulSoup(f, "lxml")


def get_adresses(rows):
    for row in rows:
        cols = row.find_all("td")
        if cols[4].string == "anonymous":
            yield cols[0].string, cols[1].string


rows = soup.find("table", id="proxylisttable").tbody.find_all("tr")

with open("proxies.txt", "w") as f:
    for addr, port in get_adresses(rows):
        f.write(f"{addr}:{port}\n")
