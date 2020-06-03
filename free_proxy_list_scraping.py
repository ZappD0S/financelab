import csv
from bs4 import BeautifulSoup, ResultSet

with open("free_proxy_list.html", 'r') as f:
    soup = BeautifulSoup(f, "lxml")


def get_adresses(rows):
    for row in rows:
        cols = row.find_all("td")
        if cols[4].string == "anonymous":
            yield cols[0].string, cols[1].string, cols[https_index].string == "yes"


table = soup.find("table", id="proxylisttable")

headers = table.thead.tr
https_header = headers.find("th", string="Https")
https_index = headers.find_all("th").index(https_header)

rows = table.tbody.find_all("tr")

with open("proxies.txt", "w") as f:
    for addr, port, is_https in get_adresses(rows):
        f.write(f"{addr}:{port}, {is_https}\n")
