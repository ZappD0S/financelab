import requests
from bs4 import BeautifulSoup
import io
import zipfile
import re

pair = 'eurusd'
year = 2018

histdata = "https://www.histdata.com"
url = f"{histdata}/download-free-forex-historical-data/?/ascii/tick-data-quotes/{pair}/{year}"
post_url = f"{histdata}/get.php"
csv_file = re.compile(r'.*\.csv')


for month in range(1, 13):
    current_url = url + f'/{month}'
    page = requests.get(current_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    form_parameters = soup.select('form#file_down input')
    payload = {}
    for parameter in form_parameters:
        payload[parameter['name']] = parameter['value']

    r = requests.post(post_url, payload, headers={'Referer': current_url})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = next(name for name in z.namelist() if csv_file.match(name))
    with open('eurusd_2018.csv', 'ab') as outfile:
        with z.open(fname) as infile:
            for line in infile:
                if line.strip():
                    outfile.write(line)
    print(f"completato mese {month}")
