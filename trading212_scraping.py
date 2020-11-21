import csv
from bs4 import BeautifulSoup, ResultSet

with open("live.trading212.com.html", 'r') as fp:
    soup = BeautifulSoup(fp, "lxml")


result: ResultSet = soup.select(".search-results-instrument .ticker span")

symbols = [node.string.strip('()') for node in result]

with open('symbols.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for symbol in symbols:
        writer.writerow([symbol])
