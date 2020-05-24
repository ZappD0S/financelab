import time
import os
import concurrent.futures
import requests
from io import BytesIO
from itertools import repeat, cycle, islice
from threading import Lock, Event
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple


MAX_REQ_PER_MIN = 5
MAX_REQ_PER_KEY = 500
DATA_FOLDER = "stock_data/"


@dataclass
class State:
    apikey: str
    lock: Lock = Lock()
    _all_reqs_completed: Event = Event()
    _started_reqs: int = 0
    _completed_reqs: int = 0
    first_req_t: Optional[float] = None
    _last_req_t: Optional[float] = None

    @property
    def last_req_t(self) -> float:
        return self._last_req_t if self._last_req_t is not None else self.first_req_t

    @last_req_t.setter
    def last_req_t(self, value: float) -> None:
        if self.first_req_t is None:
            self.first_req_t = value
        else:
            self._last_req_t = value

    def register_req_start(self) -> None:
        self._started_reqs = self._started_reqs % MAX_REQ_PER_MIN + 1
        if self._completed_reqs == 1:
            self._all_reqs_completed.clear()

    def register_req_end(self, t: float) -> None:
        self.last_req_t = t
        self._completed_reqs = self._completed_reqs % MAX_REQ_PER_MIN + 1
        if self._completed_reqs == MAX_REQ_PER_MIN:
            print("event set!")
            self._all_reqs_completed.set()

    def can_start_req(self) -> bool:
        if self._started_reqs < MAX_REQ_PER_MIN:
            return True

        if self.first_req_t is None:
            # print("mi blocco qua (1)")
            return False

        if not self._all_reqs_completed.is_set():
            # print("mi blocco qua (2)")
            return False

        delta_since_first_req = time.time() - self.first_req_t

        return (self.last_req_t - self.first_req_t) < (
            delta_since_first_req - delta_since_first_req % 60
        )

    # def time_to_wait(self) -> float:
    #     return 60 - (time.time() - self.first_req_t) % 60

    def wait(self) -> None:
        self._all_reqs_completed.wait()
        # print("i reach this point")

        time_to_wait = 60 - (time.time() - self.first_req_t) % 60
        # print(f"waiting for {time_to_wait} s")
        time.sleep(time_to_wait)


def download_symbol_data(state: State, symbol: str) -> requests.Response:
    payload = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": state.apikey,
        "datatype": "csv",
        "outputsize": "full",
    }

    with state.lock:
        while not state.can_start_req():
            # time_to_wait = state.time_to_wait()
            state.lock.release()
            state.wait()
            # time.sleep(time_to_wait)
            state.lock.acquire()

        state.register_req_start()

    # print("arrivo qua!")
    r = requests.get("https://www.alphavantage.co/query", params=payload)

    t = time.time()
    with state.lock:
        state.register_req_end(t)

    return r


def get_already_donwloaded_symbols() -> Iterable[str]:
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(file)
        if ext != ".csv":
            continue

        yield name


def get_symbols() -> Iterable[str]:
    already_downloaded_symbols = set(get_already_donwloaded_symbols())

    with open(
        os.path.join(DATA_FOLDER, "not_found_symbols.txt"), "r"
    ) as not_found_symbols_file:
        not_found_symbols = set(line.strip() for line in not_found_symbols_file)

    with open("symbols.txt", "r") as symbols_file:
        for line in symbols_file:
            if line.startswith("#"):
                continue

            symbol = line.strip()

            if symbol in already_downloaded_symbols or symbol in not_found_symbols:
                continue

            yield symbol


def get_args() -> Iterable[Tuple[State, str]]:
    with open("alphavantage_api_key.txt", "r") as keys_file:
        unique_keys = [key.strip() for key in keys_file][:1]

    states = [State(apikey=key) for key in unique_keys]
    return zip(cycle(states), islice(get_symbols(), len(unique_keys) * MAX_REQ_PER_KEY))


with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    future_to_symbol = {
        executor.submit(download_symbol_data, state, symbol): symbol
        for state, symbol in get_args()
    }

    for future in concurrent.futures.as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        r: requests.Response = future.result()

        if not r.ok:
            print(f"{symbol} download failed")
            continue

        try:
            json = r.json()
        except ValueError:
            print(f"{symbol} succesful!")
            with open(f"stock_data/{symbol}.csv", "wb") as f:
                f.write(r.content)

            continue

        if "Error Message" in json:
            print(f"{symbol} not found")
            with open(os.path.join(DATA_FOLDER, "not_found_symbols.txt"), "a") as f:
                f.write(f"{symbol}\n")
        elif "Note" in json:
            print("api limit reached")
            # print(json["Note"])
        else:
            print("boh...")
