"""Read-only probe: one hour of ES tick data (trades and top-of-book quotes).

Trades carry the price and size of each print; quote ticks carry the resting bid and
ask with their sizes. Neither is in the 5-second bars the models use today. IBKR
returns at most 1000 ticks per request, so each pass continues from the last tick's
time until the window is covered or the request budget runs out.

usage: download_ticks.py YYYY-MM-DD START_ET END_ET   (e.g. 2026-09-21 10:30 11:30)
"""
import sys
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

sys.path.insert(0, "/Users/abeadam/dev/interactive-broker-python/Updated Stats")
from download_daily import get_active_es_contract_month  # noqa: E402

PORT = 7497
CLIENT_ID = 43
EASTERN = ZoneInfo("America/New_York")
TICKS_PER_REQUEST = 1000
REQUEST_BUDGET = 400                 # keeps the probe near half of IBKR's 60-per-10-minutes pace
REQUEST_PAUSE_SECONDS = 2.5
REQUEST_TIMEOUT_SECONDS = 30
# TWS reads a timestamp without a zone as UTC, so every request time is sent in UTC.
IB_TIME_FORMAT = "%Y%m%d-%H:%M:%S"


class TickClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.done = threading.Event()
        self.rows, self.last_time = [], None

    def nextValidId(self, orderId):
        self.ready.set()

    # Connection-status notices arrive as errors on reqId -1 and must not end a request.
    STATUS_CODES = (2104, 2106, 2107, 2108, 2158)

    def error(self, reqId, errorCode, errorString, *extra):
        if errorCode in self.STATUS_CODES:
            return
        print(f"error {reqId} {errorCode} {errorString}", flush=True)
        self.done.set()

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            self.rows.append((tick.time, tick.price, tick.size))
        self.last_time = ticks[-1].time if ticks else self.last_time
        self.done.set()

    def historicalTicksBidAsk(self, reqId, ticks, done):
        for tick in ticks:
            self.rows.append((tick.time, tick.priceBid, tick.priceAsk, tick.sizeBid, tick.sizeAsk))
        self.last_time = ticks[-1].time if ticks else self.last_time
        self.done.set()


def es_contract(day: datetime) -> Contract:
    contract = Contract()
    contract.symbol, contract.secType, contract.exchange, contract.currency = "ES", "FUT", "CME", "USD"
    contract.lastTradeDateOrContractMonth = get_active_es_contract_month(day)
    contract.includeExpired = True
    return contract


def collect(client: TickClient, contract: Contract, start: datetime, end: datetime, what_to_show: str,
            columns: str, path: str) -> None:
    client.rows, client.last_time = [], None
    moment, requests = start, 0
    while moment < end and requests < REQUEST_BUDGET:
        client.done.clear()
        before = len(client.rows)
        client.reqHistoricalTicks(1, contract, moment.astimezone(timezone.utc).strftime(IB_TIME_FORMAT), "",
                                  TICKS_PER_REQUEST, what_to_show, 0, True, [])
        if not client.done.wait(REQUEST_TIMEOUT_SECONDS):
            print(f"  {what_to_show}: timed out at {moment:%H:%M:%S}", flush=True)
            break
        if len(client.rows) == before:
            print(f"  {what_to_show}: no more ticks at {moment:%H:%M:%S}", flush=True)
            break
        requests += 1
        moment = datetime.fromtimestamp(client.last_time, tz=EASTERN)
        if requests % 20 == 0:
            print(f"  {what_to_show}: {len(client.rows):,} ticks, up to {moment:%H:%M:%S}", flush=True)
        time.sleep(REQUEST_PAUSE_SECONDS)
    rows = [row for row in client.rows if start.timestamp() <= row[0] <= end.timestamp()]
    with open(path, "w") as handle:
        handle.write(columns + "\n")
        handle.writelines(",".join(map(str, row)) + "\n" for row in rows)
    print(f"{what_to_show}: {len(rows):,} ticks in {requests} requests -> {path}", flush=True)


def main() -> None:
    day, start_clock, end_clock = sys.argv[1], sys.argv[2], sys.argv[3]
    start = datetime.strptime(f"{day} {start_clock}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    end = datetime.strptime(f"{day} {end_clock}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    client = TickClient()
    client.connect("127.0.0.1", PORT, CLIENT_ID)
    threading.Thread(target=client.run, daemon=True).start()
    if not client.ready.wait(15):
        sys.exit("could not connect to TWS")
    contract = es_contract(start)
    print(f"ES {contract.lastTradeDateOrContractMonth}, {start} .. {end}", flush=True)
    collect(client, contract, start, end, "TRADES", "time,price,size", f"es_trades_{day}.csv")
    collect(client, contract, start, end, "BID_ASK", "time,bid,ask,bid_size,ask_size", f"es_quotes_{day}.csv")
    client.disconnect()


if __name__ == "__main__":
    main()
