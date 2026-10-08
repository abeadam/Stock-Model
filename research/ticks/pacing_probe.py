"""Find the real reqHistoricalTicks pacing limit instead of trusting the comment.

download_tick_history says 2.5s is "half of IBKR's 60-requests-per-10-minutes
allowance". It is not: 2.5s is 240 requests per 10 minutes, four times that limit,
and it has been running for 33-minute stretches without a pacing error. The 60/10min
figure documents reqHistoricalData, and reqHistoricalTicks is evidently not held to
it. What it IS held to decides whether a full session costs 2.4 hours or 1.

Each arm walks real pages forward so the requests are the ones the downloader makes,
not a cheaper substitute. A pacing violation arrives as an error, not a refusal, so
the arm is judged on errors raised and on whether pages keep coming back.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, "/Users/abeadam/dev/interactive-broker-python/Updated Stats")
import download_tick_history as dth
from market_calendar import NEW_YORK

PROBE_DAY = "2026-09-24"
REQUESTS_PER_ARM = 25
PAUSES = (2.5, 1.0, 0.5, 0.25)
# 162 is the historical-data pacing violation; the others accompany a throttle.
PACING_CODES = {162, 165, 366, 420}


def collect_errors(app) -> None:
    """The downloader prints errors and moves on; the probe needs to count them.

    Its handler also sets page_ready on an error, so a pacing violation surfaces as a
    page that returns no rows and ends the arm early -- worth reading alongside the codes.
    """
    app.errors = []
    original = app.error

    def record(reqId, errorCode, errorString, additional=None):
        app.errors.append((errorCode, errorString))
        return original(reqId, errorCode, errorString, additional)

    app.error = record


def arm(app, contract, start, pause: float) -> dict:
    app.rows, app.last_tick_time = [], None
    before_errors = len(app.errors)
    moment, pages, began = start, 0, time.time()
    for _ in range(REQUESTS_PER_ARM):
        seen = len(app.rows)
        if not app.request_page(contract, moment, "BID_ASK") or len(app.rows) == seen:
            break
        pages += 1
        moment = dth.next_page_start(app.last_tick_time)
        time.sleep(pause)
    raised = app.errors[before_errors:]
    return {"pause": pause, "pages": pages, "ticks": len(app.rows),
            "seconds_covered": moment.timestamp() - start.timestamp(),
            "elapsed": time.time() - began,
            "pacing": [code for code, _ in raised if code in PACING_CODES],
            "other": [code for code, _ in raised if code not in PACING_CODES]}


def main() -> None:
    day = datetime.strptime(PROBE_DAY, "%Y-%m-%d").replace(tzinfo=NEW_YORK)
    start, _ = dth.session_window(day)
    app = dth.connect(dth.TICKS_DIR)
    if app is None:
        sys.exit(dth.EXIT_NO_CONNECTION)
    collect_errors(app)
    contract = dth.es_contract(day)
    print(f"{'pause':>7}{'pages':>7}{'ticks':>9}{'sec/page':>10}{'elapsed':>9}  pacing errors")
    try:
        for pause in PAUSES:
            result = arm(app, contract, start, pause)
            per_page = result["seconds_covered"] / max(result["pages"], 1)
            print(f"{result['pause']:>7.2f}{result['pages']:>7}{result['ticks']:>9,}"
                  f"{per_page:>10.1f}{result['elapsed']:>9.0f}  "
                  f"{result['pacing'] or 'none'}"
                  f"{'  other: ' + str(result['other']) if result['other'] else ''}", flush=True)
    finally:
        app.disconnect()


if __name__ == "__main__":
    main()
