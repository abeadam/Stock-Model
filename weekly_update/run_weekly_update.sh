#!/bin/bash
#
# Weekly Stock-Model refresh — runs every Monday 08:00 via launchd.
#
# Pipeline:
#   0. Preflight: TWS/IB Gateway reachable on 127.0.0.1:7497
#   1. download_daily.py          (IBKR day-by-day historical pull into daily_data/)
#   2. futures_price.py           (indicator build)
#   3. lightgbm_model_highest.py  ] run concurrently
#      lightgbm_model_lowest.py   ]
#   4. gradient_value/prepare_data.py
#   5. gradient_value/train_predictor.py
#   6. gradient_value/optimize_thresholds.py  (produces the buy/sell grid)
#   6b. compare_backtest.py       (backtests the thresholds currently in
#                                   futures_trader.py against the new ones,
#                                   report only -- changes nothing)
#   7. propose_thresholds.py      (applies the thresholds to futures_trader.py
#                                   if they pass sanity checks; otherwise
#                                   refuses and leaves the file untouched)
#
# Failure policy: stop on first failure, log everything, exit non-zero.
# This script NEVER restarts the running trader and NEVER places an order —
# step 7 may update the two threshold constants in futures_trader.py, but the
# running process only picks them up on its next restart, which stays manual.
#
set -uo pipefail

# ---------------------------------------------------------------- configuration
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IBKR_PY="/Users/abeadam/dev/interactive-broker-python/venv/bin/python"
MODEL_PY="/Users/abeadam/dev/model/Stock-Model/.venv/bin/python"

IBKR_ROOT="/Users/abeadam/dev/interactive-broker-python"
FP_DIR="/Users/abeadam/dev/model/Stock-Model/future_prices"
BASIC_DIR="$FP_DIR/basic_model"
GRAD_DIR="$FP_DIR/gradient_value"
TRADER_PY="$IBKR_ROOT/futures_trader/futures_trader.py"

TWS_HOST="127.0.0.1"
TWS_PORT="7497"          # matches downloadData.py:252

STEP_TIMEOUT="${STEP_TIMEOUT:-14400}"   # 4h per step; override via env

LOG_DIR="$HERE/logs"
REPORT_DIR="$HERE/reports"
BACKUP_DIR="$HERE/backups"
LOCK_DIR="$HERE/.run.lock"

# launchd gives a bare environment — make it look like a login shell.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/abeadam}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$BACKUP_DIR"

RUN_ID="$(date '+%Y-%m-%d_%H%M%S')"
LOG="$LOG_DIR/run_$RUN_ID.log"
RUN_START_EPOCH="$(date '+%s')"

# ---------------------------------------------------------------------- helpers
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

cleanup() { rm -rf "$LOCK_DIR" 2>/dev/null || true; }

# Take the lock, but don't let a lock orphaned by a crash (or a kill -9) block
# every future Monday: if the recorded pid is gone, the lock is stale.
acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo $$ >"$LOCK_DIR/pid"
        return 0
    fi
    local owner
    owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
        log "Another run is already in progress (pid $owner). Exiting."
        return 1
    fi
    log "Stale lock at $LOCK_DIR (pid '${owner:-unknown}' not running) — reclaiming."
    rm -rf "$LOCK_DIR" 2>/dev/null
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo $$ >"$LOCK_DIR/pid"
        return 0
    fi
    log "Could not reclaim the lock. Remove $LOCK_DIR by hand, then rerun."
    return 1
}

die() {
    log "-------------------------------------------------------------"
    log "RUN FAILED: $*"
    log "Nothing downstream was run. futures_trader.py was NOT modified."
    log "Full log: $LOG"
    log "-------------------------------------------------------------"
    cleanup
    exit 1
}

# Refuse to start a second copy on top of a still-running one.
acquire_lock || exit 1
trap cleanup EXIT INT TERM

# Some scripts in this pipeline swallow exceptions and still exit 0
# (model_tester.py's main() catches Exception and prints a traceback).
# A clean exit code alone is therefore not proof of success.
#
# The IBKR download scripts' error() callbacks print every TWS notification as
# "Error: reqId, code, message" -- including routine, non-fatal ones: 2103/2105
# ("farm connection is broken", self-heals), 2104/2106/2108/2158 ("farm
# connection is OK" / "available upon demand"), and 162 with "HMDS query
# returned no data" (a single symbol-day genuinely has no data, e.g. a futures
# contract rollover gap -- the script already records this and moves on).
# None of those indicate the run failed, so they're excluded before checking
# for a real error.
BENIGN_IBKR_ERROR_PATTERN='(Market data farm connection|HMDS data farm connection|Sec-def data farm connection|HMDS query returned no data)'
scan_for_swallowed_errors() {
    local step_log="$1"
    if grep -qE '^Traceback \(most recent call last\):' "$step_log"; then
        log "  !! Python traceback found in output despite exit code 0"
        grep -nE '^[A-Za-z_.]*(Error|Exception):' "$step_log" | tail -5 | tee -a "$LOG"
        return 1
    fi
    if grep -E '^Error: ' "$step_log" | grep -qvE "$BENIGN_IBKR_ERROR_PATTERN"; then
        log "  !! Script reported an error despite exit code 0"
        grep -E '^Error: ' "$step_log" | grep -vE "$BENIGN_IBKR_ERROR_PATTERN" | tail -5 | tee -a "$LOG"
        return 1
    fi
    return 0
}

# run_step <label> <workdir> <interpreter> <script> [args...]
# Runs one step with a timeout, streams output to its own log, then verifies
# both the exit code and the output text.
#
# stdin comes from $STEP_STDIN (default /dev/null). Nothing in this pipeline is
# allowed to sit waiting on a human: under launchd there is no terminal, so a
# prompt would otherwise burn the whole step timeout before failing.
run_step() {
    local label="$1" workdir="$2" interp="$3" script="$4"; shift 4
    local step_log="$LOG_DIR/run_${RUN_ID}__${label}.log"
    local stdin_src="${STEP_STDIN:-/dev/null}"

    log "STEP $label — starting"
    log "  cwd: $workdir"
    log "  cmd: $interp $script $*"
    log "  log: $step_log"
    [ "$stdin_src" = "/dev/null" ] || log "  stdin: $stdin_src (menu answers)"

    ( cd "$workdir" && "$interp" "$script" "$@" ) <"$stdin_src" >"$step_log" 2>&1 &
    local pid=$!

    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$STEP_TIMEOUT" ]; then
            log "  !! STEP $label exceeded ${STEP_TIMEOUT}s — killing pid $pid"
            kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null
            die "step $label timed out after ${STEP_TIMEOUT}s (see $step_log)"
        fi
    done
    wait "$pid"; local rc=$?

    tail -5 "$step_log" | sed 's/^/    | /' | tee -a "$LOG" >/dev/null
    [ "$rc" -eq 0 ] || die "step $label exited with code $rc (see $step_log)"
    scan_for_swallowed_errors "$step_log" || die "step $label reported an error in its output (see $step_log)"

    log "STEP $label — OK"
}

# ------------------------------------------------------------------- preflight
log "============================================================="
log "Weekly Stock-Model update — run $RUN_ID"
log "============================================================="

for path in "$IBKR_PY" "$MODEL_PY" "$TRADER_PY"; do
    [ -e "$path" ] || die "required path missing: $path"
done

log "STEP 0 — preflight: TWS/IB Gateway on $TWS_HOST:$TWS_PORT"
if ! ( exec 3<>"/dev/tcp/$TWS_HOST/$TWS_PORT" ) 2>/dev/null; then
    die "cannot reach TWS/IB Gateway on $TWS_HOST:$TWS_PORT. \
Start TWS (or IB Gateway), enable API connections, and confirm the socket port \
is $TWS_PORT. Step 1 (downloadData.py) needs a live connection."
fi
log "STEP 0 — OK (port open)"

# Informational only: a running trader means the thresholds you are about to
# review are not the ones the live process has loaded.
if pgrep -f "futures_trader.py" >/dev/null 2>&1; then
    log "NOTE: futures_trader.py appears to be running right now."
    log "      Any threshold change requires restarting it to take effect."
fi

# ------------------------------------------------------------------- pipeline
# --- step 1: pull each symbol-day into daily_data/ -----------------------------
# download_daily.py fetches one symbol-day at a time and skips any file that's
# already on disk, so a weekly run only pulls the handful of trading days since
# the last one — no interactive prompts, so no menu-answering wrapper needed.
# futures_price.py reads straight out of daily_data/ (via the Stock-Model <->
# interactive-broker-python symlink), so no separate consolidation step exists.
run_step "1_download_daily_data" "$HERE" "$IBKR_PY" "$IBKR_ROOT/Updated Stats/download_daily.py"

run_step "2_futures_price" "$FP_DIR" "$MODEL_PY" "$FP_DIR/futures_price.py"

# --- step 3: the two LightGBM trainings run concurrently -----------------------
log "STEP 3 — starting lightgbm_model_highest.py and lightgbm_model_lowest.py concurrently"
HIGH_LOG="$LOG_DIR/run_${RUN_ID}__3_lgbm_highest.log"
LOW_LOG="$LOG_DIR/run_${RUN_ID}__3_lgbm_lowest.log"

( cd "$BASIC_DIR" && "$IBKR_PY" "$BASIC_DIR/lightgbm_model_highest.py" ) </dev/null >"$HIGH_LOG" 2>&1 &
HIGH_PID=$!
( cd "$BASIC_DIR" && "$IBKR_PY" "$BASIC_DIR/lightgbm_model_lowest.py" ) </dev/null >"$LOW_LOG" 2>&1 &
LOW_PID=$!
log "  highest pid=$HIGH_PID -> $HIGH_LOG"
log "  lowest  pid=$LOW_PID -> $LOW_LOG"

wait "$HIGH_PID"; HIGH_RC=$?
wait "$LOW_PID";  LOW_RC=$?
log "  highest exit=$HIGH_RC  lowest exit=$LOW_RC"

STEP3_FAIL=""
[ "$HIGH_RC" -eq 0 ] || STEP3_FAIL="lightgbm_model_highest.py exited $HIGH_RC"
[ "$LOW_RC"  -eq 0 ] || STEP3_FAIL="${STEP3_FAIL:+$STEP3_FAIL; }lightgbm_model_lowest.py exited $LOW_RC"
scan_for_swallowed_errors "$HIGH_LOG" || STEP3_FAIL="${STEP3_FAIL:+$STEP3_FAIL; }lightgbm_model_highest.py logged an error"
scan_for_swallowed_errors "$LOW_LOG"  || STEP3_FAIL="${STEP3_FAIL:+$STEP3_FAIL; }lightgbm_model_lowest.py logged an error"
[ -z "$STEP3_FAIL" ] || die "step 3 failed: $STEP3_FAIL"
log "STEP 3 — OK (both models trained)"

# --- step 3b: the retrained models must be usable by the live trader -----------
# Step 3 rewrites each model's feature-importance CSV, and that CSV is the
# contract es_features.py reads at prediction time. If a retrain selects a
# feature the live generator cannot compute, futures_trader.py keeps predicting
# with that column silently NaN-filled. Catch it here, while the cause is
# obvious, instead of discovering it weeks later in live behaviour.
run_step "3b_verify_live_features" "$HERE" "$IBKR_PY" "$HERE/verify_live_features.py" \
    --basic-model-dir "$BASIC_DIR"

run_step "4_prepare_data"       "$GRAD_DIR" "$IBKR_PY" "$GRAD_DIR/prepare_data.py"
run_step "5_train_predictor"    "$GRAD_DIR" "$IBKR_PY" "$GRAD_DIR/train_predictor.py"
run_step "6_optimize_thresholds" "$GRAD_DIR" "$IBKR_PY" "$GRAD_DIR/optimize_thresholds.py"

# --- step 6b: what would the proposed change actually buy? ---------------------
# Step 6 reports the proposal on its own, which says nothing about whether it
# beats what the trader is already using. This scores both on the same fresh
# data. It must run BEFORE step 7, which overwrites the incumbent values.
run_step "6b_compare_backtest" "$HERE" "$IBKR_PY" "$HERE/compare_backtest.py" \
    --results       "$GRAD_DIR/threshold_optimization.txt" \
    --trader        "$TRADER_PY" \
    --gradient-dir  "$GRAD_DIR" \
    --report-dir    "$REPORT_DIR" \
    --min-mtime     "$RUN_START_EPOCH" \
    --run-id        "$RUN_ID"

# --- step 7: apply thresholds if they pass sanity checks -----------------------
run_step "7_propose_thresholds" "$HERE" "$MODEL_PY" "$HERE/propose_thresholds.py" \
    --results     "$GRAD_DIR/threshold_optimization.txt" \
    --trader      "$TRADER_PY" \
    --report-dir  "$REPORT_DIR" \
    --backup-dir  "$BACKUP_DIR" \
    --min-mtime   "$RUN_START_EPOCH" \
    --run-id      "$RUN_ID"

log "============================================================="
log "RUN COMPLETE"
log "Report:   $REPORT_DIR/latest_recommendation.md"
log "Backtest: $REPORT_DIR/latest_backtest_comparison.md (incumbent vs proposed)"
log "Full log: $LOG"
log "See the report above for whether futures_trader.py was updated."
log "Restart the trader for any threshold change to take effect — this script never does."
log "============================================================="
cleanup
exit 0
