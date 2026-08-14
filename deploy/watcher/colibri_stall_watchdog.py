#!/usr/bin/env python3
"""Detects a stalled (deadlocked) `coli serve` engine process.

Runs standalone on a timer, same shape as colibri_watcher.py. Each run takes
one /proc sample of the engine process (disk read bytes + CPU ticks) and
compares it against the previous run's sample, persisted in a small state
file between invocations.

The distinguishing signal is NOT "CPU/IO went quiet" -- an idle server with
no request in flight looks identical at the /proc level (every worker thread
legitimately parked in a blocking wait). What actually means something is
frozen counters *while a real client connection is open* on the serve port:
someone is waiting on a response and nothing is happening. That combination
only shows up if there's an actual stall.

Purely mechanical detection -- no auto-restart, no judgment calls about what
caused it. On a confirmed stall this appends an ALERT line (plus a thread
wait-channel snapshot for post-mortem) to stall_watchdog.log; a human (or a
separate, explicitly-opted-in policy) decides what to do about it.
"""
import glob
import json
import os
import re
import subprocess
import time

HOST = os.environ.get("COLI_WATCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("COLI_WATCH_PORT", "8000"))
PROC_PATTERN = os.environ.get("COLI_WATCH_PROC_PATTERN", "/c/colibri")
STALL_SAMPLES = int(os.environ.get("COLI_STALL_SAMPLES", "5"))  # consecutive frozen runs before alerting
CPU_TICK_SLACK = int(os.environ.get("COLI_STALL_CPU_SLACK", "2"))  # jiffies of "noise" tolerance
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.environ.get("COLI_STALL_LOG", os.path.join(BASE, "stall_watchdog.log"))
STATE = os.path.join(BASE, "stall_watchdog_state.json")


def log_line(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(LOG, "a") as f:
        f.write(f"{ts} {msg}\n")


def find_engine_pid():
    """The engine's own PID, not the launcher's -- matched by cmdline substring
    so this works regardless of container name, host path prefix, or the
    trailing device-count argument colibri is invoked with."""
    for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(cmdline_path, "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if PROC_PATTERN in cmdline:
            return int(cmdline_path.split("/")[2])
    return None


def sample_io_cpu(pid):
    try:
        with open(f"/proc/{pid}/io") as f:
            io = f.read()
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    read_bytes = int(re.search(r"read_bytes:\s*(\d+)", io).group(1))
    fields = stat.split()
    utime, stime = int(fields[13]), int(fields[14])
    return {"read_bytes": read_bytes, "cpu_ticks": utime + stime}


def has_client_connection():
    """A real (non-loopback) ESTABLISHED connection on the serve port --
    someone other than our own health/metrics probes is waiting on a response."""
    try:
        out = subprocess.run(
            ["ss", "-tn", "state", "established", f"( sport = :{PORT} )"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return False
    for line in out.splitlines()[1:]:
        cols = line.split()
        if len(cols) >= 4 and not cols[3].startswith("127.0.0.1:"):
            return True
    return False


def thread_wchan_histogram(pid):
    counts = {}
    for wchan_path in glob.glob(f"/proc/{pid}/task/*/wchan"):
        try:
            with open(wchan_path) as f:
                w = f.read().strip() or "(running)"
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"pid": None, "sample": None, "frozen_count": 0, "alerted": False}


def save_state(state):
    with open(STATE, "w") as f:
        json.dump(state, f)


def main():
    state = load_state()
    pid = find_engine_pid()

    if pid is None:
        if state.get("pid") is not None:
            log_line(f"engine process gone (was pid {state['pid']}) -- likely restarting")
        save_state({"pid": None, "sample": None, "frozen_count": 0, "alerted": False})
        return

    if state.get("pid") != pid:
        if state.get("pid") is not None:
            log_line(f"engine pid changed {state['pid']} -> {pid} (restart detected), resetting")
        state = {"pid": pid, "sample": None, "frozen_count": 0, "alerted": False}

    sample = sample_io_cpu(pid)
    if sample is None:
        save_state(state)
        return

    if not has_client_connection():
        # Nobody's waiting on a response -- frozen counters here are normal idle.
        if state.get("alerted"):
            log_line(f"RECOVERED (or client disconnected): pid={pid} no active client connection")
        state.update(pid=pid, sample=sample, frozen_count=0, alerted=False)
        save_state(state)
        return

    prev = state.get("sample")
    if prev is not None:
        d_read = sample["read_bytes"] - prev["read_bytes"]
        d_cpu = sample["cpu_ticks"] - prev["cpu_ticks"]
        if d_read == 0 and d_cpu <= CPU_TICK_SLACK:
            state["frozen_count"] = state.get("frozen_count", 0) + 1
        else:
            if state.get("alerted"):
                log_line(f"RECOVERED: pid={pid} progress resumed (d_read_bytes={d_read} d_cpu_ticks={d_cpu})")
            state["frozen_count"] = 0
            state["alerted"] = False

    state["sample"] = sample
    state["pid"] = pid

    if state["frozen_count"] >= STALL_SAMPLES and not state.get("alerted"):
        log_line(
            f"ALERT: pid={pid} appears STALLED -- client connected, but zero disk I/O "
            f"and <= {CPU_TICK_SLACK} CPU ticks of movement across {state['frozen_count']} "
            "consecutive checks"
        )
        log_line("  thread wchan histogram:")
        for wchan, n in thread_wchan_histogram(pid):
            log_line(f"    {n:3d}  {wchan}")
        state["alerted"] = True

    save_state(state)


if __name__ == "__main__":
    main()
