#!/usr/bin/env python3
"""Persistent usage-metrics collector for a live `coli serve` instance.

Runs standalone (no LLM, no network egress beyond localhost) on a timer.
Each run: polls /health and /profile, samples RAM/swap pressure, counts new
error/4xx/5xx lines in serve.log since the last run, and appends one JSON
line to metrics.jsonl. Purely mechanical data collection -- the periodic
*analysis* of this file (looking for patterns, proposing tuning/code
changes) is a separate, human-in-the-loop step, not done here.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HOST = os.environ.get("COLI_WATCH_HOST", "192.168.2.27")
PORT = os.environ.get("COLI_WATCH_PORT", "8000")
API_KEY = os.environ.get("COLI_API_KEY", "")
BASE = os.path.dirname(os.path.abspath(__file__))
SERVE_LOG = os.environ.get("COLI_SERVE_LOG", os.path.join(BASE, "serve.log"))
METRICS = os.path.join(BASE, "watcher_metrics.jsonl")
STATE = os.path.join(BASE, "watcher_state.json")


def http_get(path, timeout=5):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}")
    if API_KEY:
        req.add_header("Authorization", f"Bearer {API_KEY}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def mem_sample():
    out = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5).stdout
    mem = out.splitlines()[1].split()
    swap = out.splitlines()[2].split()
    return {
        "mem_total_gb": round(int(mem[1]) / 2**30, 2),
        "mem_used_gb": round(int(mem[2]) / 2**30, 2),
        "mem_available_gb": round(int(mem[6]) / 2**30, 2),
        "swap_used_gb": round(int(swap[2]) / 2**30, 2),
    }


def swap_activity():
    """Sample si/so over 2s to catch active thrashing, not just 'swap has bytes in it'."""
    out = subprocess.run(["vmstat", "1", "3"], capture_output=True, text=True, timeout=8).stdout
    lines = [l.split() for l in out.strip().splitlines()[2:] if l.strip()]
    if not lines:
        return {"si_max": 0, "so_max": 0}
    si = max(int(l[6]) for l in lines)
    so = max(int(l[7]) for l in lines)
    return {"si_max": si, "so_max": so}


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"log_offset": 0, "last_profile_seq": -1}


def save_state(state):
    with open(STATE, "w") as f:
        json.dump(state, f)


def scan_new_log_lines(offset):
    errors_4xx = errors_5xx = tracebacks = tool_recovered = tool_mangled = 0
    new_offset = offset
    if os.path.exists(SERVE_LOG):
        with open(SERVE_LOG, "r", errors="replace") as f:
            f.seek(offset)
            for line in f:
                if re.search(r'HTTP/1\.1"\s+4\d\d', line):
                    errors_4xx += 1
                elif re.search(r'HTTP/1\.1"\s+5\d\d', line):
                    errors_5xx += 1
                if "Traceback" in line:
                    tracebacks += 1
                if "RECOVERED" in line:
                    tool_recovered += 1
                if "quantization-mangled" in line:
                    tool_mangled += 1
            new_offset = f.tell()
    return new_offset, {
        "errors_4xx": errors_4xx, "errors_5xx": errors_5xx,
        "tracebacks": tracebacks, "tool_calls_recovered": tool_recovered,
        "tool_calls_mangled": tool_mangled,
    }


PROFILE_WINDOW = 120  # must match PROFILE_TURNS in openai_server.py -- the server's
                       # own rolling deque size, i.e. the most turns we can ever recover


def main():
    state = load_state()
    record = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    new_turns = []

    try:
        health = http_get("/health")
        record["health"] = health
        record["reachable"] = True
    except Exception as e:
        record["reachable"] = False
        record["health_error"] = str(e)

    try:
        prof = http_get("/profile")
        turns = prof.get("turns", [])
        new_seq = prof.get("seq", -1)
        last_seq = state.get("last_profile_seq", -1)
        if last_seq >= 0 and new_seq < last_seq and turns:
            # seq went backwards -- the server restarted (profile_seq resets to 0).
            # Can't know how many turns happened before we could reach it again;
            # just pick up from here and flag the gap rather than guess.
            record["server_restarted"] = True
            new_turns = turns[-1:]
        elif last_seq >= 0 and new_seq > last_seq and turns:
            missing = new_seq - last_seq
            if missing > PROFILE_WINDOW:
                record["turns_lost"] = missing - PROFILE_WINDOW
            n_new = min(missing, len(turns))
            new_turns = turns[-n_new:] if n_new else []
        state["last_profile_seq"] = new_seq
    except Exception as e:
        record["profile_error"] = str(e)

    record["mem"] = mem_sample()
    record["swap_activity"] = swap_activity()

    new_offset, log_counts = scan_new_log_lines(state.get("log_offset", 0))
    state["log_offset"] = new_offset
    record["log_since_last"] = log_counts

    with open(METRICS, "a") as f:
        f.write(json.dumps(record) + "\n")
        for t in new_turns:
            turn_record = dict(t)
            if turn_record.get("wall_s", 0) > 0 and turn_record.get("completion_tokens", 0) > 0:
                turn_record["tok_s"] = round(turn_record["completion_tokens"] / turn_record["wall_s"], 4)
            f.write(json.dumps({"ts": record["ts"], "iso": record["iso"], "turn": turn_record}) + "\n")
    save_state(state)


if __name__ == "__main__":
    main()
