# Running colibri as a persistent service

Templates for running `coli serve` as a systemd user service, plus an
optional usage-metrics collector. Copy the `.example` files, drop the
`.example` suffix, fill in the placeholders, then:

```bash
mkdir -p ~/.config/systemd/user
cp colibri.service.example ~/.config/systemd/user/colibri.service
cp colibri-watcher.service.example ~/.config/systemd/user/colibri-watcher.service
cp colibri-watcher.timer.example ~/.config/systemd/user/colibri-watcher.timer
mkdir -p /path/to/colibri/watcher
cp watcher/colibri_watcher.py /path/to/colibri/watcher/
# edit the copied unit files: model path, host/port, RAM budget, API key
loginctl enable-linger "$USER"   # so it keeps running without an active login session
systemctl --user daemon-reload
systemctl --user enable --now colibri.service
systemctl --user enable --now colibri-watcher.timer
```

## Memory ceiling

`--ram`/`--cap` bound what colibri *asks for*; `MemoryHigh`/`MemoryMax` bound
what the cgroup actually *allows*, independent of that budget. These aren't
redundant: a host running other memory-hungry services (VMs, containers) can
still be pushed into swap thrashing if colibri's own footprint isn't fenced
off from the rest of the system. Size the ceiling above the working set you
actually observe (check `/profile` or the startup banner's `RAM_GB=...
projected peak`), with headroom, but well below your host's actual danger
zone — if colibri ever tries to exceed it, the kernel reclaims from (or
OOM-kills, which then auto-restarts) just this unit, not the whole host.

`CAP_RAISE=0` keeps the expert-cache footprint fixed at `--cap` instead of
letting it auto-grow — without it, the actual working set can drift upward
over time and outgrow whatever ceiling you picked.

## Building on hardware without AVX2

If you build directly in the same checkout you're deploying from, don't run
`make check` there: its `check` target rebuilds via `portable`, which
hardcodes an AVX2/FMA baseline (`PORTABLE_ARCH = x86-64-v3` in `c/Makefile`)
with no host-capability check. On real pre-Haswell hardware (e.g. Sandy/Ivy
Bridge) this silently overwrites a correct `ARCH=native` binary with one that
SIGILLs on that same host. Run the test suite in a separate scratch checkout,
or just rebuild with plain `make colibri` (no `ARCH=`) afterward to restore
the host-appropriate binary.

## The watcher

`watcher/colibri_watcher.py` polls `/health` and `/profile`, samples RAM/swap
pressure, and counts new error/4xx/5xx/traceback lines in the serve log,
appending one JSON line per poll (plus one line per completed inference turn)
to `watcher_metrics.jsonl`. It's a data collector only — mechanical, no
judgment calls. Turning that log into concrete tuning or code proposals is a
separate, human-in-the-loop step: read the log, look for patterns (latency
drift, swap pressure trending up, recurring errors, expert hit rate), and
decide what's worth changing.
