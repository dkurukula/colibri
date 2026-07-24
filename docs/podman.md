# Run it in podman

The project already ships a manual [`docker/`](../docker/README.md) build+run
guide. This is a more automated alternative built around one wrapper script —
useful if you'd rather not hand-write `podman run -v ... -e ...` yourself, or
prefer podman's rootless-by-default model over Docker Desktop.

```bash
COLI_MODEL=/nvme/glm52_i4 c/scripts/podman.sh chat
COLI_MODEL=/nvme/glm52_i4 c/scripts/podman.sh serve --host 0.0.0.0
COLI_MODEL=/nvme/glm52_i4 c/scripts/podman.sh run "The capital of France is"
COLI_MODEL=/nvme/glm52_i4 c/scripts/podman.sh plan

# or via make, from the repo root:
make podman-chat  COLI_MODEL=/nvme/glm52_i4
make podman-serve COLI_MODEL=/nvme/glm52_i4
```

The first invocation builds the image (`ARCH=native` by default); later runs
reuse it. The model directory is mounted read-write, so `.coli_kv` and
`.coli_usage` persist across container runs exactly as they would running the
engine directly on the host.

## Tunables (env)

| Var | Default | Meaning |
|---|---|---|
| `COLI_MODEL` | *(required)* | Model directory on the **host** — mounted to `/model` inside the container. |
| `IMAGE` | `localhost/colibri:latest` | Image tag to build/run. |
| `ARCH` | `native` | CPU tier passed as `--build-arg ARCH=...` (see [docs/tuning.md](tuning.md) for the list). |
| `RAM_GB` | engine auto-sizes it | RAM budget passed through to the engine. |
| `REPIN` / `REPIN_EPS` | engine defaults | Live re-pin gate — see [docs/tuning.md](tuning.md). |
| `PORT` | `8000` | Host port published for `serve`. |
| `REBUILD` | `0` | Set to `1` to force a rebuild even if `IMAGE` already exists. |

## Notes

- `scripts/podman.sh` only requests a TTY (`-it`) when one is actually attached
  (`[ -t 0 ] && [ -t 1 ]`) — forcing it unconditionally hangs non-interactive
  invocations (scripts, CI, a piped/non-tty SSH session) because podman then
  waits on a terminal that never shows up.
- `serve` additionally publishes `-p ${PORT}:${PORT}` and is the only
  subcommand that does; other subcommands don't open a network port.
- The build context is the whole repo; see `.containerignore` for what's
  excluded (build artifacts, `.git`, model weights, caches).
