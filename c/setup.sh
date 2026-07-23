#!/usr/bin/env bash
# colibrì — install on a fresh machine (Linux x86-64, macOS, Windows/MinGW).
# Builds the engine and runs a self-test. The MODEL (~372 GB int4) must be copied
# separately or regenerated with: coli convert --model <dir-on-ext4/NVMe>
set -e
cd "$(dirname "$0")"
echo "🐦 colibrì — setup"

UNAME_S=$(uname -s)

# 1) dependencies
command -v make >/dev/null || { echo "make is missing"; exit 1; }
case "$UNAME_S" in
Darwin)
    command -v clang >/dev/null || { echo "clang is missing (xcode-select --install)"; exit 1; }
    echo "  clang: $(clang --version | head -1) · $(sysctl -n hw.ncpu) cores"
    echo -n "  OpenMP: "
    if [ -f "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib" ]; then echo "ok (libomp)"
    else echo "libomp missing -> single-thread build (recommended: brew install libomp)"; fi
    ;;
MINGW*|MSYS*)
    command -v gcc  >/dev/null || { echo "gcc is missing (MinGW-w64). Install: pacman -S mingw-w64-x86_64-gcc make"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · MinGW-w64"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "libgomp is missing (pacman -S mingw-w64-x86_64-gcc)"; exit 1; }
    ;;
*)
    command -v gcc  >/dev/null || { echo "gcc is missing (e.g.: sudo apt install build-essential)"; exit 1; }
    echo "  gcc: $(gcc -dumpversion) · $(nproc) cores"
    echo -n "  OpenMP: "; echo 'int main(){return 0;}' | gcc -fopenmp -xc - -o /tmp/_omp 2>/dev/null && echo ok || { echo "missing (libgomp)"; exit 1; }
    ;;
esac

# 2) build: native (fast, for THIS machine). For a binary to distribute: make portable
echo "  building (ARCH=${ARCH:-native})…"
make -s glm ARCH="${ARCH:-native}"

# 3) self-test against the tiny oracle, if present
if [ -d glm_tiny ] && [ -f ref_glm.json ]; then
    r=$(SNAP=./glm_tiny TF=1 ./glm 64 16 16 2>/dev/null | grep -oE "[0-9]+/[0-9]+ positions" || true)
    echo "  engine self-test: ${r:-?}  (expected 32/32)"
fi

# 4) machine info (speed depends on THESE two numbers, not the GPU)
case "$UNAME_S" in
Darwin)
    ram=$(( $(sysctl -n hw.memsize) / 1000000000 ))
    ;;
MINGW*|MSYS*)
    # MSYS2 provides /proc/meminfo as a symlink (more reliable than the deprecated wmic)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
*)
    ram=$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo 2>/dev/null || echo "?")
    ;;
esac
echo "  RAM: ${ram} GB   (more RAM = more experts cached = faster)"
echo
echo "ready. Next steps:"
echo "  ./coli build           # (already done)"
echo "  ./coli convert --model /path/to/NVMe/glm52_i4     # generates the int4 model (hours)"
echo "  ./coli info  --model /path/to/NVMe/glm52_i4"
echo "  ./coli chat  --model /path/to/NVMe/glm52_i4 --ram <GB>"
echo
echo "IMPORTANT: keep the model on FAST storage (NVMe/ext4), NEVER on /mnt/c or network storage."
