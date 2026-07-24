.PHONY: all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin verify-real-model quickstart podman-build podman-chat podman-serve podman-run podman-plan podman-bench

all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin verify-real-model quickstart:
	$(MAKE) -C c $@

# Default, easiest way to run colibrì: builds the podman image (once) and runs
# the engine inside it. See c/scripts/podman.sh for tunables (ARCH, RAM_GB,
# REPIN, REPIN_EPS, PORT, REBUILD=1) and README "Run it in podman".
podman-build:
	bash c/scripts/podman.sh --build-only

podman-chat:
	bash c/scripts/podman.sh chat

podman-serve:
	bash c/scripts/podman.sh serve --host 0.0.0.0

podman-run:
	bash c/scripts/podman.sh run $(ARGS)

podman-plan:
	bash c/scripts/podman.sh plan

podman-bench:
	bash c/scripts/podman.sh bench $(ARGS)
