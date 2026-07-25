.PHONY: all glm portable portable-avx test check cuda-test clean install uninstall bench-cpu-tiers quickstart podman-build podman-chat podman-serve podman-run podman-plan podman-bench

all glm portable portable-avx test check cuda-test clean install uninstall bench-cpu-tiers quickstart:
	$(MAKE) -C c $@

# Automated alternative to docker/'s manual build+run guide: builds the podman
# image (once) and runs colibrì inside it. See c/scripts/podman.sh for tunables
# (ARCH, RAM_GB, REPIN, REPIN_EPS, PORT, REBUILD=1).
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
