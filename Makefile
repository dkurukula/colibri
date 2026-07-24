.PHONY: all glm portable portable-avx test check cuda-test clean install uninstall bench-cpu-tiers quickstart

all glm portable portable-avx test check cuda-test clean install uninstall bench-cpu-tiers quickstart:
	$(MAKE) -C c $@
