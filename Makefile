.PHONY: all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin quickstart

all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin quickstart:
	$(MAKE) -C c $@
