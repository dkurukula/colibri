.PHONY: all glm portable portable-avx test check cuda-test clean bench-cpu-tiers quickstart

all glm portable portable-avx test check cuda-test clean bench-cpu-tiers quickstart:
	$(MAKE) -C c $@
