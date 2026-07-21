.PHONY: all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin verify-real-model quickstart

all glm portable portable-avx test check cuda-test clean bench-cpu-tiers bench-repin verify-real-model quickstart:
	$(MAKE) -C c $@
