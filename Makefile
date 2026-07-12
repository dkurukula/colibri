.PHONY: all glm portable portable-avx test check cuda-test clean bench-cpu-tiers

all glm portable portable-avx test check cuda-test clean bench-cpu-tiers:
	$(MAKE) -C c $@
