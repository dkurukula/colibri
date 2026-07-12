.PHONY: all glm portable portable-avx test check cuda-test clean

all glm portable portable-avx test check cuda-test clean:
	$(MAKE) -C c $@
