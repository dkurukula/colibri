# Generic podman/docker image for running colibrì — see scripts/podman.sh (or
# `make podman-chat` / `make podman-serve`) for the one-command way to build and
# run this. Not tied to any specific host; pass --build-arg ARCH=... for a
# non-native CPU tier (ivybridge, x86-64-v3, ... — see c/scripts/bench_cpu_tiers.sh).
FROM docker.io/library/rockylinux:9

RUN dnf -y install \
        bash \
        findutils \
        gcc \
        git \
        libgomp \
        make \
        python3 \
        python3-pip \
    && dnf clean all \
    && python3 -m pip install --no-cache-dir "huggingface_hub[cli]<1"

ARG ARCH=native
WORKDIR /work
COPY . /work
RUN make -C c glm ARCH=${ARCH}

WORKDIR /work/c
ENV COLI_MODEL=/model
ENTRYPOINT ["python3", "coli"]
CMD ["chat"]
