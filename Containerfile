# Generic podman/docker image for running colibrì — see scripts/podman.sh (or
# `make podman-chat` / `make podman-serve`) for the one-command way to build and
# run this. Not tied to any specific host; pass --build-arg ARCH=... for a
# non-native CPU tier (see docs/tuning.md for the list). An existing docker/
# guide already ships with the project (manual build+run) — this is a more
# automated alternative built around a single wrapper script.
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
RUN make -C c colibri ARCH=${ARCH}

WORKDIR /work/c
ENV COLI_MODEL=/model
ENTRYPOINT ["python3", "coli"]
CMD ["chat"]
