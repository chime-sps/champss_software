# Stage 1: Install dependencies
# trixie: dmtlib needs GCC >= 13 (bookworm has 12)
FROM python:3.12-slim-trixie as base

ARG SLACK_APP_TOKEN

ENV SLACK_APP_TOKEN=${SLACK_APP_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]


RUN set -ex \
    && apt-get update \
    && apt-get install -yqq --no-install-recommends \
    curl \
    cmake \
    libfftw3-dev \
    git \
    libblas3 \
    liblapack3 \
    liblapack-dev \
    libblas-dev \
    build-essential \
    ca-certificates \
    vim \
    less \
    && update-ca-certificates

# Stage 2: Install Miniconda dependencies
FROM base as miniconda

WORKDIR /champss_module

ENV PATH="/champss_module/miniconda3/bin:$PATH"
ENV TEMPO2="/champss_module/miniconda3/share/tempo2"

RUN set -ex \
    && curl -O https://repo.anaconda.com/miniconda/Miniconda3-py312_24.1.2-0-Linux-x86_64.sh \
    && chmod 700 Miniconda3-py312_24.1.2-0-Linux-x86_64.sh \
    && bash Miniconda3-py312_24.1.2-0-Linux-x86_64.sh -b -p ./miniconda3 \
    && source ./miniconda3/bin/activate \
    && conda install -c conda-forge dspsr=2024.08.01=py312h3d5c965_0  

# Stage 3: Install Pip dependencies
FROM miniconda as pip

# The CPU the native code (spshuff, dmtlib) is compiled for. "native" targets the CPU of
# the machine running the build, so build on the kind of node the image will run on
# (image.yml: runs_on). An explicit target (e.g. --build-arg MARCH=sapphirerapids) lets
# another machine build for those nodes, but the numba warm-up below still compiles for
# the build machine's CPU, so its cache is only used on CPUs like it.
ARG MARCH=native

# NUMBA_CACHE_DIR: numba's compiled kernels, warmed below and shipped in the image.
# OPENBLAS/MKL: many pipeline processes share a node, so one BLAS thread each; OpenMP and
# numba thread counts are set by the pipeline from the CPUs it may use (sps_common.cpus)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    HOME="/champss_module/" \
    NUMBA_CACHE_DIR="/champss_module/numba_cache" \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
#    XDG_CACHE_HOME="/champss_module/" \
#    XDG_CONFIG_HOME="/champss_module/"

COPY . .

RUN set -ex \
    && if [ "${MARCH}" = "native" ]; then \
         export DMT_ENABLE_NATIVE_ARCH=ON; \
       else \
         export DMT_ENABLE_NATIVE_ARCH=OFF CFLAGS="-march=${MARCH}" CXXFLAGS="-march=${MARCH}"; \
       fi \
    && export SPSHUFF_MARCH="${MARCH}" \
    && python3 -m pip install --ignore-requires-python . \
    && get-data \
    && workflow workspace set champss.workspace.yml \
    && python3 download_files.py
# --ignore-requires-python: cfbm (chime-frb-beam-model) still caps python <3.12, but runs fine on 3.12
# Above "get-data" call is needed for CHIMEFRB/beam-model
# The astropy calls allow downloading of data that might be available when running the container

# Record the build CPU (the pipeline warns at start-up if it runs on a CPU lacking any of
# its instruction sets: sps_common.cpus), then compile the pipeline's numba kernels for
# it, so containers start with a full cache instead of compiling in every container and
# pool worker (sps_common.numba_warmup; the kernels' signatures are recorded from real runs
# in sps_common/numba_signatures.json)
RUN set -ex \
    && { grep -m1 "^model name" /proc/cpuinfo; grep -m1 "^flags" /proc/cpuinfo; \
         echo "march: ${MARCH}"; } > /champss_module/BUILD_CPU \
    && cat /champss_module/BUILD_CPU | cut -c1-200 \
    && mkdir -p "${NUMBA_CACHE_DIR}" \
    && python3 -m sps_common.numba_warmup warm

RUN run-stack-search-pipeline --help

# Stage 4: Cleanup to prepare for runtime
FROM pip as runtime

WORKDIR /champss_module/

RUN chmod 777 /champss_module/ \
    && chmod 777 -R /champss_module/.astropy /champss_module/.config \
    && chmod -R a+rwX /champss_module/numba_cache

RUN set -ex \
    && apt-get remove build-essential -yqq \
    && apt-get clean autoclean \
    && apt-get autoremove -yqq --purge \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf ~/.cache \
    && rm -rf /usr/share/man \
    && rm -rf /usr/share/doc \
    && rm -rf /usr/share/doc-base
