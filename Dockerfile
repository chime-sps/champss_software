# Stage 1: Install dependencies
FROM python:3.11-slim as base

ARG SLACK_APP_TOKEN

ENV SLACK_APP_TOKEN=${SLACK_APP_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -ex \
    && apt-get update \
    && apt-get install -yqq --no-install-recommends \
    curl \
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
    && curl -O https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
    && chmod 700 Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
    && bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -b -p ./miniconda3 \
    && source ./miniconda3/bin/activate \
    && conda install -c conda-forge dspsr=2024.08.01=py311ha2d4b42_0  

# Stage 3: Install Pip dependencies
FROM miniconda as pip

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100
#    XDG_CACHE_HOME="/champss_module/" \
#    XDG_CONFIG_HOME="/champss_module/"

COPY . .

RUN set -ex \
    && python3 -m pip install . \
    && python3 -c "import cfbm; import os; beam_dir = os.path.join(os.path.dirname(cfbm.__file__), 'bm_data'); os.makedirs(beam_dir, exist_ok=True); [os.remove(os.path.join(beam_dir, f)) for f in ['beam_XX_v1.h5', 'beam_YY_v1.h5'] if os.path.exists(os.path.join(beam_dir, f))]" \
    && get-data \
    && python3 -c "import h5py; import cfbm; import os; beam_dir = os.path.join(os.path.dirname(cfbm.__file__), 'bm_data'); import sys; \
failed = False; \
for f in ['beam_XX_v1.h5', 'beam_YY_v1.h5']: \
    try: \
        h5py.File(os.path.join(beam_dir, f), 'r').close(); \
        print(f'Verified {f}'); \
    except Exception as e: \
        print(f'Warning: {f} validation failed: {e}'); \
        failed = True; \
        try: os.remove(os.path.join(beam_dir, f)); \
        except: pass; \
if failed: \
    print('Re-downloading beam files...'); \
    sys.exit(1)" \
    || (get-data && python3 -c "import h5py; import cfbm; import os; beam_dir = os.path.join(os.path.dirname(cfbm.__file__), 'bm_data'); [h5py.File(os.path.join(beam_dir, f), 'r').close() for f in ['beam_XX_v1.h5', 'beam_YY_v1.h5']]") \
    && workflow workspace set champss.workspace.yml \
    && python3 download_files.py
# Above commands: 1) Create beam data directory and remove any corrupted files
#                 2) Download beam model files via get-data
#                 3) Verify HDF5 files are valid by opening them, retry once if corrupted
# The astropy calls allow downloading of data that might be available when running the container

RUN run-stack-search-pipeline --help

# Stage 4: Cleanup to prepare for runtime
FROM pip as runtime

WORKDIR /champss_module/

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
