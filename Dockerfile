FROM python:3.11-slim as runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -ex \
    && apt-get update \
    && apt-get install -yqq --no-install-recommends \
    curl \
    ssh \
    git \
    libblas3 \
    liblapack3 \
    liblapack-dev \
    libblas-dev \
    build-essential \
    ca-certificates \
    && mkdir -p ~/.ssh \
    && update-ca-certificates \
    && touch ~/.ssh/known_hosts \
    && chmod 0600 ~/.ssh/known_hosts ~/.ssh \
    && ssh-keyscan github.com >> ~/.ssh/known_hosts

COPY . /module
WORKDIR /module
ENV PATH="/module/miniconda3/bin:$PATH"
ENV TEMPO2="/module/miniconda3/share/tempo2"
RUN --mount=type=ssh,id=github_ssh_id set -ex \
    && curl -O https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
    && chmod 700 Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
    && bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -b -p ./miniconda3 \
    && source ./miniconda3/bin/activate \
    && conda install -c conda-forge dspsr \
    && conda install poetry \
    && python3 -m pip install . \
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

