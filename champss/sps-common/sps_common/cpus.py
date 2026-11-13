"""
CPUs available to this process, and thread limits for the numerical libraries.

In a container, os.cpu_count() and the libraries' own defaults see every core of the host,
not the container's share, so each pool worker's BLAS, OpenMP or numba threads can
oversubscribe the node. available_cpus() honours the CPU affinity (a cpuset), a cgroup CPU
quota (docker --cpus, swarm resources.limits.cpus) and the environment variable
CHAMPSS_CPUS: a container's share of its node, set by the swarm services
(sps_pipeline.processing), which limits threads without the throttling of a CPU quota.

limit_thread_env() must run before numpy, numba or any OpenMP library is imported: they
read these variables when they load. It only sets variables that are not already set, so
a deployment can always override them.

This module imports nothing but the standard library.
"""

import math
import os

# Written into the image at build time (see the Dockerfile): the build CPU's flags
BUILD_CPU_FILE = os.environ.get("CHAMPSS_BUILD_CPU_FILE", "/champss_module/BUILD_CPU")
# Instruction-set flags whose absence on the running CPU would break native builds
_ISA_FLAGS = (
    "sse4_2", "avx", "avx2", "fma", "bmi1", "bmi2", "f16c",
    "avx512f", "avx512dq", "avx512bw", "avx512vl", "avx512cd", "avx512_vnni",
)  # fmt: skip


def _cgroup_cpu_quota():
    """The cgroup CPU quota in CPUs (v2 cpu.max or v1 cfs files), or None if unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus():
    """
    CPUs this process may use: its affinity, capped by any cgroup quota and by
    CHAMPSS_CPUS when that is a positive integer (at least 1).
    """
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, max(1, math.ceil(quota)))
    try:
        share = int(os.environ.get("CHAMPSS_CPUS", ""))
    except ValueError:
        share = 0
    if share > 0:
        n = min(n, share)
    return max(1, n)


def limit_thread_env():
    """
    Default thread counts for the numerical libraries, unless already set:
    OMP_NUM_THREADS and NUMBA_NUM_THREADS to available_cpus() (dmtlib's FDMT is OpenMP and
    is given its count explicitly; numba's parallel kernels), and OPENBLAS_NUM_THREADS and
    MKL_NUM_THREADS to 1, since the pipeline runs many worker processes and does not rely
    on multi-threaded BLAS. Returns the values in effect.
    """
    n = available_cpus()
    defaults = {
        "OMP_NUM_THREADS": n,
        "NUMBA_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    return {key: os.environ[key] for key in defaults}


def _cpu_flags(path="/proc/cpuinfo"):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return None


def missing_build_cpu_flags(build_cpu_file=BUILD_CPU_FILE):
    """
    Instruction-set flags the image's build CPU had and this CPU lacks, or None if unknown
    (no build CPU record). Native code (spshuff, dmtlib built with -march=native) can stop
    with "illegal instruction" when any are missing. The record's "flags" line has the
    same format as /proc/cpuinfo's.
    """
    built = _cpu_flags(build_cpu_file)
    here = _cpu_flags()
    if built is None or here is None:
        return None
    return sorted((built & set(_ISA_FLAGS)) - here)
