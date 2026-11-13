"""CHIME SPS Pipeline Package."""

# Before numpy, numba or OpenMP code is imported by any pipeline module: default thread
# counts for the CPUs this process may use (a container sees all of the host's otherwise)
from sps_common.cpus import limit_thread_env

limit_thread_env()
