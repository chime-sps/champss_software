"""
Imported by the forkserver only (see sps_common.mp_start): load the compiled numba kernels
that pool workers use, so every worker inherits them instead of loading its own copy.
"""

from sps_common.mp_start import load_cached_kernels

KERNEL_MODULES = (
    "beamformer.skybeam",
    "sps_common.fast_select",
    "ps_processes.processes.ps",
    "ps_processes.utilities.rednoise_fast",
    "ps_processes.processes.harmonic_search",
    "rfi_mitigation.utilities.fast_stats",
)

LOADED = load_cached_kernels(*KERNEL_MODULES)
