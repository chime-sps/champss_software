"""load_cached_kernels loads cached numba versions and compiles nothing."""

import subprocess
import sys
import textwrap

KERNEL = """
import numpy as np
from numba import njit


@njit(cache=True)
def double(x):
    return 2 * x
"""


def run(tmp_path, code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def test_loads_what_earlier_runs_cached_and_compiles_nothing(tmp_path):
    (tmp_path / "kernels_for_test.py").write_text(KERNEL)
    # Nothing cached yet: nothing to load
    assert run(tmp_path, """
        from sps_common.mp_start import load_cached_kernels
        print(load_cached_kernels("kernels_for_test"))
    """) == ["0"]
    # An earlier run compiles and caches one version
    run(tmp_path, """
        import numpy as np, kernels_for_test as k
        k.double(np.ones(3))
    """)
    # A fresh process loads it from the cache, and a call then needs no compiling
    out = run(tmp_path, """
        import numpy as np, kernels_for_test as k
        from sps_common.mp_start import load_cached_kernels
        n = load_cached_kernels("kernels_for_test")
        k.double._can_compile = False  # a call that needed compiling would now fail
        print(n, k.double(np.ones(3)).sum(), len(k.double.stats.cache_hits))
    """)
    assert out == ["1", "6.0", "1"]


def test_parallel_kernels_are_left_alone(tmp_path):
    (tmp_path / "parallel_for_test.py").write_text(
        KERNEL.replace("@njit(cache=True)", "@njit(cache=True, parallel=True)")
    )
    run(tmp_path, """
        import numpy as np, parallel_for_test as k
        k.double(np.ones(3))
    """)
    assert run(tmp_path, """
        from sps_common.mp_start import load_cached_kernels
        print(load_cached_kernels("parallel_for_test"))
    """) == ["0"]
