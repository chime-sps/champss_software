"""Workers started through use_forkserver."""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from sps_common.mp_start import use_forkserver


def _draw(_):
    time.sleep(0.2)  # so the tasks spread over the workers
    return os.getpid(), np.random.random(), "sps_common.stack_quant" in sys.modules


def test_workers_get_preload_and_their_own_random_streams():
    use_forkserver("sps_common.stack_quant")
    with Pool(4) as pool:
        results = pool.map(_draw, range(8), chunksize=1)
    first_draw = {}
    for pid, value, preloaded in results:
        first_draw.setdefault(pid, value)
        assert preloaded
    assert len(first_draw) > 1
    # Forked from one server, the workers would share numpy's global state unless reseeded
    assert len(set(first_draw.values())) == len(first_draw)


def test_main_module_name(monkeypatch):
    import types

    from sps_common import mp_start

    script = types.ModuleType("__main__")
    script.__spec__ = None  # run by path
    monkeypatch.setitem(sys.modules, "__main__", script)
    assert mp_start._importable(("__main__", "a.b", "a.b")) == ["a.b"]
    module = types.ModuleType("__main__")
    module.__spec__ = types.SimpleNamespace(name="FFA.FFA_cli")  # python -m FFA.FFA_cli
    monkeypatch.setitem(sys.modules, "__main__", module)
    assert mp_start._importable(("__main__", "FFA.FFA_cli")) == ["FFA.FFA_cli"]
