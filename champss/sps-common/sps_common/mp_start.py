"""
Start the multiprocessing forkserver with the pipeline already imported.

The pipeline uses the "forkserver" start method. CPython 3.11's forkserver is meant to
import the main module once and fork every worker from it, but never does: it looks for a
"main_path" key in the preparation data, which names the path "init_main_from_path". So
each worker of every pool imports the main script itself, which for run-pipeline is the
whole pipeline (about 3 s, on every core, for every pool).

use_forkserver names the pipeline module to the forkserver as a preload module instead,
and starts the server at once, so its single import overlaps with whatever the pipeline
does before its first pool. Workers forked from it also reseed numpy's global random
generator, which a fresh import used to do; otherwise all would share one stream.
"""

import multiprocessing
import multiprocessing.forkserver
import os
import sys

import numpy as np


def _reseed_numpy():
    np.random.seed()


# Registered in every process that imports this module, the forkserver included, so it
# runs in each worker forked from it
os.register_at_fork(after_in_child=_reseed_numpy)


def _importable(preload):
    """preload with "__main__" replaced by its real name (python -m) or dropped."""
    modules = []
    for m in preload:
        if m == "__main__":
            spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            m = getattr(spec, "name", None)
            if not m or m == "__main__":
                continue
        if m not in modules:
            modules.append(m)
    return modules


def use_forkserver(*preload):
    """
    Use the forkserver start method, with preload imported once in the server.

    Parameters
    ==========
    preload: str
        Modules for the server to import, typically the calling pipeline module
        (pass __name__). "__main__" stands for the main module's real name when it was
        run with python -m; a script run by path cannot be imported by the server, so
        it is dropped (name the modules it imports instead).
    """
    multiprocessing.set_start_method("forkserver", force=True)
    multiprocessing.set_forkserver_preload([__name__, *_importable(preload)])
    multiprocessing.forkserver.ensure_running()


def load_cached_kernels(*module_names):
    """
    Load every compiled version of the cached numba kernels in the given modules.

    Meant for the forkserver: the first call to a numba kernel in a fresh worker loads its
    compiled code and sets up numba's runtime, ~60-70 MB of private memory and some time
    in every worker of every pool. Loaded once in the server instead, it is inherited and
    shared. Only versions found in each kernel's on-disk cache (cache=True, made by
    earlier runs) are loaded, so nothing is compiled here; a kernel with no cached
    version is left for the workers, as before. Uses numba's cache index (numba 0.58);
    any failure only means less is preloaded.

    Returns the number of compiled versions loaded.
    """
    import importlib

    loaded = 0
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        for obj in list(vars(module).values()):
            cache = getattr(obj, "_cache", None)
            if cache is None or not hasattr(obj, "compile"):
                continue
            if getattr(obj, "targetoptions", {}).get("parallel"):
                # Parallel kernels start a thread pool, which must not happen before
                # the server forks
                continue
            try:
                index = cache._cache_file._load_index()
                codegen = obj.targetctx.codegen()
                for key in index:
                    sig = key[0]
                    if cache._index_key(sig, codegen) == key:
                        obj.compile(sig)
                        loaded += 1
            except Exception:
                continue
    return loaded
