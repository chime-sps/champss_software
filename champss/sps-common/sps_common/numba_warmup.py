"""
Compile the pipeline's cached numba kernels ahead of time, for container images.

numba keys its cache (cache=True) by kernel, argument types and the CPU it compiled for,
and a fresh container starts with whatever cache its image holds. Without one, every
container compiles every kernel it uses, and so does every pool worker, since the
forkserver can only preload kernels that are already cached (mp_start.load_cached_kernels).

    python -m sps_common.numba_warmup record [manifest.json]   # from this machine's caches
    python -m sps_common.numba_warmup warm [manifest.json]     # compile into the cache here

`record` lists the argument types each kernel was compiled for in this machine's caches
(after real runs), as text, so the manifest does not depend on the numba version. `warm`
compiles each listed signature on the current CPU; run it during the image build, on the
CPU type the image will run on, with NUMBA_CACHE_DIR set as at run time. A kernel or
signature that no longer exists is reported and skipped; the build does not fail on it,
since such a kernel is simply compiled at run time as before.
"""

import importlib
import json
import os
import re
import sys
import time

MODULES = (
    "beamformer.coded",
    "beamformer.skybeam",
    "ps_processes.processes.harmonic_search",
    "ps_processes.processes.ps",
    "ps_processes.utilities.rednoise_fast",
    "rfi_mitigation.utilities.fast_stats",
    "sps_common.fast_select",
    "sps_common.stack_quant",
)
MANIFEST = os.path.join(os.path.dirname(__file__), "numba_signatures.json")
_LITERAL = re.compile(r"Literal\[(int|bool)\]\(([^()]*)\)")


def _kernels(module):
    """The cached dispatchers defined in module (not those it imports)."""
    for name, obj in sorted(vars(module).items()):
        if getattr(obj, "_cache", None) is not None and hasattr(obj, "compile"):
            if getattr(obj, "__module__", None) == module.__name__:
                yield name, obj


def record(modules=MODULES):
    """{module: {kernel: [[arg type text, ...], ...]}} from this machine's numba caches."""
    out = {}
    for m in modules:
        module = importlib.import_module(m)
        for name, kernel in _kernels(module):
            try:
                index = kernel._cache._cache_file._load_index()
            except Exception:
                continue
            sigs = sorted({tuple(repr(t) for t in key[0]) for key in index})
            if sigs:
                out.setdefault(m, {})[name] = [list(s) for s in sigs]
    return out


def parse_type(text):
    """A numba type from its repr, e.g. "Array(float32, 1, 'C', False, aligned=True)"."""
    from numba.core import types

    namespace = dict(vars(types))
    namespace.update(bool=types.boolean, literal=types.literal)
    return eval(_LITERAL.sub(r"literal(\2)", text), {"__builtins__": {}}, namespace)  # noqa: S307


def warm(manifest=MANIFEST, verbose=True):
    """Compile every signature in the manifest; returns (compiled, skipped)."""
    with open(manifest) as f:
        wanted = json.load(f)
    compiled, skipped = 0, []
    t0 = time.perf_counter()
    for m, kernels in wanted.items():
        try:
            module = importlib.import_module(m)
        except Exception as e:  # a module that no longer exists
            skipped.append(f"{m}: {e}")
            continue
        for name, sigs in kernels.items():
            kernel = getattr(module, name, None)
            if kernel is None or not hasattr(kernel, "compile"):
                skipped.append(f"{m}.{name}: not a numba kernel")
                continue
            for sig in sigs:
                try:
                    kernel.compile(tuple(parse_type(t) for t in sig))
                    compiled += 1
                except Exception as e:
                    skipped.append(f"{m}.{name}{tuple(sig)}: {type(e).__name__}: {e}"[:300])
    if verbose:
        cache_dir = os.environ.get("NUMBA_CACHE_DIR", "(next to the sources)")
        print(f"numba warm-up: {compiled} signatures compiled in {time.perf_counter() - t0:.0f} s, "
              f"cache {cache_dir}; {len(skipped)} skipped")
        for s in skipped:
            print("  skipped:", s)
    return compiled, skipped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("record", "warm"):
        print(__doc__)
        return 2
    path = argv[1] if len(argv) > 1 else MANIFEST
    if argv[0] == "record":
        sigs = record()
        with open(path, "w") as f:
            json.dump(sigs, f, indent=1, sort_keys=True)
        n = sum(len(s) for k in sigs.values() for s in k.values())
        print(f"recorded {n} signatures of {sum(len(k) for k in sigs.values())} kernels to {path}")
        return 0
    compiled, _ = warm(path)
    return 0 if compiled else 1


if __name__ == "__main__":
    sys.exit(main())
