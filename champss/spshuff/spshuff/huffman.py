# The compiled extension (huffman.cpython-*.so, built from src/huffman.cpp) sits next to
# this file and is imported in its place. This file is only reached when no extension
# was built for the running Python.
raise ImportError(
    "spshuff.huffman is not built for this Python version: reinstall spshuff"
    " (e.g. poetry install, or pip install -e champss/spshuff)."
)
