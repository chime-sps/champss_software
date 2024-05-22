# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

__version__ = "1.1"

ext_modules = [
    Pybind11Extension(
        "spshuff.huffman",
        ["src/huffman.cpp"],
        cxx_std=11,
        extra_compile_args=["-g", "-O3", "-march=native"],
        define_macros=[
            ("VERSION_INFO", __version__),
            ("MAJOR_VERSION", "1"),
            ("MINOR_VERSION", "1"),
        ],
    ),
]
# package_contents = ['huff_utils', 'l1_io', 'test_huff']


setup(
    name="spshuff",
    version=__version__,
    description="CIRADA/CHIME Slow Pulsar Search compression and file io",
    author="Alexander Roman",
    author_email="aroman@perimeterinstitue.ca",
    url="https://github.com/chime-sps/spshuff",
    long_description="""
        A huffman compression library for the CIRADA/CHIME Slow Pulsar Search (SPS).
        """,
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=["numpy"],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.6",
)
