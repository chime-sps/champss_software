from setuptools import setup, find_packages, Extension
import os

extra_compile_args = ['-std=c++11',]
extra_link_args = ['-std=c++11',]

package_contents = ['huff_utils', 'l1_io', 'test_huff']


class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()


module1 = Extension('spshuff.huffman',
                    language = 'c++',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include', '/usr/include',
                                     get_pybind_include()],
                    libraries = [],
                    library_dirs = ['/usr/local/lib', '/usr/lib', 'src'],
                    sources = ['src/huffman.cpp'],
                    extra_compile_args = extra_compile_args,
                    extra_link_args = extra_link_args)


setup(name = 'spshuff',
       version = '1.0',
       description = 'CIRADA/CHIME Slow Pulsar Search compression and file io',
       author = 'Alexander Roman',
       author_email = 'aroman@perimeterinstitue.ca',
       url = 'https://github.com/chime-sps/spshuff',
       long_description = '''
        A huffman compression library for the CIRADA/CHIME Slow Pulsar Search (SPS).
        ''',
       packages = find_packages(),
       install_requires = ['pybind11', 'numpy'],
       ext_modules = [module1,],
       python_requires = '>=3.6')
