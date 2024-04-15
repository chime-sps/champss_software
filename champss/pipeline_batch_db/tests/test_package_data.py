import pkg_resources
import sps_pipeline


def test_data_files():
    """Access package config file using pkg_resources APIs (setuptools).

    https://setuptools.readthedocs.io/en/latest/pkg_resources.html#basic-resource-access
    """

    assert pkg_resources.resource_string(sps_pipeline.__name__, "sps_config.yml")
