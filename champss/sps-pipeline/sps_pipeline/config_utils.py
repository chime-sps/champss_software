import os
from omegaconf import OmegaConf
import ast


def load_config(config_file="sps_config.yml", cli_config_string="{}"):
    """
    Combines default/user-specified config settings and applies them to loggers.

    User-specified settings are given as a YAML file in the current directory.

    The format of the file is (all sections optional):
    ```
    logging:
      format: string for the `logging.formatter`
      level: logging level for the root logger
      modules:
        module_name: logging level for the submodule `module_name`
        module_name2: etc.
    ```
    Paramaters
    ----------
    config_files: str
        Name of config file. Default: "sps_config.yml"

    Returns
    -------
    The `omegaconf` configuration object merging all the default configuration
    with the (optional) user-specified overrides.
    """
    base_config_path = os.path.dirname(__file__) + "/" + config_file
    if os.path.isfile(base_config_path):
        base_config = OmegaConf.load(base_config_path)
    else:
        base_config = OmegaConf.create()
    if os.path.exists("./" + config_file):
        user_config = OmegaConf.load("./" + config_file)
    else:
        user_config = OmegaConf.create()
    cli_config = ast.literal_eval(cli_config_string)
    return OmegaConf.merge(base_config, user_config, cli_config)
