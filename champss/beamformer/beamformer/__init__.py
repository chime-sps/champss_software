import importlib.metadata
import logging
import os
import pathlib

__version__ = importlib.metadata.version("beamformer")

# Global Path for the project
PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = PATH + "/data/"

AVAILAIBLE_POINTING_MAPS = []
filenames = pathlib.Path(DATA_PATH).glob("**/pointings_map*")
for filename in filenames:
    AVAILAIBLE_POINTING_MAPS.append(str(filename.absolute()))
CURRENT_POINTING_MAP = DATA_PATH + "pointings_map_v1-3.json"

log = logging.getLogger(__package__)


class NoSuchPointingError(ValueError):
    """Exception raised when the beamformer cannot find a pointing in its map."""


class NotEnoughDataError(ValueError):
    """Exception raised when the available data for beamforming is below a set
    threshold.
    """
