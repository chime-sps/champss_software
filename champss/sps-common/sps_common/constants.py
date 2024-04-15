# CHIME position (from wiki)
CHIME_LAT = 49.3211  # degrees
CHIME_LON = -119.6239  # degrees
CHIME_ELEV = 545.0  # meters
TELESCOPE_ROTATION_ANGLE = -0.071  # degrees
# Dispersion constant used by CHIME/FRB
DM_CONSTANT = 1.0 / 2.41e-4
# The frequency of the top part of the CHIME band
FREQ_TOP = 800.1953125
# The frequency of the bottom part of the CHIME band
FREQ_BOTTOM = 400.1953125
# The path to the SPS intensity data
STORAGE_PATH = "/data/frb-archiver/SPS/"
# One sidereal day in number of seconds
SIDEREAL_S = 86164.1
# One "civil" day in number of seconds
SEC_PER_DAY = 86400.0
# Sampling time of the SPS intensity data
TSAMP = 0.00098304
# Number of native frequency channels at L0
L0_NCHAN = 1024
# Number of native frequency channels at L1
L1_NCHAN = 16384
# Default number of time samples per msgpack file (not Huffmann encoded)
DEFAULT_MSGPACK_NTIME = 1024
# Minimum frequency for candidate search
MIN_SEARCH_FREQ = 9.70127682e-04
# Maximum frequency for candidate search
MAX_SEARCH_FREQ = 2**19 * MIN_SEARCH_FREQ
# Minimum DM for candidate search
MIN_SEARCH_DM = 0.0
# Maximum DM for candidate search
MAX_SEARCH_DM = 10000.0
