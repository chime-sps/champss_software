from astropy.time import update_leap_seconds
from astropy.utils.data import download_file

f = download_file(
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
    cache=True,
)
update_leap_seconds()
