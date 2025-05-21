import attr
import numpy as np


@attr.s(slots=True)
class Pointing(object):
    """
    Pointing class to store the properties of a pointing into the CHIME/SPS database.
    Does a self validation to ensure all inputs are valid.

    Parameters
    =======
    beam_row: int
        Beam row of the pointing.

    ra: float
        Right Ascension of the pointing

    dec: float
        Declination of the pointing

    length: int
        Length of the pointing in number of samples

    ne2001dm: float
        The line-of-sight max DM from ne2001 model

    ymw16dm: float
        The line-of-sight max DM from ymw16 model

    maxdm:float
        The max DM value to search for

    nchans: int
        The number of channels required for this pointing
    """

    beam_row = attr.ib(converter=int)
    ra = attr.ib(converter=float)
    dec = attr.ib(converter=float)
    length = attr.ib(converter=int)
    ne2001dm = attr.ib(converter=float)
    ymw16dm = attr.ib(converter=float)
    maxdm = attr.ib(converter=float)
    nchans = attr.ib(converter=int)

    @beam_row.validator
    def _validate_beam_row(self, attribute, value):
        assert 0 <= value <= 255, "invalid beam row"

    @ra.validator
    def _validate_ra(self, attribute, value):
        assert 0.0 <= value <= 360.0, "invalid ra value"

    @dec.validator
    def _validate_dec(self, attribute, value):
        assert -90.0 <= value <= 90.0, "invalid dec value"

    @length.validator
    def _validate_length(self, attribute, value):
        assert value > 0, "length of pointing must be larger than zero"

    @ne2001dm.validator
    def _validate_ne2001(self, attribute, value):
        assert value >= 0.0, "max LoS DM must be larger than zero"

    @ymw16dm.validator
    def _validate_ymw16(self, attribute, value):
        assert value >= 0.0, "max LoS DM must be larger than zero"

    @maxdm.validator
    def _validate_maxdm(self, attribute, value):
        assert value >= np.max(
            [self.ne2001dm, self.ymw16dm]
        ), "The max DM to search to must be larger than the max of ne2001 or ymw16 models"

    @nchans.validator
    def _validate_nchans(self, attribute, value):
        assert value in [
            1024,
            2048,
            4096,
            8192,
            16384,
        ], "nchans must be either 1024, 2048, 4096, 8192 or 16384 channels"
