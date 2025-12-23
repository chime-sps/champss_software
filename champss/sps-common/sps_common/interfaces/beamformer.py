import logging

import numpy as np
from attr import ib as attrib
from attr import s as attrs
from attr.converters import optional
from attr.setters import convert
from attr.validators import deep_iterable, in_, instance_of
from sps_common.constants import TSAMP
from sps_common.conversion import convert_ra_dec, unix_to_mjd
from sps_common.filterbank import write_to_filterbank

log = logging.getLogger(__name__)


@attrs(slots=True)
class Pointing:
    """
    Pointing class to store the properties of a pointing into the CHIME/SPS database.
    Does a self validation to ensure all inputs are valid.

    Parameters
    =======
    beam_row: int
        FRB beam row of the pointing. Must be between 0 and 255.

    ra: float
        Right Ascension of the pointing. Must be between 0 and 360.

    dec: float
        Declination of the pointing. Must be between -90 and 90.

    length: int
        Length of the pointing in number of time samples. Must be larger than 0.

    ne2001dm: float
        The line-of-sight max DM from ne2001 model. Must be larger than 0.

    ymw16dm: float
        The line-of-sight max DM from ymw16 model. Must be larger than 0.

    maxdm:float
        The max DM value to search for the pointing. Must be larger than both ne2001dm and ymw16dm.

    nchans: int
        The number of channels required for this pointing. Must be either 1024, 2048, 4096, 8192, or 16384.

    pointing_id: str or None
        The id of the pointing, if extracted from sps-databases, otherwise will be None.
    """

    beam_row = attrib(converter=int)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    length = attrib(converter=int)
    ne2001dm = attrib(converter=float)
    ymw16dm = attrib(converter=float)
    maxdm = attrib(converter=float)
    nchans = attrib(
        converter=int,
        validator=in_(
            [
                1024,
                2048,
                4096,
                8192,
                16384,
            ]
        ),
    )
    pointing_id = attrib(default=None, converter=optional(str), on_setattr=convert)

    def __attrs_post_init__(self):
        if self.maxdm <= np.max([self.ne2001dm, self.ymw16dm]):
            raise ValueError(
                f"The attribute maxdm must be larger than both ne2001dm and ymw16dm."
            )

    @beam_row.validator
    def _validate_beam_row(self, attribute, value):
        if not 0 <= value <= 255:
            raise ValueError(
                f"Beam row attribute ({attribute.name}={value}) must be between 0 and"
                " 255."
            )

    @ra.validator
    def _validate_ra(self, attribute, value):
        if not 0.0 <= value <= 360.0:
            raise ValueError(
                f"Right ascension attribute ({attribute.name}={value}) must be between"
                " 0 and 360."
            )

    @dec.validator
    def _validate_dec(self, attribute, value):
        if not -90.0 <= value <= 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) must be between -90"
                " and 90."
            )

    @length.validator
    def _validate_length(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Number of time samples attribute ({attribute.name}={value}) must be"
                " larger than 0."
            )

    @ne2001dm.validator
    @ymw16dm.validator
    def _validate_dm(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) must be larger than 0."
            )


@attrs
class ActivePointing:
    """
    Interface for active pointings with all the information required by SkyBeamCreation.

    Parameters
    ----------
    ra: float
        The Right Ascension of the active pointing. Must be between 0 and 360.

    dec: float
        The Declination of the active pointing. Must be between -90 and 90.

    nchan: int
        The number of channels in the active pointing. Must be either 1024, 2048, 4096, 8192, or 16384.

    ntime: int
        The number of time samples in the active pointing. Must be larger than 0.

    maxdm: float
        The maximum DM to search to for the active pointing. Must be larger than 0.

    max_beams: list(dict)
        The FRB beams that transited the active pointing and their transit time. Individual dict entry in the list must
        contain the keys 'beam', 'utc_start', and 'utc_end' to signify the FRB beam used, the start and end transit
        time in unix utc time.

    beam_row: int
        The FRB beam row of the active pointing. Must be between 0 and 255.

    obs_id: str or None
        The observation ID of the active pointing. None if a database entry is not produced.

    pointing_id: str or None
        The id of the pointing, if extracted from sps-databases, otherwise will be None.
    """

    ra = attrib(converter=float)
    dec = attrib(converter=float)
    nchan = attrib(
        converter=int,
        validator=in_(
            [
                1024,
                2048,
                4096,
                8192,
                16384,
            ]
        ),
    )
    ntime = attrib(converter=int)
    maxdm = attrib(converter=float)
    max_beams = attrib(
        validator=deep_iterable(
            member_validator=instance_of(dict), iterable_validator=instance_of(list)
        )
    )
    beam_row = attrib(converter=int)
    sub_pointing = attrib(default=0, converter=int)
    obs_id = attrib(default=None, converter=optional(str), on_setattr=convert)
    pointing_id = attrib(default=None, converter=optional(str), on_setattr=convert)

    @ra.validator
    def _validate_ra(self, attribute, value):
        if not 0.0 <= value <= 360.0:
            raise ValueError(
                f"Right ascension attribute ({attribute.name}={value}) must be between"
                " 0 and 360."
            )

    @dec.validator
    def _validate_dec(self, attribute, value):
        if not -90.0 <= value <= 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) must be between -90"
                " and 90."
            )

    @ntime.validator
    def _validate_ntime(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Number of time samples attribute ({attribute.name}={value}) must be"
                " larger than 0."
            )

    @maxdm.validator
    def _validate_maxdm(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) must be larger than 0."
            )

    @max_beams.validator
    def _validate_max_beams(self, attribute, value):
        for val in value:
            if not "beam" and "utc_start" and "utc_end" in val.keys():
                raise KeyError(
                    f"The dict keys in the elements of ({attribute.name}) does not"
                    " contain 'beam', 'utc_start', and 'utc_end'."
                )

    @beam_row.validator
    def _validate_beam_row(self, attribute, value):
        if not 0 <= value <= 255:
            raise ValueError(
                f"Beam row attribute ({attribute.name}={value}) must be between 0 and"
                " 255."
            )


@attrs
class SkyBeam:
    """
    Interface for skybeam to be sent to dedispersion process.

    Parameters
    ----------
    spectra: np.ndarray
        The spectra of the skybeam formed. Must have shape (nchan, ntime).

    ra: float
        The Right Ascension of the skybeam formed. Must be between 0 and 360.

    dec: float
        The Declination of the skybeam formed. Must be between -90 and 90.

    nchan: int
        The number of channels in the skybeam formed. Must be either 1024, 2048, 4096, 8192, or 16384.

    ntime: int
        The number of time samples in the skybeam formed. Must be larger than 0.

    maxdm: float
        The maximum DM to search to for the skybeam formed. Must be larger than 0.

    beam_row: int
        The FRB beam row of the skybeam formed. Must be between 0 and 255.

    utc_start: float
        The unix utc start time of the skybeam formed.

    obs_id: str or None
        The observation ID of the skybeam formed. None if a database entry is not created for the observation.

    nbits: int
        The number of bits to save the skybeam formed into a filterbank. Default = 32.
    """

    spectra = attrib(validator=instance_of(np.ndarray))
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    nchan = attrib(
        converter=int,
        validator=in_(
            [
                1024,
                2048,
                4096,
                8192,
                16384,
            ]
        ),
    )
    ntime = attrib(converter=int)
    maxdm = attrib(converter=float)
    beam_row = attrib(converter=int)
    utc_start = attrib(converter=float)
    obs_id = attrib(default=None, converter=optional(str), on_setattr=convert)
    pointing_id = attrib(default=None, converter=optional(str), on_setattr=convert)
    nbits = attrib(default=32, converter=int)

    def __attrs_post_init__(self):
        if self.spectra.shape != (self.nchan, self.ntime):
            raise ValueError(
                f"The attribute spectra does not have the right number of channels and"
                f" number of time samples."
            )

    @spectra.validator
    def _validate_spectra(self, attribute, value):
        if value.size <= 0:
            raise ValueError(
                f"Spectra attribute ({attribute.name}={value}) must have size larger"
                " than 0."
            )
        if value.ndim != 2:
            raise ValueError(
                f"Spectra attribute ({attribute.name}={value}) must number of dimension"
                " of 2."
            )

    @ra.validator
    def _validate_ra(self, attribute, value):
        if not 0.0 <= value <= 360.0:
            raise ValueError(
                f"Right ascension attribute ({attribute.name}={value}) must be between"
                " 0 and 360."
            )

    @dec.validator
    def _validate_dec(self, attribute, value):
        if not -90.0 <= value <= 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) must be between -90"
                " and 90."
            )

    @ntime.validator
    def _validate_ntime(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Number of time samples attribute ({attribute.name}={value}) must be"
                " larger than 0."
            )

    @maxdm.validator
    def _validate_maxdm(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) must be larger than 0."
            )

    @beam_row.validator
    def _validate_beam_row(self, attribute, value):
        if not 0 <= value <= 255:
            raise ValueError(
                f"Beam row attribute ({attribute.name}={value}) must be between 0 and"
                " 255."
            )

    def write(self, oname):
        """
        Write spectra to filterbank format.

        Parameters
        =======
        oname: str
            The output filename of the spectra.
        """
        srcra, srcdec = convert_ra_dec(self.ra, self.dec)
        if srcdec.startswith("-"):
            srcname = f"J{srcra[:4]}{srcdec[:3]}"
        else:
            srcname = f"J{srcra[:4]}+{srcdec[:2]}"
        start_mjd = unix_to_mjd(self.utc_start)
        write_to_filterbank(
            self.spectra,
            self.nchan,
            self.ntime,
            TSAMP,
            self.beam_row,
            start_mjd,
            self.nbits,
            srcname,
            srcra,
            srcdec,
            oname,
        )

    def append(self, oname):
        """
        Append spectra to an existing filterbank file.

        Parameters
        =======
        oname: str
            The output filename of the filterbank to append to.
        """
        from sps_common.filterbank import append_spectra

        with open(oname, "ab") as fb_out:
            append_spectra(fb_out, self.spectra, nbits=self.nbits, verbose=False)
