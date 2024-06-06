import datetime

import attr
import matplotlib.colors as colors
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from beamformer.utilities import dm
from beamformer.utilities.common import beammod
from matplotlib import pyplot as plt
from sps_common import constants
from sps_common.interfaces.beamformer import Pointing

sidereal_s = constants.SIDEREAL_S


@attr.s(slots=True)
class PointingMapper:
    """
    PointingMapper class to create a sky pointing map.

    Parameters
    =======
    beams: np.array
        Array of FRB beam rows to use (Default: np.arange(0, 224))

    ref_time: datetime.datetime
        Reference time to determine RA, Dec of the pointings (Default: datetime.datetime(2021, 3, 20, 20, 5, 17, tzinfo=datetime.timezone.utc)

    zenith_angle: float
        Zenith angle of CHIME in Dec (Default: 49.32)

    exp: bool
        Boolean operator on whether an exponential modelling of LoS max DM is used (Default: True)

    excess: float
        Excess DM to add to LoS max DM (Default: 50.0)

    excess_fac: float
        Excess DM factor to multiply if exponential modelling is not used (Default: 2.0)

    ra_spacing: float
        Spacing between pointings in the RA direction in degrees (Default: 0.32)

    chunk_size: int
        Chunk size in number of time samples to determine the range of possible pointing length (Default: 40960)
    """

    beams: np.ndarray = attr.ib(default=np.arange(0, 224))
    ref_time: datetime.datetime = attr.ib(
        default=datetime.datetime(2021, 3, 20, 20, 5, 17, tzinfo=datetime.timezone.utc)
    )
    zenith_angle: float = attr.ib(default=49.32)
    exp: bool = attr.ib(default=True)
    excess: float = attr.ib(default=50.0)
    excess_fac: float = attr.ib(default=2.0)
    extragalactic: bool = attr.ib(default=True)
    ra_spacing: float = attr.ib(default=0.32)
    chunk_size: int = attr.ib(default=40960)
    y_pos = attr.ib(factory=list)
    dm_check = dm.DMMap()

    @beams.validator
    def _validate_beam_range(self, attribute, value):
        assert value.max() < 256, "beam cannot be greater than 255"
        assert value.min() > -1, "beam cannot be less than 0"

    @zenith_angle.validator
    def _validate_zenith_angle(self, attribute, value):
        assert -90 < value < 90, "declination of zenith must be between -90 and 90"

    def __attrs_post_init__(self):
        self.y_pos = [beammod.reference_angles[x] for x in self.beams]

    def get_pointing_map(self):
        """Get sky pointing map with a list of RA, Dec and max DM to search to."""
        pointings = []
        for index, beam in enumerate(self.beams):
            no_ra = (
                int(
                    1
                    / (
                        self.ra_spacing
                        / (
                            360.0
                            * np.cos(np.deg2rad(self.y_pos[index] + self.zenith_angle))
                        )
                    )
                )
                + 1
            )
            length = self.get_length(index)
            for r in range(no_ra):
                ra, dec = beammod.get_equatorial_from_position(
                    0,
                    self.y_pos[index],
                    self.ref_time + datetime.timedelta(seconds=r * sidereal_s / no_ra),
                )
                ne2001dm, ymw16dm = self.get_ne2001_ymw16(ra, dec)
                maxdm = self.get_max_dm(
                    ra,
                    dec,
                    ne2001dm,
                    ymw16dm,
                    exp=self.exp,
                    excess=self.excess,
                    excess_fac=self.excess_fac,
                    extragalactic=self.extragalactic,
                )
                nchans = self.get_nchans(maxdm)
                pointing = Pointing(
                    ra=ra,
                    dec=dec,
                    beam_row=beam,
                    length=length,
                    ne2001dm=ne2001dm,
                    ymw16dm=ymw16dm,
                    maxdm=maxdm,
                    nchans=nchans,
                )
                pointings.append(attr.asdict(pointing))
        return pointings

    def get_ne2001_ymw16(self, ra, dec):
        if ra > 359:
            ne2001 = self.dm_check.get_dm_ne2001(dec, 359) + (
                self.dm_check.get_dm_ne2001(dec, 1)
                - self.dm_check.get_dm_ne2001(dec, 359)
            ) * ((ra - 359) / 2)
            ymw16 = self.dm_check.get_dm_ymw16(dec, 359) + (
                self.dm_check.get_dm_ymw16(dec, 1)
                - self.dm_check.get_dm_ymw16(dec, 359)
            ) * ((ra - 359) / 2)
        elif ra < 1:
            ne2001 = self.dm_check.get_dm_ne2001(dec, 1) - (
                self.dm_check.get_dm_ne2001(dec, 1)
                - self.dm_check.get_dm_ne2001(dec, 359)
            ) * ((ra - 1) / 2)
            ymw16 = self.dm_check.get_dm_ymw16(dec, 1) - (
                self.dm_check.get_dm_ymw16(dec, 1)
                - self.dm_check.get_dm_ymw16(dec, 359)
            ) * ((ra - 1) / 2)
        else:
            ne2001 = self.dm_check.get_dm_ne2001(dec, ra).sum()
            ymw16 = self.dm_check.get_dm_ymw16(dec, ra).sum()
        return ne2001, ymw16

    def get_max_dm(
        self,
        ra,
        dec,
        ne2001dm,
        ymw16dm,
        exp=True,
        excess=50.0,
        excess_fac=2.0,
        extragalactic=True,
    ):
        """
        Get LoS max DM from a dict of RA, Dec, ne2001 and ymw16 max DM.

        Inputs :
        pointing - a dict of ra, dec, beam, ne2001dm and ymw16dm
        exp - Boolean input for exponential modelling of max DM to search
        excess - Excess DM to add to the LoS Max DM
        excess_fac - Excess factor to LoS Max DM if exp=False
        extragalactic - Boolean input for extending LoS max DM to M31 and M33
        """
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        gb = coord.galactic.b.value
        if exp:
            if np.abs(gb) < 15:
                maxdm = (np.maximum(ne2001dm, ymw16dm)) * np.exp(
                    0.0313 * np.abs(gb) + 0.223
                ) + excess
            else:
                maxdm = (np.maximum(ne2001dm, ymw16dm)) * 2 + excess
        else:
            maxdm = (np.maximum(ne2001dm, ymw16dm)) * excess_fac + excess
        if extragalactic:
            # Andromeda galaxy
            if 8.355 <= ra <= 13.012 and 40.269 <= dec <= 42.269:
                maxdm += 400
            # M33
            elif 22.761 <= ra <= 24.156 and 30.285 <= dec <= 31.035:
                maxdm += 400
        return maxdm

    def get_nchans(self, maxdm):
        """
        Inputs a pointing dict with ra, dec, maxdm and returns nchans.

        Inputs
        pointing - A dict of ra, dec, maxdm
        """
        if maxdm <= 212.5:
            nchans = 1024
        elif maxdm <= 425:
            nchans = 2048
        elif maxdm <= 850:
            nchans = 4096
        elif maxdm <= 1700:
            nchans = 8192
        else:
            nchans = 16384
        return nchans

    def get_length(self, beam_index):
        """
        Get the length of a beam for a given FRB beam row in multiple of chunks of n
        samples.

        Inputs
        beam - FRB beam row number between 0 and 255
        chunk_size - size of chunk in number of samples
        """
        length = 0
        ra_now, dec_now = beammod.get_equatorial_from_position(
            0, self.y_pos[beam_index], self.ref_time
        )
        for td in np.arange(-22000, 22000, (self.chunk_size // 4) * 0.00098304):
            x_now, y_now = beammod.get_position_from_equatorial(
                ra_now, dec_now, self.ref_time + datetime.timedelta(seconds=td)
            )
            if -0.6 < x_now < 1.0 and y_now < (90 - self.zenith_angle):
                length += self.chunk_size // 4
        length = (length // self.chunk_size) * self.chunk_size
        return int(length)

    def plot_dm_nchan_map(self, pointings, save=True):
        """Plots the all sky max DM and nchans requirements maps."""
        fig = plt.figure(figsize=(15, 20))
        ax = fig.add_subplot(211, projection="aitoff")
        ax2 = fig.add_subplot(212, projection="aitoff")
        set_a = []
        set_b = []
        set_c = []
        set_d = []
        set_e = []
        total_length = 0
        length_a = 0
        length_b = 0
        length_c = 0
        length_d = 0
        length_e = 0

        for p in pointings:
            if p["ra"] > 180:
                p["ra"] -= 360
            total_length += p["length"]
            if p["nchans"] == 1024:
                set_a.append(p)
                length_a += p["length"]
            elif p["nchans"] == 2048:
                set_b.append(p)
                length_b += p["length"]
            elif p["nchans"] == 4096:
                set_c.append(p)
                length_c += p["length"]
            elif p["nchans"] == 8192:
                set_d.append(p)
                length_d += p["length"]
            elif p["nchans"] == 16384:
                set_e.append(p)
                length_e += p["length"]

        ave_nchans = (
            len(set_a) * 1024
            + len(set_b) * 2048
            + len(set_c) * 4096
            + len(set_d) * 8192
            + len(set_e) * 16384
        ) / len(pointings)
        ax.set_title(f"Average number of channels per pointing: {ave_nchans}")
        ax.scatter(
            [np.deg2rad(a["ra"]) for a in set_a],
            [np.deg2rad(a["dec"]) for a in set_a],
            color="lavender",
            s=3,
            label="DM < 212.5, {:.2f} %% of pointings, {:.2f} %% of data".format(
                (len(set_a) * 100 / len(pointings)), (length_a * 100 / total_length)
            ),
        )
        ax.scatter(
            [np.deg2rad(b["ra"]) for b in set_b],
            [np.deg2rad(b["dec"]) for b in set_b],
            color="skyblue",
            s=3,
            label="DM < 425, {:.2f} %% of pointings, {:.2f} %% of data".format(
                (len(set_b) * 100 / len(pointings)), (length_b * 100 / total_length)
            ),
        )
        ax.scatter(
            [np.deg2rad(c["ra"]) for c in set_c],
            [np.deg2rad(c["dec"]) for c in set_c],
            color="lime",
            s=3,
            label="DM < 850, {:.2f} %% of pointings, {:.2f} %% of data".format(
                (len(set_c) * 100 / len(pointings)), (length_c * 100 / total_length)
            ),
        )
        ax.scatter(
            [np.deg2rad(d["ra"]) for d in set_d],
            [np.deg2rad(d["dec"]) for d in set_d],
            color="orange",
            s=3,
            label="DM < 1700, {:.2f} %% of pointings, {:.2f} %% of data".format(
                (len(set_d) * 100 / len(pointings)), (length_d * 100 / total_length)
            ),
        )
        ax.scatter(
            [np.deg2rad(e["ra"]) for e in set_e],
            [np.deg2rad(e["dec"]) for e in set_e],
            color="red",
            s=3,
            label="DM > 1700, {:.2f} %% of pointings, {:.2f} %% of data".format(
                (len(set_e) * 100 / len(pointings)), (length_e * 100 / total_length)
            ),
        )
        ax.legend(markerscale=3.0, loc="lower right")
        ax.grid()

        ave_dm = sum(p["maxdm"] for p in pointings) / len(pointings)
        ax2.set_title(f"Average max DM to search : {ave_dm}")
        dm_plot = ax2.scatter(
            [np.deg2rad(p["ra"]) for p in pointings],
            [np.deg2rad(p["dec"]) for p in pointings],
            c=[p["maxdm"] for p in pointings],
            norm=colors.LogNorm(vmin=100, vmax=10000),
        )
        ax2.grid()
        cbar = fig.colorbar(dm_plot, ticks=[100, 500, 1000, 5000, 10000])
        cbar.set_label("DM value")

        if save:
            plt.savefig("pointings_dm_nchans.png", bbox_inches="tight", dpi=150)
        else:
            plt.show()
