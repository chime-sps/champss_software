import datetime as dt
import logging
import shutil
from glob import glob
from os import path, remove

import pytz
from attr import ib as attribute
from attr import s as attrs
from attr.validators import instance_of
from sps_pipeline import utils

log = logging.getLogger(__package__)


@attrs(slots=True)
class CleanUp:
    """
    Class object to delete intermediate products after processing.

    Parameters
    =======
    beamform: bool
        Remove the beamformed filterbank file. Default = True

    dedisp: bool
        Remove the dedispersed time series. Default = True

    ps: bool
        Remove the power spectra file. Default = True

    ps_detections: bool
        Remove the power spectra detections file. Default = True

    candidates: bool
        Remove the candidates file. Default = False
    """

    beamform = attribute(validator=instance_of(bool), default=True)
    dedisp = attribute(validator=instance_of(bool), default=True)
    ps = attribute(validator=instance_of(bool), default=True)
    ps_detections = attribute(validator=instance_of(bool), default=True)
    candidates = attribute(validator=instance_of(bool), default=False)

    def remove_files(self, pointing):
        date = utils.transit_time(pointing).date()
        log.info(f"Cleanup ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}")
        file_path = path.join(
            date.strftime("%Y/%m/%d"),
            f"{pointing.ra:.02f}_{pointing.dec:.02f}",
        )
        if self.beamform:
            if path.exists(
                path.join(
                    file_path,
                    (
                        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}.fil"
                    ),
                )
            ):
                remove(
                    path.join(
                        file_path,
                        (
                            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}.fil"
                        ),
                    )
                )
        if self.dedisp:
            files_to_remove = glob(
                path.join(
                    file_path,
                    (
                        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}*.dat"
                    ),
                )
            )
            files_to_remove.extend(
                glob(
                    path.join(
                        file_path,
                        (
                            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}*.inf"
                        ),
                    )
                )
            )
            for f in files_to_remove:
                remove(f)
        if self.ps:
            if path.exists(
                path.join(
                    file_path,
                    (
                        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra.hdf5"
                    ),
                )
            ):
                remove(
                    path.join(
                        file_path,
                        (
                            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra.hdf5"
                        ),
                    )
                )
        if self.ps_detections:
            if path.exists(
                path.join(
                    file_path,
                    (
                        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_detections.hdf5"
                    ),
                )
            ):
                remove(
                    path.join(
                        file_path,
                        (
                            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_detections.hdf5"
                        ),
                    )
                )
        if self.candidates:
            if path.exists(
                path.join(
                    file_path,
                    (
                        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_candidates.hdf5"
                    ),
                )
            ):
                remove(
                    path.join(
                        file_path,
                        (
                            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_candidates.hdf5"
                        ),
                    )
                )
