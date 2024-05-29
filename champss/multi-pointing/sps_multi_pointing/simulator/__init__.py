import copy
import os

import numpy as np

from champss.multi-pointing.sps_multi_pointing.simulator.pointing import Pointing
from champss.multi-pointing.sps_multi_pointing.simulator.sources import (
    generate_known_sources,
    generate_source,
)
from champss.multi-pointing.sps_multi_pointing.simulator.utils import (
    make_single_pointing_candidate_collection,
)

reference_angles = np.load(os.path.dirname(__file__) + "/reference_angles.npy")


class PointingGrid:
    def __init__(self, num_rows=5, num_cols=6, beam_row=128):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.beam_row = beam_row

    def create(self):
        pulsars = generate_source(num_sources=1)
        consistent_rfi = generate_source(num_sources=5, source="RFI")
        pulsar_pointing_ids = []
        psr_row, psr_col = (
            np.random.choice(range(self.num_rows)),
            np.random.choice(range(self.num_cols)),
        )
        ra, dec = self.get_ra_dec(psr_row, psr_col)
        ppi = f"{ra:.3f}_{dec:.3f}"
        self.main_ppi = ppi
        pulsar_pointing_ids.append(ppi)
        if np.random.choice(["bright", "bright", "dim"]) == "bright":
            if psr_row + 1 < self.num_rows and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row + 1, psr_col)
                pulsar_pointing_ids.append(f"{side_ra:.3f}_{side_dec:.3f}")
            if psr_row - 1 > 0 and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row - 1, psr_col)
                pulsar_pointing_ids.append(f"{side_ra:.3f}_{side_dec:.3f}")
            if psr_col + 1 < self.num_cols and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row, psr_col + 1)
                pulsar_pointing_ids.append(f"{side_ra:.3f}_{side_dec:.3f}")
            if psr_col - 1 > 0 and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row, psr_col - 1)
                pulsar_pointing_ids.append(f"{side_ra:.3f}_{side_dec:.3f}")
        print("Injecting pulsar in: ", pulsar_pointing_ids)
        num_consistent_rfi_ids = np.random.choice(range(self.num_rows * self.num_cols))
        consistent_rfi_ids = []
        for _ in range(num_consistent_rfi_ids):
            ra, dec = self.get_ra_dec(
                np.random.choice(range(self.num_rows)),
                np.random.choice(range(self.num_cols)),
            )
            ppi = f"{ra:.3f}_{dec:.3f}"
            if ppi not in consistent_rfi_ids:
                consistent_rfi_ids.append(ppi)

        ks = np.load(
            os.path.dirname(__file__)
            + "/../known_source_sifter/ks_database_210203_1541.npy"
        )
        ks = ks[ks["pos_ra_deg"] > 270]
        ks = ks[ks["pos_ra_deg"] < 300]
        ks = ks[ks["pos_dec_deg"] > 22]
        ks = ks[ks["pos_dec_deg"] < 28]
        ks = ks[np.asarray([not s.isnumeric() for s in ks["source_name"]])]
        ks = ks[ks["spin_period_s"] > 0.02]
        ksources = []
        for i in range(10):
            idx = i * len(ks) // 10
            ksources.append(ks[idx])
        np.savez("injected_known_sources.npz", sources=ksources)
        known_pulsar_pointing_ids = {}
        for s in ksources:
            kss = generate_known_sources([s])
            psr_row, psr_col = self.get_row_col(s["pos_ra_deg"], s["pos_dec_deg"])
            ra, dec = self.get_ra_dec(psr_row, psr_col)
            ppi = f"{ra:.3f}_{dec:.3f}"
            print(
                s["source_name"],
                psr_row,
                psr_col,
                s["pos_ra_deg"],
                s["pos_dec_deg"],
                ppi,
            )
            known_pulsar_pointing_ids[ppi] = kss
            if np.random.choice(["bright", "bright", "dim"]) == "bright":
                if psr_row + 1 < self.num_rows and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row + 1, psr_col)
                    known_pulsar_pointing_ids[
                        f"{side_ra:.3f}_{side_dec:.3f}"
                    ] = kss
                if psr_row - 1 > 0 and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row - 1, psr_col)
                    known_pulsar_pointing_ids[
                        f"{side_ra:.3f}_{side_dec:.3f}"
                    ] = kss
                if psr_col + 1 < self.num_cols and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row, psr_col + 1)
                    known_pulsar_pointing_ids[
                        f"{side_ra:.3f}_{side_dec:.3f}"
                    ] = kss
                if psr_col - 1 > 0 and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row, psr_col - 1)
                    known_pulsar_pointing_ids[
                        f"{side_ra:.3f}_{side_dec:.3f}"
                    ] = kss

        print(known_pulsar_pointing_ids.keys())
        count = 0
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                ra, dec = self.get_ra_dec(r, c)
                pointing_id = f"{ra:.3f}_{dec:.3f}"
                # print(pointing_id, pointing_id in known_pulsar_pointing_ids)
                rfi = []
                psrs = []
                if pointing_id in pulsar_pointing_ids:
                    psrs = pulsars
                    # if not self.main_ppi == pointing_id:
                    #    psrs[0] = self.change_brightness(psrs[0], factor=0.75)
                if len(psrs):
                    print("Pulsars: ", psrs)
                if pointing_id in known_pulsar_pointing_ids:
                    count += 1
                    print(count)
                    psrs += known_pulsar_pointing_ids[pointing_id]
                if len(psrs):
                    print("with Known Pulsars: ", psrs)
                if pointing_id in consistent_rfi_ids:
                    rfi = copy.deepcopy(consistent_rfi)
                    # factor = np.random.random() + np.random.choice([0, 0, 0, 1])
                    # for i in range(len(rfi)):
                    #    rfi[i] = self.change_brightness(rfi[i], factor=factor)
                # add some random RFI
                rfi += generate_source(num_sources=10, source="RFI")
                self.create_pointing(pointing_id, psrs, rfi)
                if count % 100 == 0:
                    print(count)
                count += 1

    def get_ra_dec(self, row, col):
        dec = reference_angles[int(row + self.beam_row)] + 49.32
        ra = col * 0.32 / np.cos(np.deg2rad(dec)) + 270
        return ra, dec

    def get_row_col(self, ra, dec):
        decs = []
        ras = []
        rows = []
        cols = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                _ra, _dec = self.get_ra_dec(r, c)
                decs.append(_dec)
                ras.append(_ra)
                rows.append(r)
                cols.append(c)
        row_idx = np.argmin(np.abs(np.asarray(decs) - dec))
        col_idx = np.argmin(np.abs(np.asarray(ras) - ra))
        return rows[row_idx], cols[col_idx]

    def create_pointing(self, pointing_id, pulsars, rfi):
        p = Pointing(pulsars, rfi)
        groups, gs = p.create()

        ra, dec = pointing_id.split("_")
        spcc = make_single_pointing_candidate_collection(
            ra=float(ra), dec=float(dec), group_summary=gs, groups=groups
        )
        spcc.write(f"{pointing_id}_sim_ps_candidates.npz")

        if len(pulsars):
            print("Pulsars:", pulsars)
        np.save(f"{pointing_id}_sim_pulsars", pulsars)

    def change_brightness(self, source, factor=0.8):
        source["max_power"] *= factor
        if source["max_power"] < 5.0:
            source["max_power"] /= factor
            return source
        if factor < 1:
            delta_dm = source["max_dm"] - source["dm"]
            source["max_dm"] = max(
                source["dm"] + 1, source["max_dm"] - delta_dm * factor
            )
            source["min_dm"] = min(
                source["dm"] - 1, source["min_dm"] + delta_dm * factor
            )
        else:
            delta_dm = source["max_dm"] - source["dm"]
            source["max_dm"] = source["max_dm"] + delta_dm * factor
            source["min_dm"] = max(0, source["min_dm"] - delta_dm * factor)

        source["num_harm"] = 0
        return source
