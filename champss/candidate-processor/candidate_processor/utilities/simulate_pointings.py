import numpy as np
import copy
import beam_model

config = beam_model.current_config
beammod = beam_model.current_model_class(config)
np.random.seed(123456)


class Group(object):
    def __init__(
        self,
        f,
        dm,
        max_dm,
        min_dm,
        max_power,
        delta_freq=9.70127682e-04,
        delta_dm=0.125,
        shape="RFI",
    ):
        self.freq = f
        self.dm = dm
        self.delta_dm = delta_dm
        self.delta_freq = delta_freq
        self.max_dm = max_dm
        self.min_dm = min_dm
        self.max_power = max_power
        self.shape = shape
        self.dtype = [("f", "<f8"), ("dm", "<f8"), ("dc", "<f8"), ("hhat", "<f8")]
        self.min_power = 5.0
        self.num_dms = int((max_dm - min_dm) / delta_dm)

    def create(self):
        dms = np.linspace(self.min_dm, self.max_dm, self.num_dms)
        hhats = self.prof_shape()
        freq_choices = np.arange(
            self.freq - 4 * self.delta_freq,
            self.freq + 4 * self.delta_freq,
            self.delta_freq,
        )
        group = []
        for i, d in enumerate(dms):
            group.append((np.random.choice(freq_choices), d, 0, hhats[i]))
        return np.asarray(group, dtype=self.dtype)

    def prof_shape(self):
        if self.shape == "ASTRO":
            values = [self.pulsar_shape(i) for i in range(self.num_dms)]
        if self.shape == "RFI":
            choice = np.random.choice(["decline", "random", "flat"])
            if choice == "decline":
                values = [self.decline_shape(i) for i in range(self.num_dms)]
            elif choice == "random":
                values = [self.random_shape(i) for i in range(self.num_dms)]
            else:
                values = [self.max_power - np.random.random()] * self.num_dms
                values[self.num_dms // 2] = self.max_power
        return np.asarray(values)

    def pulsar_shape(self, i):
        decay = (abs(i - self.num_dms // 2)) ** 0.25
        if decay < 1:
            decay = 1
        return max(self.min_power, self.max_power / decay)

    def decline_shape(self, i):
        decay = (i + 1) ** 0.3333
        return max(self.min_power, self.max_power / decay)

    def random_shape(self, i):
        decay = np.random.random() * 10
        if decay < 1:
            decay = 1.5
        return max(self.min_power, self.max_power / decay)


class GroupSummary(object):
    def __init__(
        self, group_id, freq, dm, max_dm, min_dm, max_power, num_harm=0, rfi=False
    ):
        self.group_id = group_id
        self.f = freq
        self.dm = dm
        self.max_dm = max_dm
        self.min_dm = min_dm
        self.max_power = max_power
        self.num_harm = num_harm
        self.rfi = rfi
        self.shape = "RFI" if self.rfi else "ASTRO"

    def create(self):
        groups = {}
        group_summary = {}
        gs = {
            "f": self.f,
            "dc": 0,
            "dm": self.dm,
            "min_dm": self.min_dm,
            "max_dm": self.max_dm,
            "hhat": self.max_power,
            "rfi": self.rfi,
            "harmonics": {},
        }
        g = Group(
            f=gs["f"],
            dm=gs["dm"],
            min_dm=gs["min_dm"],
            max_dm=gs["max_dm"],
            max_power=gs["hhat"],
            shape=self.shape,
        )
        groups[self.group_id] = g.create()
        gs["dm"] = groups[self.group_id]["dm"][np.argmax(groups[self.group_id]["hhat"])]
        harm = copy.deepcopy(gs)
        for h in range(self.num_harm):
            harm["f"] = harm["f"] / 2
            harm["hhat"] = harm["hhat"] * 0.8
            g = Group(
                f=harm["f"],
                dm=harm["dm"],
                min_dm=harm["min_dm"],
                max_dm=harm["max_dm"],
                max_power=harm["hhat"],
                shape=self.shape,
            )
            groups[self.group_id + h + 1] = g.create()
            harm["dm"] = groups[self.group_id + h + 1]["dm"][
                np.argmax(groups[self.group_id + h + 1]["hhat"])
            ]
            gs["harmonics"][self.group_id + h + 1] = harm
            harm = copy.deepcopy(harm)
        group_summary[self.group_id] = gs
        return groups, group_summary


class Pointing(object):
    def __init__(self, pulsars=[], rfi=[]):
        self.pulsars = pulsars
        self.rfi = rfi

    def create(self):
        groups = {}
        group_summary = {}
        group_id = 1
        for item in self.pulsars + self.rfi:
            item["group_id"] = group_id
            gs = GroupSummary(**item)
            group, group_s = gs.create()
            for g in group:
                groups[g] = group[g]
            for g in group_s:
                group_summary[g] = group_s[g]
            group_id = item["num_harm"] + group_id + 1
        return groups, group_summary


def narrow_band(dm):
    max_dm = dm + np.random.choice(range(50, 100))
    min_dm = max(0, dm - np.random.choice(range(50, 100)))
    return max_dm, min_dm


def broad_band(dm):
    max_dm = dm + np.random.choice(range(2, 5))
    min_dm = max(0, dm - np.random.choice(range(2, 5)))
    return max_dm, min_dm


def generate_known_sources(sources):
    data = []
    for s in sources:
        dm = s["dm"]
        freq = 1 / s["spin_period_s"]
        signal_type = "broad"
        rfi = False
        max_power = max(10, np.random.random() * np.random.choice([15, 20, 20, 30]))
        if max_power <= 15:
            num_harm = 0
        elif max_power <= 25:
            num_harm = 5
        else:
            num_harm = 10
        max_dm, min_dm = broad_band(dm)
        data.append(
            dict(
                freq=freq,
                dm=dm,
                max_dm=max_dm,
                min_dm=min_dm,
                max_power=max_power,
                num_harm=num_harm,
                rfi=rfi,
            )
        )
    return data


def generate_source(num_sources=10, source="ASTRO"):
    data = []
    for _ in range(num_sources):
        freq = np.random.random() * np.random.choice(range(0, 50))
        dm = (
            np.random.random()
            * np.random.choice([10, 10, 10, 10, 10, 50, 50, 100, 100, 200, 300])
            + 10
        )
        signal_type = np.random.choice(["narrow", "broad", "broad", "broad"])
        rfi = True
        max_power = max(
            7.5, np.random.random() * np.random.choice([10, 10, 10, 10, 10, 10, 20, 20])
        )
        if max_power <= 15:
            num_harm = 0
        elif max_power <= 25:
            num_harm = 2
        else:
            num_harm = 10
        if source == "ASTRO":
            dm = (
                np.random.random()
                * np.random.choice([50, 50, 100, 100, 100, 100, 100, 150, 200, 300])
                + 10
            )
            freq = np.random.random() * np.random.choice(range(3)) + 0.125
            signal_type = "broad"
            rfi = False
            max_power = max(10, np.random.random() * np.random.choice([15, 20, 20, 30]))
            if max_power <= 15:
                num_harm = 0
            elif max_power <= 25:
                num_harm = 5
            else:
                num_harm = 10
        if signal_type == "narrow":
            max_dm, min_dm = narrow_band(dm)
        else:
            max_dm, min_dm = broad_band(dm)
        data.append(
            dict(
                freq=freq,
                dm=dm,
                max_dm=max_dm,
                min_dm=min_dm,
                max_power=max_power,
                num_harm=num_harm,
                rfi=rfi,
            )
        )
    return data


class PointingGrid(object):
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
        ppi = "{:.3f}_{:.3f}".format(ra, dec)
        self.main_ppi = ppi
        pulsar_pointing_ids.append(ppi)
        if np.random.choice(["bright", "bright", "dim"]) == "bright":
            if psr_row + 1 < self.num_rows and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row + 1, psr_col)
                pulsar_pointing_ids.append("{:.3f}_{:.3f}".format(side_ra, side_dec))
            if psr_row - 1 > 0 and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row - 1, psr_col)
                pulsar_pointing_ids.append("{:.3f}_{:.3f}".format(side_ra, side_dec))
            if psr_col + 1 < self.num_cols and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row, psr_col + 1)
                pulsar_pointing_ids.append("{:.3f}_{:.3f}".format(side_ra, side_dec))
            if psr_col - 1 > 0 and np.random.choice([True, False]):
                side_ra, side_dec = self.get_ra_dec(psr_row, psr_col - 1)
                pulsar_pointing_ids.append("{:.3f}_{:.3f}".format(side_ra, side_dec))
        print("Injecting pulsar in: ", pulsar_pointing_ids)
        num_consistent_rfi_ids = np.random.choice(range(self.num_rows * self.num_cols))
        consistent_rfi_ids = []
        for _ in range(num_consistent_rfi_ids):
            ra, dec = self.get_ra_dec(
                np.random.choice(range(self.num_rows)),
                np.random.choice(range(self.num_cols)),
            )
            ppi = "{:.3f}_{:.3f}".format(ra, dec)
            if ppi not in consistent_rfi_ids:
                consistent_rfi_ids.append(ppi)

        ks = np.load("ks_database.npy")
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
            ppi = "{:.3f}_{:.3f}".format(ra, dec)
            print(
                s["source_name"],
                psr_row,
                psr_col,
                s["pos_ra_deg"],
                s["pos_dec_deg"],
                ppi,
            )
            main_ppi = ppi
            known_pulsar_pointing_ids[ppi] = kss
            if np.random.choice(["bright", "bright", "dim"]) == "bright":
                if psr_row + 1 < self.num_rows and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row + 1, psr_col)
                    known_pulsar_pointing_ids[
                        "{:.3f}_{:.3f}".format(side_ra, side_dec)
                    ] = kss
                if psr_row - 1 > 0 and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row - 1, psr_col)
                    known_pulsar_pointing_ids[
                        "{:.3f}_{:.3f}".format(side_ra, side_dec)
                    ] = kss
                if psr_col + 1 < self.num_cols and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row, psr_col + 1)
                    known_pulsar_pointing_ids[
                        "{:.3f}_{:.3f}".format(side_ra, side_dec)
                    ] = kss
                if psr_col - 1 > 0 and np.random.choice([True, False]):
                    side_ra, side_dec = self.get_ra_dec(psr_row, psr_col - 1)
                    known_pulsar_pointing_ids[
                        "{:.3f}_{:.3f}".format(side_ra, side_dec)
                    ] = kss

        print(known_pulsar_pointing_ids.keys())
        count = 0
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                ra, dec = self.get_ra_dec(r, c)
                pointing_id = "{:.3f}_{:.3f}".format(ra, dec)
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
                    rfi = consistent_rfi
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
        dec = beammod.reference_angles[int(row + self.beam_row)] + 49.32
        ra = col * 0.32 / np.cos(np.deg2rad(dec)) + 270
        return ra, dec

    def get_row_col(self, ra, dec):
        diffs = []
        # ra_dec = []
        # ptgs = []
        decs = []
        ras = []
        idxs = []
        rows = []
        cols = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                _ra, _dec = self.get_ra_dec(r, c)
                decs.append(_dec)
                ras.append(_ra)
                rows.append(r)
                cols.append(c)
        # print(min(decs), max(decs))
        # print(min(ras), max(ras))
        row_idx = np.argmin(np.abs(np.asarray(decs) - dec))
        col_idx = np.argmin(np.abs(np.asarray(ras) - ra))
        return rows[row_idx], cols[col_idx]

    def create_pointing(self, pointing_id, pulsars, rfi):
        p = Pointing(pulsars, rfi)
        groups, gs = p.create()
        if len(pulsars):
            print(pulsars)
        np.savez(
            "{}_fake_pointing.npz".format(pointing_id),
            group=groups,
            group_summary=gs,
            pulsars=pulsars,
        )

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

        if source["max_power"] <= 15:
            num_harm = 0
        elif source["max_power"] <= 25:
            source["num_harm"] = 2
        else:
            source["num_harm"] = 10
        return source


def main():
    p = PointingGrid(num_rows=16, num_cols=90, beam_row=59)
    p.create()


if __name__ == "__main__":
    main()
