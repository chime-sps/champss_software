import copy
import numpy as np

from sps_common.constants import (
    MIN_SEARCH_FREQ,
    MAX_SEARCH_FREQ,
    MIN_SEARCH_DM,
    MAX_SEARCH_DM,
)


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
        self.dtype = [("f", "<f8"), ("dm", "<f8"), ("dc", "<f8"), ("sigma", "<f8")]
        self.min_power = 5.0
        self.num_dms = int((max_dm - min_dm) / delta_dm)

    def create(self):
        dms = np.linspace(self.min_dm, self.max_dm, self.num_dms)
        sigmas = self.prof_shape()
        freq_choices = np.arange(
            self.freq - 4 * self.delta_freq,
            self.freq + 4 * self.delta_freq,
            self.delta_freq,
        )
        group = []
        for i, d in enumerate(dms):
            g = (np.random.choice(freq_choices), d, 0, sigmas[i])
            # only accept simulated detections within the search range
            if (
                MIN_SEARCH_FREQ <= g[0] <= MAX_SEARCH_FREQ
                and MIN_SEARCH_DM <= g[1] <= MAX_SEARCH_DM
            ):
                group.append(g)
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
            "sigma": self.max_power,
            "rfi": self.rfi,
            "harmonics": {},
        }
        g = Group(
            f=gs["f"],
            dm=gs["dm"],
            min_dm=gs["min_dm"],
            max_dm=gs["max_dm"],
            max_power=gs["sigma"],
            shape=self.shape,
        )
        groups[self.group_id] = g.create()
        # set the group frequency and DM to be the detections with the highest sigma
        gs["f"] = groups[self.group_id]["f"][np.argmax(groups[self.group_id]["sigma"])]
        gs["dm"] = groups[self.group_id]["dm"][
            np.argmax(groups[self.group_id]["sigma"])
        ]
        harm = copy.deepcopy(gs)
        for h in range(self.num_harm):
            harm["f"] = harm["f"] / 2
            harm["sigma"] = harm["sigma"] * 0.8
            g = Group(
                f=harm["f"],
                dm=harm["dm"],
                min_dm=harm["min_dm"],
                max_dm=harm["max_dm"],
                max_power=harm["sigma"],
                shape=self.shape,
            )
            groups[self.group_id + h + 1] = g.create()
            harm["dm"] = groups[self.group_id + h + 1]["dm"][
                np.argmax(groups[self.group_id + h + 1]["sigma"])
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
            if not (
                MIN_SEARCH_FREQ <= gs.f <= MAX_SEARCH_FREQ
                and MIN_SEARCH_DM <= gs.dm <= MAX_SEARCH_DM
            ):
                continue
            group, group_s = gs.create()
            for g in group:
                groups[g] = group[g]
            for g in group_s:
                group_summary[g] = group_s[g]
            group_id = item["num_harm"] + group_id + 1
        return groups, group_summary
