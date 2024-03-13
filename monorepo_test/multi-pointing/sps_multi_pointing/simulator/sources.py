import numpy as np


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
        rfi = False
        max_power = max(10, np.random.random() * np.random.choice([15, 20, 20, 30]))
        num_harm = 0
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
        freq = np.random.random() * np.random.choice(range(1, 50))
        if freq < 9.70127682e-04:
            freq = 9.70127682e-04
        if freq > 524288 * 9.70127682e-04:
            freq = 524288 * 9.70127682e-04
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
        num_harm = 0
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
            num_harm = 0
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
