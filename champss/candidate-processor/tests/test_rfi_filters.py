import datetime

import numpy as np
from candidate_processor.utilities.rfi_filters import (
    apply_rfi_filters,
    dm_curve_shape,
    merged_config,
)
from sps_common.interfaces import SinglePointingCandidate
from sps_common.interfaces.single_pointing import SearchAlgorithm


def make_candidate(curve, rel_err=0.02, manual=""):
    dms = np.arange(len(curve)) * 0.1
    sigmas = np.tile(np.asarray(curve, float)[:, None], (1, 5))
    features = np.array(
        [(rel_err,)], dtype=[("dm_sigma_FitGauss_amplitude_rel_err", float)]
    )
    return SinglePointingCandidate(
        freq=85.0,
        dm=float(dms[np.argmax(curve)]),
        sigma=float(np.max(curve)),
        ra=74.53,
        dec=16.98,
        features=features,
        detection_statistic=SearchAlgorithm.power_spec,
        obs_id=["test"],
        harmonics_info=None,
        dm_freq_sigma={
            "sigmas": sigmas,
            "dms": dms,
            "freqs": np.linspace(84.9, 85.1, 5),
        },
        datetimes=[datetime.datetime(2026, 8, 26)],
        manual_candidate=manual,
    )


def pulsar_curve():
    dms = np.arange(301) * 0.1
    return 2 + 18 * np.exp(-0.5 * ((dms - 15) / 1.5) ** 2)


def wobbly_curve():
    rng = np.random.default_rng(0)
    dms = np.arange(301) * 0.1
    curve = 3 + rng.normal(0, 0.5, len(dms))
    for centre in (3, 9, 14, 20, 25):
        curve += 18 * np.exp(-0.5 * ((dms - centre) / 0.15) ** 2)
    return curve


def test_dm_curve_shape_pulsar_vs_wobbly():
    smooth = dm_curve_shape(make_candidate(pulsar_curve()).dm_freq_sigma)
    wobbly = dm_curve_shape(make_candidate(wobbly_curve()).dm_freq_sigma)
    assert smooth["npeaks"] == 1
    assert smooth["roughness"] < 0.03
    assert wobbly["npeaks"] >= 4
    assert wobbly["roughness"] > 0.03


def test_apply_rfi_filters_flags():
    config = merged_config(
        {
            "enabled": True,
            "one_day_burst": {"enabled": False},
            "fit_failed": {"source": "feature"},
        }
    )
    pulsar = make_candidate(pulsar_curve())
    wobbly = make_candidate(wobbly_curve())
    failed_fit = make_candidate(pulsar_curve(), rel_err=2000.0)
    manual = make_candidate(wobbly_curve(), manual="J0000+00")
    n = apply_rfi_filters([pulsar, wobbly, failed_fit, manual], config)
    assert n == 2
    assert not pulsar.rfi_filters["flagged"]
    assert wobbly.rfi_filters["wobbly"] and wobbly.rfi_filters["flagged"]
    assert failed_fit.rfi_filters["fit_failed"] and not failed_fit.rfi_filters["wobbly"]
    assert manual.rfi_filters == {}
    assert not wobbly.rfi  # flag_as_rfi is off by default

    # default: own Gaussian fit to the (convolved) DM curve instead of the feature;
    # for a clean pulsar-like curve that fit succeeds
    default = merged_config({"enabled": True, "one_day_burst": {"enabled": False}})
    candidate = make_candidate(pulsar_curve(), rel_err=2000.0)
    apply_rfi_filters([candidate], default)
    assert not candidate.rfi_filters["fit_failed"]
    assert candidate.rfi_filters["feature_fit_amplitude_rel_err"] == 2000.0


def test_one_day_burst(tmp_path):
    folder = tmp_path / "2026" / "08" / "26" / "74.53_16.98"
    folder.mkdir(parents=True)
    np.savez(
        folder / "74.53_16.98_0_power_spectra_candidates.npz",
        candidate_0={"freq": 84.98, "sigma": 11.3},
        cand_count=1,
    )
    candidate = make_candidate(pulsar_curve())
    config = merged_config({"enabled": True, "flag_as_rfi": True})
    apply_rfi_filters([candidate], config, daily_basepath=str(tmp_path))
    assert candidate.rfi_filters["daily_hit_days"] == 1
    assert candidate.rfi_filters["one_day_burst"]
    assert candidate.rfi


def test_convolved_map_and_fit():
    from candidate_processor.utilities.rfi_filters import candidate_map, fit_dm_curve

    candidate = make_candidate(pulsar_curve())
    dfs = dict(candidate.dm_freq_sigma)
    # plain map peaks low, convolved map at the candidate sigma; the convolved
    # array is allocated larger than the filled DM/frequency range
    dfs["sigmas"] = dfs["sigmas"] * 0.25
    conv = np.zeros((dfs["sigmas"].shape[0] + 20, 9))
    conv[: dfs["sigmas"].shape[0], :7] = np.tile(pulsar_curve()[:, None], (1, 7))
    dfs["convolve_arr"] = conv
    dfs["convolve_freqs"] = np.linspace(84.95, 85.05, 7)
    sigmas, dms, freqs, is_conv = candidate_map(dfs)
    assert is_conv and sigmas.shape == (301, 7)
    fit = fit_dm_curve(dfs, 85.0, 15.0, float(pulsar_curve().max()))
    assert fit["convolved"]
    assert abs(fit["mu"] - 15.0) < 0.2 and fit["amplitude_rel_err"] < 0.1
    plain = fit_dm_curve(
        dfs, 85.0, 15.0, float(pulsar_curve().max()), use_convolved=False
    )
    assert not plain["convolved"]
    # bounded to the candidate sigma, the fit to the low plain map cannot reach it
    assert plain["amplitude"] > 2 * dfs["sigmas"].max()


def test_feature_datagetter_convolved():
    from types import SimpleNamespace

    from candidate_processor.feature_generator import FitDataGetter

    candidate = make_candidate(pulsar_curve())
    hrc = SimpleNamespace(
        dm_freq_sigma=candidate.dm_freq_sigma, freq=85.0, dm=15.0, sigma=20.0
    )
    x, y, point = FitDataGetter(datacode="dm_sigma_convolved").get(hrc)
    assert len(x) == len(y) == 301 and point == [15.0, 20.0]
    x, y, point = FitDataGetter(datacode="freq_sigma_convolved").get(hrc)
    assert len(x) == len(y) == 5 and point == [85.0, 20.0]
