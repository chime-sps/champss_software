import datetime as dt
import logging
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec
import warnings


log = logging.getLogger(__name__)


def truncate_number_string(number, max_decimal=6):
    if isinstance(number, str):
        # is already string
        return number
    base_string = f"{number}"
    split_string = base_string.split(".")
    if len(split_string) == 1:
        return base_string
    else:
        decimals = min(len(split_string[-1]), max_decimal)
        return f"{number:.{decimals}f}"


def plot_text(fig, panel, grid_points, candidate):
    ax_text = fig.add_subplot(grid_points)
    ax_text.set_axis_off()
    text = ""
    if "values" in panel.keys():
        for value in panel["values"]:
            if not value:
                text += "\n"
                continue
            if isinstance(getattr(candidate, value), dict):
                val_dict = getattr(candidate, value)
                for key in val_dict:
                    val_string = truncate_number_string(val_dict[key])
                    text += f"{key}: {val_string} \n"
            else:
                val_string = truncate_number_string(getattr(candidate, value))
                text += f"{value}: {val_string} \n"
    if "best_candidate_values" in panel.keys():
        for value in panel["best_candidate_values"]:
            if not value:
                continue
            if isinstance(getattr(candidate.best_candidate, value), dict):
                val_dict = getattr(candidate.best_candidate, value)
                for key in val_dict:
                    val_string = truncate_number_string(val_dict[key])
                    text += f"{key}: {val_string} \n"
            else:
                val_string = truncate_number_string(
                    getattr(candidate.best_candidate, value)
                )
                text += f"{value}: {val_string} \n"
    ax_text.text(0, 1, text, ha="left", va="top", fontsize=10)


def plot_dm_freq_3(fig, panel, grid_points, candidate):
    dm_vals = candidate.dm_freq_sigma["dms"]
    freq_vals = candidate.dm_freq_sigma["freqs"]
    sgs = grid_points.subgridspec(1, 4)

    current_subplot_index = 0
    ax_2d = fig.add_subplot(sgs[current_subplot_index])
    ax_2d.imshow(
        candidate.dm_freq_sigma["sigmas"].astype(float),
        aspect="auto",
        origin="lower",
        extent=[freq_vals.min(), freq_vals.max(), dm_vals.min(), dm_vals.max()],
    )
    if "nharm" in candidate.dm_freq_sigma:
        current_subplot_index += 1
        alpha_nharm = np.clip(
            candidate.dm_freq_sigma["sigmas"]
            / np.nanmax(candidate.dm_freq_sigma["sigmas"]),
            0,
            1,
        ).astype(float)
        alpha_nharm = np.nan_to_num(alpha_nharm)
        ax_2d_nharm = fig.add_subplot(sgs[current_subplot_index])
        ax_2d_nharm.imshow(
            candidate.dm_freq_sigma["nharm"].astype(int),
            aspect="auto",
            origin="lower",
            extent=[freq_vals.min(), freq_vals.max(), dm_vals.min(), dm_vals.max()],
            vmin=0,
            vmax=32,
            cmap="Reds",
            alpha=alpha_nharm,
        )
        ax_2d_nharm.set_title("Best Harmonic", fontsize=10)
        ax_2d_nharm.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_2d_nharm.set_ylabel("DM", fontsize=8)
        ax_2d_nharm.xaxis.set_tick_params(rotation=30, labelsize=8)
        ax_2d_nharm.ticklabel_format(useOffset=False)

    current_subplot_index += 1
    ax_dm = fig.add_subplot(sgs[current_subplot_index])
    ax_dm.plot(dm_vals, candidate.dm_max_curve, "--", c="grey")
    ax_dm.plot(dm_vals, candidate.dm_best_curve, c="black")
    ax_dm.grid()
    ax_dm.ticklabel_format(useOffset=False)

    current_subplot_index += 1
    ax_freq = fig.add_subplot(sgs[current_subplot_index])
    ax_freq.plot(freq_vals, candidate.freq_max_curve, "--", c="grey")
    ax_freq.plot(freq_vals, candidate.freq_best_curve, c="black")
    ax_freq.grid()
    ax_dm.ticklabel_format(useOffset=False)

    if panel.get("plot_fit", False):
        if "dm_sigma_FitGaussWidth_gauss_sigma" in candidate.features.dtype.names:
            ax_dm.plot(
                dm_vals,
                gauss(
                    dm_vals,
                    candidate.sigma,
                    candidate.dm,
                    candidate.features["dm_sigma_FitGaussWidth_gauss_sigma"][0],
                ),
                c="red",
                alpha=1,
            )
        if "freq_sigma_FitGaussWidth_gauss_sigma" in candidate.features.dtype.names:
            ax_freq.plot(
                freq_vals,
                gauss(
                    freq_vals,
                    candidate.sigma,
                    candidate.freq,
                    candidate.features["freq_sigma_FitGaussWidth_gauss_sigma"][0],
                ),
                c="red",
                alpha=1,
            )
        if "dm_sigma_FitGauss_gauss_sigma" in candidate.features.dtype.names:
            ax_dm.plot(
                dm_vals,
                gauss(
                    dm_vals,
                    candidate.features["dm_sigma_FitGauss_amplitude"][0],
                    candidate.features["dm_sigma_FitGauss_mu"][0],
                    candidate.features["dm_sigma_FitGauss_gauss_sigma"][0],
                ),
                c="blue",
                alpha=1,
            )
        if "freq_sigma_FitGauss_gauss_sigma" in candidate.features.dtype.names:
            ax_freq.plot(
                freq_vals,
                gauss(
                    freq_vals,
                    candidate.features["freq_sigma_FitGauss_amplitude"][0],
                    candidate.features["freq_sigma_FitGauss_mu"][0],
                    candidate.features["freq_sigma_FitGauss_gauss_sigma"][0],
                ),
                c="blue",
                alpha=1,
            )

    ax_2d.set_title("Sigma", fontsize=10)

    ax_2d.set_xlabel("Frequency (Hz)", fontsize=8)
    ax_2d.set_ylabel("DM", fontsize=8)
    ax_dm.set_xlabel("DM", fontsize=8)
    ax_dm.set_ylabel("Sigma", fontsize=8)
    ax_freq.set_xlabel("Frequency (Hz)", fontsize=8)
    ax_freq.set_ylabel("Sigma", fontsize=8)
    ax_dm.ticklabel_format(useOffset=False)

    ax_2d.xaxis.set_tick_params(rotation=30, labelsize=8)
    ax_dm.xaxis.set_tick_params(rotation=30, labelsize=8)
    ax_freq.xaxis.set_tick_params(rotation=30, labelsize=8)

    ax_2d.ticklabel_format(useOffset=False)
    ax_freq.ticklabel_format(useOffset=False)

    ax_dm.sharey(ax_freq)


def plot_raw_harm_1d(fig, panel, grid_points, candidate):
    freq_vals = candidate.raw_harmonic_powers_array["freqs"][0, :, 0]
    freq_vals_trunc = np.trim_zeros(freq_vals, trim="b")
    harm_vals = np.arange(1, len(freq_vals_trunc) + 1)
    raw_powers = candidate.best_raw_harmonic_powers[: len(freq_vals_trunc)]
    harm_sigma_curve = candidate.harm_sigma_curve[: len(freq_vals_trunc)]
    zero_sigmas = raw_powers[raw_powers == 0]
    zero_harms = harm_vals[raw_powers == 0]
    sgs = grid_points.subgridspec(2, 1)

    ax_raw = fig.add_subplot(sgs[0])
    ax_raw.plot(harm_vals, raw_powers, c="black")
    ax_raw.scatter(
        zero_harms,
        zero_sigmas,
        facecolors="none",
        edgecolors="red",
        s=160,
        linewidths=3,
    )
    ax_raw.grid()

    ax_sigma = fig.add_subplot(sgs[1], sharex=ax_raw)
    ax_sigma.plot(harm_vals, harm_sigma_curve, c="black")
    ax_sigma.grid()

    ax_raw.set_ylabel("Power")
    ax_sigma.set_ylabel("Sigma")
    ax_sigma.set_xlabel("Harmonic")

    # Second x axis
    f = lambda x: x * freq_vals[0]
    g = lambda x: x / freq_vals[0]
    ax_raw2 = ax_raw.secondary_xaxis("top", functions=(f, g))
    ax_raw2.set_xlabel("Frequency (Hz)")

    ax_raw.sharex(ax_sigma)


def plot_dm_sigma_1d(fig, panel, grid_points, candidate):
    dms = candidate.dm_sigma_1d["dms"]
    sigmas = candidate.dm_sigma_1d["sigmas"]
    ax_dm_sigma = fig.add_subplot(grid_points)
    ax_dm_sigma.plot(dms, sigmas, c="black")
    ax_dm_sigma.set_xlabel("DM")
    ax_dm_sigma.set_ylabel("Sigma")
    ax_dm_sigma.grid()


def plot_raw_harm_all(fig, panel, grid_points, candidate):
    raw_powers = candidate.raw_harmonic_powers_array["powers"][:, :, 0]
    dm_values = candidate.raw_harmonic_powers_array["dms"]

    ax_raw = fig.add_subplot(grid_points)
    ax_raw.plot(dm_values, raw_powers, c="black")
    ax_raw.grid()

    ax_raw.set_xlabel("DM")
    ax_raw.set_ylabel("Power")


def plot_scatter_positions(fig, panel, grid_points, candidate):
    ra_vals = candidate.position_sigmas[:, 0]
    dec_vals = candidate.position_sigmas[:, 1]
    sigma_pos_vals = candidate.position_sigmas[:, 2]
    dist_pos_vals = candidate.position_sigmas[:, 3]
    # Fixes weird scaling in plot
    if len(dist_pos_vals) == 1:
        dist_pos_vals = np.zeros(1)

    dm_vals = candidate.all_dms
    freq_vals = candidate.all_freqs
    # sigma_vals = candidate.all_sigmas
    sgs = grid_points.subgridspec(2, 2)
    ax_pos = fig.add_subplot(sgs[0, 0])
    ax_pos.scatter(
        ra_vals, dec_vals, s=sigma_pos_vals * 10, alpha=0.7, edgecolors="black"
    )

    ax_dist = fig.add_subplot(sgs[1, 0])
    ax_dist.scatter(dist_pos_vals, sigma_pos_vals, alpha=0.7, edgecolors="black")

    ax_dm_freq = fig.add_subplot(sgs[0, 1])
    ax_dm_freq.scatter(freq_vals, dm_vals, alpha=0.7, edgecolors="black")

    ax_time_sigma = fig.add_subplot(sgs[1, 1])
    ax_time_sigma.hlines(
        candidate.all_sigmas,
        candidate.all_date_ranges[:, 0] - dt.timedelta(minutes=5),
        candidate.all_date_ranges[:, 1] + dt.timedelta(minutes=5),
    )

    # Catch warning created by autodatelocator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax_time_sigma.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax_time_sigma.xaxis.get_major_locator())
        )
        ax_time_sigma.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=5)
        )
        ax_time_sigma.autoscale_view()

    # Label wil overlap with xticks otherwise
    # ax_time_sigma.set_xlabel("Date")
    ax_time_sigma.set_ylabel("Sigma")

    ax_pos.set_xlabel("RA")
    ax_pos.set_ylabel("Dec")
    ax_dist.set_xlabel("Distance to Centroid (°)")
    ax_dist.set_ylabel("Sigma")
    ax_dm_freq.set_xlabel("Frequency (Hz)")
    ax_dm_freq.set_ylabel("DM")

    for axis in [ax_pos, ax_dist, ax_dm_freq]:
        axis.ticklabel_format(useOffset=False)
        axis.grid()
        axis.xaxis.set_tick_params(rotation=30, labelsize=8)

    ax_time_sigma.grid()
    # ax_time_sigma.xaxis.set_tick_params(rotation=30, labelsize=8)


def plot_detections_freq_dm(fig, panel, grid_points, candidate):
    freqs = candidate.detections[:, 1]
    dms = candidate.detections[:, 0]
    ax_det_freq_dm = fig.add_subplot(grid_points)
    ax_det_freq_dm.scatter(freqs, dms, c="black", alpha=0.4)
    ax_det_freq_dm.set_xlabel("Frequency")
    ax_det_freq_dm.set_ylabel("DM")
    ax_det_freq_dm.grid()


def plot_candidate(
    sp_candidate, config, folder, prefix="single_point_cand", mp_cand=None
):
    """
    Plots a candidate.

    Parameters
    ----------
    sp_candidate: SinglePointingCandidate
                Single pointing candidate that should be plotted.
                If a multi pointing candidate is plotted this is the
                strongest single pointing candidate of that multi pointing
                candidate.
    config: str or dict
            Path to the yml config containing the plot layout ir
            dictionary containing the plot layout
    folder: str
            Folder where plot should be saved
    prefix: str
            Prefix that is used in the file name of the plot
    mp_cand: MultiPointingCandidate or None
            Multi pointing candidate that should be used for the plot
    """
    if type(config) == str:
        config = yaml.safe_load(open(config))
    elif type(config) != dict:
        log.error("Plot config needs to be either a string or a dict")
        exit(1)

    os.makedirs(folder, exist_ok=True)

    file_name = (
        f"{folder}/{prefix}_{sp_candidate.ra:.2f}_"
        f"{sp_candidate.dec:.2f}_{sp_candidate.sigma:06.2f}_"
        f"{sp_candidate.freq:.5f}_{sp_candidate.dm:.3f}_"
        f"{sp_candidate.datetimes[-1].strftime('%Y%m%d')}.png"
    )
    fig = plt.figure(figsize=config["figure"]["figsize"], layout="constrained")
    gs = GridSpec(**config["figure"]["grid"], figure=fig)
    for panel in config["panels"]:
        cand_type = panel.get("cand_type", "single")
        if cand_type == "multi":
            current_data_source = mp_cand
        else:
            current_data_source = sp_candidate
        grid_points = gs[
            panel["row_range"][0] : panel["row_range"][1],
            panel["col_range"][0] : panel["col_range"][1],
        ]
        globals()[panel["type"]](fig, panel, grid_points, current_data_source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(file_name)
    plt.close()
    return file_name


def gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
