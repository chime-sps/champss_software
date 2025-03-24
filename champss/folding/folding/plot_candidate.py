import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.optimize import curve_fit
from astropy.time import Time

from folding.archive_utils import clean_foldspec, get_SN, readpsrarch
from matplotlib.gridspec import GridSpec
from sps_databases.db_api import get_nearby_known_sources
from sps_multi_pointing.known_source_sifter import known_source_filters
from multiday_search.phase_aligned_search import phase_loop


def compute_accel_steps(
    dts, f0, npbin, vmax=500 * u.km / u.s, Pbmin=2 * u.hour, phase_accuracy=1.0 / 256
):
    # P = 1.0 / f0
    # dphase = P / npbin
    M_f0 = npbin * phase_accuracy
    M_f0 = int(np.max((M_f0, 1)))  # To make sure M_f0 does not return 0

    dfmax = f0 * (vmax / (c * u.m / u.s)).decompose()
    f1max = (2 * np.pi * dfmax * u.Hz / Pbmin).to(u.s**-2).value
    dfmax = dfmax.value

    # factor of 2, since we reference to central observation
    Tobs = max(dts) - min(dts)
    f0_points = 2 * int(dfmax * Tobs * npbin / M_f0)
    f1_points = 2 * int(0.5 * f1max * Tobs**2 * npbin / M_f0)

    f0s = np.linspace(-dfmax, dfmax, f0_points, endpoint=True)
    f1s = np.linspace(-f1max, f1max, f1_points, endpoint=True)
    print(f"Acceleration search with {len(f0s)} F0, {len(f1s)} F1 trials")
    return f0s, f1s


def plot_candidate_archive(
    fn,
    sigma,
    dm,
    f0,
    ra,
    dec,
    coord_path,
    accel_search=True,
    dm_search=True,
    known=" ",
    foldpath="/data/chime/sps/archives/plots/folded_candidate_plots",
):
    data, F, T, psr, tel = readpsrarch(fn)

    fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze())
    taxis = ((T - T[0]) * u.day).to(u.min)
    T0 = Time(T[0], format="mjd")

    fs_bg = fs - np.median(fs, axis=-1, keepdims=True)
    fs_bg = fs_bg[1:-1]

    ngate = fs.shape[-1]
    nc = 256
    binf = fs_bg.shape[1] // nc
    fs_bin = np.copy(fs_bg)
    maskchan = np.argwhere(fs_bin.mean(0).mean(-1) == 0).squeeze()
    fs_bin[:, maskchan] = np.nan
    fs_bin = np.nanmean(
        fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1] // binf, binf, -1), 2
    )
    fs_bin[np.isnan(fs_bin)] = 0

    fig = plt.figure(figsize=(24, 20))

    gs = GridSpec(28, 36)

    ax0 = fig.add_subplot(gs[0:4, 0:9])
    ax1 = fig.add_subplot(gs[4:16, 0:9])
    ax2 = fig.add_subplot(gs[16:, 0:9])

    ax3top = fig.add_subplot(gs[8, 11:17])
    ax3 = fig.add_subplot(gs[9:17, 11:17])
    ax3left = fig.add_subplot(gs[9:17, 10])

    ax4top = fig.add_subplot(gs[19, 11:17])
    ax4 = fig.add_subplot(gs[20:, 11:17])
    ax4left = fig.add_subplot(gs[20:, 10])

    axtext = fig.add_subplot(gs[0, 4])
    axtext.axis("off")

    ax_kstext = fig.add_subplot(gs[7, 10:17])
    ax_kstext.axis("off")

    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.4)

    if accel_search:
        dts = taxis.to(u.s).value
        dts = dts - np.median(dts)
        npbin = fs_bin.shape[-1]
        f0s, f1s = compute_accel_steps(dts, f0, npbin)

        prof2D = np.mean(fs_bin.squeeze(), 1)
        chi2_grid = phase_loop(prof2D, dts, f0s, f1s)
        i_f0, i_f1 = np.unravel_index(np.argmax(chi2_grid), chi2_grid.shape)
        f0_best = f0s[i_f0]
        f1_best = f1s[i_f1]
        f0_slice = chi2_grid[:, i_f1]
        f1_slice = chi2_grid[i_f0]

        npbin = fs_bin.shape[-1]
        dphis = f0_best * dts + 0.5 * f1_best * dts**2
        i_phis = (dphis * npbin).astype("int")
        for i in range(fs_bin.shape[0]):
            fs_bin[i] = np.roll(fs_bin[i], -i_phis[i], axis=-1)

        ax3.pcolormesh(f0s, f1s, chi2_grid.T)
        ax3.scatter(f0_best, f1_best, color="w", marker="*")
        ax3top.plot(f0s, f0_slice)
        ax3left.plot(-f1_slice, f1s)
        ax3left.set_yticks([])
        ax3left.set_xticks([])
        ax3top.set_xticks([])
        ax3top.set_yticks([])
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel("F1 (Hz/s)", fontsize=16)
        ax3.set_xlabel(r"$\Delta$F0 (Hz)", fontsize=16)
        ax3top.set_xlim(min(f0s), max(f0s))
        ax3left.set_ylim(min(f1s), max(f1s))

        F1_scinot = ax3.yaxis.get_offset_text()
        F1_scinot.set_x(1.15)

        def gaussian(x, x0, sigma, A, C):
            return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C

        try:
            p0 = [
                f0s[np.argmax(f0s)],
                0.0002,
                np.max(f0_slice) - np.median(f0_slice),
                np.median(f0_slice),
            ]
            popt, pcov = curve_fit(gaussian, f0s, f0_slice, p0=p0)
            # w = popt[1]
            # xerr = np.sqrt(pcov[0][0])
        except Exception as e:
            print(e)
    else:
        ax3.axis("off")
        ax3top.axis("off")
        ax3left.axis("off")

    if dm_search:
        freq = F[::binf]
        f_ref = np.max(freq)
        dDM = 0.125
        nDM = int(np.ceil(2 * dm / dDM))
        DMs = np.linspace(0, nDM * dDM, nDM)
        DMs -= np.mean(DMs)
        P = (1 / f0) * u.s
        fs_fp = fs_bin.mean(0)
        DMprofs = np.zeros((nDM, fs_bin.shape[-1]))

        for i, DMi in enumerate(DMs):
            t_delay = (1 / 2.41e-4) * DMi * (1.0 / f_ref**2 - 1.0 / freq**2)
            pshifts = (t_delay / P).decompose().value
            pshifts = (pshifts * npbin).astype("int")

            fs_shifted = np.zeros_like(fs_fp)
            for j, p in enumerate(pshifts):
                fs_shifted[j] = np.roll(fs_fp[j], int(pshifts[j]))
            DMprofs[i] = fs_shifted.mean(0)

        DMprofs = np.tile(DMprofs, (1, 2))
        DM_slice = np.max(DMprofs, axis=-1)
        DM_prof = DMprofs[np.argmax(DM_slice)]

        ax4.pcolormesh(np.linspace(0, 2, 2 * npbin), dm + DMs, DMprofs)
        ax4left.plot(-DM_slice, dm + DMs)
        ax4top.plot(np.linspace(0, 2, len(DM_prof)), DM_prof)
        ax4left.set_yticks([])
        ax4left.set_xticks([])
        ax4top.set_xticks([])
        ax4top.set_yticks([])
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.set_xlabel("Phase", fontsize=16)
        ax4.set_ylabel(r"DM (pc cm$^3$)", fontsize=16)
        ax4top.set_xlim(0, 2)
        ax4left.set_ylim(min(dm + DMs), max(dm + DMs))

    else:
        ax4.axis("off")
        ax4top.axis("off")
        ax4left.axis("off")

    profile = np.nanmean(np.nanmean(fs_bin, 0), 0)
    SNR_val, SNprof = get_SN(profile, return_profile=True)
    fs_bin = np.tile(fs_bin, (1, 2))
    SNprof = np.tile(SNprof, 2)

    vfmin = np.nanmean(fs_bin) - 1 * np.nanstd(np.nanmean(fs_bin, 0))
    vfmax = np.nanmean(fs_bin) + 3 * np.nanstd(np.nanmean(fs_bin, 0))
    vtmin = np.nanmean(fs_bin) - 1 * np.nanstd(np.nanmean(fs_bin, 1))
    vtmax = np.nanmean(fs_bin) + 3 * np.nanstd(np.nanmean(fs_bin, 1))

    radius = 5
    sources = get_nearby_known_sources(ra, dec, radius)
    pos_diffs = []
    for source in sources:
        pos_diff = known_source_filters.angular_separation(
            ra, dec, source.pos_ra_deg, source.pos_dec_deg
        )[1]
        pos_diffs.append(pos_diff)
    i_order = np.argsort(pos_diffs)
    sources_ordered = [sources[i] for i in i_order]

    num_ks = 16  # Max number of ks displayed in table
    ks_params = []
    for source in sources_ordered:
        ks_name = source.source_name
        # ks_epoch = source.spin_period_epoch
        ks_ra = round(source.pos_ra_deg, 2)
        ks_dec = round(source.pos_dec_deg, 2)
        ks_f0 = round(1 / source.spin_period_s, 4)
        ks_dm = round(source.dm, 2)
        ks_survey = source.survey[:1]
        if not ks_survey:
            ks_survey = ["N/A"]
        pos_diff = known_source_filters.angular_separation(
            ra, dec, source.pos_ra_deg, source.pos_dec_deg
        )[1]
        if np.abs(dm - ks_dm) < dm / 10.0:
            ks_param = [
                ks_name,
                round(pos_diff, 2),
                ks_ra,
                ks_dec,
                ks_f0,
                ks_dm,
                ks_survey[0],
            ]
            if len(ks_params) < num_ks:
                ks_params.append(ks_param)

    ks_text = f"Closest {len(ks_params)} known sources within {radius} degrees, $\Delta$DM < 10%\n"

    column_labels = ["Name", "$\Delta Pos.$", "RA", "Dec", "F0", "DM", "Survey(s)"]
    ks_df = pd.DataFrame(ks_params, columns=column_labels)

    ax1.imshow(
        np.nanmean(fs_bin, 0),
        aspect="auto",
        interpolation="nearest",
        vmin=vfmin,
        vmax=vfmax,
        extent=[0, 2, F[-1], F[1]],
    )
    ax2.imshow(
        np.nanmean(fs_bin, 1),
        aspect="auto",
        interpolation="nearest",
        vmin=vtmin,
        vmax=vtmax,
        origin="lower",
        extent=[0, 2, 0, max(taxis.to(u.min).value)],
    )

    ax1.set_ylabel("Obs Frequency (MHz)", fontsize=16)
    ax1.set_xticks([])

    ax2.set_ylabel("Time (min)", fontsize=16)
    ax2.set_xlabel("Phase", fontsize=16)

    phaseaxis = np.linspace(0, 2, 2 * ngate, endpoint=False)
    dp = phaseaxis[1] - phaseaxis[0]
    phaseaxis += dp / 2.0
    ax0.plot(phaseaxis, SNprof)
    ax0.set_xlim(0, 2)
    ax0.set_xticks([])
    if sigma is not None:
        sigma = round(sigma, 2)
    else:
        sigma = 0.0
    cand_params_text = [
        [rf"{psr}", f"Date: {T0.isot[:10]}", " "],
        [rf"RA (deg): {ra:,.5g}", f"f0: {f0:.5f}", f"Incoh. $\\sigma$: {sigma:.2f}"],
        [
            rf"DEC (deg): {dec:,.5g}",
            f"DM: {dm:.2f}",
            f"Folded $\\sigma$: {SNR_val:.2f}",
        ],
    ]

    ax_kstext.text(
        0,
        0.8,
        ks_text,
        fontsize=8,
        va="top",
        ha="left",
        transform=ax_kstext.transAxes,
    )

    if len(ks_df.values) > 0:
        ks_param_table = ax_kstext.table(
            cellText=ks_df.values,
            colLabels=ks_df.columns,
            colColours=["lavender"] * len(ks_df.columns),
            cellLoc="left",
            loc="top",
        )
        ks_param_table.auto_set_font_size(False)
        ks_param_table.set_fontsize(8)
        ks_param_table.auto_set_column_width(col=list(range(len(ks_df.columns))))

    cand_param_table = axtext.table(
        cellText=cand_params_text, cellLoc="left", loc="top", edges="open"
    )
    cand_param_table.auto_set_font_size(False)
    cand_param_table.set_fontsize(10)
    cand_param_table.scale(10, 1.25)

    if not known:
        plotstring = f"cand_{round(dm, 2)}_{round(f0, 2)}_{T0.isot[:10]}.png"
        plotstring_radec = f"cand_{round(ra, 2)}_{round(dec, 2)}_{round(dm, 2)}_{round(f0, 2)}_{T0.isot[:10]}.png"
    else:
        plotstring = f"{psr}_{T0.isot[:10]}.png"
        plotstring_radec = f"{psr}_{T0.isot[:10]}.png"

    plt.savefig(coord_path + plotstring, dpi=fig.dpi, bbox_inches="tight")

    img_path = f"{foldpath}/{T0.isot[:10]}-plots/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        print(f"Directory '{img_path}' already exists.")
    plot_fname = img_path + plotstring_radec
    plt.savefig(plot_fname, dpi=fig.dpi, bbox_inches="tight")
    plt.close()

    return SNprof, SNR_val, plot_fname
