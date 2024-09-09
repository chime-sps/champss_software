import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from folding.archive_utils import clean_foldspec, get_SN, readpsrarch
from sps_databases.db_api import get_nearby_known_sources


def plot_candidate_archive(
    fn,
    sigma,
    dm,
    f0,
    ra,
    dec,
    coord_path,
    known=" ",
    foldpath="/data/chime/sps/archives",
):
    data, F, T, psr, tel = readpsrarch(fn)
    print(data.shape)

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

    profile = np.nanmean(np.nanmean(fs_bin, 0), 0)
    SNR_val, SNprof = get_SN(profile, return_profile=True)

    vfmin = np.nanmean(fs_bin) - 1 * np.nanstd(np.nanmean(fs_bin, 0))
    vfmax = np.nanmean(fs_bin) + 3 * np.nanstd(np.nanmean(fs_bin, 0))
    vtmin = np.nanmean(fs_bin) - 1 * np.nanstd(np.nanmean(fs_bin, 1))
    vtmax = np.nanmean(fs_bin) + 3 * np.nanstd(np.nanmean(fs_bin, 1))

    fig = plt.figure(figsize=(12, 8))
    ax0 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2)

    plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.4)

    ax0.set_title(f"{psr} {T0.isot[:10]}", fontsize=18)

    ax1.imshow(
        np.nanmean(fs_bin, 0),
        aspect="auto",
        interpolation="nearest",
        vmin=vfmin,
        vmax=vfmax,
        extent=[0, 1, F[-1], F[1]],
    )
    ax3.imshow(
        np.nanmean(fs_bin, 1),
        aspect="auto",
        interpolation="nearest",
        vmin=vtmin,
        vmax=vtmax,
        origin="lower",
        extent=[0, 1, 0, max(taxis.to(u.min).value)],
    )

    ax1.set_ylabel("Frequency (MHz)", fontsize=18)
    ax1.set_xlabel("Phase", fontsize=18)

    ax3.set_ylabel("Time (min)", fontsize=18)
    ax3.set_xlabel("Phase", fontsize=18)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    phaseaxis = np.linspace(0, 1, ngate, endpoint=False)
    ax0.plot(phaseaxis, SNprof)
    ax0.set_xlim(0, 1)
    ax0.set_xticks([])
    if known.strip():
        print(f"Known pulsar {known} detected")
        parameters_text = (
            f"{known} \n"
            f"Incoherent $\\sigma$: {sigma}\n"
            f"Folded $\\sigma$: {SNR_val}\n"
            f"RA (deg): {ra}\n"
            f"DEC (deg): {dec}\n"
            f"DM: {dm}\n"
            f"f0: {f0}\n"
        )
    else:
        parameters_text = (
            f"Incoherent $\\sigma$: {sigma}\n"
            f"Folded $\\sigma$: {SNR_val}\n"
            f"RA (deg): {ra}\n"
            f"DEC (deg): {dec}\n"
            f"DM: {dm}\n"
            f"f0: {f0}\n"
        )

    ax0.text(
        1.05,
        1.05,
        parameters_text,
        transform=ax0.transAxes,
        fontsize=18,
        va="top",
        ha="left",
        backgroundcolor="white",
    )

    radius = 5
    sources = get_nearby_known_sources(ra, dec, radius)

    ks_text = f"Known sources within {radius} degrees\n"

    for source in sources:
        ks_name = source.source_name
        ks_epoch = source.spin_period_epoch
        if ks_epoch == 45000.0:
            published = False
        else:
            published = True
        ks_ra = round(source.pos_ra_deg, 2)
        ks_dec = round(source.pos_dec_deg, 2)
        ks_f0 = round(1 / source.spin_period_s, 4)
        ks_dm = round(source.dm, 2)
        if published:
            ks_text += f"{ks_name}: ra={ks_ra}, dec={ks_dec}, dm={ks_dm}, f0={ks_f0} \n"
        else:
            ks_text += (
                f"{ks_name}: ra={ks_ra}, dec={ks_dec}, dm={ks_dm}, f0={ks_f0},"
                " Unpublished \n"
            )

    def get_text_height(text, fontsize=10):
        renderer = fig.canvas.get_renderer()
        t = fig.text(0.5, 0.01, text, ha="center", fontsize=fontsize)
        bbox = t.get_window_extent(renderer)
        fig_height = fig.get_size_inches()[1] * fig.dpi
        text_height = bbox.height / fig_height
        t.remove()
        return text_height

    text_height = get_text_height(ks_text)

    bottom_margin = 0.1 + text_height
    plt.subplots_adjust(bottom=bottom_margin, top=0.9, left=0.1, right=0.9)

    fig.text(0.1, 0.01, ks_text, ha="left", fontsize=10)

    # ax1.text(
    #     0.0,
    #     -0.3,
    #     ks_text,
    #     transform=ax1.transAxes,
    #     fontsize=10,
    #     va="top",
    #     ha="left",
    #     backgroundcolor="white",
    # )

    plt.savefig(coord_path + f"/{psr}_{T0.isot[:10]}_{round(dm,2)}_{round(f0,2)}.png")

    img_path = f"{foldpath}/plots/folded_candidate_plots/{T0.isot[:10]}-plots/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        print(f"Directory '{img_path}' already exists.")
    plot_fname = img_path + f"{psr}_{T0.isot[:10]}_{round(dm,2)}_{round(f0,2)}.png"
    plt.savefig(plot_fname)
    plt.close()

    return SNprof, SNR_val, plot_fname
