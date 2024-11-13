import os

import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.time import Time
from folding.archive_utils import clean_foldspec, get_SN, readpsrarch
from sps_databases.db_api import get_nearby_known_sources
from sps_multi_pointing.known_source_sifter import known_source_filters

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
    
    gs = GridSpec(3, 3, height_ratios=[1, 2, 2])  

    ax0 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax1 = fig.add_subplot(gs[1:, 0])  # Second and third rows, first column
    ax2 = fig.add_subplot(gs[1:, 1])  # Second and third rows, second column

    ax3 = fig.add_subplot(gs[0, 1])  # First row, first column
    ax3.axis('off')  

    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.4)

    ax0.set_title(f"{psr} {T0.isot[:10]}", fontsize=18)


    ax1.imshow(
        np.nanmean(fs_bin, 0),
        aspect="auto",
        interpolation="nearest",
        vmin=vfmin,
        vmax=vfmax,
        extent=[0, 1, F[-1], F[1]],
    )
    ax2.imshow(
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

    ax2.set_ylabel("Time (min)", fontsize=18)
    ax2.set_xlabel("Phase", fontsize=18)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    phaseaxis = np.linspace(0, 1, ngate, endpoint=False)
    ax0.plot(phaseaxis, SNprof)
    ax0.set_xlim(0, 1)
    ax0.set_xticks([])
    if known.strip():
        print(f"Known pulsar {known} detected")
        parameters_text = (
            f"{known} \n"
            f"Incoherent $\\sigma$: {sigma:.2f}\n"
            f"Folded $\\sigma$: {SNR_val:.2f}\n"
            f"RA (deg): {ra:,.5g}\n"
            f"DEC (deg): {dec:,.5g}\n"
            f"DM: {dm:.2f}\n"
            f"f0: {f0}"
        )
        txt_height = 1.4
    else:
        parameters_text = (
            f"Incoherent $\\sigma$: {sigma:.2f}\n"
            f"Folded $\\sigma$: {SNR_val:.2f}\n"
            f"RA (deg): {ra:,.5g}\n"
            f"DEC (deg): {dec:,.5g}\n"
            f"DM: {dm:.2f}\n"
            f"f0: {f0}"
        )
        txt_height = 1.3

    ax3.text(
        0.0,
        txt_height,
        parameters_text,
        fontsize=10,
        va="top",
        ha="left",
        backgroundcolor="white",
    )

    radius = 5
    sources = get_nearby_known_sources(ra, dec, radius)
    
    ks_text = [f"Known sources within {radius} degrees\n"]
    source_texts = []
    for source in sources:
        ks_name = source.source_name
        ks_epoch = source.spin_period_epoch
        ks_ra = round(source.pos_ra_deg, 2)
        ks_dec = round(source.pos_dec_deg, 2)
        ks_f0 = round(1 / source.spin_period_s, 4)
        ks_dm = round(source.dm, 2)
        ks_survey = source.survey
        pos_diff = known_source_filters.angular_separation(ra, dec, ks_ra, ks_dec)[1]
        # pos_diff = np.sqrt((ra - ks_ra)**2 + (dec - ks_dec)**2) 
        source_texts.append([pos_diff,f"{ks_name}: pos_diff={pos_diff:.4f}, ra={ks_ra}, dec={ks_dec}, dm={ks_dm}, f0={ks_f0}, survey={ks_survey} \n"])
    source_texts.sort(key=lambda x: x[0])
    ks_text.extend([text[1] for text in source_texts])
    ks_text = ' '.join(ks_text)

    ax3.text(1.25, 1.2, ks_text, fontsize=8, ha='left', va='top')

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
