import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.optimize import curve_fit
from astropy.time import Time
from astropy.coordinates import SkyCoord

from folding.archive_utils import clean_foldspec, get_SN, readpsrarch
from folding.known_source_matching import find_matching_sources
from matplotlib.gridspec import GridSpec
from multiday_search.phase_aligned_search import phase_loop
from numba import njit, prange, set_num_threads


def compute_accel_steps(
    dts, f0, npbin, vmax=500 * u.km / u.s, Pbmin=2 * u.hour, phase_accuracy=1.0 / 256,
    stack_binary_search=False, f1_maxbins=2048,
):
    """
    Compute the f0 and f1 search grid for acceleration search.

    By default, restrict f0 search range for +- one PS bin (e.g. +- one phase wrap). 
    Set stack_binary_search=True for binary candidates identified in power spectrum stacks,
    where f0 range is set by maximum orbital velocity.

    Parameters
    ----------
    dts : array
        Time offsets in seconds
    f0 : float
        Spin frequency in Hz
    npbin : int
        Number of phase bins
    vmax : astropy.Quantity
        Maximum orbital velocity (for binary systems)
    Pbmin : astropy.Quantity
        Minimum binary orbital period
    phase_accuracy : float
        Required phase accuracy (determines step size)
    stack_binary_search : bool
        If True, use velocity-based f0 range for binary candidates from stacks.
        If False (default), limit f0 search to spectral resolution for single-day obs.
    f0_bins : int
        Number of frequency bins (±) to search around f0 (default 3)

    Returns
    -------
    f0s : array
        F0 offset grid
    f1s : array
        F1 grid
    """
    Tobs = max(dts) - min(dts)
    if phase_accuracy < 1.0 / npbin:
        print("Warning: phase accuracy finer than one phase is unnecessary, "
              "setting to one phase bin.")
        phase_accuracy = np.max([phase_accuracy, 1.0 / npbin])

    # Calculate dfmax based on velocity (for binary systems)
    dfmax_velocity = f0 * (vmax / (c * u.m / u.s)).decompose().value

    if stack_binary_search:
        # Use velocity-based limit for binary candidates from stacks
        dfmax = dfmax_velocity
        print(f"Stack binary search mode: dfmax = {dfmax:.2e} Hz")
        f0_points = 2 * int(0.5 * dfmax * Tobs / phase_accuracy)
    else:
        f0_points = 2 * int(1 / phase_accuracy)
        dfmax = 1.0 / Tobs

    # f1 always uses binary orbital limit
    f1max = (2 * np.pi * dfmax_velocity * u.Hz / Pbmin).to(u.s**-2).value

    f1_points = 2 * int(0.5 * f1max * Tobs**2 / phase_accuracy)

    # Ensure at least a minimal search grid
    f0_points = max(f0_points, 3)
    f1_points = max(f1_points, 3)
    if f1_points > f1_maxbins:
        n_f1_downsample = int(np.ceil(f1_points / f1_maxbins))
        f1_points = f1_points // n_f1_downsample
        print(f"Warning: f1_points exceeds {f1_maxbins}, downsampling by factor "
                f"{n_f1_downsample} to {f1_points} points.")

    f0s = np.linspace(-dfmax, dfmax, f0_points, endpoint=True)
    f1s = np.linspace(-f1max, f1max, f1_points, endpoint=True)
    print(f"Acceleration search with {len(f0s)} F0, {len(f1s)} F1 trials")
    return f0s, f1s


@njit(parallel=True)
def dm_shift_loop(fs_fp, DMs, freq, f_ref, P_sec, npbin):
    """
    Optimized DM shifting loop using numba njit.

    Parameters
    ----------
    fs_fp : ndarray
        Frequency-phase array of shape (nfreq, nphase)
    DMs : ndarray
        Array of DM offsets to try
    freq : ndarray
        Frequency array in MHz
    f_ref : float
        Reference frequency in MHz
    P_sec : float
        Period in seconds
    npbin : int
        Number of phase bins

    Returns
    -------
    DMprofs : ndarray
        DM trial profiles of shape (nDM, nphase)
    """
    set_num_threads(8)

    nDM = len(DMs)
    nfreq = len(freq)
    nphase = fs_fp.shape[-1]
    DMprofs = np.zeros((nDM, nphase))

    DM_constant = 1.0 / 2.41e-4

    for i in prange(nDM):
        DMi = DMs[i]
        # Calculate time delays in seconds for each frequency
        t_delay = DM_constant * DMi * (1.0 / f_ref**2 - 1.0 / freq**2)
        # Convert to phase shifts
        pshifts = ((t_delay / P_sec) * npbin).astype(np.int32)

        # Shift and accumulate
        for j in range(nfreq):
            DMprofs[i] += np.roll(fs_fp[j], pshifts[j])

        # Average over frequency
        DMprofs[i] /= nfreq

    return DMprofs


def plot_candidate_archive(
    fn,
    coord_path,
    cand_info=None,
    accel_search=True,
    dm_search=True,
    foldpath="/data/chime/sps/archives/plots/folded_candidate_plots",
):
    """
    Plot a folded candidate archive.

    Parameters
    ----------
    fn : str
        Path to the archive file
    coord_path : str
        Path to save coordinate-based plot
    cand_info : dict, optional
        Dictionary containing optional candidate information:
        - 'sigma': Pipeline sigma of candidate
        - 'known': Name of known pulsar (empty string if unknown)
        - 'ap': Active pointing list from PointingStrategist
    accel_search : bool
        Whether to perform acceleration search (default True)
    dm_search : bool
        Whether to perform DM search (default True)
    foldpath : str
        Base path for saving folded candidate plots
    """
    if cand_info is None:
        cand_info = {}
    sigma = cand_info.get('sigma', None)
    known = cand_info.get('known', ' ')
    ap = cand_info.get('ap', None)

    # Extract info from active pointing if available
    max_beam = None
    dm_ne2001 = None
    dm_ymw16 = None
    maxdm = None
    if ap is not None and len(ap) > 0:
        ap0 = ap[0]
        if hasattr(ap0, 'max_beams') and ap0.max_beams:
            max_beam = ap0.max_beams[0].get('beam', None)
        if hasattr(ap0, 'ne2001dm'):
            dm_ne2001 = ap0.ne2001dm
        if hasattr(ap0, 'ymw16dm'):
            dm_ymw16 = ap0.ymw16dm
        if hasattr(ap0, 'maxdm'):
            maxdm = ap0.maxdm
    data, params = readpsrarch(fn)
    F = params["F"]
    T = params["T"]
    psr = params["psr"]
    dm = params["dm"]
    ra = params["ra"]
    dec = params["dec"]
    f0 = params["f0"]

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

    fig = plt.figure(figsize=(12, 22))

    gs = GridSpec(30, 18)

    # Text info at top (rows 0-1)
    axtext = fig.add_subplot(gs[0:1, 0:9])
    axtext.axis("off")

    # Main plots below text
    ax0 = fig.add_subplot(gs[1:5, 0:9])
    ax1 = fig.add_subplot(gs[5:17, 0:9])
    ax2 = fig.add_subplot(gs[17:, 0:9])

    # F0-F1 grid
    ax3top = fig.add_subplot(gs[11, 11:17])
    ax3 = fig.add_subplot(gs[12:19, 11:17])
    ax3left = fig.add_subplot(gs[12:19, 10])

    # DM-phase grid
    ax4top = fig.add_subplot(gs[21, 11:17])
    ax4 = fig.add_subplot(gs[22:, 11:17])
    ax4left = fig.add_subplot(gs[22:, 10])

    ax_kstext = fig.add_subplot(gs[0:1, 10:17])
    ax_kstext.axis("off")

    # Sky position scatterplot
    ax_sky = fig.add_subplot(gs[3:9, 10:17])
    ax_sky.yaxis.tick_right()
    ax_sky.yaxis.set_label_position("right")
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.4)

    # Initialize search result variables
    f0_best = None
    f1_best = None
    dm_best = None

    if accel_search:
        dts = taxis.to(u.s).value
        dts = dts - np.median(dts)
        npbin = fs_bin.shape[-1]
        f0s, f1s = compute_accel_steps(dts, f0, npbin)

        prof2D = np.mean(fs_bin.squeeze(), 1)
        chi2_grid = phase_loop(prof2D, dts, f0s, f1s, metric=1)
        i_f0, i_f1 = np.unravel_index(np.argmax(chi2_grid), chi2_grid.shape)
        f0_best = f0s[i_f0]
        f1_best = f1s[i_f1]
        f0_slice = chi2_grid[:, i_f1]
        f1_slice = chi2_grid[i_f0]

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
        DM_constant = 1.0 / 2.41e-4
        tsamp_pbin = 1 / (f0*npbin)
        dDM = 2 * tsamp_pbin / (DM_constant * (1.0 / np.min(freq)**2 - 1.0 / f_ref**2))
        nDM = int(np.ceil(2 * dm / dDM))
        print(f"DM search over {nDM} trials with dDM = {dDM:.2f} pc cm^-3")
        DMs = np.linspace(0, nDM * dDM, nDM)
        DMs -= np.mean(DMs)
        P = (1 / f0) * u.s
        fs_fp = fs_bin.mean(0)

        # Use optimized njit function for DM shifting
        P_sec = P.to(u.s).value
        DMprofs = dm_shift_loop(fs_fp, DMs, freq, f_ref, P_sec, npbin)

        DMprofs = np.tile(DMprofs, (1, 2))
        DM_slice = np.max(DMprofs, axis=-1)
        DM_prof = DMprofs[np.argmax(DM_slice)]
        dm_best = dm + DMs[np.argmax(DM_slice)]

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

    # Find and categorize nearby known sources
    ks_result = find_matching_sources(ra, dec, f0, T, max_beam=max_beam)
    ks_df = pd.DataFrame(ks_result['ks_df_data'], columns=ks_result['column_labels'])
    likely_match_count = ks_result['likely_match_count']
    ks_is_psr_scraper = ks_result['ks_is_psr_scraper']
    ks_positions_match = ks_result['ks_positions_match']
    ks_positions_arc = ks_result['ks_positions_arc']
    ks_positions_scraper = ks_result['ks_positions_scraper']
    ks_positions_other = ks_result['ks_positions_other']
    arc_positions = ks_result['arc_positions']

    # Plot sky position scatter plot
    # Plot the arc if available
    if arc_positions is not None:
        arc_ras = arc_positions[:, 0]
        arc_decs = arc_positions[:, 1]
        # Handle RA wrap-around (plot in two segments if needed)
        ra_break_idx = np.argmax(np.abs(np.diff(arc_ras)))
        if np.abs(np.diff(arc_ras))[ra_break_idx] > 100.0:
            ax_sky.plot(arc_ras[:ra_break_idx + 1], arc_decs[:ra_break_idx + 1],
                        c='tab:orange', ls=':', lw=1.5, zorder=1, label='EW Arc')
            ax_sky.plot(arc_ras[ra_break_idx + 1:], arc_decs[ra_break_idx + 1:],
                        c='tab:orange', ls=':', lw=1.5, zorder=1)
        else:
            ax_sky.plot(arc_ras, arc_decs, c='tab:orange', ls=':', lw=1.5, zorder=1, label='EW Arc')

    ax_sky.scatter(ra, dec, s=150, c='k', marker='*', label='Candidate', zorder=10)
    dra_arc = 0
    if ks_positions_match:
        ras_match, decs_match = zip(*ks_positions_match)
        dra_arc = np.max(np.abs(ra - np.array(ras_match)))
        ax_sky.scatter(ras_match, decs_match, s=80, c='red', marker='o', label='Harmonic match', zorder=5)
    if ks_positions_arc:
        ras_arc, decs_arc = zip(*ks_positions_arc)
        ax_sky.scatter(ras_arc, decs_arc, s=60, c='tab:orange', marker='d', label='Arc source', zorder=6)
    if ks_positions_scraper:
        ras_scraper, decs_scraper = zip(*ks_positions_scraper)
        ax_sky.scatter(ras_scraper, decs_scraper, s=50, c='blue', marker='s', label='Scraper', zorder=4)
    if ks_positions_other:
        ras_other, decs_other = zip(*ks_positions_other)
        ax_sky.scatter(ras_other, decs_other, s=50, c='gray', marker='o', label='Other', zorder=3)
    xlim_sky = max([5/np.cos(dec*u.deg), dra_arc])
    ax_sky.set_xlim(ra - xlim_sky, ra + xlim_sky)
    ax_sky.set_ylim(dec - 5, dec + 5)
    ax_sky.set_xlabel('RA (deg)', fontsize=12)
    ax_sky.set_ylabel('Dec (deg)', fontsize=12)
    ax_sky.legend(loc='upper right', fontsize=8)
    ax_sky.set_aspect('equal', adjustable='datalim')

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

    # Compute galactic coordinates
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    gal_l = coord.galactic.l.deg
    gal_b = coord.galactic.b.deg

    # Format optional values
    sigma_str = f"{sigma:.2f}" if sigma is not None else "N/A"
    beam_str = f"{max_beam}" if max_beam is not None else "N/A"
    ne2001_str = f"{dm_ne2001:.1f}" if dm_ne2001 is not None else "N/A"
    ymw16_str = f"{dm_ymw16:.1f}" if dm_ymw16 is not None else "N/A"
    maxdm_str = f"{maxdm:.1f}" if maxdm is not None else "N/A"
    dm_best_str = f"{dm_best:.2f}" if dm_best is not None else "N/A"
    f0_best_str = f"{f0_best:.2e}" if f0_best is not None else "N/A"
    f1_best_str = f"{f1_best:.2e}" if f1_best is not None else "N/A"

    # 5 columns x 3 rows layout
    cand_params_text = [
        [rf"{psr}", f"{T0.isot[:10]}", f"DM$_{{max}}$: {maxdm_str}", f"Beam: {beam_str}", f"DM$_{{best}}$: {dm_best_str}"],
        [f"Fold $\sigma$: {SNR_val:.2f}", f"RA: {ra:.2f}", f"f0: {f0:.5f}", rf"$g_l$: {gal_l:.2f}", f"$\\Delta$F0: {f0_best_str}"],
        [rf"PS $\sigma$: {sigma_str}", f"Dec: {dec:.2f}", f"DM: {dm:.2f}", rf"$g_b$: {gal_b:.2f}", f"F1: {f1_best_str}"],
    ]

    cand_param_table = axtext.table(
        cellText=cand_params_text, cellLoc="left", loc="top", edges="open",
        bbox=[-0.1, 0.5, 1.1, 2]
    )
    cand_param_table.auto_set_font_size(False)
    cand_param_table.set_fontsize(10)
    cand_param_table.auto_set_column_width(col=list(range(5)))

    # Render known sources table
    if len(ks_df.values) > 0:
        ks_table = ax_kstext.table(
            cellText=ks_df.values,
            colLabels=ks_df.columns,
            colColours=["lavender"] * len(ks_df.columns),
            cellLoc="center",
            loc="center",
        )
        ks_table.auto_set_font_size(False)
        ks_table.set_fontsize(9)
        ks_table.auto_set_column_width(col=list(range(len(ks_df.columns))))

        # Color code the rows based on match type
        for row_idx, is_psr_scraper in enumerate(ks_is_psr_scraper):
            is_harmonic_match = row_idx < likely_match_count

            # Determine color: purple if both, red if harmonic only, blue if scraper only
            if is_harmonic_match and is_psr_scraper:
                color = "purple"
            elif is_harmonic_match:
                color = "red"
            elif is_psr_scraper:
                color = "blue"
            else:
                continue  # No special color

            for col_idx in range(len(ks_df.columns)):
                ks_table[(row_idx + 1, col_idx)].set_text_props(color=color)

        # Add color legend below table
        legend_text = (
            "Red: likely match  |  Blue: unpublished  |  Purple: both"
        )
        ax_kstext.text(0.5, -0.05, legend_text, transform=ax_kstext.transAxes,
                      fontsize=9, ha='center', va='top')

    if not known.strip():
        plotstring = f"cand_{f0:.02f}_{dm:.02f}_{T0.isot[:10]}.png"
        plotstring_radec = (
            f"cand_{ra:.02f}_{dec:.02f}_{f0:.02f}_{dm:.02f}_{T0.isot[:10]}.png"
        )
    else:
        plotstring = f"{psr}_{T0.isot[:10]}.png"
        plotstring_radec = f"{psr}_{T0.isot[:10]}.png"

    plt.savefig(f'{coord_path}/{plotstring}', dpi=fig.dpi, bbox_inches="tight")

    img_path = f"{foldpath}/{T0.isot[:10]}-plots/"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        print(f"Directory '{img_path}' already exists.")
    plot_fname = img_path + plotstring_radec
    plt.savefig(plot_fname, dpi=fig.dpi, bbox_inches="tight")
    plt.close()

    return SNprof, SNR_val, plot_fname
