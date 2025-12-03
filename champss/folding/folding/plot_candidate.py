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
        phase_wraps = 2
        f0_points = phase_wraps * int(1 / phase_accuracy)
        dfmax = phase_wraps / Tobs

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
    accel_search=False,
    dm_search=True,
    f0search=True,
    foldpath="/data/chime/sps/archives/plots/folded_candidate_plots",
    plot_bw=False,
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
    f0search : dict, optional
        If provided, performs F0 search instead of accel_search.
        Dictionary containing parameters to pass to compute_accel_steps.
        Common parameters include:
        - 'stack_binary_search': bool, use velocity-based f0 range
        - 'vmax': astropy.Quantity, maximum orbital velocity
        - 'Pbmin': astropy.Quantity, minimum binary period
        - 'phase_accuracy': float, required phase accuracy
    foldpath : str
        Base path for saving folded candidate plots
    plot_bw : bool
        If True, use black/white color scheme (black lines, Greys colormap).
        If False (default), use color scheme (tab:blue lines, viridis colormap).
    """
    if cand_info is None:
        cand_info = {}
    sigma = cand_info.get('sigma', None)
    known = cand_info.get('known', ' ')
    ap = cand_info.get('ap', None)

    # Set color scheme based on plot_bw
    if plot_bw:
        plot_color = 'black'
        cmap_name = 'Greys'
    else:
        plot_color = 'tab:blue'
        cmap_name = 'viridis'

    data, params = readpsrarch(fn)
    F = params["F"]
    T = params["T"]
    psr = params["psr"]
    dm = params["dm"]
    ra = params["ra"]
    dec = params["dec"]
    f0 = params["f0"]
    P0 = 1/f0

    # Extract info from active pointing if available
    max_beam = None
    dm_ne2001 = None
    dm_ymw16 = None
    maxdm = None
    if ap is not None and len(ap) > 0:
        ap0 = ap[0]
        if hasattr(ap0, 'max_beams') and ap0.max_beams:
            max_beam = ap0.max_beams[0].get('beam', None)
        if hasattr(ap0, 'maxdm'):
            maxdm = ap0.maxdm

    try:
        from beamformer.utilities.dm import DMMap
        dmm = DMMap()
        dm_ne2001 = dmm.get_dm_ne2001(latitude=dec, longitude=ra)
        dm_ymw16 = dmm.get_dm_ymw16(latitude=dec, longitude=ra)
    except (ImportError, Exception) as e:
        print(e)

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

    # F0-phase grid
    ax3top = fig.add_subplot(gs[16, 11:17])
    ax3 = fig.add_subplot(gs[17:23, 11:17])
    ax3left = fig.add_subplot(gs[17:23, 10])

    # DM-phase grid
    ax4top = fig.add_subplot(gs[23, 11:17])
    ax4 = fig.add_subplot(gs[24:, 11:17])
    ax4left = fig.add_subplot(gs[24:, 10])

    # F0-1 grid
    ax_acceltop = fig.add_subplot(gs[9, 11:17])
    ax_accel = fig.add_subplot(gs[10:15, 11:17])
    ax_accelleft = fig.add_subplot(gs[10:15, 10])

    ax_kstext = fig.add_subplot(gs[0, 10:17])
    ax_kstext.axis("off")

    # Sky position scatterplot
    ax_sky = fig.add_subplot(gs[3:8, 10:17])
    ax_sky.yaxis.tick_right()
    ax_sky.yaxis.set_label_position("right")
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.4)

    # Initialize search result variables
    f0_best = None
    f1_best = None
    dm_best = None

    if accel_search is not None:
        dts = taxis.to(u.s).value
        dts = dts - np.median(dts)
        npbin = fs_bin.shape[-1]
        f0s, f1s = compute_accel_steps(dts, f0, npbin)
        print(f"Acceleration search with {len(f0s)} F0, {len(f1s)} F1 trials")

        prof2D = np.mean(fs_bin.squeeze(), 1)
        chi2_grid = phase_loop(prof2D, dts, f0s, f1s, metric=1)
        i_f0, i_f1 = np.unravel_index(np.argmax(chi2_grid), chi2_grid.shape)
        f0_best = f0s[i_f0]
        f1_best = f1s[i_f1]
        f0_slice = chi2_grid[:, i_f1]
        f1_slice = chi2_grid[i_f0]

        dphis = f0_best * dts + 0.5 * f1_best * dts**2
        i_phis = (dphis * npbin).astype("int")
        if not f0search:
            for i in range(fs_bin.shape[0]):
                fs_bin[i] = np.roll(fs_bin[i], -i_phis[i], axis=-1)
        else:
            print("Computing accel grid, but only showing profile from F0 search")
<<<<<<< HEAD

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
        DM_prof = DMprofs[i_dm_best]

        ax4.pcolormesh(np.linspace(0, 2, 2 * npbin), dm + DMs, DMprofs, cmap=cmap_name)
        ax4left.plot(-DM_SNs, dm + DMs, color=plot_color)
        ax4top.plot(np.linspace(0, 2, len(DM_prof)), DM_prof, color=plot_color)
        ax4left.set_yticks([])
        ax4left.set_xticks([])
        ax4top.set_xticks([])
        ax4top.set_yticks([])
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.set_xlabel("Phase", fontsize=12)
        ax4.set_ylabel(r"DM (pc cm$^3$)", fontsize=12)
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
        cmap=cmap_name,
    )
    ax2.imshow(
        np.nanmean(fs_bin, 1),
        aspect="auto",
        interpolation="nearest",
        vmin=vtmin,
        vmax=vtmax,
        origin="lower",
        extent=[0, 2, 0, max(taxis.to(u.min).value)],
        cmap=cmap_name,
    )

    ax1.set_ylabel("Obs Frequency (MHz)", fontsize=16)
    ax1.set_xticks([])

    ax2.set_ylabel("Time (min)", fontsize=16)
    ax2.set_xlabel("Phase", fontsize=16)

    phaseaxis = np.linspace(0, 2, 2 * ngate, endpoint=False)
    dp = phaseaxis[1] - phaseaxis[0]
    phaseaxis += dp / 2.0
    ax0.plot(phaseaxis, SNprof, color=plot_color)
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
    df0_best_str = f"{f0_best:.2e}" if f0_best is not None else "N/A"
    f0_best_str = f"{f0+f0_best:.5f}" if f0_best is not None else "N/A"
    f1_best_str = f"{f1_best:.5f}" if f1_best is not None else "N/A"
    P0_best = 1/(f0+f0_best) if f0_best is not None else "N/A"
    P0_best_str = f"{P0_best:.5f}" if P0_best is not None else "N/A"

    # 5 columns x 4 rows layout
    cand_params_text = [
        [rf"{psr}", f"RA: {ra:.2f}", rf"$g_l$: {gal_l:.2f}", f"DM$_{{max}}$: {maxdm_str}", f"$\\Delta$F0: {df0_best_str}"],
        [rf"{T0.isot[:10]}", f"Dec: {dec:.2f}", rf"$g_b$: {gal_b:.2f}", f"DM$_{{ne2001}}$: {ne2001_str}", f"F0$_{{best}}$: {f0_best_str}"],
        [rf"PS $\sigma$: {sigma_str}", f"DM: {dm:.2f}", f"Beam: {beam_str}", f"DM$_{{ymw16}}$: {ymw16_str}",f"P0$_{{best}}$: {P0_best_str}"],
        [rf"Fold $\sigma$: {SNR_val:.2f}", f"f0: {f0:.5f}", f"P0: {P0:.5f}", f"DM$_{{best}}$: {dm_best_str}" , f"F1: {f1_best_str}"],
    ]

    cand_param_table = axtext.table(
        cellText=cand_params_text, cellLoc="left", loc="top", edges="open",
        bbox=[-0.1, 0.5, 1.1, 2]
    )
    cand_param_table.auto_set_font_size(False)
    cand_param_table.set_fontsize(10)
    cand_param_table.auto_set_column_width(col=list(range(5)))
    foldsigma_cell = cand_param_table[(3, 0)]
    foldsigma_cell.get_text().set_weight('bold')

    # Render known sources table
    if len(ks_df.values) > 0:
        fig = ax_kstext.get_figure()

        # First create a temporary table to measure its height
        tmp_table = ax_kstext.table(
            cellText=ks_df.values,
            colLabels=ks_df.columns,
            loc="center"
        )
        tmp_table.auto_set_font_size(False)
        tmp_table.set_fontsize(9)
        tmp_table.auto_set_column_width(col=list(range(len(ks_df.columns))))

        fig.canvas.draw()

        # Get height of table in axes coordinates
        bbox = tmp_table.get_window_extent(fig.canvas.get_renderer())
        bbox_axes = bbox.transformed(ax_kstext.transAxes.inverted())
        table_height = bbox_axes.height

        # Remove temporary table
        tmp_table.remove()

        # Create final table, aligned to axis top
        ks_table = ax_kstext.table(
            cellText=ks_df.values,
            colLabels=ks_df.columns,
            cellLoc="center",
            loc=None,   # manually positioning!
            bbox=[-0.1, 2.5 - table_height, 1.2, table_height]  # top aligned!
        )
        ks_table.auto_set_font_size(False)
        ks_table.set_fontsize(9)
        ks_table.auto_set_column_width(col=list(range(len(ks_df.columns))))

        # Color/style code the rows based on match type
        for row_idx, is_psr_scraper in enumerate(ks_is_psr_scraper):
            is_harmonic_match = row_idx < likely_match_count

            if plot_bw:
                # Greyscale mode: use text styling instead of colors
                # Harmonic match: bold, Scraper: italic, Both: bold+italic
                weight = 'bold' if is_harmonic_match else 'normal'
                style = 'italic' if is_psr_scraper else 'normal'

                if weight == 'normal' and style == 'normal':
                    continue  # No special styling

                for col_idx in range(len(ks_df.columns)):
                    ks_table[(row_idx + 1, col_idx)].set_text_props(weight=weight, style=style)
            else:
                # Color mode: purple if both, red if harmonic only, blue if scraper only
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

        bbox = ks_table.get_window_extent(fig.canvas.get_renderer())
        bbox_ax = bbox.transformed(ax_kstext.transAxes.inverted())

        y_below = bbox_ax.y0 - 0.03

        if plot_bw:
            legend_text = (
                "Nearby Sources. Bold: likely match  |  Italic: unpublished  |  Bold+Italic: both"
            )
        else:
            legend_text = (
                "Nearby Sources. Red: likely match  |  Blue: unpublished  |  Purple: both"
            )
        if num_sources_not_displayed == 1:
            legend_text += f"\n {num_sources_not_displayed} additional source not displayed"
        if num_sources_not_displayed > 1:
            legend_text += f"\n {num_sources_not_displayed} additional sources not displayed"
        ax_kstext.text(
            0.5, y_below, legend_text,
            ha='center', va='top',
            fontsize=9,
            transform=ax_kstext.transAxes
        )

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
