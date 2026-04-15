# Re-define classes / functions in phase_aligned_search

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.constants import au, c
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    EarthLocation,
    SkyCoord,
    get_body_barycentric,
)
from astropy.time import Time
from beamformer.utilities.common import find_closest_pointing
from folding.utilities.archives import get_SN
from multiday_search.load_profiles import load_unwrapped_archives, unwrap_profiles
from numba import njit, prange, set_num_threads
from numpy import unravel_index
from scipy.ndimage import uniform_filter


@njit(parallel=True)
def phase_loop(profiles, dts, f0s, f1s, metric=0):
    """
    Calculates chi-squared or S/N of the sum of the phase-shifted profiles.

    Parameters
    ----------
    profiles : ndarray
        2D array of shape [ntime, nphase]
    dts : ndarray
        Time offsets in seconds
    f0s : ndarray
        F0 offset grid
    f1s : ndarray
        F1 grid
    metric : int
        0 for chi-squared (default), 1 for S/N with boxcar smoothing

    Returns
    -------
    grid : ndarray
        2D grid of chi-squared or S/N values
    """
    set_num_threads(8)

    npbin = profiles.shape[1]
    Nf1 = len(f1s)
    Nf0 = len(f0s)
    grid = np.zeros((Nf0, Nf1))

    # Pre-compute noise estimate from summed profiles
    sigma_off = np.std(profiles.sum(0))

    # Pre-compute boxcar widths (powers of 2 up to npbin//4) and scaled noise
    max_exp = int(np.log2(npbin // 4))
    n_widths = max_exp + 1
    widths = np.zeros(n_widths, dtype=np.int64)
    scaled_sigmas = np.zeros(n_widths)
    for iw in range(n_widths):
        widths[iw] = 2 ** iw
        scaled_sigmas[iw] = sigma_off / np.sqrt(widths[iw])

    for i, f0i in enumerate(f0s):
        profsums = np.zeros((Nf1, npbin))
        for j in prange(Nf1):
            f1j = f1s[j]
            dphis = f0i * dts + 0.5 * f1j * dts**2
            i_phis = (dphis * npbin).astype("int")

            for k, prof in enumerate(profiles):
                profsums[j] += np.roll(prof, -i_phis[k])

            if metric == 0:
                # Chi-squared
                grid[i, j] = np.sum(
                    (profsums[j] - np.mean(profsums[j])) ** 2 / sigma_off**2
                )
            else:
                # S/N with boxcar smoothing over powers of 2
                prof = profsums[j]
                prof_mean = np.mean(prof)
                snmax = 0.0

                # Cumulative sum for efficient boxcar computation
                cumsum = np.zeros(npbin + 1)
                for idx in range(npbin):
                    cumsum[idx + 1] = cumsum[idx] + prof[idx]

                for iw in range(n_widths):
                    width = widths[iw]
                    # Find max of boxcar-filtered profile
                    boxcar_max = -1e30
                    for idx in range(npbin):
                        end_idx = idx + width
                        if end_idx <= npbin:
                            val = (cumsum[end_idx] - cumsum[idx]) / width
                        else:
                            val = (cumsum[npbin] - cumsum[idx] + cumsum[end_idx - npbin]) / width
                        if val > boxcar_max:
                            boxcar_max = val

                    sn = (boxcar_max - prof_mean) / scaled_sigmas[iw]
                    if sn > snmax:
                        snmax = sn

                grid[i, j] = snmax

    return grid


class ExploreGrid:
    def __init__(self, data, f0_lims, f1_lims, f0_points, f1_points):
        self.f0_lims = f0_lims
        self.f1_lims = f1_lims
        self.profiles = data["profiles"]
        self.ngate = len(data["profiles"][0])
        self.dts = data["times"]
        self.f0_incoherent = data["F0"]
        self.P0_incoherent = 1 / self.f0_incoherent
        self.DM = data["DM"]
        self.RA = data["RA"]
        self.DEC = data["DEC"]
        self.directory = data["directory"]
        self.archives = data["archives"]
        self.psr_name = data["psr"]
        self.PEPOCH = data["PEPOCH"]
        self.candidate_sigma = data.get("candidate_sigma", None)

        self.f0_points = f0_points
        self.f1_points = f1_points
        f0_ax = np.linspace(*self.f0_lims, self.f0_points) - self.f0_incoherent
        f1_ax = np.linspace(*self.f1_lims, self.f1_points)
        self.f0_ax = f0_ax          # ΔF0 axis (Hz)
        self.f1_ax = f1_ax          # F1 axis  (s/s)
        self.f0s, self.f1s = np.meshgrid(f0_ax, f1_ax)

        self.chi2_grid = phase_loop(self.profiles, self.dts, f0_ax, f1_ax)

        index_of_maximum = unravel_index(self.chi2_grid.argmax(), self.chi2_grid.shape)

        df0_best = f0_ax[index_of_maximum[0]]
        f0_best = -df0_best + self.f0_incoherent
        f1_best = f1_ax[index_of_maximum[1]]
        self.max_indeces = index_of_maximum

        self.optimal_parameters = (f0_best, f1_best)
        self.profiles_aligned = unwrap_profiles(
            self.profiles, self.dts, df0_best, f1_best
        )
        self.SNmax = get_SN(self.profiles_aligned.sum(0))

    def output(self):
        print("f0: " + str(self.optimal_parameters[0]))
        print("f1: " + str(self.optimal_parameters[1]))
        print("SNR: " + str(np.max(self.SNmax)))
        return self.f0s, self.f1s, self.chi2_grid, self.optimal_parameters

    def plot(self, fullplot=True):
        """
        Grid layout
        -----------
        xx11112222
        3455556666
        3455556666
        3455557788
        3455557788

        1 – summed profile (phase, shared x with 5)
        2 – chi² vs ΔF0 marginal (shared x with 6)
        3 – cumulative S/N vs MJD (shared y with 5)
        4 – per-day S/N vs MJD (shared y with 5)
        5 – phase vs MJD  (main 2-D panel)
        6 – ΔF0 vs F1 search grid (shared x with 2)
        7 – phase vs frequency  (fullplot only)
        8 – phase vs time       (fullplot only)
        """
        plt.rcParams.update({"font.size": 14})

        # ------------------------------------------------------------------
        # Build figure and axes
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(26, 16))
        from matplotlib.gridspec import GridSpec as GS
        gs = GS(5, 10, figure=fig,
                height_ratios=[1, 2, 2, 2, 2],
                width_ratios=[1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                hspace=0.06, wspace=0.08)

        # create ax5 first so others can share its axes
        ax5 = fig.add_subplot(gs[1:5, 2:6])
        ax1 = fig.add_subplot(gs[0,   2:6], sharex=ax5)
        ax3 = fig.add_subplot(gs[1:5, 0  ], sharey=ax5)
        ax4 = fig.add_subplot(gs[1:5, 1  ], sharey=ax5)
        ax2 = fig.add_subplot(gs[0,   6:10])
        ax6 = fig.add_subplot(gs[1:3, 6:10], sharex=ax2)
        ax7 = fig.add_subplot(gs[3:5, 6:8 ])
        ax8 = fig.add_subplot(gs[3:5, 8:10])

        # ------------------------------------------------------------------
        # Panel 5 – phase vs MJD  (gap-correct with pcolormesh)
        # ------------------------------------------------------------------
        obs_mjds  = self.PEPOCH + self.dts / 86400.0
        mjd_start = obs_mjds.min()
        sort_idx  = np.argsort(obs_mjds)
        sorted_mjds     = obs_mjds[sort_idx]
        sorted_profiles = self.profiles_aligned[sort_idx]

        day_indices = np.round(obs_mjds - mjd_start).astype(int)
        n_days      = int(day_indices.max()) + 1

        profile2D_gapped = np.zeros((n_days, self.ngate * 2))
        for k in range(len(self.profiles_aligned)):
            profile2D_gapped[day_indices[k], :] = np.tile(self.profiles_aligned[k], 2)

        phase_edges = np.linspace(0, 2, self.ngate * 2 + 1)
        mjd_edges   = mjd_start + np.arange(n_days + 1)

        vmin5 = np.nanmean(profile2D_gapped) - np.nanstd(profile2D_gapped)
        vmax5 = np.nanmean(profile2D_gapped) + 3 * np.nanstd(profile2D_gapped)
        ax5.pcolormesh(phase_edges, mjd_edges, profile2D_gapped,
                       shading='flat', vmin=vmin5, vmax=vmax5)
        ax5.set_xlabel("Phase")
        ax5.set_ylabel("MJD")
        plt.setp(ax5.get_yticklabels(), visible=False)

        # ------------------------------------------------------------------
        # Panel 1 – summed profile
        # ------------------------------------------------------------------
        profile_total = self.profiles_aligned.sum(0)
        phase2 = np.linspace(0, 2, self.ngate * 2, endpoint=False)
        ax1.plot(phase2, np.tile(profile_total, 2), color='k', lw=1)
        ax1.set_xlim(0, 2)
        ax1.set_yticks([])
        plt.setp(ax1.get_xticklabels(), visible=False)

        # ------------------------------------------------------------------
        # Panel 3 – cumulative S/N vs MJD
        # ------------------------------------------------------------------
        cumulative_sn = np.zeros(len(sorted_mjds))
        for k in range(len(sorted_mjds)):
            cumprof = sorted_profiles[:k + 1].sum(0)
            cumulative_sn[k] = float(np.max(get_SN(cumprof)))

        ax3.plot(cumulative_sn, sorted_mjds, color='tab:blue', lw=1)
        ax3.invert_xaxis()
        ax3.set_xlabel("Cum.\nS/N", fontsize=11)
        ax3.set_ylabel("MJD")
        ax3.xaxis.set_major_locator(plt.MaxNLocator(3))

        # ------------------------------------------------------------------
        # Panel 4 – per-day S/N vs MJD
        # ------------------------------------------------------------------
        per_day_sn = np.array([float(np.max(get_SN(p))) for p in sorted_profiles])
        ax4.plot(per_day_sn, sorted_mjds, color='tab:orange', lw=0.8,
                 marker='o', ms=3)
        ax4.invert_xaxis()
        ax4.set_xlabel("Day\nS/N", fontsize=11)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.xaxis.set_major_locator(plt.MaxNLocator(3))

        # ------------------------------------------------------------------
        # Panel 2 – chi² vs ΔF0 (marginal over F1)
        # ------------------------------------------------------------------
        f0_uhz       = 1e6 * self.f0_ax
        chi2_marginal = np.max(self.chi2_grid, axis=1)
        f0best_uhz   = 1e6 * (self.optimal_parameters[0] - self.f0_incoherent)

        ax2.plot(f0_uhz, chi2_marginal, color='k', lw=1)
        ax2.axvline(f0best_uhz, color='tab:orange', ls='--', lw=1)
        ax2.set_ylabel(r"$\chi^{2}$")
        ax2.set_yticks([])
        plt.setp(ax2.get_xticklabels(), visible=False)

        # ------------------------------------------------------------------
        # Panel 6 – ΔF0 vs F1 search grid
        # ------------------------------------------------------------------
        f1best_1e15 = 1e15 * self.optimal_parameters[1]
        ax6.pcolormesh(f0_uhz, 1e15 * self.f1_ax, self.chi2_grid.T,
                       shading='auto')
        ax6.scatter(f0best_uhz, f1best_1e15,
                    color='tab:orange', marker='x', s=60, zorder=5)
        ax6.set_ylabel(r"$f_1$ ($10^{-15}$ s$^{-2}$)")
        ax6.set_xlabel(r"$\Delta f_0$ ($\mu$Hz)")

        # ------------------------------------------------------------------
        # Panels 7 & 8 – phase vs frequency / time (fullplot only)
        # ------------------------------------------------------------------
        if fullplot:
            data_T, data_F = load_unwrapped_archives(
                self.archives, self.optimal_parameters
            )
            vfmin = np.nanmean(data_F) - 2 * np.nanstd(data_F)
            vfmax = np.nanmean(data_F) + 5 * np.nanstd(data_F)
            vtmin = np.nanmean(data_T) - 2 * np.nanstd(data_T)
            vtmax = np.nanmean(data_T) + 5 * np.nanstd(data_T)

            data_T2 = np.tile(data_T, (1, 2))
            data_F2 = np.tile(data_F, (1, 2))

            ax7.imshow(data_F2, aspect='auto', interpolation='nearest',
                       vmin=vfmin, vmax=vfmax,
                       extent=[0, 2, 400, 800])
            ax7.set_xlabel("Phase")
            ax7.set_ylabel("Freq (MHz)")

            ax8.imshow(data_T2, aspect='auto', interpolation='nearest',
                       vmin=vtmin, vmax=vtmax,
                       extent=[0, 2, 0, data_T.shape[0]])
            ax8.set_xlabel("Phase")
            ax8.set_ylabel("T (subints)")
            ax8.yaxis.tick_right()
            ax8.yaxis.set_label_position('right')
        else:
            ax7.axis('off')
            ax8.axis('off')

        # ------------------------------------------------------------------
        # Candidate parameter table (suptitle area)
        # ------------------------------------------------------------------
        SNR      = float(np.max(self.SNmax))
        F0_best  = round(self.optimal_parameters[0], 6)
        F1plot   = f"{self.optimal_parameters[1]:.1e}"
        P0       = 1 / self.f0_incoherent
        P0_best  = 1 / self.optimal_parameters[0]

        gal_coord = SkyCoord(
            ra=self.RA * u.degree, dec=self.DEC * u.degree, frame="icrs"
        )
        gal_l = gal_coord.galactic.l.deg
        gal_b = gal_coord.galactic.b.deg
        pointing     = find_closest_pointing(self.RA, self.DEC)
        max_dm       = pointing.maxdm
        beam_str     = f"{pointing.beam_row}"
        start_date   = Time(self.PEPOCH + min(self.dts) / 86400.0, format="mjd").isot[:10]
        dm_ne2025_str = f"{pointing.ne2025dm:.1f}"
        dm_ymw16_str  = f"{pointing.ymw16dm:.1f}"
        sigma_str     = (f"σ: {self.candidate_sigma:.2f}"
                         if self.candidate_sigma is not None else "")

        cand_params_text = [
            [f"{self.psr_name}", f"RA: {self.RA:.2f}", f"gl: {gal_l:.2f}",
             f"DMmax: {max_dm:.1f}", f"F0_best: {F0_best}"],
            [f"{start_date}", f"Dec: {self.DEC:.2f}", f"gb: {gal_b:.2f}",
             f"DM_ne2025: {dm_ne2025_str}", f"P0_best: {P0_best:.5f}"],
            [sigma_str, f"DM: {self.DM:.2f}", f"Beam: {beam_str}",
             f"DM_ymw16: {dm_ymw16_str}", f"f1: {F1plot}"],
            [f"SNR: {SNR:.2f}", f"f0: {self.f0_incoherent:.5f}",
             f"P0: {P0:.5f}", f"Nday: {len(self.profiles)}",
             f"PEPOCH: {self.PEPOCH:.2f}"],
        ]

        table_ax = fig.add_axes([0.05, 0.92, 0.9, 0.06])
        table_ax.axis("off")
        param_table = table_ax.table(
            cellText=cand_params_text, cellLoc="left", loc="center", edges="open",
        )
        param_table.auto_set_font_size(False)
        param_table.set_fontsize(13)
        param_table.scale(1, 1.6)

        plot_name = (f"{self.directory}/phase_search_"
                     f"{round(self.DM, 2)}_{round(self.f0_incoherent, 2)}.png")
        print(f"Saving diagnostic plot to {plot_name}")
        plt.savefig(plot_name, bbox_inches="tight")
        plt.close()
        return plot_name
