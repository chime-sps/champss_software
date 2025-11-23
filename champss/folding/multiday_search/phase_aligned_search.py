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
from folding.archive_utils import get_SN
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

        self.f0_points = f0_points
        self.f1_points = f1_points
        f0_ax = np.linspace(*self.f0_lims, self.f0_points) - self.f0_incoherent
        f1_ax = np.linspace(*self.f1_lims, self.f1_points)
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
        plt.rcParams.update({"font.size": 25})
        if fullplot:
            fig, axs = plt.subplots(
                2,
                3,
                figsize=(20, 20),
                gridspec_kw={"width_ratios": [1, 2, 2], "height_ratios": [1, 2]},
            )
        else:
            fig, axs = plt.subplots(
                2, 2, figsize=(20, 20), gridspec_kw={"width_ratios": [1, 2]}
            )
        profile = self.profiles_aligned.sum(0)
        profile2D = self.profiles_aligned
        profile = np.tile(profile, 2)
        profile2D = np.tile(profile2D, (1, 2))

        # Search grid plot
        f0best_plot = (self.optimal_parameters[0] - self.f0_incoherent) * 1e6
        f1best_plot = self.optimal_parameters[1] * 1e15
        axs[0, 0].pcolormesh(1e6 * self.f0s, 1e15 * self.f1s, self.chi2_grid.T)
        axs[0, 0].scatter(
            -f0best_plot, f1best_plot, color="tab:orange", marker="x", s=20
        )
        axs[0, 0].set_xlabel(r"$\Delta f_0 (\mu Hz)$")
        axs[0, 0].set_ylabel(r"$f_1$ (1e-15 s/s)")

        # Aliasing plot
        axs[1, 0].plot(
            np.max(self.chi2_grid, axis=1).T, (self.f0s[0]) / 1e-6, "b", alpha=0.6
        )
        axs[1, 0].set_ylabel(r"$\Delta f_0  (\mu Hz)$")
        axs[1, 0].set_xlabel(r"$\chi^{2}$")

        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(wspace=0.4)
        axs[1, 0].set_xticks(
            axs[1, 0].get_xticks(), axs[1, 0].get_xticklabels(), rotation=45, ha="right"
        )

        # Total summed pulse profile plot
        axs[0, 1].plot(
            np.linspace(0, 2, self.ngate * 2), profile, color="k", label="Aligned sum"
        )
        axs[0, 1].label_outer()
        axs[0, 1].set_xlim(0, 2)

        # Days vs Phase vs Intensity plot
        axs[1, 1].imshow(
            profile2D,
            aspect="auto",
            interpolation="Nearest",
            extent=[0, 2, 0, len(self.profiles_aligned)],
        )
        axs[1, 1].set_xlabel("Phase")
        axs[1, 1].set_ylabel("Days")
        axs[1, 1].xaxis.get_major_ticks()[0].label1.set_visible(False)

        SNR = float(np.max(self.SNmax))
        F0plot = round(self.optimal_parameters[0], 6)
        F1plot = round(self.optimal_parameters[1] * 1e15, 2) / 1e15
        param_txt1 = f"$f_0$: {F0plot}\n$f_1$: {F1plot}\nSNR: {round(SNR, 2)}"

        param_txt2 = (
            f"RA (deg): {round(self.RA, 2)}\n"
            f"DEC (deg): {round(self.DEC, 2)}\n"
            f"DM: {round(self.DM, 2)}"
        )

        gal_coord = SkyCoord(
            ra=self.RA * u.degree, dec=self.DEC * u.degree, frame="icrs"
        )
        l = gal_coord.galactic.l.deg
        b = gal_coord.galactic.b.deg
        pointing = find_closest_pointing(self.RA, self.DEC)
        max_dm = pointing.maxdm

        param_txt3 = (
            f"$\\ell$ (deg): {round(l, 2)}\n"
            f"$\\it{{b}}$ (deg): {round(b, 2)}\n"
            f"Max DM: {round(max_dm,2)}"
        )

        axs[0, 1].text(
            0,
            1.3,
            param_txt1,
            transform=axs[0, 0].transAxes,
            fontsize=25,
            va="top",
            ha="left",
            backgroundcolor="white",
        )
        axs[0, 1].text(
            1.7,
            1.3,
            param_txt2,
            transform=axs[0, 0].transAxes,
            fontsize=25,
            va="top",
            ha="left",
            backgroundcolor="white",
        )
        axs[0, 1].text(
            4.2,
            1.3,
            param_txt3,
            transform=axs[0, 0].transAxes,
            fontsize=25,
            va="top",
            ha="left",
            backgroundcolor="white",
        )

        if fullplot:
            data_T, data_F = load_unwrapped_archives(
                self.archives, self.optimal_parameters
            )

            vfmin = np.nanmean(data_F) - 2 * np.nanstd(data_F)
            vfmax = np.nanmean(data_F) + 5 * np.nanstd(data_F)
            vtmin = np.nanmean(data_T) - 2 * np.nanstd(data_T)
            vtmax = np.nanmean(data_T) + 5 * np.nanstd(data_T)

            data_T = np.tile(data_T, (1, 2))
            data_F = np.tile(data_F, (1, 2))
            axs[1, 2].imshow(
                data_F,
                aspect="auto",
                interpolation="nearest",
                vmin=vfmin,
                vmax=vfmax,
                extent=[0, 2, 400, 800],
            )
            axs[1, 2].set_ylabel("Frequency (MHz)")
            axs[1, 2].set_xlabel("phase")
            axs[1, 2].yaxis.tick_left()
            axs[1, 2].yaxis.set_label_position("left")
            axs[1, 2].xaxis.get_major_ticks()[0].label1.set_visible(False)

            # Time vs phase plot, stacked observations

            axs[0, 2].imshow(
                data_T,
                aspect="auto",
                interpolation="nearest",
                extent=[0, 2, 0, data_T.shape[0]],
            )
            axs[0, 2].set_ylabel("T (subints)")
            axs[0, 2].set_xlabel("phase")
            axs[0, 2].yaxis.tick_left()
            axs[0, 2].yaxis.set_label_position("left")
            axs[0, 2].xaxis.get_major_ticks()[0].label1.set_visible(False)

        plot_name = f"{self.directory}/phase_search_{round(self.DM,2)}_{round(self.f0_incoherent,2)}.png"
        print(f"Saving diagnostic plot to {plot_name}")
        plt.savefig(plot_name, bbox_inches="tight")
        plt.close()
        return plot_name
