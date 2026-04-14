"""
Circular-orbit grid search over the dF0 vs MJD S/N panel.

The orbital model is:
    F_spin(t) = F0_base + A * cos(2π * (t - T0) * F_binary)

where:
    t        – MJD (days)
    F0_base  – rest (barycentre) spin frequency offset, in the same units as
               the f0_offsets axis of the SN_F0_MJD panel (Hz)
    A        – semi-amplitude in spin frequency (Hz)
    F_binary – orbital frequency in cycles/day
    T0       – reference MJD of the orbit (parameterised as phase φ = T0 × F_binary
               so φ ∈ [0, 1))

Grid ranges
-----------
  F0_base  : every f0_offsets bin
  F_binary : 1/(10 × T_total) to 0.5 c/d, spacing 1/T_total
  A        : integer multiples of dF0_step: dF0_step × {1, 2, …, nA}
             where nA = round(F0_range / (4 × dF0_step))
  phase    : ceil(nA × 2π) uniform values in [0, 1), so that one step
             shifts the cosine by ≤ 1 dF bin at the largest amplitude
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Numba search kernel
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _orbital_sn_grid(SN_F0_MJD, f0_offsets, mjds_days, F_bins, phases, A_vals):
    """
    Evaluate the orbital S/N grid.

    Parameters
    ----------
    SN_F0_MJD : float64 array (nF0, ndays)
    f0_offsets : float64 array (nF0,)   – dF0 axis in Hz
    mjds_days  : float64 array (ndays,) – observation MJDs
    F_bins     : float64 array (nFbin,) – trial orbital frequencies, c/d
    phases     : float64 array (nPhase,)– trial phases ∈ [0, 1)
    A_vals     : float64 array (nA,)    – trial semi-amplitudes, Hz

    Returns
    -------
    grid : float64 array (nF0, nFbin, nPhase, nA)
        Summed S/N for each orbital parameter combination.
    """
    nF0, ndays = SN_F0_MJD.shape
    nFbin  = len(F_bins)
    nPhase = len(phases)
    nA     = len(A_vals)

    grid  = np.zeros((nF0, nFbin, nPhase, nA))
    dF0   = f0_offsets[1] - f0_offsets[0]
    F0min = f0_offsets[0]
    two_pi = 2.0 * np.pi

    for iF0 in prange(nF0):
        F0_base = f0_offsets[iF0]
        for iFbin in range(nFbin):
            F_bin = F_bins[iFbin]
            for iPhase in range(nPhase):
                phi = phases[iPhase]
                for iA in range(nA):
                    A = A_vals[iA]
                    total_sn = 0.0
                    for j in range(ndays):
                        # F_orbit = A * cos(2π*(mjd*F_bin − φ))
                        F_orbit = A * np.cos(two_pi * (mjds_days[j] * F_bin - phi))
                        F_total = F0_base + F_orbit
                        # Nearest-bin lookup (f0_offsets is uniformly spaced)
                        k = int((F_total - F0min) / dF0 + 0.5)
                        if k < 0:
                            k = 0
                        elif k >= nF0:
                            k = nF0 - 1
                        total_sn += SN_F0_MJD[k, j]
                    grid[iF0, iFbin, iPhase, iA] = total_sn

    return grid


# ---------------------------------------------------------------------------
# High-level search driver
# ---------------------------------------------------------------------------

def run_orbital_search(SN_F0_MJD, f0_offsets, mjds):
    """
    Build the orbital parameter grid and search it.

    Parameters
    ----------
    SN_F0_MJD : ndarray (nF0, ndays)
    f0_offsets : ndarray (nF0,)   – dF0 axis in Hz
    mjds       : ndarray (ndays,) – observation MJDs

    Returns
    -------
    result : dict or None
        None if no orbit is preferred (best S/N ≤ 1.5 × incoherent baseline).
        Otherwise a dict with keys:

          SN_grid      – (nF0, nFbin, nPhase, nA) S/N array
          F_bins       – (nFbin,) trial orbital frequencies
          phases       – (nPhase,) trial phases
          A_vals       – (nA,)    trial amplitudes
          best_sn      – float
          baseline_sn  – float (best incoherent sum)
          F0_best      – Hz  (best f0_offset)
          F_bin_best   – c/d
          phase_best   – [0, 1)
          A_best       – Hz
          indices      – (iF0, iFbin, iPhase, iA)
    """
    T_total = float(mjds[-1] - mjds[0])  # days
    if T_total <= 0 or len(mjds) < 2:
        return None

    dF0_step = float(f0_offsets[1] - f0_offsets[0])
    F0_range = float(f0_offsets[-1] - f0_offsets[0])

    # Orbital frequency grid (cycles / day)
    F_bin_min = 1.0 / (10.0 * T_total)
    F_bin_max = 0.5          # 2-day minimum period
    dF_bin    = 1.0 / T_total
    F_bins = np.arange(F_bin_min, F_bin_max + 0.5 * dF_bin, dF_bin)
    if len(F_bins) == 0:
        return None

    # A grid: exact multiples of dF0_step up to F0_range / 4
    nA     = max(4, int(round(F0_range / (4.0 * dF0_step))))
    A_vals = dF0_step * np.arange(1, nA + 1, dtype=np.float64)

    # Phase grid: spacing chosen so that one φ-step shifts the cosine by at
    # most 1 dF bin at the largest amplitude.  The maximum rate of change is
    # A_max * 2π (at the cosine's steepest point), so dφ = dF0_step/(A_max*2π)
    # and N_phi = ceil(A_max * 2π / dF0_step) = ceil(nA * 2π).
    N_phi  = int(np.ceil(nA * 2.0 * np.pi))
    phases = np.linspace(0.0, 1.0, N_phi, endpoint=False)

    SN_grid = _orbital_sn_grid(
        SN_F0_MJD.astype(np.float64),
        f0_offsets.astype(np.float64),
        mjds.astype(np.float64),
        F_bins.astype(np.float64),
        phases.astype(np.float64),
        A_vals.astype(np.float64),
    )

    # Incoherent baseline: best coherent sum at any constant F0
    baseline_sn = float(np.max(np.sum(SN_F0_MJD, axis=1)))
    best_sn     = float(SN_grid.max())

    if best_sn <= 1.5 * baseline_sn:
        return None

    idx = np.unravel_index(int(np.argmax(SN_grid)), SN_grid.shape)
    iF0, iFbin, iPhase, iA = idx

    return dict(
        SN_grid     = SN_grid,
        F_bins      = F_bins,
        phases      = phases,
        A_vals      = A_vals,
        best_sn     = best_sn,
        baseline_sn = baseline_sn,
        F0_best     = float(f0_offsets[iF0]),
        F_bin_best  = float(F_bins[iFbin]),
        phase_best  = float(phases[iPhase]),
        A_best      = float(A_vals[iA]),
        indices     = idx,
    )


# ---------------------------------------------------------------------------
# Plotting helpers (called from SemicoherentFoldSearch.plot)
# ---------------------------------------------------------------------------

def overlay_orbit(ax_2d, ax_top, orbit, f0_offsets, mjds, SN_F0_MJD):
    """
    Overlay the best-fit orbit on the dF0 vs MJD 2D panel and add the
    orbit-summed S/N profile to the top marginal in tab:orange.

    Parameters
    ----------
    ax_2d       : Axes – the dF0 × MJD pcolormesh axes (x=dF0, y=MJD)
    ax_top      : Axes – the marginal axes above ax_2d
    orbit       : dict returned by run_orbital_search
    f0_offsets  : ndarray (nF0,)
    mjds        : ndarray (ndays,)
    SN_F0_MJD  : ndarray (nF0, ndays)
    """
    F0_best    = orbit['F0_best']
    F_bin_best = orbit['F_bin_best']
    phase_best = orbit['phase_best']
    A_best     = orbit['A_best']

    # Dense curve for the 2D overlay (x=dF0, y=MJD – transposed layout)
    mjd_dense = np.linspace(float(mjds.min()), float(mjds.max()), 500)
    F_curve = F0_best + A_best * np.cos(
        2.0 * np.pi * (mjd_dense * F_bin_best - phase_best)
    )
    ax_2d.plot(F_curve, mjd_dense, color='tab:orange', lw=1.5, alpha=0.85, zorder=5)

    # Orbit-summed profile for the top marginal
    nF0    = len(f0_offsets)
    dF0    = f0_offsets[1] - f0_offsets[0]
    F0min  = float(f0_offsets[0])
    orbit_profile = np.zeros(nF0)
    for j, mjd in enumerate(mjds):
        F_orb = F0_best + A_best * np.cos(
            2.0 * np.pi * (mjd * F_bin_best - phase_best)
        )
        k = int(round((F_orb - F0min) / dF0))
        k = max(0, min(nF0 - 1, k))
        orbit_profile[k] += SN_F0_MJD[k, j]

    ax_top.plot(f0_offsets, orbit_profile, color='tab:orange', lw=1, alpha=0.8,
                label='orbit sum')


def add_corner_plot(fig, gs_corner, orbit, f0_offsets):
    """
    Fill the 4×4 corner plot (lower triangle = 2D S/N contours,
    diagonal = 1D marginals, upper triangle = parameter text).

    Parameters
    ----------
    fig       : Figure
    gs_corner : GridSpecFromSubplotSpec – 4 rows × 4 cols
    orbit     : dict returned by run_orbital_search
    f0_offsets : ndarray (nF0,)
    """
    SN_grid = orbit['SN_grid']          # (nF0, nFbin, nPhase, nA)
    F_bins  = orbit['F_bins']
    phases  = orbit['phases']
    A_vals  = orbit['A_vals']

    param_axes  = [f0_offsets, F_bins, phases, A_vals]
    param_labels = [r'$\Delta F_0$ (Hz)', r'$F_\mathrm{bin}$ (c/d)',
                    r'Phase', r'$A$ (Hz)']
    best_vals = [orbit['F0_best'], orbit['F_bin_best'],
                 orbit['phase_best'], orbit['A_best']]

    nparams = 4

    axes = [[None] * nparams for _ in range(nparams)]

    for row in range(nparams):
        for col in range(nparams):
            if col > row:
                continue   # upper triangle – handled separately
            ax = fig.add_subplot(gs_corner[row, col])
            axes[row][col] = ax

            if row == col:
                # 1-D marginal: max over all other axes
                other = tuple(k for k in range(nparams) if k != row)
                profile = SN_grid.max(axis=other)
                ax.plot(param_axes[row], profile, color='tab:blue', lw=1)
                ax.axvline(best_vals[row], color='tab:orange', lw=1, ls='--')
                ax.set_xlim(param_axes[row][0], param_axes[row][-1])
                ax.set_yticks([])
            else:
                # 2-D S/N contour: max over remaining axes
                other = tuple(k for k in range(nparams) if k != row and k != col)
                sn_2d = SN_grid.max(axis=other)
                # sn_2d has shape corresponding to axes [col, row] (col < row)
                # pcolormesh wants (x, y) = (col_axis, row_axis)
                ax.pcolormesh(param_axes[col], param_axes[row],
                              sn_2d.T, cmap='viridis', shading='auto')
                ax.axhline(best_vals[row], color='tab:orange', lw=0.8, alpha=0.8)
                ax.axvline(best_vals[col], color='tab:orange', lw=0.8, alpha=0.8)

            # Axis labels only on edges
            if row == nparams - 1:
                ax.set_xlabel(param_labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(param_labels[row], fontsize=9)
            else:
                if row != col:
                    ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    # Upper-triangle cell (0, nparams-1): print orbital parameters
    ax_text = fig.add_subplot(gs_corner[0, nparams - 1])
    ax_text.axis('off')
    T0_mjd = orbit['phase_best'] / orbit['F_bin_best'] if orbit['F_bin_best'] != 0 else 0.0
    P_orb  = 1.0 / orbit['F_bin_best'] if orbit['F_bin_best'] != 0 else float('inf')
    lines = [
        r'Best-fit orbit',
        r'',
        rf"$\Delta F_0 = {orbit['F0_best']:.3e}$ Hz",
        rf"$P_\mathrm{{orb}} = {P_orb:.2f}$ d",
        rf"$\phi = {orbit['phase_best']:.3f}$",
        rf"$A = {orbit['A_best']:.3e}$ Hz",
        r'',
        rf"S/N$_\mathrm{{orb}}$ = {orbit['best_sn']:.1f}",
        rf"S/N$_\mathrm{{base}}$ = {orbit['baseline_sn']:.1f}",
    ]
    ax_text.text(0.05, 0.95, '\n'.join(lines),
                 transform=ax_text.transAxes,
                 va='top', ha='left', fontsize=9,
                 family='monospace')
