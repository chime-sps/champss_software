"""
Semi-coherent fold panel search.

Takes .npz panel files saved by plot_candidate_archive (one per day) and
produces a many-panel diagnostic figure:

Left column
-----------
  Top:    2D image of S/N vs DM and MJD, with a marginal panel above
          summing over MJD (mean S/N vs DM).
  Bottom: 2D image of S/N vs dF0 and MJD, with a marginal panel above
          summing over MJD (mean S/N vs dF0).

Right column
------------
  Two candidate rows (best and second-best day by fold S/N), each with:

  Left pair  – phase vs time panels
    Top:    summed profile (collapsed over sub-integrations for both days)
    Mid:    phase-vs-time for the primary day of this candidate
    Bottom: phase-vs-time for the secondary day

  Right pair – phase vs frequency panels
    Top:    summed profile collapsed over frequency (best DM)
    Mid:    phase-vs-frequency at best DM
    Bottom: phase-vs-frequency at DM = 0 (no dispersion correction)
"""

import os

import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_dm_shift(fs_freq_phase, dm_offset, freq, f0):
    """
    Apply an incoherent DM shift to a frequency-phase array.

    Parameters
    ----------
    fs_freq_phase : ndarray, shape (nfreq, nphase)
        Frequency-phase array at the current DM.
    dm_offset : float
        DM offset to apply in pc cm^-3 (negative to un-dedisperse).
    freq : ndarray
        Frequency array in MHz, shape (nfreq,).
    f0 : float
        Spin frequency in Hz.

    Returns
    -------
    shifted : ndarray, shape (nfreq, nphase)
    """
    DM_constant = 1.0 / 2.41e-4
    f_ref = float(freq.max())
    P_sec = 1.0 / f0
    nphase = fs_freq_phase.shape[1]
    t_delay = DM_constant * dm_offset * (1.0 / f_ref**2 - 1.0 / freq**2)
    pshifts = ((t_delay / P_sec) * nphase).astype(np.int32)
    shifted = np.zeros_like(fs_freq_phase)
    for j in range(len(freq)):
        shifted[j] = np.roll(fs_freq_phase[j], pshifts[j])
    return shifted


def _load_panels(npz_files):
    """Load and time-sort panel dicts from a list of .npz paths."""
    panels = []
    for fn in sorted(npz_files):
        d = np.load(fn, allow_pickle=True)
        panels.append({k: d[k] for k in d.files})
    panels.sort(key=lambda p: float(p['mjd']))
    return panels


def _build_sn_grids(panels):
    """
    Build 2D S/N arrays across (DM or dF0) × MJD from loaded panels.

    Returns
    -------
    mjds       : ndarray (ndays,)
    dm_abs     : ndarray (nDM,) or None  – absolute DM values
    SN_DM_MJD : ndarray (nDM, ndays) or None
    f0_offsets : ndarray (nF0,) or None  – dF0 offsets in Hz
    SN_F0_MJD : ndarray (nF0, ndays) or None
    """
    mjds = np.array([float(p['mjd']) for p in panels])
    ndays = len(panels)

    # ---- DM grid ----
    has_dm = all('DM_SNs' in p and len(p['DM_SNs']) > 0 for p in panels)
    SN_DM_MJD = None
    dm_abs = None
    if has_dm:
        ref_dms = panels[0]['DMs']
        dm_nominal = float(panels[0]['dm'])
        dm_offsets_ref = ref_dms - dm_nominal
        nDM = len(ref_dms)
        SN_DM_MJD = np.zeros((nDM, ndays))
        for j, p in enumerate(panels):
            sn = p['DM_SNs']
            if len(sn) == nDM:
                SN_DM_MJD[:, j] = sn
            else:
                dm_off_j = p['DMs'] - float(p['dm'])
                SN_DM_MJD[:, j] = np.interp(dm_offsets_ref, dm_off_j, sn,
                                             left=np.nan, right=np.nan)
        dm_abs = ref_dms

    # ---- F0 grid ----
    has_f0 = all('F0_SNs' in p and len(p['F0_SNs']) > 0 for p in panels)
    SN_F0_MJD = None
    f0_offsets = None
    if has_f0:
        f0_offsets = panels[0]['f0s']
        nF0 = len(f0_offsets)
        SN_F0_MJD = np.zeros((nF0, ndays))
        for j, p in enumerate(panels):
            sn = p['F0_SNs']
            if len(sn) == nF0:
                SN_F0_MJD[:, j] = sn
            else:
                SN_F0_MJD[:, j] = np.interp(f0_offsets, p['f0s'], sn,
                                             left=np.nan, right=np.nan)

    return mjds, dm_abs, SN_DM_MJD, f0_offsets, SN_F0_MJD


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SemicoherentFoldSearch:
    """
    Semi-coherent fold panel search over N days of .npz panel files.

    Parameters
    ----------
    npz_files : list of str
        Paths to panel .npz files produced by plot_candidate_archive.
    """

    def __init__(self, npz_files):
        self.npz_files = list(npz_files)
        self.panels = _load_panels(self.npz_files)
        (self.mjds,
         self.dm_abs,
         self.SN_DM_MJD,
         self.f0_offsets,
         self.SN_F0_MJD) = _build_sn_grids(self.panels)

        per_day_sn = np.array([float(p['SN']) for p in self.panels])
        self.best_day_indices = list(np.argsort(per_day_sn)[::-1])

    # ------------------------------------------------------------------
    # Internal plotting helpers
    # ------------------------------------------------------------------

    def _plot_left_dm(self, ax_top, ax_2d, cmap, color):
        if self.SN_DM_MJD is None:
            ax_top.axis('off')
            ax_2d.axis('off')
            return
        # Transposed: DM on x-axis, MJD on y-axis
        ax_2d.pcolormesh(self.dm_abs, self.mjds, self.SN_DM_MJD.T, cmap=cmap)
        ax_2d.set_xlabel(r'DM (pc cm$^{-3}$)', fontsize=9)
        ax_2d.set_ylabel('MJD', fontsize=9)

        mean_sn = np.nanmean(self.SN_DM_MJD, axis=1)   # mean over MJD → shape (nDM,)
        ax_top.plot(self.dm_abs, mean_sn, color=color, lw=1)
        ax_top.set_xlim(self.dm_abs[0], self.dm_abs[-1])
        ax_top.set_xticks([])
        ax_top.set_ylabel('Mean S/N', fontsize=7)

    def _plot_left_f0(self, ax_top, ax_2d, cmap, color):
        if self.SN_F0_MJD is None:
            ax_top.axis('off')
            ax_2d.axis('off')
            return
        # Transposed: dF0 on x-axis, MJD on y-axis
        ax_2d.pcolormesh(self.f0_offsets, self.mjds, self.SN_F0_MJD.T, cmap=cmap)
        ax_2d.set_xlabel(r'$\Delta F_0$ (Hz)', fontsize=9)
        ax_2d.set_ylabel('MJD', fontsize=9)

        mean_sn = np.nanmean(self.SN_F0_MJD, axis=1)   # mean over MJD → shape (nF0,)
        ax_top.plot(self.f0_offsets, mean_sn, color=color, lw=1)
        ax_top.set_xlim(self.f0_offsets[0], self.f0_offsets[-1])
        ax_top.set_xticks([])
        ax_top.set_ylabel('Mean S/N', fontsize=7)

    def _plot_candidate_row(self, panel_a, panel_b,
                            ax_prof_t,
                            ax_tp_a_nom, ax_tp_a_corr,
                            ax_tp_b_nom, ax_tp_b_corr,
                            ax_prof_f,
                            ax_fp_a_best, ax_fp_a_dm0,
                            ax_fp_b_best, ax_fp_b_dm0,
                            cmap, color):
        """
        Fill one candidate row.

        Left pair (phase vs time, 4 panels):
          ax_tp_a_nom  – day A at nominal fold F0
          ax_tp_a_corr – day A phase-corrected to best dF0 (+F1 if searched)
          ax_tp_b_nom  – day B at nominal fold F0
          ax_tp_b_corr – day B phase-corrected to best dF0 (+F1 if searched)

        Right pair (phase vs frequency, 4 panels — mirrors left):
          ax_fp_a_best – day A at best DM
          ax_fp_a_dm0  – day A at DM = 0
          ax_fp_b_best – day B at best DM
          ax_fp_b_dm0  – day B at DM = 0
        """
        # ---- Left pair: phase vs time ----
        profile_a = panel_a.get('profile', None)

        if profile_a is not None:
            nphase = profile_a.shape[0]
            phase_ax2 = np.linspace(0, 2, 2 * nphase, endpoint=False)

            prof_sum = profile_a.copy()
            prof_b = panel_b.get('profile', None)
            if prof_b is not None and prof_b.shape[0] == nphase:
                prof_sum = prof_sum + prof_b
            ax_prof_t.plot(phase_ax2, np.tile(prof_sum, 2), color=color, lw=1)
            ax_prof_t.set_xlim(0, 2)
            ax_prof_t.set_xticks([])
            ax_prof_t.set_title(
                f"MJD {float(panel_a['mjd']):.2f}  S/N={float(panel_a['SN']):.1f}",
                fontsize=7, pad=2)

            def _draw_tp(ax, fs_tp, label):
                if fs_tp is None or fs_tp.size == 0:
                    ax.axis('off')
                    return
                fs_tiled = np.tile(fs_tp, (1, 2))
                vmin = np.nanmean(fs_tp) - np.nanstd(fs_tp)
                vmax = np.nanmean(fs_tp) + 3 * np.nanstd(fs_tp)
                ax.imshow(fs_tiled, aspect='auto', interpolation='nearest',
                          extent=[0, 2, 0, fs_tp.shape[0]],
                          cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax.set_ylabel(label, fontsize=6)
                ax.set_xticks([])

            for panel, ax_nom, ax_corr in [
                (panel_a, ax_tp_a_nom, ax_tp_a_corr),
                (panel_b, ax_tp_b_nom, ax_tp_b_corr),
            ]:
                mjd_str = f"MJD {float(panel['mjd']):.2f}"
                f0_best = float(panel.get('f0_best', 0.0))
                f1_best = float(panel.get('f1_best', 0.0))
                if f1_best != 0.0:
                    corr_label = f"{mjd_str}\ndF0={f0_best:.2e} F1={f1_best:.1e}"
                else:
                    corr_label = f"{mjd_str}\ndF0={f0_best:.2e}"

                _draw_tp(ax_nom,  panel.get('fs_time_phase_nominal', None),
                         f"{mjd_str} (nominal)")
                _draw_tp(ax_corr, panel.get('fs_time_phase', None), corr_label)

            ax_tp_b_corr.set_xlabel('Phase', fontsize=8)
            ax_tp_b_corr.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        else:
            for ax in (ax_prof_t, ax_tp_a_nom, ax_tp_a_corr,
                       ax_tp_b_nom, ax_tp_b_corr):
                ax.axis('off')

        # ---- Right pair: phase vs frequency (both days) ----
        def _draw_fp(ax, panel, dm_offset, label):
            fs_fp = panel.get('fs_freq_phase', None)
            freq  = panel.get('freq', None)
            if fs_fp is None or freq is None or len(freq) == 0:
                ax.axis('off')
                return
            f0 = float(panel['f0'])
            fs = _apply_dm_shift(fs_fp, dm_offset, freq, f0)
            fs_tiled = np.tile(fs, (1, 2))
            vmin = np.nanmean(fs) - np.nanstd(fs)
            vmax = np.nanmean(fs) + 3 * np.nanstd(fs)
            ax.imshow(fs_tiled, aspect='auto', interpolation='nearest',
                      extent=[0, 2, freq.min(), freq.max()],
                      cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_ylabel(f'Freq (MHz)\n{label}', fontsize=6)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_xticks([])
            return fs.mean(0)   # profile summed over frequency

        # Summed profile for top panel: best-DM profiles from both days
        prof_f_sum = None
        nphase_f = None
        for panel in (panel_a, panel_b):
            fs_fp = panel.get('fs_freq_phase', None)
            freq  = panel.get('freq', None)
            if fs_fp is None or freq is None or len(freq) == 0:
                continue
            dm      = float(panel['dm'])
            dm_best = float(panel.get('dm_best', dm))
            f0      = float(panel['f0'])
            fs = _apply_dm_shift(fs_fp, dm_best - dm, freq, f0)
            prof = fs.mean(0)
            if prof_f_sum is None:
                prof_f_sum = prof.copy()
                nphase_f = len(prof)
            elif len(prof) == nphase_f:
                prof_f_sum += prof

        if prof_f_sum is not None:
            phase_ax2 = np.linspace(0, 2, 2 * nphase_f, endpoint=False)
            ax_prof_f.plot(phase_ax2, np.tile(prof_f_sum, 2), color=color, lw=1)
            ax_prof_f.set_xlim(0, 2)
            ax_prof_f.set_xticks([])
        else:
            ax_prof_f.axis('off')

        for panel, ax_best, ax_dm0 in [
            (panel_a, ax_fp_a_best, ax_fp_a_dm0),
            (panel_b, ax_fp_b_best, ax_fp_b_dm0),
        ]:
            dm      = float(panel.get('dm', 0))
            dm_best = float(panel.get('dm_best', dm))
            mjd_str = f"MJD {float(panel['mjd']):.2f}"
            _draw_fp(ax_best, panel, dm_best - dm, f"{mjd_str} DM={dm_best:.1f}")
            _draw_fp(ax_dm0,  panel, -dm,          f"{mjd_str} DM=0")

        ax_fp_b_dm0.set_xlabel('Phase', fontsize=8)
        ax_fp_b_dm0.set_xticks([0, 0.5, 1.0, 1.5, 2.0])

    # ------------------------------------------------------------------
    # Public plot method
    # ------------------------------------------------------------------

    def plot(self, output_dir=None, plot_bw=False):
        """
        Create and save the semi-coherent diagnostic figure.

        Parameters
        ----------
        output_dir : str, optional
            Directory for the output PNG.  Defaults to the directory of the
            first npz file.
        plot_bw : bool
            Use a greyscale colour scheme if True.

        Returns
        -------
        plot_name : str
            Path to the saved figure.
        """
        cmap = 'Greys_r' if plot_bw else 'viridis'
        color = 'black' if plot_bw else 'tab:blue'

        panels = self.panels
        ndays = len(panels)
        p0 = panels[0]

        # Layout: 17 columns × 21 rows
        #
        #   Cols  0– 7: left DM / F0 panels          (8 cols)
        #   Col   8   : empty gap                    (1 col)
        #   Cols  9–12: best-day phase-time panels    (4 cols)
        #   Cols 13–16: best-day phase-freq panels    (4 cols)
        #
        #   Rows  0– 1: DM marginal / right profile  (2 rows)
        #   Rows  2– 9: DM 2D image / day-A panels   (8 rows, 4 each)
        #   Row  10   : 1-grid gap                   (tiny height)
        #   Rows 11–12: F0 marginal                  (2 rows)
        #   Rows 13–20: F0 2D image / day-B panels   (8 rows, 4 each)
        #
        # Right panels span all rows, split by the same gap:
        #   Rows  0– 1: summed profiles (time + freq)
        #   Rows  2– 5: day A nominal F0  /  day A best DM
        #   Rows  6– 9: day A corrected   /  day A DM=0
        #   Row  10   : gap (visual break between day A and day B)
        #   Rows 11–14: day B nominal F0  /  day B best DM
        #   Rows 15–18: day B corrected   /  day B DM=0
        #   Rows 19–20: (unused)

        n_rows = 21
        height_ratios = [1] * n_rows
        height_ratios[10] = 0.25   # tiny visual gap between DM and F0 sections

        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(n_rows, 17, figure=fig,
                      height_ratios=height_ratios,
                      hspace=0.06, wspace=0.06)

        # Left column – DM × MJD (transposed: DM on x, MJD on y)
        ax_dm_top = fig.add_subplot(gs[0:2,  0:8])
        ax_dm_2d  = fig.add_subplot(gs[2:10, 0:8])

        # Left column – F0 × MJD (transposed: dF0 on x, MJD on y)
        ax_f0_top = fig.add_subplot(gs[11:13, 0:8])
        ax_f0_2d  = fig.add_subplot(gs[13:21, 0:8])

        self._plot_left_dm(ax_dm_top, ax_dm_2d, cmap, color)
        self._plot_left_f0(ax_f0_top, ax_f0_2d, cmap, color)

        # Single best-day candidate panel (cols 9-16, all rows)
        # Col 8 is left empty as a 1-grid horizontal gap
        ax_prof_t    = fig.add_subplot(gs[0:2,   9:13])
        ax_tp_a_nom  = fig.add_subplot(gs[2:6,   9:13])
        ax_tp_a_corr = fig.add_subplot(gs[6:10,  9:13])
        ax_tp_b_nom  = fig.add_subplot(gs[11:15, 9:13])
        ax_tp_b_corr = fig.add_subplot(gs[15:19, 9:13])

        ax_prof_f    = fig.add_subplot(gs[0:2,   13:17])
        ax_fp_a_best = fig.add_subplot(gs[2:6,   13:17])
        ax_fp_a_dm0  = fig.add_subplot(gs[6:10,  13:17])
        ax_fp_b_best = fig.add_subplot(gs[11:15, 13:17])
        ax_fp_b_dm0  = fig.add_subplot(gs[15:19, 13:17])

        if len(self.best_day_indices) >= 1:
            idx_a = self.best_day_indices[0]
            others = [i for i in self.best_day_indices if i != idx_a]
            idx_b = others[0] if others else idx_a
            panel_a = panels[idx_a]
            panel_b = panels[idx_b]

            self._plot_candidate_row(
                panel_a, panel_b,
                ax_prof_t,
                ax_tp_a_nom, ax_tp_a_corr,
                ax_tp_b_nom, ax_tp_b_corr,
                ax_prof_f,
                ax_fp_a_best, ax_fp_a_dm0,
                ax_fp_b_best, ax_fp_b_dm0,
                cmap, color,
            )
        else:
            for ax in (ax_prof_t, ax_tp_a_nom, ax_tp_a_corr,
                       ax_tp_b_nom, ax_tp_b_corr,
                       ax_prof_f, ax_fp_a_best, ax_fp_a_dm0,
                       ax_fp_b_best, ax_fp_b_dm0):
                ax.axis('off')

        f0_str = f"{float(p0['f0']):.5f}"
        dm_str = f"{float(p0['dm']):.2f}"
        plt.suptitle(
            f'Semi-coherent fold panel search  |  N={ndays} days  |  '
            f'f0={f0_str} Hz  |  DM={dm_str} pc/cm³',
            fontsize=11, y=1.005,
        )

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.npz_files[0]))
        os.makedirs(output_dir, exist_ok=True)

        plot_name = os.path.join(
            output_dir,
            f'semicoherent_{f0_str}_{dm_str}_N{ndays}.png',
        )
        plt.savefig(plot_name, dpi=150, bbox_inches='tight',
                    pil_kwargs={'optimize': True})
        plt.close()
        print(f"Saved semi-coherent diagnostic plot to {plot_name}")
        return plot_name


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("npz_files", nargs=-1, required=True)
@click.option(
    "--output-dir", "-o",
    default=None, type=str,
    help="Directory for the output diagnostic plot.  "
         "Defaults to the directory containing the first npz file.",
)
@click.option(
    "--plot-bw", is_flag=True,
    help="Use a black/white (Greys_r) colour scheme.",
)
def main(npz_files, output_dir, plot_bw):
    """
    Semi-coherent fold panel search from N days of npz panel files.

    NPZ_FILES should be the *_panels.npz files written by plot_candidate_archive
    (one per day of observation).
    """
    search = SemicoherentFoldSearch(list(npz_files))
    plot_name = search.plot(output_dir=output_dir, plot_bw=plot_bw)
    print(f"Done: {plot_name}")


if __name__ == "__main__":
    main()
