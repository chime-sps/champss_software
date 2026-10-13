"""
Run the multi-day folding search on local archives, without the database.

This mirrors the folding step of fold_candidate and the coherent and semi-coherent
searches of confirm_cand, but takes archives and an ephemeris directly instead of a
FollowUpSource ID. Needs psrchive, dspsr and pam, e.g. inside the champss_software
image with /champss_module/miniconda3/bin first in PATH.

Subcommands
-----------
fold:   fold filterbank files with dspsr like fold_candidate, create the FT archive
        and the diagnostic plot plus .searchpanels.npz with plot_candidate_archive.
        With --clean the archive is RFI-cleaned first.
clean:  RFI-clean existing archives into a new directory and recreate the FT archive,
        plot and .searchpanels.npz.
beamform-fold: beamform CHAMPSS raw data like fold_candidate and fold with dspsr at a
        chosen number of phase bins, then optionally clean and create the products.
panels: copy existing archives into a new directory and recreate the FT archive, plot
        and .searchpanels.npz, e.g. with a different --f0-vmax.
search: run the phase-aligned coherent F0/F1 search and the semi-coherent fold panel
        search (with orbital search) on a set of archives.
z2search: semi-coherent search that sums the per-day Z^2_n power (instead of the per-day
        best S/N) over days, along constant-F0 lines and circular-orbit tracks. Z^2 is a
        calibrated quadratic statistic, so weak signals that are not detected on single
        days still add up, like in a power-spectrum stack.
"""

import datetime
import os
import shutil
import subprocess

import click
import numpy as np


def robust_z(values, valid):
    """Robust z-score of values (median/MAD over the valid entries)."""
    med = np.median(values[valid])
    mad = 1.4826 * np.median(np.abs(values[valid] - med))
    return (values - med) / max(mad, 1e-30)


def clean_archive(archive_in, archive_out, cell_sigma=6.0, chan_sigma=4.0, max_flagged=0.5):
    """
    Zero-weight and zero RFI in a folded archive and write the result to archive_out.

    Flags (iteratively, with median/MAD statistics):
    - channels whose typical noise level is zero or an outlier,
    - single (subint, channel) cells whose noise is an outlier (bursts),
    - channels whose time-summed profile has more phase structure than noise allows
      (persistent narrowband periodic RFI, which the per-cell variance flagging in
      clean_foldspec does not catch because each cell looks normal),
    - isolated live channels inside masked bands (fewer than 2 live channels among the
      4 neighbours), which border RFI bands and are usually contaminated,
    - channels or subints with more than max_flagged of their cells flagged.
    Flagged data are set to zero weight and zero amplitude, since readpsrarch ignores
    the weights.
    """
    import psrchive

    arch = psrchive.Archive.load(archive_in)
    data = arch.get_data().sum(1)  # total intensity, (nsub, nchan, nbin)
    nsub, nchan, nbin = data.shape
    data = data - np.median(data, axis=-1, keepdims=True)
    cell_noise = 1.4826 * np.median(np.abs(data), axis=-1)
    flagged = arch.get_weights() == 0

    for _ in range(3):
        good = ~flagged
        # Channel noise level, dead channels and bandpass outliers
        chan_noise = np.array(
            [np.median(cell_noise[good[:, c], c]) if good[:, c].any() else 0.0 for c in range(nchan)]
        )
        valid_chan = chan_noise > 0
        flagged[:, ~valid_chan] = True
        flagged[:, valid_chan & (np.abs(robust_z(chan_noise, valid_chan)) > cell_sigma)] = True

        # Bursts in single cells, relative to their channel's noise level
        ratio = np.where(chan_noise[None, :] > 0, cell_noise / np.maximum(chan_noise[None, :], 1e-30), 0)
        good = ~flagged
        flagged |= good & (robust_z(ratio, good) > cell_sigma)

        # Persistent phase structure in the time-summed profile of each channel
        good = ~flagged
        ngood = good.sum(0)
        prof = (data * good[..., None]).sum(0)
        expected = np.maximum(chan_noise * np.sqrt(np.maximum(ngood, 1)), 1e-30)
        structure = np.std(prof / expected[:, None], axis=-1)
        valid_chan = ngood > 0
        flagged[:, valid_chan & (robust_z(structure, valid_chan) > chan_sigma)] = True

        # Isolated live channels inside masked bands
        live = ~flagged.all(0)
        live_padded = np.pad(live, 2)
        live_neighbours = sum(np.roll(live_padded, shift)[2:-2] for shift in (-2, -1, 1, 2))
        flagged[:, live & (live_neighbours < 2)] = True

        # Mostly bad channels or subints, counting only channels and subints that are
        # not fully flagged already (a large part of the band can be masked upstream)
        dead_chan = flagged.all(0)
        dead_sub = flagged.all(1)
        if (~dead_sub).any():
            flagged[:, flagged[~dead_sub].mean(0) > max_flagged] = True
        if (~dead_chan).any():
            flagged[flagged[:, ~dead_chan].mean(1) > max_flagged, :] = True

    for isub in range(nsub):
        integ = arch.get_Integration(isub)
        for ichan in np.flatnonzero(flagged[isub]):
            integ.set_weight(int(ichan), 0.0)
            for ipol in range(arch.get_npol()):
                integ.get_Profile(ipol, int(ichan)).scale(0)
    arch.unload(archive_out)

    nchan_zapped = int(flagged.all(0).sum())
    print(
        f"Cleaned {os.path.basename(archive_in)}: {flagged.mean():.1%} of cells flagged, "
        f"{nchan_zapped} of {nchan} channels fully zapped"
    )
    return flagged


def make_products(archive, outdir, f0_vmax=None):
    """
    Create the FT archive, the diagnostic plot and the .searchpanels.npz file.

    f0_vmax (km/s) widens the per-day F0 search, which the semi-coherent search reads
    from the .searchpanels.npz files, to the Doppler range of that orbital velocity.
    By default it covers +-2 phase wraps over the observation.
    """
    import astropy.units as u
    from folding.plot_candidate import plot_candidate_archive

    f0search = True
    if f0_vmax:
        f0search = {"stack_binary_search": True, "vmax": f0_vmax * u.km / u.s}
    subprocess.run(["pam", "-T", "-F", archive, "-e", "FT"], check=True)
    SNprof, SN_arr, plot_fname, npz_file = plot_candidate_archive(
        archive, outdir, cand_info={}, foldpath=os.path.join(outdir, "plots"),
        f0search=f0search,
    )
    print(f"{archive}: S/N {SN_arr}, plot {plot_fname}, panels {npz_file}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    pass


@cli.command()
@click.argument("filterbanks", nargs=-1, required=True)
@click.option("--par", required=True, type=str, help="Ephemeris to fold with.")
@click.option("--outdir", required=True, type=str, help="Directory for the archives.")
@click.option("--num-threads", default=8, type=int, help="Threads used by dspsr.")
@click.option("--overwrite", is_flag=True, help="Refold even if the archive exists.")
@click.option("--clean", is_flag=True, help="Clean RFI in the archive before the FT and plots.")
@click.option(
    "--f0-vmax",
    default=None,
    type=float,
    help="Widen the per-day F0 search to the Doppler range of this velocity (km/s).",
)
@click.option("--nbin", default=None, type=int, help="Number of phase bins passed to dspsr.")
def fold(filterbanks, par, outdir, num_threads, overwrite, clean, f0_vmax, nbin):
    """Fold FILTERBANKS with dspsr in the same way as fold_candidate."""
    os.makedirs(outdir, exist_ok=True)
    for fil in filterbanks:
        base = os.path.join(outdir, os.path.splitext(os.path.basename(fil))[0])
        archive = base + ".ar"
        if overwrite or not os.path.isfile(archive):
            # Same settings as Fold.fold: 10 s subintegrations, single archive
            cmd = ["dspsr", "-t", str(num_threads), "-L", "10", "-A", "-k", "chime",
                   "-E", par, "-O", base, fil]
            if nbin:
                cmd[1:1] = ["-b", str(nbin)]
            print(" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.isfile(archive):
                print(f"dspsr failed for {fil}:\n{result.stderr}")
                continue
            if clean:
                clean_archive(archive, archive)
        make_products(archive, outdir, f0_vmax=f0_vmax)


@cli.command("beamform-fold")
@click.argument("dates", nargs=-1, required=True)
@click.option("--ra", required=True, type=float, help="RA of the candidate (deg).")
@click.option("--dec", required=True, type=float, help="Dec of the candidate (deg).")
@click.option("--par", required=True, type=str, help="Ephemeris to fold with.")
@click.option("--outdir", required=True, type=str, help="Directory for the archives.")
@click.option("--datpath", default="/mnt/beegfs-client/raw", type=str, help="Raw data path.")
@click.option("--nbin", default=16, type=int, help="Number of phase bins passed to dspsr.")
@click.option("--clean", is_flag=True, help="Clean RFI in the archive before the FT and plots.")
@click.option(
    "--f0-vmax",
    default=None,
    type=float,
    help="Widen the per-day F0 search to the Doppler range of this velocity (km/s).",
)
@click.option("--control-par", default=None, type=str, help="Also fold at this control ephemeris.")
@click.option("--control-outdir", default=None, type=str, help="Directory for the control archives.")
@click.option(
    "--filterbank-dir",
    default=None,
    type=str,
    help="Keep the beamformed filterbanks here and reuse existing ones (by pointing and date).",
)
@click.option(
    "--nchan",
    default=None,
    type=int,
    help="Number of channels to beamform with (default: chosen from the DM like fold_candidate).",
)
@click.option(
    "--exact-coords",
    is_flag=True,
    help="Beamform at exactly --ra/--dec instead of the nearest pointing-map position.",
)
def beamform_fold(
    dates, ra, dec, par, outdir, datpath, nbin, clean, f0_vmax, control_par, control_outdir,
    filterbank_dir, nchan, exact_coords,
):
    """
    Beamform CHAMPSS raw data on DATES (YYYYMMDD) like fold_candidate and fold with dspsr.

    Unlike fold_candidate, the number of phase bins can be set, which matters when
    the period is only a few CHAMPSS samples long.
    """
    import datetime as dt

    from folding.fold import Fold
    from folding.utilities.database import scrape_ephemeris
    from sps_pipeline.pipeline import load_config

    os.makedirs(outdir, exist_ok=True)
    eph = scrape_ephemeris(par)
    for date_str in dates:
        date = dt.datetime.strptime(date_str, "%Y%m%d")
        folder = Fold(
            ra=ra,
            dec=dec,
            f0=eph["f0"],
            dm=float(eph["dm"]),
            date=date,
            ephem_path=par,
            foldpath=outdir,
            datpath=datpath,
            config=load_config(),
            filterbank_to_ram=False,
            coord_path=outdir,
            nchan=nchan,
            exact_coords=exact_coords,
        )
        folder.setup_paths()
        if filterbank_dir:
            os.makedirs(filterbank_dir, exist_ok=True)
            nchan_tag = f"_{nchan}ch" if nchan else ""
            folder.fil = os.path.join(filterbank_dir, f"{ra:.02f}_{dec:.02f}_{date_str}{nchan_tag}.fil")
        if not folder.beamform():
            print(f"{date_str}: beamforming failed, skipping")
            continue
        folds = [(par, outdir, folder.archive_fname)]
        if control_par:
            os.makedirs(control_outdir, exist_ok=True)
            control_base = os.path.join(
                control_outdir, f"control_{date.year}-{date.month:02}-{date.day:02}"
            )
            folds.append((control_par, control_outdir, control_base))
        for fold_par, fold_outdir, base in folds:
            cmd = ["dspsr", "-t", str(folder.num_threads), "-L", "10", "-A", "-k", "chime",
                   "-b", str(nbin), "-E", fold_par, "-O", base, folder.fil]
            print(" ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            archive = base + ".ar"
            if result.returncode != 0 or not os.path.isfile(archive):
                print(f"{date_str}: dspsr failed:\n{result.stderr}")
                continue
            if clean:
                clean_archive(archive, archive)
            make_products(archive, fold_outdir, f0_vmax=f0_vmax)
        if not filterbank_dir:
            folder.cleanup()


@cli.command()
@click.argument("archives", nargs=-1, required=True)
@click.option("--outdir", required=True, type=str, help="Directory for the copies and products.")
@click.option(
    "--f0-vmax",
    default=None,
    type=float,
    help="Widen the per-day F0 search to the Doppler range of this velocity (km/s).",
)
def panels(archives, outdir, f0_vmax):
    """Copy ARCHIVES (.ar) to OUTDIR and recreate their FT archive, plot and panels."""
    os.makedirs(outdir, exist_ok=True)
    for ar in archives:
        copied = os.path.join(outdir, os.path.basename(ar))
        shutil.copy2(ar, copied)
        make_products(copied, outdir, f0_vmax=f0_vmax)


@cli.command("clean")
@click.argument("archives", nargs=-1, required=True)
@click.option("--outdir", required=True, type=str, help="Directory for the cleaned archives.")
@click.option("--cell-sigma", default=6.0, type=float, help="Threshold for noisy cells/channels.")
@click.option("--chan-sigma", default=4.0, type=float, help="Threshold for persistent structure.")
@click.option(
    "--f0-vmax",
    default=None,
    type=float,
    help="Widen the per-day F0 search to the Doppler range of this velocity (km/s).",
)
def clean_cmd(archives, outdir, cell_sigma, chan_sigma, f0_vmax):
    """Clean RFI in ARCHIVES (.ar), writing new archives, FT files and panels to OUTDIR."""
    os.makedirs(outdir, exist_ok=True)
    for ar in archives:
        cleaned = os.path.join(outdir, os.path.basename(ar))
        clean_archive(ar, cleaned, cell_sigma=cell_sigma, chan_sigma=chan_sigma)
        make_products(cleaned, outdir, f0_vmax=f0_vmax)


@cli.command()
@click.argument("archives", nargs=-1, required=True)
@click.option("--par", required=True, type=str, help="Ephemeris the archives were folded with.")
@click.option("--outdir", required=True, type=str, help="Directory for copies and results.")
@click.option(
    "--phase-accuracy",
    default=1.0 / 64,
    type=float,
    help="Required accuracy in pulse phase, which determines step sizes.",
)
@click.option("--coherent/--no-coherent", default=True, help="Run the coherent search.")
@click.option(
    "--semicoherent/--no-semicoherent", default=True, help="Run the semi-coherent search."
)
@click.option("--candidate-sigma", default=None, type=float, help="Sigma shown in plots.")
def search(archives, par, outdir, phase_accuracy, coherent, semicoherent, candidate_sigma):
    """Run the coherent and semi-coherent multi-day search on ARCHIVES (.ar files)."""
    from folding.utilities.archives import read_par
    from multiday_search.load_profiles import load_profiles
    from multiday_search.phase_aligned_search import ExploreGrid
    from multiday_search.semicoherent_foldpanels import SemicoherentFoldSearch

    # load_profiles writes reference par files next to the archives and may rewrite
    # their ephemeris with pam -m, so work on copies
    archive_dir = os.path.join(outdir, "archives")
    os.makedirs(archive_dir, exist_ok=True)
    local_archives = []
    for ar in sorted(archives):
        stem = os.path.splitext(ar)[0]
        for src in (ar, stem + ".FT", stem + ".searchpanels.npz"):
            if os.path.isfile(src):
                shutil.copy2(src, archive_dir)
        local_archives.append(os.path.join(archive_dir, os.path.basename(ar)))
    if len(local_archives) < 2:
        raise click.UsageError("At least 2 archives are needed for the multi-day search.")
    print(f"Searching {len(local_archives)} archives")

    par_vals = read_par(par)
    DM_incoherent = par_vals["DM"]
    F0_incoherent = par_vals["F0"]

    data = load_profiles(local_archives)
    data["candidate_sigma"] = candidate_sigma
    data["directory"] = outdir
    print(f"Loaded {len(data['profiles'])} profiles")

    if coherent:
        # Same search grid as confirm_cand
        dF0 = 1 / 86164.1  # 1 day alias
        f0_lims = (F0_incoherent - dF0, F0_incoherent + dF0)
        delta_f0max = 2 * dF0
        f1_max = 2e-12
        f1_lims = (-f1_max, f1_max)
        delta_f1max = 2 * f1_max

        T = data["Tmax_from_reference"]
        npbin = data["npbin"]
        M_f0 = int(np.max((npbin * phase_accuracy, 1)))
        f0_points = 2 * int(delta_f0max * T * npbin / M_f0)
        f1_points = 2 * int(np.max((0.5 * delta_f1max * T**2 * npbin / M_f0, 1)))

        print(f"Running coherent search with {f0_points} f0 bins, {f1_points} f1 bins")
        explore_grid = ExploreGrid(data, f0_lims, f1_lims, f0_points, f1_points)
        f0s, f1s, chi2_grid, optimal_parameters = explore_grid.output()
        np.savez(
            os.path.join(outdir, f"cand_{F0_incoherent}_{DM_incoherent}_explore_grid.npz"),
            f0s=f0s,
            f1s=f1s,
            chi2_grid=chi2_grid,
            SN=np.max(explore_grid.SNmax),
            nobs=len(data["profiles"]),
        )
        plot_name = explore_grid.plot(fullplot=False)
        explore_grid.plot(fullplot=True)
        print(
            f"Coherent search: S/N {np.max(explore_grid.SNmax):.2f}, "
            f"F0 {optimal_parameters[0]}, F1 {optimal_parameters[1]}, plot {plot_name}"
        )

        optimal_par_file = os.path.join(
            outdir, os.path.basename(par).replace(".par", "_optimal.par")
        )
        with open(par) as input, open(optimal_par_file, "w") as output:
            output.write("# Created: " + str(datetime.datetime.now()) + "\n")
            output.write("# F0 and F1 from CHAMPSS coherent search\n")
            for line in input:
                key = line.split()[0] if line.split() else ""
                if key == "F0":
                    output.write(f"F0 {optimal_parameters[0]} 1 \n")
                    output.write(f"F1 {-optimal_parameters[1]} 1 \n")
                elif key in ("RAJ", "DECJ"):
                    output.write(f"{line.strip()} 1 \n")
                elif key == "PEPOCH":
                    output.write(f"PEPOCH {data['PEPOCH']} 0 \n")
                else:
                    output.write(line)
        print(f"Wrote {optimal_par_file}")

    if semicoherent:
        npz_files = [os.path.splitext(ar)[0] + ".searchpanels.npz" for ar in local_archives]
        npz_files = [f for f in npz_files if os.path.exists(f)]
        if len(npz_files) < 2:
            print("Fewer than 2 .searchpanels.npz files found, skipping semi-coherent search")
        else:
            print(f"Running semi-coherent search on {len(npz_files)} panel files")
            sc_plot = SemicoherentFoldSearch(npz_files).plot(
                output_dir=outdir, orbital_search=True
            )
            print(f"Semi-coherent plot: {sc_plot}")


def day_harmonics(fname, dms, max_harm, fmin=None, fmax=None):
    """
    Per-subint Fourier harmonics of the channel-summed profile of one archive.

    Each channel is whitened by its robust noise, dedispersed for every trial DM by
    rotating its harmonics, and summed with the archive weights. With fmin/fmax only the
    channels in that frequency range (MHz) are used.

    Returns
    -------
    C : complex ndarray (ndm, nsub, nharm), harmonics 1..nharm of the summed profiles
    tau : ndarray (nsub,), subint times in s relative to the middle of the observation
    mjd : float, middle of the observation
    harms : ndarray (nharm,), harmonic numbers
    is_nyquist : bool ndarray (nharm,), harmonic is the Nyquist bin (real only)
    f0 : float, folding frequency
    """
    import psrchive

    arch = psrchive.Archive.load(fname)
    if arch.get_dedispersed():
        raise ValueError(f"{fname} is already dedispersed")
    data = arch.get_data().sum(1)  # (nsub, nchan, nbin)
    weights = arch.get_weights()
    nsub, nchan, nbin = data.shape
    integ0 = arch.get_Integration(0)
    freqs = np.array([integ0.get_centre_frequency(c) for c in range(nchan)])
    f0 = float(arch.get_ephemeris().get_value("F0"))
    mjds = np.array([arch.get_Integration(j).get_epoch().in_days() for j in range(nsub)])
    mjd_mid = 0.5 * (mjds.min() + mjds.max())
    tau = (mjds - mjd_mid) * 86400.0

    good = weights > 0
    if fmin is not None:
        good &= (freqs >= fmin)[None, :]
    if fmax is not None:
        good &= (freqs <= fmax)[None, :]
    data = data - data.mean(-1, keepdims=True)
    chan_sigma = np.zeros(nchan)
    for c in range(nchan):
        if good[:, c].any():
            vals = data[good[:, c], c].ravel()
            chan_sigma[c] = 1.4826 * np.median(np.abs(vals - np.median(vals)))
    usable = chan_sigma > 0
    scale = np.where(usable, 1.0 / np.maximum(chan_sigma, 1e-30), 0.0)
    data = data * (good * scale[None, :])[..., None]

    nharm = min(max_harm, nbin // 2)
    harms = np.arange(1, nharm + 1)
    is_nyquist = (nbin % 2 == 0) & (harms == nbin // 2)
    spec = np.fft.rfft(data, axis=-1)[..., 1 : nharm + 1]  # (nsub, nchan, nharm)

    DM_CONSTANT = 1.0 / 2.41e-4
    fref = freqs[usable].max()
    C = np.empty((len(dms), nsub, nharm), dtype=complex)
    for i, dm in enumerate(dms):
        delay = DM_CONSTANT * dm * (freqs**-2 - fref**-2)  # s, later at low frequency
        rot = np.exp(2j * np.pi * (delay * f0)[:, None] * harms[None, :])  # (nchan, nharm)
        C[i] = np.einsum("jck,ck->jk", spec, rot)
    return C, tau, mjd_mid, harms, is_nyquist, f0


def z2_vs_f0(C, tau, harms, is_nyquist, dfs, use_harms):
    """
    Z^2 of one day versus F0 offset, for each trial DM.

    The noise level of every harmonic is estimated from the median power over subints,
    so that under noise each normal harmonic contributes chi^2 with 2 dof and the
    Nyquist harmonic chi^2 with 1 dof.

    Returns Z2 (ndm, nf0) and the number of degrees of freedom.
    """
    sel = np.isin(harms, use_harms)
    active = np.abs(C).sum((0, 2)) > 0  # subints with data
    C = C[:, active][..., sel]
    tau = tau[active]
    harms_sel, nyq_sel = harms[sel], is_nyquist[sel]
    # median of chi^2_2 / 2 is ln 2, of chi^2_1 it is 0.4549 (times the mean)
    med_factor = np.where(nyq_sel, 0.4549, np.log(2))
    sigma2 = np.median(np.abs(C) ** 2, axis=1) / med_factor  # (ndm, nharm)

    # Drop subints with extreme harmonic power (e.g. RFI bursts that survived cleaning),
    # which would otherwise dominate the sum at every trial F0. A weak pulsar cannot
    # reach this level in a single subint.
    from scipy.stats import chi2

    dof_sub = int(np.sum(np.where(nyq_sel, 1, 2)))
    q = (np.where(nyq_sel, 1, 2) * np.abs(C) ** 2 / sigma2[:, None, :]).sum(-1)  # (ndm, nsub)
    keep = (q < chi2.isf(1e-4, dof_sub)).all(0)
    if not keep.all():
        C, tau = C[:, keep], tau[keep]
        sigma2 = np.median(np.abs(C) ** 2, axis=1) / med_factor
    nsub = C.shape[1]
    # A signal at F0 + df drifts to earlier phase (by df * tau turns), which multiplies
    # harmonic k by exp(+2 pi i k df tau); undo that so that dfs are true F0 offsets.
    phase = np.exp(-2j * np.pi * dfs[:, None, None] * tau[None, :, None] * harms_sel[None, None, :])
    A = np.einsum("djk,fjk->dfk", C, phase)  # (ndm, nf0, nharm)
    # The Nyquist coefficient is real per subint but becomes complex once the subints
    # are rotated; its power keeps mean 1 (counted as 1 dof), with an only approximately
    # chi^2 tail.
    power = np.abs(A) ** 2 / (nsub * sigma2[:, None, :])
    contrib = np.where(nyq_sel, power, 2 * power)
    dof = int(np.sum(np.where(nyq_sel, 1, 2)))
    return contrib.sum(-1), dof


def chi2_sigma(value, dof):
    """Gaussian-equivalent significance of a chi^2 value with dof degrees of freedom."""
    from scipy.stats import chi2, norm

    return float(norm.isf(max(chi2.sf(value, dof), 1e-300)))


@cli.command()
@click.argument("archives", nargs=-1, required=True)
@click.option("--outdir", required=True, type=str, help="Directory for results and plot.")
@click.option("--label", default="z2", type=str, help="Name used for output files.")
@click.option("--harmonics", default="1,2,3,4", type=str, help="Harmonics to sum, e.g. '1' or '1,2'.")
@click.option("--f0-range", default=0.06, type=float, help="Search +- this F0 offset (Hz).")
@click.option("--oversample", default=4, type=int, help="F0 steps per 1/T of a day.")
@click.option(
    "--dms",
    default=None,
    type=str,
    help="Comma separated trial DMs (default: the archive DM).",
)
@click.option("--orbital/--no-orbital", default=True, help="Run the orbital track search.")
@click.option(
    "--control-windows",
    default=0,
    type=int,
    help="Also search this many windows of the same width offset from the folding F0 on "
    "either side, as controls for the trial factor of the orbital search.",
)
@click.option("--band", default=None, type=str, help="Only use channels in 'fmin,fmax' (MHz).")
def z2search(archives, outdir, label, harmonics, f0_range, oversample, dms, orbital,
             control_windows, band):
    """Semi-coherent Z^2 search over ARCHIVES (.ar, folded with the same ephemeris)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import psrchive
    from multiday_search.orbital_search import run_orbital_search

    os.makedirs(outdir, exist_ok=True)
    archives = sorted(archives)
    use_harms = np.array([int(h) for h in harmonics.split(",")])
    if dms:
        dm_list = np.array([float(x) for x in dms.split(",")])
    else:
        dm_list = np.array([psrchive.Archive.load(archives[0]).get_dispersion_measure()])

    # F0 grid from the longest observation
    fmin, fmax = (float(x) for x in band.split(",")) if band else (None, None)
    days = [day_harmonics(f, dm_list, use_harms.max(), fmin, fmax) for f in archives]
    T = max(np.ptp(d[1]) for d in days)
    df = 1.0 / (T * oversample)
    # Window centres: the folding F0 and control windows on either side, spaced by
    # twice the window half-width so that the windows do not overlap
    centres = [0.0] + [sign * 2 * f0_range * i for i in range(1, control_windows + 1) for sign in (1, -1)]
    full_range = f0_range + 2 * f0_range * control_windows
    dfs = np.arange(-full_range, full_range + 0.5 * df, df)
    print(f"{len(archives)} days, {len(dfs)} F0 trials (step {df:.2e} Hz), {len(dm_list)} DMs, "
          f"harmonics {list(use_harms)}")

    Z2 = []
    dofs = []
    mjds = []
    for (C, tau, mjd, harms, nyq, f0), fname in zip(days, archives):
        z, dof = z2_vs_f0(C, tau, harms, nyq, dfs, use_harms)
        Z2.append(z)
        dofs.append(dof)
        mjds.append(mjd)
        print(f"  {os.path.basename(fname)}: MJD {mjd:.2f}, dof {dof}, "
              f"max Z2 {z.max():.1f} (mean {z.mean():.2f})")
    Z2 = np.array(Z2)  # (nday, ndm, nf0)
    mjds = np.array(mjds)
    f0 = days[0][5]
    dof_total = int(np.sum(dofs))

    result = dict(dfs=dfs, f0=f0, dms=dm_list, mjds=mjds, Z2=Z2, dofs=dofs,
                  archives=archives, harmonics=use_harms, centres=centres)
    n_indep = 2 * f0_range * T * len(dm_list)
    rows = []
    for c in centres:
        win = np.abs(dfs - c) <= f0_range + 0.5 * df
        total = Z2[:, :, win].sum(0)  # (ndm, nwin)
        idm, if0 = np.unravel_index(np.argmax(total), total.shape)
        row = dict(centre=c, const_z2=total[idm, if0],
                   const_sigma=chi2_sigma(total[idm, if0], dof_total),
                   const_f0=f0 + dfs[win][if0], const_dm=dm_list[idm])
        if orbital and len(archives) > 2:
            orbit = run_orbital_search(Z2[:, idm, win].T, dfs[win] - c, mjds)
            if orbit is not None:
                row.update(orbit_z2=orbit["best_sn"],
                           orbit_sigma=chi2_sigma(orbit["best_sn"], dof_total),
                           orbit_Porb=1 / orbit["F_bin_best"], orbit_A=orbit["A_best"],
                           orbit_dF0=orbit["F0_best"] + c)
        rows.append(row)
        if c == 0.0:
            total0, idm0 = total, idm
    result["windows"] = rows

    print(f"Constant-F0 sum: about {n_indep:.0f} independent trials per window; "
          f"orbital tracks: many more")
    print(f"{'window':>10} {'const sigma':>11} {'at F0 (Hz)':>12} {'DM':>7} {'orbit sigma':>11} "
          f"{'P_orb (d)':>9} {'A (mHz)':>8}")
    for r in rows:
        tag = "CANDIDATE" if r["centre"] == 0.0 else f"{r['centre'] * 1e3:+.0f} mHz"
        print(f"{tag:>10} {r['const_sigma']:11.2f} {r['const_f0']:12.5f} {r['const_dm']:7.2f} "
              f"{r.get('orbit_sigma', np.nan):11.2f} {r.get('orbit_Porb', np.nan):9.2f} "
              f"{r.get('orbit_A', np.nan) * 1e3:8.2f}")
    if len(rows) > 1 and "orbit_sigma" in rows[0]:
        others = np.array([r["orbit_sigma"] for r in rows[1:]])
        print(f"Candidate orbital sigma {rows[0]['orbit_sigma']:.2f} ranks "
              f"{1 + (others > rows[0]['orbit_sigma']).sum()} of {len(rows)} windows "
              f"(controls: median {np.median(others):.2f}, max {others.max():.2f})")
    zero = np.argmin(np.abs(dfs))
    print("At the folding F0: " + ", ".join(
        f"DM {d:.2f}: {chi2_sigma(Z2[:, i, zero].sum(), dof_total):.2f} sigma"
        for i, d in enumerate(dm_list)))

    np.savez(os.path.join(outdir, f"{label}_z2search.npz"), **result)

    # Plot the candidate window: constant-F0 sum in sigma and per-day Z2
    win = np.abs(dfs) <= f0_range + 0.5 * df
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    ax0.plot(dfs[win] * 1e3, [chi2_sigma(v, dof_total) for v in total0[idm0]], color="tab:blue")
    ax0.set_ylabel("constant-F0 sum (sigma)")
    ax0.set_title(f"{label}: Z2 over {len(archives)} days, F0 {f0:.4f} Hz, DM {dm_list[idm0]:.2f}, "
                  f"harmonics {','.join(map(str, use_harms))}")
    ax1.pcolormesh(dfs[win] * 1e3, np.arange(len(mjds)), Z2[:, idm0, win], shading="nearest")
    ax1.set_yticks(np.arange(len(mjds)))
    ax1.set_yticklabels([f"{m:.1f}" for m in mjds])
    ax1.set_ylabel("MJD")
    ax1.set_xlabel("F0 offset (mHz)")
    plot_name = os.path.join(outdir, f"{label}_z2search.png")
    fig.savefig(plot_name, dpi=120, bbox_inches="tight")
    print(f"Plot: {plot_name}")


if __name__ == "__main__":
    cli()
