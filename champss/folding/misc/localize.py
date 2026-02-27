import glob
from datetime import datetime

import chime_frb_constants
import click
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from beamformer.skybeam import SkyBeamFormer
from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import beammod as bm
from folding.utilities.archives import *
from lmfit import Model
from scipy.ndimage import uniform_filter


def beam_transit_model(t_transit, ra, dec, beamarg, spectrum=[], nchan=2048):
    """
    This function computes the frequency-averaged beam model I(t) for a transit at a
    given beam number, for a given RA, DEC, and source spectrum.

    Parameters
    ----------
    t_transit: array of astropy.time.Time objects

    ra: float
    Right ascension of the source in degrees

    dec: float
    Declination of the source in degrees

    beamarg: int
    Beam number

    spectrum: array
    1D I(f) spectrum of the source

    nchan: int
    Number of frequency channels
    """

    # Create a beam model, choose a beam number, and generate the frequencies as before
    freqs = chime_frb_constants.FREQ
    binf = len(freqs) // nchan
    freqs = freqs[::binf]

    # Setting the RA and Dec coordinates, choosing a time, and getting the corresponding
    # x and y coordinates
    I_array = []
    xs = []
    ys = []

    for i in range(len(t_transit)):
        j = i * 4 // len(t_transit)
        beam_no = [j * 1000 + beamarg]
        ti = t_transit[i]

        x, y = bm.get_position_from_equatorial(ra, dec, ti.datetime)
        pos = np.array([x, y])
        xs.append(x)
        ys.append(y)

        sensitivities = bm.get_sensitivity(beam_no, pos, freqs)
        sensitivities = np.squeeze(sensitivities)
        I_array.append(sensitivities)

    beamref = [1000 + beamarg]
    pos_beam = bm.get_beam_positions(beamref, freqs=[600])
    pos_beam = pos_beam.squeeze()
    I_control = bm.get_sensitivity(beamref, pos_beam, freqs)
    I_control = I_control.squeeze()

    I_array = np.array(I_array)
    I_array = I_array / I_control[np.newaxis, :]
    if len(spectrum) > 0:
        I_array = I_array * spectrum[np.newaxis, :] / np.mean(spectrum)
    Imodel = I_array.mean(-1)

    return Imodel


def beam_dec_model(t_transit, ra, dec, A=1, nchan=2048):
    SNs = np.zeros(len(beams))
    for i, beamarg in enumerate(beams):
        model_ts = beam_transit_model(
            t_transit, ra, dec, beamarg=beamarg, spectrum=spec
        )
        S = np.sum(model_ts)
        SNs[i] = S

    return A * SNs


@click.command()
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
)
@click.option("--ra", type=float, help="RA", required=True)
@click.option("--dec", type=float, help="DEC", required=True)
@click.option("--nday", type=int, default=21, help="Number of days to fold")
def main(ra, dec, date, nday):
    """
    Code to localize new pulsar using folded SPS data.

    Requisite is to have folded the data for the candidate in the max beam, as well as
    the beam above and below in declination.

    All archives must have the same ephemeris / parfile installed, there must be slices
    of .F files for each day, and a combined archive added.T for each beam which has
    units of (time(days), pol, freq, phase), (where pol=1 is a dummy axis)
    """
    year = date.year
    month = date.month
    day = date.day

    pst = PointingStrategist(create_db=False)
    ap = pst.get_single_pointing(ra, dec, datetime(year, month, day))
    beamarg = ap[0].max_beams[0]["beam"]

    ra_dir = round(ap[0].ra, 2)
    dec_dir = round(ap[0].dec, 2)

    # Load the data of the promary beam
    fpath = f"/data/chime/sps/archives/candidates/{ra_dir}_{dec_dir}"

    # Load the full array, get the spectrum
    fname_combfs = f"{fpath}/added.T"

    # plot_foldspec(fname_combfs)

    data, params = readpsrarch(fname_combfs)
    F, T = params["F"], params["T"]
    fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze())
    fs -= np.median(fs, axis=-1, keepdims=True)
    if F[1] - F[0] < 0:
        F = F[::-1]
        fs = fs[:, ::-1]

    dynspec = create_dynspec(fs)
    global spec
    spec = dynspec.mean(0)

    files = np.sort(glob.glob(f"{fpath}/cand_231*2023-1*.F"))
    for i, fn in enumerate(files):
        data, params = readpsrarch(fn)
        F, T = params["F"], params["T"]
        fs = data.squeeze()
        if i == 0:
            fs_comb = np.copy(fs)
        else:
            fs_comb += fs[: fs_comb.shape[0]]

    taxis = ((T - T[0]) * u.day).to(u.s).value

    # data cube will have shape [time, pol, freq, phase]
    # squeeze to remove dummpy pol axis, since it is length 1
    # Sum frequency axis
    data_timephase = np.copy(fs_comb)
    data_timephase = data_timephase - np.median(data_timephase, axis=-1, keepdims=True)

    # Step in ra
    dra_range = np.linspace(-1, 1, 100)
    I_rastep = np.zeros(len(dra_range))
    t_transit = Time(T, format="mjd")

    prof = data_timephase.mean(0)
    data_ts = (data_timephase * prof[np.newaxis, :]).mean(-1)
    data_ts = data_ts / np.max(data_ts)

    for i, dra in enumerate(dra_range):
        rai = ra + dra
        model_ts = beam_transit_model(
            t_transit, rai, dec, spectrum=spec, beamarg=beamarg
        )
        S = np.sum(data_ts * model_ts)
        I_rastep[i] = S

    ramax = ra + dra_range[np.argmax(I_rastep)]

    # Fit in Dec, search +- one beam
    global beams
    beams = []
    profs = []

    for ddec in [-0.34, 0, 0.34]:
        ap = pst.get_single_pointing(ra, dec + ddec, datetime(year, month, day))
        beami = ap[0].max_beams[0]["beam"]
        beams.append(beami)
        ra_dir = round(ap[0].ra, 2)
        dec_dir = round(ap[0].dec, 2)

        fname = f"/data/chime/sps/archives/candidates/{ra_dir}_{dec_dir}/added.T"

        data, params = readpsrarch(fname)
        F, T = params["F"], params["T"]
        fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze())

        fs -= np.median(fs, axis=-1, keepdims=True)
        if F[1] - F[0] < 0:
            F = F[::-1]
            fs = fs[:, ::-1]
        prof = fs[:nday].mean(0).mean(0)
        profs.append(prof)

    SNs = []
    SNmax, SNindex, ibin = get_SN(profs[1])

    for profile in profs:
        ngate = len(profile)
        prof_filtered = uniform_filter(profile, ibin)
        profsort = np.sort(prof_filtered)
        prof_N = profsort[: 3 * ngate // 4]
        std = np.std(prof_N)
        mean = np.mean(prof_N)
        SNprof = (prof_filtered - mean) / std
        SNs.append(SNprof[SNindex])

    # Fit in declination
    fmodel = Model(beam_dec_model)
    params = fmodel.make_params(ra=ra, dec=dec, A=1, nchan=len(spec))
    params["ra"].vary = False
    params["nchan"].vary = False
    result = fmodel.fit(SNs, params, t_transit=t_transit)

    decfit = result.params["dec"].value
    Afit = result.params["A"].value
    SNs_beam = beam_dec_model(t_transit, ra, dec=decfit, A=Afit)
    plt.figure(figsize=(10, 10))

    plt.suptitle(f"Best Fit RA = {ramax:.2f}, Dec = {decfit:.2f}", y=0.95, fontsize=16)

    plt.subplot(221)
    for i, prof in enumerate(profs):
        plt.plot(prof, label=beams[i], alpha=0.5)
    plt.legend()
    plt.xlim(0, len(prof))
    plt.xlabel("phase bins")
    plt.ylabel("S/N")

    plt.subplot(223)
    plt.imshow(data_timephase, aspect="auto", interpolation="nearest")
    plt.ylabel("T (subints)")
    plt.xlabel("phase bins")

    plt.subplot(222)
    plt.scatter(beams, SNs, alpha=0.5, label="data")
    plt.scatter(beams, SNs_beam, alpha=0.5, label="beam model")
    plt.legend()
    plt.xticks(beams)
    plt.xlabel("beam")
    plt.ylabel("S/N")

    plt.subplot(224)
    plt.plot(ra + dra_range, I_rastep)
    plt.axvline(ra, linestyle="dotted")
    plt.xlabel("RA (deg)")
    plt.ylabel("S (arb)")

    plt.savefig(f"PSRlocalization_{ra}_{dec}.png", bbox_inches="tight", dpi=300)

    print("RA", ramax, "pointing RA", ra)
    print("Dec", decfit, "pointing Dec", dec)

    coords = SkyCoord(ramax, decfit, unit="deg")
    ra_hms = coords.icrs.ra.hms
    dec_dms = coords.icrs.dec.dms
    print(ra_hms, dec_dms)
    print(coords.to_string("hmsdms", sep=":"))


if __name__ == "__main__":
    main()
