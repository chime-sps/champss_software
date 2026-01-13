import math

import numpy as np
import pandas as pd
from folding.utilities.database import add_candidate_to_fsdb
from sps_databases import db_utils
from pathlib import Path
import click


def get_indices_within_radius(lati, loni, latitude, longitude, rad_deg):
    indices = []
    for i in range(len(latitude)):
        lat = latitude[i]
        lon = longitude[i]
        dist_deg = haversine_distance(lat, lon, lati, loni)
        if dist_deg <= rad_deg:
            indices.append(i)
    return indices


def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the differences between the latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Calculate the haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    dist_rad = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_deg = dist_rad * 180 / np.pi

    return dist_deg


def Filter(
    cand_obs_date,
    min_sigma=7.0,
    min_dm=2.0,
    min_f0=1e-2,
    save_candidates=True,
    db_port=27017,
    db_host="sps-archiver1",
    db_name="sps",
    write_to_db=False,
    basepath="/data/chime/sps/sps_processing",
    foldpath="/data/chime/sps/archives",
    class_threshold=0,
):
    """
    Read a day's worth of multi-pointing candidates, retrieve a set of the most
    promising candidates to fold. Filters based on a set of DM, F0, sigma, and spatial
    tests.

    Args:
        cand_obs_date (datetime): The observation date.
        min_sigma (float, optional): The minimum sigma threshold for filtering candidates. Defaults to 7.0.
        min_dm (float, optional): The minimum DM threshold for filtering candidates. Defaults to 2.0 pc/cm^3.
        min_f0 (float, optional): The minimum F0 threshold for filtering candidates. Defaults to 0.01 Hz.
        save_candidates (bool, optional): Whether to save the filtered candidates to a npz file. Defaults to True.
        db_port (int, optional): The port number for the database connection
        db_host (str, optional): The host name for the database connection
        db_name (str, optional): The name of the database
        write_to_db (bool, optional): Whether to write the candidates to the FollowUpSources database. Defaults to False.

    Notes:
        - This function reads a CSV file containing candidate data for a specific observation date.
        - It applies several filters based on DM, F0, sigma, and spatial tests to select the most promising candidates.
        - The filtered candidates are saved to a npz file and/or written to a database if specified.
    """

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    date_str = cand_obs_date.strftime("%Y%m%d")
    fname = f"{basepath}/mp_runs/daily_{date_str}/all_mp_cands_predicted.csv"
    df = pd.read_csv(fname)
    if class_threshold == 0.0:
        SNthresh = min_sigma
        DMthresh = min_dm
        F0cut = min_f0

        i_SNcutabove = np.flatnonzero(df["sigma"] > SNthresh)
        i_DMcutabove = np.flatnonzero(df["mean_dm"] > DMthresh)

        # Filter on known_sources
        ra = df["best_ra"]
        dec = df["best_dec"]

        i_ks = np.flatnonzero(df["known_source_label"] > 0.5)
        dDMtol = 0.1
        i_ksclusters = []

        for j in range(len(i_ks)):
            rai = ra[i_ks[j]]
            deci = dec[i_ks[j]]

            i_nearks = get_indices_within_radius(rai, deci, ra, dec, rad_deg=1)
            dm_ks = df["known_source_dm"][i_ks[j]]
            f0_ks = 1 / df["known_source_p0"][i_ks[j]]
            source = df["known_source_name"][i_ks[j]]
            print(source, dm_ks, f0_ks)

            i_dmmatch = np.flatnonzero(np.abs(df["mean_dm"] - dm_ks) / dm_ks < dDMtol)
            i_ksmatch = np.intersect1d(i_nearks, i_dmmatch)
            i_ksclusters = np.concatenate((i_ksclusters, i_ksmatch))

        i_ksclusters = np.unique(np.array(i_ksclusters))

        # Filter on position spread.
        # Currently only filtering on weighted RA spread, since we have limited beams
        dra = df["std_ra"]

        # Arbitrary, but tested that only RFI and extremely bright known sources
        # (e.g. B0329+54, B2217+47) are above this threshold
        i_posspread = np.flatnonzero(dra > 5)
        i_posspread = np.intersect1d(i_posspread, i_SNcutabove)
        i_sndmposcut = np.intersect1d(i_posspread, i_DMcutabove)
        fsub = df["mean_freq"][i_sndmposcut].values
        dmsub = df["mean_dm"][i_sndmposcut].values

        # Uniform bins of .5%
        # perhaps better to do iteratively?
        bins = []
        bini = 10 ** (-2)
        bins.append(bini)
        binmax = 1e2
        stepsize = 0.01
        step = 1 + stepsize

        while bini < binmax:
            bini = bini * step
            bins.append(bini)

        hist_f0clusters = np.histogram(fsub, bins=bins)
        histbins = hist_f0clusters[1]
        bincounts = hist_f0clusters[0]

        i_f0cluster = np.flatnonzero(bincounts > 2)
        birdie_cands = []

        # Flag narrow freq bins with large DM spread of candidates
        for i in i_f0cluster:
            bclow = histbins[i]
            bchigh = histbins[i + 1]
            bcmid = (bchigh + bclow) / 2.0
            bcwidth = (bchigh - bclow) / 2.0

            isub = np.flatnonzero(np.abs(fsub - bcmid) <= bcwidth)
            dDM = np.std(dmsub[isub]) / np.mean(dmsub[isub])
            if dDM > dDMtol:
                birdie = (bcmid, bcwidth)
                birdie_cands.append(birdie)

        # Zap these bins for all candidates
        birdie_cands = np.array(birdie_cands)
        i_birdies = []
        if birdie_cands.shape[0] > 0:
            for i in range(birdie_cands.shape[0]):
                birdie = birdie_cands[i]
                f0 = birdie[0]
                fwidth = birdie[1]
                i_birdie = np.flatnonzero(np.abs(df["mean_freq"] - f0) < fwidth)
                i_birdies = np.concatenate((i_birdies, i_birdie))
            i_birdies = i_birdies.astype("int")

        # Combine all filters, indeces of candidates to remove
        i_SNcutbelow = np.flatnonzero(df["sigma"] < SNthresh)
        i_DMcutbelow = np.flatnonzero(df["mean_dm"] < DMthresh)
        i_F0cut = np.flatnonzero(df["mean_freq"] < F0cut)

        i_bad = np.concatenate(
            [
                i_SNcutbelow,
                i_DMcutbelow,
                i_F0cut,
                i_ksclusters,
                i_ks,
                i_birdies,
                i_posspread,
            ]
        )

        i_bad = np.unique(i_bad).astype("int")
        i_good = np.arange(len(df["mean_freq"].values))
        i_good = np.delete(i_good, i_bad)

        # New dataframe for Candidates that pass all filters
        df1 = df.iloc[i_good]

        # Restrict to only one candidate per pointing
        cand_ra = df1["best_ra"].values
        cand_dec = df1["best_dec"].values
        cand_sigma = df1["sigma"].values
        i_matched = []
        for i in range(len(cand_ra)):
            rai = cand_ra[i]
            deci = cand_dec[i]
            i_match = np.flatnonzero((cand_ra == rai) & (cand_dec == deci))
            if len(i_match) > 1:
                sigma_sub = cand_sigma[i_match]
                isub_brightest = np.argmax(sigma_sub)
                i_matched.append(i_match[isub_brightest])
            else:
                i_matched.append(i_match[0])
        i_matched = np.unique(i_matched)

        # Final dataframe
        print(
            f"{len(i_matched)} candidates after filtering, from {len(df)} candidates before"
            " filtering."
        )
        df_filtered = df1.iloc[i_matched]
    else:
        df_filtered = df[df["prediction"] > class_threshold]
        df_filtered = df_filtered[df_filtered["sigma"] > min_sigma]
    ras = df_filtered["best_ra"].values
    decs = df_filtered["best_dec"].values
    f0s = df_filtered["mean_freq"].values
    dms = df_filtered["mean_dm"].values
    sigmas = df_filtered["sigma"].values
    known = df_filtered["known_source_name"].values
    cand_paths = df_filtered["file_name"].values

    if write_to_db:
        print("Writing candidates to FollowUpSource database...")
        for i in range(len(decs)):
            add_candidate_to_fsdb(
                date_str,
                ras[i],
                decs[i],
                f0s[i],
                dms[i],
                sigmas[i],
                cand_paths[i],
                "sd_candidate",
            )

    npz_filename = f"{foldpath}/candidates/filtered_cands/cands_{date_str}_filtered"
    Path(f"{foldpath}/candidates/filtered_cands/").mkdir(parents=True, exist_ok=True)
    if save_candidates:
        print("Saving filtered data...")
        np.savez(
            npz_filename,
            sigmas=sigmas,
            dms=dms,
            ras=ras,
            decs=decs,
            f0s=f0s,
            known=known,
        )


def filter_df_with_freq_histogram(
    df, binmax=1e2, bini=10 ** (-2), stepsize=0.01, dDMtol=0.1
):
    fsub = df["mean_freq"].values
    dmsub = df["mean_dm"].values
    dDMtol = 0.1

    bins = []
    bins.append(bini)
    step = 1 + stepsize

    while bini < binmax:
        bini = bini * step
        bins.append(bini)

    hist_f0clusters = np.histogram(fsub, bins=bins)
    histbins = hist_f0clusters[1]
    bincounts = hist_f0clusters[0]

    i_f0cluster = np.flatnonzero(bincounts > 2)
    birdie_cands = []

    # Flag narrow freq bins with large DM spread of candidates
    for i in i_f0cluster:
        bclow = histbins[i]
        bchigh = histbins[i + 1]
        bcmid = (bchigh + bclow) / 2.0
        bcwidth = (bchigh - bclow) / 2.0

        isub = np.flatnonzero(np.abs(fsub - bcmid) <= bcwidth)
        dDM = np.std(dmsub[isub]) / np.mean(dmsub[isub])
        if dDM > dDMtol:
            birdie = (bcmid, bcwidth)
            birdie_cands.append(birdie)

    birdie_cands = np.array(birdie_cands)
    i_birdies = []
    if birdie_cands.shape[0] > 0:
        for i in range(birdie_cands.shape[0]):
            birdie = birdie_cands[i]
            f0 = birdie[0]
            fwidth = birdie[1]
            i_birdie = np.flatnonzero(np.abs(df["mean_freq"] - f0) < fwidth)
            i_birdies = np.concatenate((i_birdies, i_birdie))
        i_birdies = i_birdies.astype("int")
    i_good = np.arange(len(df["mean_freq"].values))
    i_good = np.delete(i_good, i_birdies)
    df_output = df.iloc[i_good]
    return df_output


def filter_mp_df(
    df,
    sigma_min=6,
    class_min=0.9,
    only_use_strongest_in_cluster=True,
    filter_with_histogram=True,
):
    df_filtered = df[df["known_source_label"] == 0]
    if filter_with_histogram:
        df_filtered = filter_df_with_freq_histogram(df_filtered)
    df_filtered = df_filtered[df_filtered["prediction"] > class_min]
    df_filtered = df_filtered[df_filtered["sigma"] > sigma_min]
    if only_use_strongest_in_cluster:
        df_filtered = df_filtered[df_filtered["strongest_in_cluster"] == 1]
    df_filtered["folded"] = True
    return df_filtered


def write_df_to_fsdb(df, date):
    for index, row in df.iterrows():
        fs = add_candidate_to_fsdb(
            date.strftime("%Y%m%d"),
            row["best_ra"],
            row["best_dec"],
            row["mean_freq"],
            row["mean_dm"],
            row["sigma"],
            row["file_name"],
            "sd_candidate",
        )
        df.loc[index, "fs_id"] = str(fs["_id"])
    return df


@click.command()
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
)
@click.option(
    "--min_sigma",
    type=float,
    default=7.0,
    help=(
        "Minimum sigma to filter candidates. Default = 7, apparent inflection point in"
        " Ncands"
    ),
)
@click.option(
    "--min_dm",
    type=float,
    default=2.0,
    help="Minimum DM to filter candidates. Default = 2",
)
@click.option(
    "--min_f0",
    type=float,
    default=0.01,
    help="Minimum F0 to filter candidates. Default = 0.01",
)
@click.option(
    "--save_candidates",
    is_flag=True,
    help="Save filtered candidates to npz file. Default = True",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--write-to-db",
    is_flag=True,
    help="Update FollowUpSources Database with new candidates. Default = False",
)
def main(
    date,
    min_sigma,
    min_dm,
    min_f0,
    save_candidates,
    db_port,
    db_host,
    db_name,
    write_to_db,
):
    Filter(
        date,
        min_sigma,
        min_dm,
        min_f0,
        save_candidates,
        db_port,
        db_host,
        db_name,
        write_to_db,
    )


if __name__ == "__main__":
    main()
