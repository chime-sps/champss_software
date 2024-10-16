import datetime
import sys

import click
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
from rfi_mitigation.cleaners.periodic import StaticPeriodicFilter
from scipy.signal import find_peaks
from sps_common import constants
from sps_databases import db_utils, models
from tqdm.contrib.concurrent import process_map


def create_report(
    start_date,
    end_date,
    freqs,
    mask_normed,
    mask_copy,
    new_birdies,
    peaks,
    peak_properties,
):
    date_string = f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}'
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"birdie_report_{date_string}.pdf")
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    axs[0, 0].plot(freqs, mask_normed)
    axs[0, 1].hist(mask_normed)
    axs[1, 0].plot(freqs, mask_copy)
    axs[1, 1].hist(mask_copy)
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[0, 1].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Mask Fraction")
    axs[1, 0].set_ylabel("Mask Fraction")
    axs[0, 0].set_title("Without static birdies filter.")
    axs[1, 0].set_title("With static birdies filter.")

    plt.tight_layout()

    pdf.savefig(fig)
    ranges = [[0, 110], [90, 210], [190, 310], [290, 410], [390, 510]]
    for single_range in ranges:
        fig, axs = plt.subplots(2, 1, figsize=(15, 4))
        axs[0].plot(freqs, mask_normed)
        axs[0].grid()
        axs[0].set_xlim(single_range[0], single_range[1])
        axs[1].plot(freqs, mask_copy)
        axs[1].grid()
        axs[1].set_xlim(single_range[0], single_range[1])
        axs[0].set_xlabel("Frequency (Hz)")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Mask Fraction")
        axs[1].set_ylabel("Mask Fraction")
        axs[0].set_title("Without static birdies filter.")
        axs[1].set_title("With static birdies filter.")
        plt.tight_layout()
        pdf.savefig(fig)

    pdf.close()

    np.savez(
        f"arrays_{date_string}",
        mask_raw=mask_normed,
        mask_filtered=mask_copy,
        peaks=peaks,
        peak_properties=peak_properties,
    )
    with open(f"birdie_recommendations_{date_string}.yml", "w") as outfile:
        yaml.dump(new_birdies, outfile, default_flow_style=False)


def add_to_overall_mask(mask, obs):
    # Needed to load birdies from disk, unfortunately quite slow
    # Could use multiprocessing for faster loading
    obs_obj = models.Observation.from_db(obs)
    if obs["birdies_position"] is not None:
        mask[obs["birdies_position"]] += 1
    return mask


def return_birdies(obs):
    # Needed to load birdies from disk, unfortunately quite slow
    # Could use multiprocessing for faster loading
    obs_obj = models.Observation.from_db(obs)
    return obs["birdies_position"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    default="sps-processing",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--end-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    default=datetime.datetime.now(),
    help="Last date of obsvervations to test.",
)
@click.option(
    "--ndays",
    default=10,
    type=int,
    help="Number of says to process.",
)
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help="Mask fraction threshold for the birdie recommendations.",
)
def main(db_port, db_name, db_host, end_date, ndays, threshold):
    freqs = np.fft.rfftfreq(2**20, d=constants.TSAMP)[:-1]
    start_date = end_date - datetime.timedelta(days=ndays)
    db = db_utils.connect(name=db_name, host=db_host, port=db_port)
    used_obs = list(
        db.observations.find({"datetime": {"$gt": start_date, "$lt": end_date}})
    )

    if len(used_obs):
        print(f"{len(used_obs)} observations found.")
    else:
        print("No observations found")
        sys.exit()
    mask = np.zeros(freqs.shape)

    use_multi = True
    if use_multi:
        all_birdies = process_map(return_birdies, used_obs, max_workers=8, chunksize=1)
        for single_birdies in all_birdies:
            if single_birdies is not None:
                mask[single_birdies] += 1
    else:
        for index, obs in tqdm.tqdm(enumerate(used_obs)):
            mask = add_to_overall_mask(mask, obs)

    mask_normed = mask / mask.max()

    rfi_filter = StaticPeriodicFilter()

    filtered_indices = rfi_filter.apply_static_mask(freqs)
    mask_copy = mask_normed.copy()
    mask_copy[filtered_indices] = 0

    peaks, peak_properties = find_peaks(mask_copy, height=threshold)

    new_birdies = {}
    for index in range(len(peaks)):
        peak = peaks[index]
        freq = freqs[peak]
        height = peak_properties["peak_heights"][index]
        name = f"birdie{index}"
        birdie = {
            "frequency": float(freq),
            "width": 0.01,
            "int_nharm": 1,
            "frac_nharm": 0,
            "note": (
                f"Derived from dynamic birdies. Filtered in {(height*100):.1f}% of"
                f' observations between {start_date.strftime("%Y%m%d")} and'
                f' {end_date.strftime("%Y%m%d")}.'
            ),
        }
        new_birdies[name] = birdie

    create_report(
        start_date,
        end_date,
        freqs,
        mask_normed,
        mask_copy,
        new_birdies,
        peaks,
        peak_properties,
    )


if __name__ == "__main__":
    main()
