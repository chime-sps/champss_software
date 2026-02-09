from sps_databases import db_utils, db_api
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os
from pprint import pformat


def create_report_pdf(date, db_host, db_port, db_name, basepath):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    daily_run = db_api.get_daily_run(date)
    date_string = date.strftime("%Y/%m/%d")
    all_proc = list(db.processes.find({"date": date_string}))
    all_obs = list(
        db.observations.find(
            {"datetime": {"$gt": date, "$lt": date + datetime.timedelta(days=1)}}
        )
    )
    all_obs_pointings = list(
        [db.pointings.find_one({"_id": obs["pointing_id"]}) for obs in all_obs]
    )
    all_obs_ra = [point["ra"] for point in all_obs_pointings]
    all_obs_dec = [point["dec"] for point in all_obs_pointings]
    plotted_quantities = [
        "status",
        "status",
        "mask_fraction",
        "num_detections",
        "num_clusters",
    ]
    quantity_source = [
        "Process",
        "Observation",
        "Observation",
        "Observation",
        "Observation",
    ]

    plot_folder = basepath + "/quality_reports/" + date.strftime("%Y/%m/")
    file_name = plot_folder + f"quality_report_{date.strftime('%Y%m%d')}.pdf"
    os.makedirs(plot_folder, exist_ok=True)

    with PdfPages(file_name) as pdf:
        try:
            rows = len(plotted_quantities)
            fig1, axs = plt.subplots(rows, 2, figsize=(20, 5 * rows), rasterized=True)
            fig1.suptitle(f"Quality Report {date_string} Process/Observation Metrics")
            for index, quant, source in zip(
                np.arange(rows), plotted_quantities, quantity_source
            ):
                if source == "Process":
                    all_vals = np.array([proc.get(quant) for proc in all_proc]).astype(
                        float
                    )
                    ra = [proc.get("ra") for proc in all_proc]
                    dec = [proc.get("dec") for proc in all_proc]
                else:
                    all_vals = np.array([obs.get(quant) for obs in all_obs]).astype(
                        float
                    )
                    ra = all_obs_ra
                    dec = all_obs_dec
                ra_dec_index = 0
                status_hist_index = 1
                if quant == "status":
                    im1 = axs[index, ra_dec_index].scatter(
                        ra, dec, c=all_vals, alpha=1, vmin=0, vmax=5, cmap=plt.cm.jet
                    )
                    hist = axs[index, status_hist_index].hist(
                        all_vals, bins=np.arange(7) - 0.5
                    )
                else:
                    im1 = axs[index, ra_dec_index].scatter(
                        ra, dec, c=all_vals, alpha=0.5
                    )
                    hist = axs[index, status_hist_index].hist(all_vals, bins=20)
                plt.colorbar(im1, ax=axs[index, ra_dec_index])
                axs[index, ra_dec_index].set_xlabel("Ra")
                axs[index, ra_dec_index].set_ylabel("Dec")
                axs[index, status_hist_index].bar_label(hist[2])
                axs[index, status_hist_index].set_yscale("log")
                axs[index, status_hist_index].set_xlabel(f"{source}:{quant}")
                axs[index, ra_dec_index].set_title(f"{source}: {quant}")
                axs[index, status_hist_index].set_title(f"Total: {len(all_vals)}")
                if quant == "status":
                    axs[index, status_hist_index].set_title(f"Total: {len(all_vals)}")
                else:
                    axs[index, status_hist_index].set_title(
                        f"Total: {len(all_vals)}, Median: {np.nanmedian(all_vals):.2f}, Mean: {np.nanmean(all_vals):.2f}"
                    )
            plt.tight_layout()

            pdf.savefig(fig1)
            plt.close(fig1)
        except:
            pass
        try:
            df = pd.read_csv(
                daily_run.classification_result["output_file"], index_col=0
            )
            fig2, axs = plt.subplots(2, 1, figsize=(12, 12), rasterized=True)
            fig2.suptitle(f"Quality Report {date_string} Multi-Pointing Metrics")
            col1 = "best_ra"
            col2 = "best_dec"
            z = "sigma"
            axs[0].scatter(
                df[col1], df[col2], c=np.log10(df[z]), alpha=0.05, marker="."
            )
            axs[0].set_xlabel(col1)
            axs[0].set_ylabel(col2)
            axs[0].set_xlim(0, 360)
            axs[0].set_ylim(-12, 90)

            col1 = "mean_freq"
            col2 = "mean_dm"
            z = "sigma"
            axs[1].scatter(
                df[col1], df[col2], c=np.log10(df[z]), alpha=0.05, marker="."
            )
            axs[1].set_xlabel(col1)
            axs[1].set_ylabel(col2)
            axs[1].set_xscale("log")
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)
        except:
            pass

        try:
            fig3 = plt.figure(figsize=(15, 8))
            try:
                # Some daily run entries include a very long list
                daily_run.pipeline_result.pop("rfi_processeses")
            except:
                pass
            fig3.text(
                0.05,
                0.95,
                pformat(daily_run.__dict__, indent=2, width=80),
                ha="left",
                va="top",
                family="monospace",
                fontsize=10,
            )
            plt.axis("off")
            pdf.savefig(fig3)
            plt.close(fig3)
        except:
            pass
    return file_name
