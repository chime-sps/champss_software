"""
Create summary PDF plots for multi-day candidate analysis.
"""

import logging
import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

log = logging.getLogger(__name__)


def create_summary_pdf(source, plot_name, output_dir):
    """
    Create a multi-page PDF summary combining all plots.

    Parameters
    ----------
    source : FollowUpSource
        The followup source object from database
    plot_name : str
        Path to the phase search plot
    output_dir : str
        Directory to save the summary PDF

    Returns
    -------
    pdf_path : str
        Path to the created PDF file
    """
    from sps_common.interfaces import MultiPointingCandidate

    plot_paths = []

    # Add original candidate plots from NPZ files
    if source.path_to_candidates:
        # Save current rcParams to restore later
        original_rcParams = plt.rcParams.copy()

        log.info(f"Generating candidate plots from {len(source.path_to_candidates)} NPZ file(s)...")

        for i, cand_path in enumerate(source.path_to_candidates):
            if os.path.exists(cand_path):
                try:
                    # Reset matplotlib font size to default before plotting candidate
                    plt.rcParams.update({"font.size": 10})

                    # Load and plot the candidate
                    mpc = MultiPointingCandidate.read(cand_path)
                    plot_path = mpc.plot_candidate(path=output_dir)
                    plot_paths.append(plot_path)
                    log.info(f"  Generated candidate plot: {plot_path}")
                except Exception as e:
                    log.warning(f"Could not plot candidate from {cand_path}: {e}")

        # Restore original rcParams
        plt.rcParams.update(original_rcParams)

    # Add phase search plot
    if plot_name and os.path.exists(plot_name):
        plot_paths.append(plot_name)

    # Add daily folded plots (sorted by date)
    if source.folding_history:
        folding_sorted = sorted(source.folding_history, key=lambda x: x["date"])
        for entry in folding_sorted:
            plot_path = entry.get("path_to_plot", "")
            if plot_path and os.path.exists(plot_path):
                plot_paths.append(plot_path)

    # Create PDF
    os.makedirs(output_dir, exist_ok=True)
    source_name = source.source_name.replace("/", "_").replace(" ", "_")
    pdf_path = os.path.join(output_dir, f"{source_name}_summary.pdf")

    log.info(f"Creating summary PDF with {len(plot_paths)} plots: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        for plot_path in plot_paths:
            try:
                img = mpimg.imread(plot_path)
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                log.error(f"Error adding plot {plot_path}: {e}")

    log.info(f"Summary PDF created: {pdf_path}")
    return pdf_path
