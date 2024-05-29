import sys

import click
import numpy as np
from sps_common.interfaces import MultiPointingCandidate
from sps_multi_pointing.classifier.trainer import MlpTrainer, SvmTrainer
from sps_multi_pointing.simulator import PointingGrid
from sps_multi_pointing.simulator.utils import relabel_simulated_candidates


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cols",
    default=90,
    type=click.types.IntRange(4, 256),
    help="Number of columns of simulated pointings",
)
@click.option(
    "--rows",
    default=16,
    type=click.types.IntRange(1, 256),
    help="Number of rows of simulated pointings",
)
@click.option(
    "--beam-row",
    default=59,
    type=click.types.IntRange(0, 223),
    help="Beam row at the centre of the simulated area",
)
@click.option(
    "--seed",
    type=click.IntRange(0, 2**32 - 1),
    required=False,
    help="Seed for replicable random number generation (default: [123456])",
)
@click.option(
    "--random-seed",
    "-r",
    is_flag=True,
    default=False,
    help="Use a random starting value of the random number generator",
)
def main(seed, random_seed, rows, cols, beam_row):
    """
    Simulate the output of single-pointing pipeline for a rectangular patch of sky of
    size ROWS x COLS, centered on BEAM_ROW.

    `SinglePointingCandidateCollection`s are saved as RA_Dec_sim_ps_candidates.npz,
    with each pointing also having a companion file to record the actual injected
    pulsars and named RA_Dec_sim_pulsars.npy. Some of the injected pulsars may be
    drawn from a small database of known pulsar sources and will be saved into file
    "injected_known_sources.npz".
    """
    if seed is not None and random_seed:
        print("Cannot both set the RNG seed and request a random seed")
        sys.exit(1)

    if seed is None:
        seed = 123456

    if not random_seed:
        np.random.seed(seed)

    p = PointingGrid(num_rows=rows, num_cols=cols, beam_row=beam_row)
    p.create()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("candidates_path", type=click.Path(exists=True, file_okay=False))
@click.argument("sim_pulsars_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--freq-diff",
    type=float,
    default=3.5 * 9.70127682e-04,
    help="The frequency difference range where a source is matched to a pulsar",
)
@click.option(
    "--dm-diff",
    type=float,
    default=1.0,
    help="The DM difference range where a source is matched to a pulsar",
)
@click.option(
    "--delete/--no-delete",
    is_flag=True,
    default=False,
    help="Whether to delete the original labelled candidates",
)
def relabel(candidates_path, sim_pulsars_path, freq_diff, dm_diff, delete):
    """
    Relabel simulated candidates based on whether they are matched to the injected
    pulsars. The output MultiPointingCandidate .npz files will be written with extra
    prefix of 'relabelled'.

    The inputs should be the directories where the MultiPointingCandidate .npz files and
    the simulated pulsars' .npy files are stored.

    There are options to change the default frequency and DM difference to match the
    candidates to a pulsar from the default of 3.5 * 9.70127682e-04 Hz and 1.0 pc/cc.

    There is also an option to delete the original .npz files after writing out the new
    MultiPointingCandidate .npz files.
    """
    relabel_simulated_candidates(
        candidates_path, sim_pulsars_path, freq_diff, dm_diff, delete
    )


if __name__ == "__main__":
    main()
