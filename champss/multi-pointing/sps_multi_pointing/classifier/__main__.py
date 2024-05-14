import click
import glob
import os

from .trainer import SvmTrainer, MlpTrainer
from sps_common.interfaces import MultiPointingCandidate


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("candidates_path", type=click.Path(exists=True, file_okay=False))
@click.option("--trainer", type=click.Choice(["SVM", "MLP"], case_sensitive=False))
def train_cands(candidates_path, trainer):
    """
    Script to train a classifier from a set of MultiPointingCandidates from a given directory.
    The script will produce a classifier as a .pickle file and a .txt file showing the metrics on
    the quality of the classifier produced.

    Currently two classification algorithms are being used for training : Support Vector Machine (SVM)
    and Multilayer Perceptron (MLP).
    """
    candidate_files = glob.glob(
        os.path.join(candidates_path, "Multi_Pointing_Groups*.npz")
    )
    mpc_list = []
    for cand in candidate_files:
        mpc_list.append(MultiPointingCandidate.read(cand))
    if trainer == "SVM":
        training_algorithm = SvmTrainer()
    elif trainer == "MLP":
        training_algorithm = MlpTrainer()
    training_algorithm.train(
        mpc_list,
        compute_metrics=True,
        save_model=True,
        filename=str(trainer) + "_simulated_classifier.pickle",
    )
