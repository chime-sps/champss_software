import glob
import pytest
from sps_multi_pointing import grouper, classifier, known_source_sifter
from sps_common.interfaces.single_pointing import SinglePointingCandidateCollection
from sps_common.interfaces.multi_pointing import (
    CandidateClassification,
    CandidateClassificationLabel,
    KnownSourceLabel,
)


@pytest.fixture(scope="module")
def single_pointing_candidates():
    files = glob.glob("*_sim_ps_candidates.npz")
    assert files
    sp_cands = []
    for file in files:
        spcc = SinglePointingCandidateCollection.read(file)
        sp_cands.extend(spcc.candidates)
    return sp_cands


@pytest.fixture(scope="module")
def multi_pointing_candidates(single_pointing_candidates):
    sp_grouper = grouper.SinglePointingCandidateGrouper()
    assert single_pointing_candidates
    return sp_grouper.group(single_pointing_candidates)


def test_grouper(simulator, multi_pointing_candidates):
    assert multi_pointing_candidates


def test_classifier(simulator, multi_pointing_candidates):
    cand_classifier = classifier.CandidateClassifier()

    for cand in multi_pointing_candidates:
        cand_classifier.classify(cand)
        # After classifying, every candidate should have a classification label
        assert cand.classification

        # Since we're using only the DummyClassifier right now, we can also
        # check its expected output, but long term this is not going to work
        assert cand.classification.label == CandidateClassificationLabel.Astro
        assert cand.classification.grade == 1


def test_known_source_sifter(simulator, multi_pointing_candidates):
    kss = known_source_sifter.known_source_sifter.KnownSourceSifter(threshold=0.997)

    for cand in multi_pointing_candidates:
        kss.classify(cand, pos_filter=True)

        # After classifying, every candidate should have a source label,
        # whether KNOWN or UNKNOWN
        assert cand.known_source
