"""The RFI pipelines on a compact layout (only some channels held) against the full layout."""

import warnings
from multiprocessing import shared_memory

import numpy as np
import pytest
from rfi_mitigation.pipeline import RFIGlobalPipeline, RFIPipeline

NCHAN, NTIME = 1024, 8 * 1024


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    spectra = rng.exponential(1, (NCHAN, NTIME)).astype(np.float32) * 50
    spectra[300:310] *= rng.normal(1, 0.8, (10, NTIME)).astype(np.float32) ** 2  # RFI-like
    spectra[100, 2048:3072] *= 40  # an impulsive burst
    mask = rng.random((NCHAN, NTIME)) < 0.05
    dead = np.zeros(NCHAN, bool)
    dead[rng.choice(NCHAN, 480, replace=False)] = True
    dead[:64] = True  # band edge, in the known-bad list too
    spectra[dead] = 0
    mask[dead] = True
    spectra[mask & (rng.random((NCHAN, NTIME)) < 0.5)] = 0
    return spectra, mask, np.flatnonzero(~dead)


class Shared:
    def __init__(self, spectra, mask):
        self.blocks = [shared_memory.SharedMemory(create=True, size=a.nbytes) for a in (spectra, mask)]
        self.spectra = np.ndarray(spectra.shape, np.float32, buffer=self.blocks[0].buf)
        self.mask = np.ndarray(mask.shape, bool, buffer=self.blocks[1].buf)
        self.spectra[:], self.mask[:] = spectra, mask

    def close(self):
        del self.spectra, self.mask
        for b in self.blocks:
            b.close()
            b.unlink()


def run(pipeline_cls, masks, spectra, mask, channels=None, slices=None, **more):
    shared = Shared(spectra, mask)
    try:
        pipeline = pipeline_cls(masks)
        names = (shared.blocks[0].name, shared.blocks[1].name, spectra.shape, "float32")
        extra = {} if channels is None else dict(channels=channels, nchan=NCHAN)
        extra.update(more)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            if slices is None:
                returned = pipeline.clean(*names, **extra)
                assert returned == np.count_nonzero(shared.mask)
            else:
                for s in slices:
                    pipeline.clean(*names, s, **extra)
        return shared.mask.copy(), shared.spectra.copy()
    finally:
        shared.close()


def check(full_result, compact_result, channels):
    full_mask, full_spectra = full_result
    held_mask, held_spectra = compact_result
    np.testing.assert_array_equal(held_mask, full_mask[channels])
    np.testing.assert_array_equal(held_spectra, full_spectra[channels])
    absent = np.setdiff1d(np.arange(NCHAN), channels)
    assert full_mask[absent].all() and not full_spectra[absent].any()


def test_chunk_pipeline_matches_full_layout():
    spectra, mask, channels = make_data()
    masks = {"badchan": True, "sk": True}
    slices = [slice(i, i + 1024) for i in range(0, NTIME, 1024)]
    full = run(RFIPipeline, masks, spectra, mask, slices=slices)
    compact = run(RFIPipeline, masks, spectra[channels], mask[channels], channels, slices)
    assert (full[0] != mask).any()  # the cleaners did flag something
    check(full, compact, channels)


def test_global_pipeline_matches_full_layout():
    spectra, mask, channels = make_data(1)
    spectra[700] *= 3  # a channel with too much variance
    full = run(RFIGlobalPipeline, {"stddev": True}, spectra, mask)
    compact = run(RFIGlobalPipeline, {"stddev": True}, spectra[channels], mask[channels], channels)
    assert full[0][700].all()
    check(full, compact, channels)


@pytest.mark.parametrize("scales", [[1024], [512, 1024]])
def test_sk_flags_match_the_full_mask_flow(scales):
    from rfi_mitigation.cleaners.cleaners import SpectralKurtosisCleaner
    from rfi_mitigation.utilities.cleaner_utils import combine_cleaner_masks

    spectra, mask, channels = make_data(2)
    spectra, mask = spectra[:, :1024], mask[:, :1024]
    kwargs = dict(rfi_threshold_sigma=25, plot_diagnostics=False)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        # As the pipeline did: full arrays, the full-size cleaner mask, a stacked OR
        full = SpectralKurtosisCleaner(spectra.shape, scales, **kwargs)
        full.clean(spectra, mask)
        expected = combine_cleaner_masks(np.array([mask, full.get_mask()]))[0]
        assert (expected != mask).any()
        applied = mask.copy()
        full.apply_to(applied)
        np.testing.assert_array_equal(applied, expected)
        # Held channels only, set in place
        compact = SpectralKurtosisCleaner(spectra.shape, scales, **kwargs)
        held_spectra, held_mask = spectra[channels], mask[channels]
        compact.clean(held_spectra, held_mask, channels=channels)
        compact.apply_to(held_mask, channels=channels)
    np.testing.assert_array_equal(held_mask, expected[channels])
    np.testing.assert_array_equal(compact.get_mask(), full.get_mask())


def test_expanded_chunks_give_the_same_mask_as_held_rows(monkeypatch):
    spectra, mask, channels = make_data(3)
    masks = {"badchan": True, "sk": True}
    slices = [slice(i, i + 1024) for i in range(0, NTIME, 1024)]
    assert not RFIPipeline(masks).needs_whole_arrays()
    direct = run(RFIPipeline, masks, spectra[channels], mask[channels], channels, slices)
    # The path taken when an enabled cleaner needs arrays with every channel in them
    monkeypatch.setattr(RFIPipeline, "needs_whole_arrays", lambda self: True)
    expanded = run(RFIPipeline, masks, spectra[channels], mask[channels], channels, slices)
    assert (direct[0] != mask[channels]).any()
    np.testing.assert_array_equal(direct[0], expanded[0])
    np.testing.assert_array_equal(direct[1], expanded[1])


@pytest.mark.parametrize("compact", [False, True])
def test_global_pipeline_count_with_a_known_initial_count(compact):
    spectra, mask, channels = make_data(1)
    spectra[700] *= 3
    if compact:
        spectra, mask, extra = spectra[channels], mask[channels], {"channels": channels}
    else:
        extra = {}
    # run() checks the returned count against the mask
    run(RFIGlobalPipeline, {"stddev": True}, spectra, mask,
        initial_masked=int(np.count_nonzero(mask)), **extra)
