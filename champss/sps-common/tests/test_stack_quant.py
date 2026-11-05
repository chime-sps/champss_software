import h5py
import numpy as np
import pytest
from sps_common.stack_quant import (
    QuantizedStack,
    quant_bias,
    read_quantized,
    round_power,
    write_quantized,
)

STEP = 0.25


@pytest.fixture
def h5(tmp_path):
    with h5py.File(tmp_path / "stack.h5", "w") as f:
        yield f


def noise_stack(days, shape=(5, 1000), seed=1):
    rng = np.random.default_rng(seed)
    spectra = sum(round_power(rng.exponential(1, shape), STEP) for _ in range(days))
    return spectra.astype(np.float32)


def test_bias_matches_rounded_noise():
    x = np.random.default_rng(0).exponential(1, 5000000)
    assert round_power(x, STEP).mean() == pytest.approx(quant_bias(STEP), abs=2e-3)


@pytest.mark.parametrize("days", [1, 30, 64, 365])
def test_round_trip_is_exact_for_sums_of_rounded_days(h5, days):
    spectra = noise_stack(days)
    write_quantized(h5, spectra, STEP, days)
    assert h5["power spectra"].dtype == np.uint16
    assert len(h5["overflow row"]) == 0
    np.testing.assert_array_equal(read_quantized(h5, raw=True), spectra)


def test_noise_stacks_compress_well_beyond_255(h5):
    days = 64
    spectra = noise_stack(days, shape=(4, 65536))
    write_quantized(h5, spectra, STEP, days)
    # The information content is ~7.5 bits per cell; plain uint16 would be 16
    assert h5["power spectra"].id.get_storage_size() < 1.1 * spectra.size


def test_values_beyond_16_bits_are_kept_exactly(h5):
    spectra = np.full((3, 10), 32 * STEP * 4, np.float32)
    spectra[1, 4], spectra[2, 7] = 1e6, 0.0
    write_quantized(h5, spectra, STEP, 32)
    assert h5["overflow row"].shape == (1,)
    np.testing.assert_array_equal(read_quantized(h5, raw=True), spectra)


def test_reading_part_of_a_stack_applies_only_its_overflow(h5):
    spectra = np.ones((4, 6), np.float32)
    spectra[0, 1], spectra[3, 2] = 2e5, 3e5
    write_quantized(h5, spectra, STEP, 1)
    stack = QuantizedStack(h5)
    np.testing.assert_array_equal(stack.rows(1, 3, raw=True), spectra[1:3])
    np.testing.assert_array_equal(stack.rows(2, 4, raw=True), spectra[2:])
    for i in range(4):
        np.testing.assert_array_equal(stack.rows(i, i + 1, raw=True)[0], spectra[i])


def test_decoding_corrects_the_bias(h5):
    spectra = np.full((1, 4), 4.0, np.float32)
    write_quantized(h5, spectra, STEP, 4)
    np.testing.assert_allclose(read_quantized(h5), 4.0 / quant_bias(STEP))


def test_masked_bins_stay_zero(h5):
    spectra = noise_stack(32, shape=(2, 50))
    spectra[:, ::7] = 0
    write_quantized(h5, spectra, STEP, 32)
    back = read_quantized(h5)
    assert not back[:, ::7].any()
    assert back[:, 1::7].all()


def test_block_boundaries_keep_overflow_in_the_right_rows(h5, monkeypatch):
    import sps_common.stack_quant as sq

    monkeypatch.setattr(sq, "READ_BLOCK_ROWS", 2)
    spectra = noise_stack(4, shape=(7, 20))
    spectra[[0, 2, 3, 6], [1, 5, 5, 19]] = [1e5, 2e5, 3e5, 4e5]
    write_quantized(h5, spectra, STEP, 4)
    np.testing.assert_array_equal(read_quantized(h5, raw=True), spectra)
    np.testing.assert_array_equal(read_quantized(h5, 1, 6, raw=True), spectra[1:6])
