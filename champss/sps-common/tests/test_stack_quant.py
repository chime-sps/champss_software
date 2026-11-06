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


# The numpy versions that the compiled loops replaced, which they must match exactly


def reference_codes(spectra, step, num_days):
    """The stored codes and overflow list, as write_quantized made them with numpy."""
    offset = int(round(num_days / step))
    inv_step = np.float32(1 / step)
    stored, over_rows, over_bins, over_codes = [], [], [], []
    for i, spectrum in enumerate(spectra):
        k = np.rint(np.asarray(spectrum, dtype=np.float32) * inv_step).astype(np.int32)
        x = k - np.int32(offset)
        z = (x << 1) ^ (x >> 31)
        over = np.flatnonzero(z > 65535)
        over_rows += [i] * len(over)
        over_bins += over.tolist()
        over_codes += k[over].tolist()
        z[over] = 0
        stored.append(z.astype(np.uint16))
    return np.array(stored), over_rows, over_bins, over_codes


def reference_rows(stored, over, offset, scale):
    z = stored.astype(np.int32)
    codes = ((z >> 1) ^ -(z & 1)) + np.int32(offset)
    codes[over[0], over[1]] = over[2]
    out = np.empty(stored.shape, np.float32)
    np.multiply(codes, scale, out=out, casting="unsafe")
    return out


def awkward_stack(step, days, seed):
    """Noise plus negative values, overflow-sized values and masked zeros."""
    rng = np.random.default_rng(seed)
    spectra = (rng.exponential(days, (9, 3001)) * 1.37).astype(np.float32)
    spectra[rng.random(spectra.shape) < 0.05] = 0
    spectra[1, 5:9] = [-3.3, -1e4, 2e5, 7e4]
    spectra[[0, 4, 8], [0, 1500, 3000]] = [5e5, 9e4, 1.25e6]
    return spectra


@pytest.mark.parametrize("step", [0.25, 0.3, 1.0])
@pytest.mark.parametrize("days", [1, 7, 32])
def test_written_codes_match_the_numpy_encoding(h5, step, days):
    spectra = awkward_stack(step, days, seed=days)
    write_quantized(h5, spectra, step, days)
    stored, rows, bins, codes = reference_codes(spectra, step, days)
    np.testing.assert_array_equal(h5["power spectra"][()], stored)
    assert h5["overflow row"][()].tolist() == rows
    assert h5["overflow bin"][()].tolist() == bins
    assert h5["overflow code"][()].tolist() == codes


@pytest.mark.parametrize("step", [0.25, 0.3])
@pytest.mark.parametrize("raw", [True, False])
def test_rows_match_the_numpy_decoding(h5, monkeypatch, step, raw):
    import sps_common.stack_quant as sq

    monkeypatch.setattr(sq, "READ_BLOCK_ROWS", 4)
    spectra = awkward_stack(step, 7, seed=3)
    write_quantized(h5, spectra, step, 7)
    stored, rows, bins, codes = reference_codes(spectra, step, 7)
    scale = np.float32(step if raw else step / quant_bias(step))
    expected = reference_rows(stored, (rows, bins, codes), h5.attrs["quant_offset"], scale)
    np.testing.assert_array_equal(read_quantized(h5, raw=raw), expected)
    np.testing.assert_array_equal(read_quantized(h5, 2, 7, raw=raw), expected[2:7])


@pytest.mark.parametrize("step", [0.25, 0.3])
def test_adding_a_day_matches_rounding_and_adding_row_by_row(h5, monkeypatch, step):
    import sps_common.stack_quant as sq

    monkeypatch.setattr(sq, "READ_BLOCK_ROWS", 4)
    write_quantized(h5, awkward_stack(step, 5, seed=4), step, 5)
    stack = QuantizedStack(h5)
    day = awkward_stack(step, 1, seed=5) / np.float32(3)
    day[2, 10] = 4e5  # new overflow in a row with none stored
    expected = np.array(
        [round_power(day[i], step) + stack.rows(i, i + 1, raw=True)[0] for i in range(9)]
    )
    stack.add_rounded(day, step)
    np.testing.assert_array_equal(day, expected)


@pytest.mark.parametrize("bad", [2.0**29, np.nan, np.inf])
def test_unstorable_power_is_refused(h5, bad):
    spectra = np.ones((3, 10), np.float32)
    spectra[2, 3] = bad
    with pytest.raises(ValueError, match="row 2"):
        write_quantized(h5, spectra, STEP, 1)
