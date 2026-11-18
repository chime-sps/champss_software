import os

import h5py
import numpy as np
import pytest
from ps_processes.processes import ps_stack
from ps_processes.processes.ps_stack import PowerSpectraStack


class FakeSpectra:
    """Minimal stand-in for PowerSpectra: writes the fields check_file_keys expects."""

    def __init__(self, value, ra=12.34, dec=56.78, fail=None):
        self.ra, self.dec, self.value, self.fail = ra, dec, value, fail

    def write(self, filename, nbit=32):
        if self.fail == "crash":
            with open(filename, "wb") as f:
                f.write(b"\x00" * 128)  # partially written file, then the "node dies"
            raise RuntimeError("simulated crash during write")
        if self.fail == "garbage":
            with open(filename, "wb") as f:
                f.write(b"\x00" * 128)  # write "succeeds" but the file is unreadable
            return
        with h5py.File(filename, "w") as f:
            for key in ["datetimes", "dms", "frequency labels"]:
                f[key] = np.arange(3)
            f["power spectra"] = np.full(4, self.value)


def read_value(path):
    with h5py.File(path, "r") as f:
        return f["power spectra"][0]


@pytest.fixture
def stacker():
    return PowerSpectraStack(update_db=False, qc=False)


def temp_files(folder):
    return [f for f in os.listdir(folder) if f.endswith("_temp.hdf5")]


def test_write_new_stack(stacker, tmp_path):
    path = tmp_path / "stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(path))
    assert read_value(path) == 1
    assert temp_files(tmp_path) == []


def test_replace_existing_stack(stacker, tmp_path):
    path = tmp_path / "stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(path))
    stacker.write_stack_safely(FakeSpectra(2), str(path))
    assert read_value(path) == 2
    assert temp_files(tmp_path) == []


def test_crash_during_write_keeps_previous_stack(stacker, tmp_path):
    path = tmp_path / "stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(path))
    with pytest.raises(RuntimeError):
        stacker.write_stack_safely(FakeSpectra(2, fail="crash"), str(path))
    assert read_value(path) == 1
    # a leftover temp file from a crash is cleaned up by the next write
    stacker.write_stack_safely(FakeSpectra(3), str(path))
    assert read_value(path) == 3
    assert temp_files(tmp_path) == []


def test_unreadable_new_stack_is_not_used(stacker, tmp_path):
    path = tmp_path / "stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(path))
    with pytest.raises(OSError):
        stacker.write_stack_safely(FakeSpectra(2, fail="garbage"), str(path))
    assert read_value(path) == 1
    assert temp_files(tmp_path) == []


def test_old_stack_only_deleted_after_new_one_is_written(stacker, tmp_path):
    old = tmp_path / "old_20260905_9_stack.hdf5"
    new = tmp_path / "new_20260906_10_stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(old))

    with pytest.raises(RuntimeError):
        stacker.replace_spectra(FakeSpectra(2, fail="crash"), str(new), str(old))
    assert read_value(old) == 1 and not new.exists()

    stacker.replace_spectra(FakeSpectra(2), str(new), str(old))
    assert read_value(new) == 2 and not old.exists()


def test_replace_spectra_same_path_two_arguments(stacker, tmp_path):
    # file_pipeline.py and test_ps.py call replace_spectra(pspec, path)
    path = tmp_path / "stack.hdf5"
    stacker.write_stack_safely(FakeSpectra(1), str(path))
    stacker.replace_spectra(FakeSpectra(2), str(path))
    assert read_value(path) == 2


def test_data_is_fsynced_before_the_rename(stacker, tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(
        ps_stack.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))
    )
    monkeypatch.setattr(
        ps_stack.os,
        "replace",
        lambda a, b: (events.append("replace"), real_replace(a, b)),
    )
    stacker.write_stack_safely(FakeSpectra(1), str(tmp_path / "stack.hdf5"))
    # file fsync, then the atomic rename, then fsync of the folder
    assert events == ["fsync", "replace", "fsync"]
