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


# --- corrupt existing stacks -----------------------------------------------------
import datetime  # noqa: E402

from ps_processes import CorruptStackError  # noqa: E402


class FakeStackDb:
    def __init__(self, path):
        self.datapath_month = str(path)
        self.pointing_id = "pointing123"


def daily_spectra(value=5):
    pspec = FakeSpectra(value)
    pspec.datetimes = [datetime.datetime(2026, 9, 12, tzinfo=datetime.timezone.utc)]
    pspec.num_days = 1
    pspec.obs_id = ["obs1"]
    return pspec


@pytest.fixture
def stack_setup(tmp_path, monkeypatch):
    """A stack folder <tmp>/stack with an existing stack file registered in the 'db'."""
    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    old = stack_dir / "12.34_56.78_20260906_11_power_spectra_stack.hdf5"
    stacker = PowerSpectraStack(update_db=True, qc=False, basepath=str(tmp_path))
    db_calls, slack = [], []
    monkeypatch.setattr(
        PowerSpectraStack, "get_ps_stack_db", lambda self, pspec: FakeStackDb(old)
    )
    monkeypatch.setattr(
        ps_stack.db_api,
        "update_ps_stack",
        lambda pointing_id, payload: db_calls.append((pointing_id, payload)),
    )
    monkeypatch.setattr(ps_stack, "notify_slack", slack.append)
    monkeypatch.setattr(
        PowerSpectraStack, "update_obs_database", lambda self, *args, **kwargs: None
    )
    return stacker, old, tmp_path, db_calls, slack


def test_corrupt_stack_is_moved_and_new_stack_started(stack_setup):
    stacker, old, root, db_calls, slack = stack_setup
    old.write_bytes(b"\x00" * 1000)  # like the stacks broken on 2026-09-20
    pspec = daily_spectra(5)

    result = stacker.stack(pspec)

    moved = root / "stack_corrupt" / old.name
    assert not old.exists() and moved.read_bytes() == b"\x00" * 1000
    new = root / "stack" / "12.34_56.78_20260912_1_power_spectra_stack.hdf5"
    assert read_value(new) == 5 and result is pspec
    # db: first reset, then pointed to the new stack
    assert db_calls[0][1] == {
        "datapath_month": "",
        "datetimes_month": [],
        "num_days_month": 0,
    }
    assert db_calls[-1][1]["datapath_month"] == str(new)
    assert (
        len(slack) == 1 and str(moved) in slack[0] and "starts a new stack" in slack[0]
    )
    assert not stacker.lock.is_locked


def test_stack_broken_while_reading_is_moved_but_not_used(stack_setup, monkeypatch):
    stacker, old, root, db_calls, slack = stack_setup
    stacker.write_stack_safely(FakeSpectra(1), str(old))  # structure looks fine

    def failing_readout(self, pspec, path):
        pspec.value = -1  # part of the old stack was already added
        raise OSError("Can't read data (wrong B-tree signature)")

    monkeypatch.setattr(
        PowerSpectraStack, "stack_power_spectra_readout", failing_readout
    )
    with pytest.raises(CorruptStackError):
        stacker.stack(daily_spectra(5))

    assert not old.exists() and (root / "stack_corrupt" / old.name).exists()
    # no new stack from the half-summed spectra
    assert [f for f in os.listdir(root / "stack") if f.endswith(".hdf5")] == []
    assert db_calls == [
        (
            "pointing123",
            {"datapath_month": "", "datetimes_month": [], "num_days_month": 0},
        )
    ]
    assert len(slack) == 1 and "could not be stacked" in slack[0]
    assert not stacker.lock.is_locked


def test_corrupt_file_name_collision_keeps_both(stack_setup):
    stacker, old, root, db_calls, slack = stack_setup
    (root / "stack_corrupt").mkdir()
    (root / "stack_corrupt" / old.name).write_bytes(b"earlier")
    old.write_bytes(b"\x00" * 10)
    stacker.stack(daily_spectra())
    files = sorted(os.listdir(root / "stack_corrupt"))
    assert len(files) == 2 and files[0] == old.name
    assert (root / "stack_corrupt" / old.name).read_bytes() == b"earlier"


def test_healthy_stack_is_not_quarantined(stack_setup, monkeypatch):
    stacker, old, root, db_calls, slack = stack_setup
    stacker.write_stack_safely(FakeSpectra(1), str(old))

    def readout(self, pspec, path):
        pspec.value += read_value(path)
        pspec.datetimes = pspec.datetimes * 12  # the real readout adds the stack days
        pspec.num_days = 12
        return pspec

    monkeypatch.setattr(PowerSpectraStack, "stack_power_spectra_readout", readout)
    stacker.stack(daily_spectra(5))
    assert not (root / "stack_corrupt").exists() and slack == []
    new = root / "stack" / "12.34_56.78_20260912_12_power_spectra_stack.hdf5"
    assert read_value(new) == 6 and not old.exists()
