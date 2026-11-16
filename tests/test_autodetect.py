"""Automatic sampling-rate and shaft-speed detection for uploaded files."""
import numpy as np
import pytest

from conftest import needs_model
from intellimaintain.ml import autodetect
from intellimaintain.ml.signal_io import load_signal_csv


def test_fs_and_rpm_from_file_name():
    assert autodetect.fs_from_name("pump_48k.csv") == 48000
    assert autodetect.fs_from_name("run_fs=25600.csv") == 25600
    assert autodetect.fs_from_name("outer_race_3hp.csv") is None
    assert autodetect.rpm_from_name("motor_1750rpm.csv") == 1750
    assert autodetect.rpm_from_name("outer_race_007_3hp.csv") == 1730   # CWRU load -> speed
    assert autodetect.rpm_from_name("data.csv") is None


def test_file_values_win():
    d = autodetect.detect(np.zeros(4096), "whatever.csv", fs_file=25600, rpm_file=1480)
    assert (d.fs, d.rpm, d.rpm_confidence) == (25600, 1480, "high") and "file" in d.fs_source


def test_cwru_file_number():
    d = autodetect.detect(np.zeros(4096), "X131_DE_time 131.mat")
    assert d.fs == 12000 and d.rpm == 1772 and "CWRU file 131" in d.rpm_source
    assert autodetect.detect(np.zeros(4096), "97.mat").fs == 48000          # CWRU normal files are 48 kHz


def test_rpm_estimated_from_a_synthetic_spectrum():
    fs, rpm = 12000, 1490.0
    t = np.arange(4 * fs) / fs
    fr = rpm / 60
    x = sum(a * np.sin(2 * np.pi * h * fr * t) for h, a in ((1, 1.0), (2, 0.6), (3, 0.3)))
    x = x + 0.2 * np.sin(2 * np.pi * 50 * t) + np.random.default_rng(1).normal(0, 0.05, len(t))  # mains hum + noise
    d = autodetect.detect(x, "unknown.csv")
    assert abs(d.rpm - rpm) / rpm < 0.02 and d.rpm_confidence == "medium" and "estimated" in d.rpm_source


def test_no_shaft_line_falls_back_and_says_so():
    d = autodetect.detect(np.random.default_rng(0).normal(size=24000), "noise.csv")
    assert d.rpm == autodetect.DEFAULT_RPM and d.rpm_confidence == "low" and "please check" in d.rpm_source


def test_time_column_gives_fs():
    fs = 25600
    csv = "time_s,vibration_g\n" + "\n".join(f"{i / fs:.7f},{np.sin(i / 7):.5f}" for i in range(3000))
    assert load_signal_csv(csv.encode()).fs == fs


@needs_model
def test_real_cwru_speed_estimate():
    from intellimaintain.ml import cwru
    files = cwru.available()
    if not files:
        pytest.skip("CWRU data not downloaded")
    errs = []
    for c in files:
        rpm, _, _ = autodetect.estimate_rpm(cwru.load_file(c)[:24000], 12000)
        errs.append(abs(rpm - c.rpm) / c.rpm)
    assert np.mean(np.array(errs) < 0.03) >= 0.9   # 95 % on the 40 recordings
