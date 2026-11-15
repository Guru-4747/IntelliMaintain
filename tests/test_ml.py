import numpy as np
import pytest

from conftest import needs_model, synthetic_bearing_signal
from intellimaintain import config
from intellimaintain.ml import cwru
from intellimaintain.ml.features import FEATURE_NAMES, signal_features, sliding_windows, window_features
from intellimaintain.ml.physics import cross_check, defect_frequencies, physics_check
from intellimaintain.ml.signal_io import load_signal_csv
from intellimaintain.schemas import ClassifyResult


def test_windowing_count_and_overlap():
    x = np.arange(10_000, dtype=float)
    w = sliding_windows(x)
    step = int(config.WINDOW * (1 - config.OVERLAP))
    assert w.shape == (1 + (10_000 - config.WINDOW) // step, config.WINDOW)
    assert w[1, 0] == step  # 50 % overlap


def test_windowing_too_short():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros(100))


def test_feature_values_on_known_signals():
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal((1, 2048))
    f = dict(zip(FEATURE_NAMES, window_features(gauss)[0]))
    assert 2.6 < f["kurtosis"] < 3.4          # Gaussian -> ~3
    sine = np.sin(2 * np.pi * (12000 / 2048 * 8) * np.arange(2048) / 12000)[None, :]  # whole periods
    f2 = dict(zip(FEATURE_NAMES, window_features(sine)[0]))
    assert abs(f2["crest_factor"] - np.sqrt(2)) < 0.05
    bands = [f2[k] for k in FEATURE_NAMES if k.startswith("fft_band")]
    assert abs(sum(bands) - 1) < 1e-6 and np.argmax(bands) == 0


def test_feature_frame_has_no_nan():
    df = signal_features(synthetic_bearing_signal())
    assert list(df.columns) == FEATURE_NAMES and not df.isna().any().any()


def test_defect_frequencies_match_cwru_reference():
    f = defect_frequencies(1797)
    fr = 1797 / 60
    assert f["bpfi_hz"] / fr == pytest.approx(5.4152, abs=2e-3)
    assert f["bpfo_hz"] / fr == pytest.approx(3.5848, abs=2e-3)
    assert f["bsf_hz"] / fr == pytest.approx(2.357, abs=2e-3)
    assert f["ftf_hz"] / fr == pytest.approx(0.3983, abs=1e-3)
    assert f["bpfo_hz"] + f["bpfi_hz"] == pytest.approx(9 * fr)


def test_defect_frequencies_validation():
    with pytest.raises(ValueError):
        defect_frequencies(0)


def test_physics_detects_synthetic_outer_race():
    rpm = 1797
    bpfo = defect_frequencies(rpm)["bpfo_hz"]
    ff = physics_check(synthetic_bearing_signal(bpfo), rpm)
    assert ff.physics_suggests == "Outer race"
    status, _ = cross_check("Outer race", ff)
    assert status == "consistent"
    status, _ = cross_check("Inner race", ff)
    assert status == "conflict"
    assert cross_check("Ball", ff)[0] == "conflict"  # very strong race-fault lines overrule a ball label
    weak = ff.model_copy(update={"envelope_match": {"Outer race": 12.0, "Inner race": 9.0, "Ball": 4.0}})
    assert cross_check("Ball", weak)[0] == "inconclusive"


def test_physics_on_noise_is_normal():
    ff = physics_check(np.random.default_rng(1).standard_normal(24000), 1797)
    assert ff.physics_suggests is None
    assert cross_check("Normal", ff)[0] == "consistent"


def test_catalog_complete():
    assert len(cwru.CATALOG) == 40
    assert sorted({c.label for c in cwru.CATALOG}) == sorted(cwru.LABELS)
    assert cwru.BY_NUMBER[97].fs == 48_000 and cwru.BY_NUMBER[105].fs == 12_000
    assert cwru.fault_type_of("OR_014") == "Outer race" and cwru.fault_size_of("IR_021") == 0.021


def test_csv_reader_variants():
    a = load_signal_csv(b"# fs=12000\n# rpm=1750\ntime,DE\n0,0.1\n0.0000833,0.2\n0.000166,0.3\n")
    assert a.fs == 12000 and a.rpm == 1750 and a.column == "DE" and len(a.x) == 3
    b = load_signal_csv(b"0.1\n0.2\n0.3\n0.4\n")
    assert len(b.x) == 4 and b.fs is None
    c = load_signal_csv(b"t,value\n0,1\n0.001,2\n0.002,3\n")
    assert c.column == "value" and c.fs == 1000
    with pytest.raises(ValueError):
        load_signal_csv(b"a,b\nx,y\n")


@needs_model
def test_classify_signal_contract(demo_signal):
    from intellimaintain.ml.classifier import classify_signal

    r = classify_signal(demo_signal)
    assert isinstance(r, ClassifyResult)
    assert r.label in cwru.LABELS and 0 <= r.confidence <= 1 and 0 <= r.anomaly_score <= 1
    assert abs(sum(r.class_probabilities.values()) - 1) < 1e-3
    assert r.n_windows > 0 and sum(r.window_votes.values()) == r.n_windows


@needs_model
def test_classify_resamples_and_rejects_short(demo_signal):
    from intellimaintain.ml.classifier import classify_signal

    from scipy.signal import resample_poly
    x48 = resample_poly(demo_signal, 4, 1)
    assert classify_signal(x48, fs=48000).n_windows == classify_signal(demo_signal).n_windows
    with pytest.raises(ValueError):
        classify_signal(np.zeros(500))


@needs_model
@pytest.mark.skipif(not cwru.available(), reason="CWRU data not downloaded")
def test_classifier_on_real_held_out_recordings():
    from intellimaintain.ml import classifier

    old = classifier.ACTIVE_MODEL_PATH
    classifier.ACTIVE_MODEL_PATH = classifier.CROSSLOAD_MODEL_PATH  # model never saw 3 HP
    try:
        for num in (100, 108, 133, 225):
            c = cwru.BY_NUMBER[num]
            r = classifier.classify_signal(cwru.load_file(c)[:24000])
            assert r.label == c.label, (num, r.label)
    finally:
        classifier.ACTIVE_MODEL_PATH = old


# ------------------------------------------------------------------ spectrogram CNN
def test_spectrogram_shape_and_standardisation():
    from intellimaintain.ml.spectrogram import IMG_SHAPE, signal_spectrogram, window_spectrograms

    img = window_spectrograms(np.random.default_rng(0).standard_normal((3, 2048)))
    assert img.shape == (3, 1, *IMG_SHAPE) and img.dtype == np.float32
    assert np.allclose(img.mean(axis=(1, 2, 3)), 0, atol=1e-4) and np.allclose(img.std(axis=(1, 2, 3)), 1, atol=1e-2)
    t, f, db = signal_spectrogram(synthetic_bearing_signal())
    assert db.shape == (len(f), len(t))


def test_cnn_forward_and_temperature():
    torch = pytest.importorskip("torch")
    from intellimaintain.ml import cnn

    m = cnn.build_model(10)
    out = m(torch.zeros(4, 1, 64, 64))
    assert tuple(out.shape) == (4, 10)
    z = np.array([[4.0, 0.0], [0.0, 4.0]])
    assert np.allclose(cnn.softmax(z).sum(1), 1) and cnn.softmax(z, 4.0)[0, 0] < cnn.softmax(z, 1.0)[0, 0]


@pytest.mark.skipif(not (config.MODELS_DIR / "cnn_classifier.pt").exists(), reason="run scripts/train_cnn.py")
def test_classify_signal_with_cnn(demo_signal):
    from intellimaintain.ml.classifier import classify_signal

    r = classify_signal(demo_signal, model="cnn")
    assert r.model_name.startswith("Spectrogram CNN") and r.label in cwru.LABELS
    assert abs(sum(r.class_probabilities.values()) - 1) < 1e-3
    with pytest.raises(ValueError):
        classify_signal(demo_signal, model="svm")


@pytest.mark.skipif(not (config.MODELS_DIR / "cnn_classifier_crossload.pt").exists() or not cwru.available(),
                    reason="needs CWRU data and scripts/train_cnn.py")
def test_cnn_on_real_held_out_recordings():
    from intellimaintain.ml import classifier

    old = classifier.ACTIVE_MODEL_PATH
    classifier.ACTIVE_MODEL_PATH = classifier.CROSSLOAD_MODEL_PATH
    try:
        hits = sum(classifier.classify_signal(cwru.load_file(cwru.BY_NUMBER[n])[:24000], model="cnn").label
                   == cwru.BY_NUMBER[n].label for n in (100, 108, 133, 225))
        assert hits >= 3
    finally:
        classifier.ACTIVE_MODEL_PATH = old
