"""Create demo CSV files (2 s each) from held-out 3 HP CWRU recordings for the app.

Usage: python scripts/make_samples.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intellimaintain import config  # noqa: E402
from intellimaintain.ml import cwru  # noqa: E402

SAMPLES = [
    (100, "normal_3hp.csv", None),
    (108, "inner_race_007_3hp.csv", None),
    (133, "outer_race_007_3hp.csv", None),
    (225, "ball_021_3hp.csv", None),
    (200, "outer_race_014_3hp_physics_conflict.csv", None),
    (172, "inner_race_014_3hp_noisy_5db.csv", 5),
]


def main(seconds: float = 2.0, offset_s: float = 3.0) -> None:
    rng = np.random.default_rng(0)
    n = int(seconds * config.FS)
    start = int(offset_s * config.FS)
    for num, name, snr_db in SAMPLES:
        c = cwru.BY_NUMBER[num]
        x = cwru.load_file(c)[start:start + n]
        if snr_db is not None:
            x = x + rng.normal(0, np.sqrt(np.mean(x**2) / 10 ** (snr_db / 10)), size=len(x))
        t = np.arange(len(x)) / config.FS
        out = config.SAMPLES_DIR / name
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# fs={config.FS}\n# rpm={c.rpm}\n# source=CWRU {num}.mat label={c.label} load={c.load_hp}HP"
                    f"{f' noise={snr_db}dB' if snr_db else ''}\n")
            f.write("time,DE\n")
            for ti, xi in zip(t, x):
                f.write(f"{ti:.6f},{xi:.6f}\n")
        print(f"wrote {out.relative_to(config.ROOT)}  ({c.label}, {len(x)} samples)")


if __name__ == "__main__":
    main()
