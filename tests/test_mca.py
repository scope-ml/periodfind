"""CPU tests for the multi-criterion ascent (MCA) period finder.

Synthetic sparse multiband light curves: 14 nights, 1-3 bands a night, a few
points each, as in DP2. Each test checks something the method should do
exactly: find a known period from a nearby start, merge duplicate starts,
return NaN rows for unusable input, and give the same row for an object
whether it is run alone or in a batch.

Run with: pytest tests/test_mca.py -v
"""

import numpy as np
import pytest

from periodfind.cpu import MultiCriterionAscent


def make_curve(period=0.31, amp=0.3, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t, m, b = [], [], []
    for night in range(14):
        base = 60000.0 + 4 * night + np.floor(2 * rng.random())
        for band in range(3):
            if rng.random() < 0.6:
                tt = base + 0.1 + 0.25 * rng.random(6)
                t.extend(tt)
                b.extend([band] * 6)
                m.extend(
                    18.0
                    + 0.3 * band
                    + amp * np.sin(2 * np.pi * tt / period)
                    + noise * rng.normal(size=6)
                )
    o = np.argsort(t)
    return np.array(t)[o], np.array(m)[o], np.array(b, dtype=np.uint8)[o]


def run(curves, periods, **kw):
    mca = MultiCriterionAscent(f_max=40.0, top=5, **kw)
    return mca, mca.calc(*[[c[k] for c in curves] for k in range(3)], periods)


def test_finds_period_from_nearby_start():
    period = 0.31
    _, out = run([make_curve(period, seed=1)], [[period * 1.003, 0.21, 0.47]])
    assert out[0, 1] >= 1
    assert abs(out[0, 2] / period - 1) < 1e-3


def test_duplicate_starts_are_merged():
    _, out = run([make_curve(seed=2)], [[0.31, 0.31, 0.31]])
    assert out[0, 0] == 1


def test_best_first_and_nan_padded():
    mca, out = run([make_curve(seed=3)], [[0.31, 0.25, 0.47, 0.6]])
    n_opt = int(out[0, 1])
    scores = out[0, 2 + mca.top : 2 + mca.top + min(n_opt, mca.top)]
    assert np.all(np.diff(scores) <= 0)
    assert np.all(np.isnan(out[0, 2 + min(n_opt, mca.top) : 2 + mca.top]))


def test_too_few_points_is_nan_row():
    t = np.array([60000.0, 60001.0, 60002.0])
    _, out = run([(t, t * 0 + 18.0, np.zeros(3, np.uint8))], [[0.5]])
    assert out[0, 1] == 0
    assert np.all(np.isnan(out[0, 2:]))


def test_batch_matches_single():
    curves = [make_curve(seed=10 + i) for i in range(3)]
    periods = [[0.31, 0.25]] * 3
    _, batched = run(curves, periods)
    for i, c in enumerate(curves):
        _, one = run([c], [periods[i]])
        np.testing.assert_array_equal(batched[i], one[0])


def test_columns_match_output_width():
    mca, out = run([make_curve(seed=4)], [[0.31]])
    assert len(mca.columns) == out.shape[1] == 2 + 2 * mca.top
    assert mca.columns[:3] == ["n_starts", "n_optima", "period_1"]


def test_rejects_bad_input():
    t, m, b = make_curve(seed=5)
    mca = MultiCriterionAscent()
    with pytest.raises(ValueError):
        mca.calc([t], [m[:-1]], [b], [[0.31]])
    with pytest.raises(ValueError):
        mca.calc([t], [m], [b], [[0.31], [0.2]])
    with pytest.raises(ValueError):
        MultiCriterionAscent(top=0)
