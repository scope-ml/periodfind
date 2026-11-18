"""CPU tests for the ellipsoidal (BEER) fit.

Light curves are drawn from the BEER model itself, so each test checks a
quantity the fit should return exactly up to noise: the amplitudes per band,
T0 on the branch with positive beaming, the orbital period among candidates,
and the sign of delta_bic.

Run with: pytest tests/test_ellipsoidal_fit.py -v
"""

import numpy as np
import pytest

from periodfind.cpu import EllipsoidalFit

PERIOD = 0.37
# (A_ellip, A_refl, A_beam) per band, with chromatic ellipsoidal amplitudes
AMPS = [(0.030, 0.004, 0.002), (0.025, 0.003, 0.0015), (0.022, 0.002, 0.0012)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_curve(period=PERIOD, t0=60000.123, amps=AMPS, n_points=3000, noise=0.001, seed=0):
    """A BEER light curve in magnitudes, sampled over 1000 days."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(60000.0, 61000.0, n_points))
    b = rng.integers(0, len(amps), n_points).astype(np.uint8)
    ae, ar, ab = (np.array(a)[b] for a in zip(*amps))
    phi = 2 * np.pi * (t - t0) / period
    f = 1 - ae * np.cos(2 * phi) - ar * np.cos(phi) + ab * np.sin(phi)
    m = 15.0 + b - 2.5 * np.log10(f) + rng.normal(0, noise, n_points)
    e = np.full(n_points, noise)
    return t, m, e, b


def phase_diff(a, b, period):
    d = np.mod((a - b) / period, 1.0)
    return min(d, 1.0 - d)


def row(fit, out, i=0):
    return dict(zip(fit.columns, out[i]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recovers_amplitudes_and_t0():
    t0 = 60000.123
    t, m, e, b = make_curve(t0=t0)
    fit = EllipsoidalFit(n_band=3)
    out = fit.calc([t], [m], [e], [b], [np.array([PERIOD / 2])])

    assert out.shape == (1, 6 + 6 * 3)
    r = row(fit, out)
    assert r["period"] == pytest.approx(PERIOD, rel=1e-12)
    assert phase_diff(r["t0"], t0, PERIOD) < 2e-3
    assert r["n_points"] == 3000
    for k, (ae, ar, ab) in enumerate(AMPS):
        assert abs(r[f"A_ellip_{k}"] - ae) < 5 * r[f"A_ellip_err_{k}"]
        assert abs(r[f"A_refl_{k}"] - ar) < 5 * r[f"A_refl_err_{k}"]
        assert abs(r[f"A_beam_{k}"] - ab) < 5 * r[f"A_beam_err_{k}"]
    assert r["delta_bic"] > 0
    assert r["delta_bic_flat"] > r["delta_bic"]


@pytest.mark.parametrize("shift", [0.0, 0.25, 0.5, 0.75])
def test_t0_follows_beaming(shift):
    """Every quarter-cycle offset lands on the branch with positive Ae and Ab."""
    t0 = 60000.05 + shift * PERIOD
    t, m, e, b = make_curve(t0=t0, seed=1)
    fit = EllipsoidalFit(n_band=3)
    r = row(fit, fit.calc([t], [m], [e], [b], [[PERIOD / 2]]))
    assert phase_diff(r["t0"], t0, PERIOD) < 2e-3
    assert r["A_ellip_0"] > 0 and r["A_beam_0"] > 0


def test_pure_ellipsoidal_prefers_half_period():
    """With no odd harmonics, a sinusoid at P/2 describes the curve as well."""
    t, m, e, b = make_curve(amps=[(0.03, 0.0, 0.0)] * 2, seed=2)
    fit = EllipsoidalFit(n_band=2)
    r = row(fit, fit.calc([t], [m], [e], [b], [[PERIOD / 2]]))
    assert r["delta_bic"] < 0


def test_picks_orbital_period_among_candidates():
    t, m, e, b = make_curve(seed=3)
    fit = EllipsoidalFit(n_band=3)
    r = row(fit, fit.calc([t], [m], [e], [b], [[0.21, PERIOD / 2, 0.29, PERIOD]]))
    assert r["period"] == pytest.approx(PERIOD, rel=1e-12)


def test_double_periods_off_takes_orbital_periods():
    t, m, e, b = make_curve(seed=4)
    fit = EllipsoidalFit(n_band=3, double_periods=False)
    r = row(fit, fit.calc([t], [m], [e], [b], [[PERIOD]]))
    assert r["period"] == pytest.approx(PERIOD, rel=1e-12)
    assert r["A_ellip_0"] == pytest.approx(AMPS[0][0], abs=5e-4)


def test_unused_band_is_nan():
    t, m, e, b = make_curve(amps=AMPS[:2], seed=5)
    fit = EllipsoidalFit(n_band=3)
    r = row(fit, fit.calc([t], [m], [e], [b], [[PERIOD / 2]]))
    assert np.isfinite(r["A_ellip_0"]) and np.isfinite(r["A_ellip_1"])
    assert np.isnan(r["A_ellip_2"])


def test_no_usable_band_is_nan():
    t, m, e, b = make_curve(n_points=6, seed=6)
    fit = EllipsoidalFit(n_band=3)
    out = fit.calc([t], [m], [e], [b], [[PERIOD / 2]])
    assert np.all(np.isnan(out[0, :5]))
    assert out[0, 5] == 0


def test_batch_matches_single():
    curves = [make_curve(t0=60000.1 + 0.03 * i, seed=10 + i) for i in range(4)]
    periods = [[PERIOD / 2, 0.3]] * 4
    fit = EllipsoidalFit(n_band=3)
    batched = fit.calc(*[[c[k] for c in curves] for k in range(4)], periods)
    for i, c in enumerate(curves):
        one = fit.calc([c[0]], [c[1]], [c[2]], [c[3]], [periods[i]])
        np.testing.assert_array_equal(batched[i], one[0])


def test_columns_match_output_width():
    fit = EllipsoidalFit(n_band=2)
    assert len(fit.columns) == 6 + 6 * 2
    assert fit.columns[6:9] == ["A_ellip_0", "A_ellip_err_0", "A_beam_0"]


def test_rejects_mismatched_inputs():
    t, m, e, b = make_curve(n_points=50)
    fit = EllipsoidalFit()
    with pytest.raises(ValueError):
        fit.calc([t], [m[:-1]], [e], [b], [[PERIOD / 2]])
    with pytest.raises(ValueError):
        fit.calc([t], [m], [e], [b], [[]])
    with pytest.raises(ValueError):
        EllipsoidalFit(n_t0=2)
