"""CPU tests for sampled Template Fit.

Like Template Fit, this returns the single best (period, template, phase,
amplitude) per light curve rather than a periodogram, so these check the fit.

The templates are built here rather than loaded, so the tests carry no data
files. One of them is a narrow dip, which is the case a truncated Fourier
representation cannot hold and a sampled one can.

Run with: pytest tests/test_template_fit_sampled.py -v
"""

import numpy as np
import pytest

from periodfind.cpu import TemplateFitSampled

N_BAND = 3
N_PHASE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_bank(n_band=N_BAND, n_phase=N_PHASE, ratios=(1.0, 1.0, 1.0)):
    """One template: a cosine, scaled per band.

    ``ratios`` is what a chromatic bank encodes: the relative band amplitudes
    are fixed by the template and the fit may only scale all of them together.
    """
    ph = np.arange(n_phase) / n_phase
    s = np.zeros((1, n_band, n_phase))
    for b in range(n_band):
        s[0, b] = ratios[b] * np.cos(2 * np.pi * ph)
    return s


def cosine_and_dip_bank(n_band=N_BAND, n_phase=N_PHASE):
    """A cosine and a narrow dip, so the fit has a choice of shape."""
    ph = np.arange(n_phase) / n_phase
    s = np.zeros((2, n_band, n_phase))
    for b in range(n_band):
        s[0, b] = np.cos(2 * np.pi * ph)
        s[1, b] = np.where(ph < 1.0 / 16.0, -1.0, 0.0)
    return s


def make_curve(
    period,
    samples,
    template=0,
    amplitude=0.3,
    n_points=900,
    t_span=60.0,
    noise=0.005,
    seed=0,
    phase0=0.0,
):
    """A light curve drawn from one template, in several bands."""
    rng = np.random.default_rng(seed)
    n_band, n_phase = samples.shape[1], samples.shape[2]
    t = np.sort(rng.uniform(0, t_span, n_points))
    b = rng.integers(0, n_band, n_points).astype(np.uint8)
    k = np.floor(np.mod(t / period - phase0, 1.0) * n_phase).astype(int)
    k = np.clip(k, 0, n_phase - 1)
    m = np.empty(n_points)
    for band in range(n_band):
        sel = b == band
        m[sel] = 15.0 + band + amplitude * samples[template, band][k[sel]]
    m += rng.normal(0, noise, n_points)
    return t, m, b


def trial_periods(true_period, n=60, margin=0.4):
    lo = max(true_period * (1 - margin), 0.01)
    hi = true_period * (1 + margin)
    p = np.linspace(lo, hi, n)
    return np.unique(np.concatenate([p, [true_period]]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recovers_known_period():
    s = cosine_bank()
    period = 1.7
    t, m, b = make_curve(period, s)
    tfs = TemplateFitSampled(s)
    out = tfs.calc([t], [m], [b], [trial_periods(period)])

    assert out.shape == (1, 5)
    assert out[0, 1] == pytest.approx(period, rel=1e-9)
    assert 0.0 <= out[0, 0] < 0.05
    assert out[0, 4] > 0.0


def test_picks_the_sharp_template():
    """A narrow dip is held as a narrow dip.

    A template truncated to a few harmonics cannot represent a feature this
    narrow, and such a fit then prefers the smooth shape or a wrong period.
    """
    s = cosine_and_dip_bank()
    period = 0.83
    t, m, b = make_curve(period, s, template=1, amplitude=0.4, seed=3)
    tfs = TemplateFitSampled(s)
    out = tfs.calc([t], [m], [b], [trial_periods(period)])

    assert out[0, 2] == 1
    assert out[0, 1] == pytest.approx(period, rel=1e-9)


def test_amplitude_is_shared_across_bands():
    """A curve that breaks the template's band ratios fits worse.

    The fit has one amplitude for all bands, so a bank encoding fixed ratios
    cannot absorb a curve that violates them. That is what makes colour a
    constraint rather than a free parameter.
    """
    s = cosine_bank(ratios=(1.0, 0.5, 0.25))
    period = 1.1
    t, m, b = make_curve(period, s, amplitude=0.4, seed=5)
    tfs = TemplateFitSampled(s)
    matching = tfs.calc([t], [m], [b], [trial_periods(period)])[0, 0]

    m2 = m.copy()
    m2[b == 2] = 17.0 + 4.0 * (m2[b == 2] - 17.0)
    breaking = tfs.calc([t], [m2], [b], [trial_periods(period)])[0, 0]

    assert matching < breaking


def test_phase_shift_is_recovered():
    s = cosine_bank()
    period = 2.3
    t, m, b = make_curve(period, s, phase0=0.25, seed=11, noise=0.002)
    tfs = TemplateFitSampled(s)
    out = tfs.calc([t], [m], [b], [trial_periods(period)])

    assert out[0, 1] == pytest.approx(period, rel=1e-9)
    # a quarter turn, as an index into the phase grid
    assert abs(int(out[0, 3]) - N_PHASE // 4) <= 2


def test_batch_matches_individual():
    s = cosine_and_dip_bank()
    periods = [0.9, 1.4, 2.1]
    curves = [make_curve(p, s, template=i % 2, seed=20 + i) for i, p in enumerate(periods)]
    trials = [trial_periods(p) for p in periods]
    tfs = TemplateFitSampled(s)

    batched = tfs.calc(
        [c[0] for c in curves], [c[1] for c in curves], [c[2] for c in curves], trials
    )
    for i, c in enumerate(curves):
        one = tfs.calc([c[0]], [c[1]], [c[2]], [trials[i]])
        assert np.allclose(batched[i], one[0], equal_nan=True)


def test_rejects_mismatched_inputs():
    s = cosine_bank()
    t, m, b = make_curve(1.0, s)
    tfs = TemplateFitSampled(s)
    with pytest.raises(ValueError):
        tfs.calc([t], [m], [b], [])
    with pytest.raises(ValueError):
        tfs.calc([t], [m[:-1]], [b], [np.array([1.0])])


def test_rejects_a_grid_that_is_not_a_power_of_two():
    """The transform is radix-2, so the grid has to be one.

    Checked at construction rather than at fit time, where it would surface as
    a wrong answer instead of an error.
    """
    with pytest.raises(ValueError):
        TemplateFitSampled(np.zeros((1, N_BAND, 100)))
