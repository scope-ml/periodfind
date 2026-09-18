# Template banks

`TemplateFitSampled` does not ship a bank. It takes one as an argument, because
a bank is a scientific choice — which physics, which parameter grid, which
passbands — and that choice belongs to the analysis rather than to the library.

This page describes what the fitter expects and how to build it.

## What a bank is

A single array:

```
samples.shape == (n_template, n_band, n_phase)
```

`samples[k, b, p]` is template `k`'s magnitude in band `b` at phase
`p / n_phase`, for phase running from 0 to 1 over one cycle. `float64`.

Three constraints:

- **`n_phase` must be a power of two.** The transform is radix-2. 512 is a
  reasonable default; 1024 or 2048 if your templates have features narrower
  than about a hundredth of a cycle.
- **`n_band` must match the band codes in your light curves.** Band `b` in the
  bank is band index `b` in the `bands` array you pass to `calc`.
- **Every template needs the same grid.** A bank is one array, so a bank of
  mixed resolutions has to be resampled onto a common grid first.

Two things the fitter does *not* care about:

- **The zero point.** An offset is fitted per band, so adding a constant to a
  template changes nothing. Subtracting each band's mean is tidy but optional.
- **The overall scale.** One amplitude is fitted, shared across bands, so
  scaling a whole template by any positive factor changes nothing.

What it *does* care about is the **ratio between bands**. Because the amplitude
is shared, a template that is twice as deep in `u` as in `r` asserts that the
star behaves that way, and a light curve that disagrees fits worse. This is
what makes colour a constraint rather than a free parameter per band, and it
means the bands of a template must be generated with a consistent set of
passbands.

## Building one

Any model that produces a light curve will do. The recipe is the same
regardless:

1. Generate one cycle of the model in every band, at whatever phase sampling
   the model naturally produces.
2. Convert to magnitudes if the model produces flux.
3. Resample onto a common phase grid of `n_phase` points.
4. Stack the templates into `(n_template, n_band, n_phase)`.

```python
import numpy as np
import periodfind

n_phase = 512
grid = np.arange(n_phase) / n_phase

samples = np.empty((n_template, n_band, n_phase))
for k, model in enumerate(models):
    for b in range(n_band):
        phase, mag = model.curve(band=b)        # one cycle, any sampling
        samples[k, b] = np.interp(grid, phase, mag, period=1.0)

tfs = periodfind.TemplateFitSampled(samples=samples)
out = tfs.calc(times, mags, bands, periods)
```

`out` has one row per curve: `fvu, period, template, shift, amplitude`. The
shift is an index into the phase grid, so the fitted phase is
`shift / n_phase` of a cycle.

## Choosing `n_phase`

The grid sets the finest feature a template can hold, and it is also the phase
resolution of the fit, since every grid point is an available shift.

It is worth checking what your own templates lose. Take a template at its
native resolution, resample it to the grid, and compare:

```python
loss = np.abs(native - np.interp(grid, native_phase, native)).max()
```

For smooth curves — pulsators, contact binaries — this is negligible at any
reasonable grid. For a narrow eclipse it is not: the light curve is flat for
most of the cycle and the information is concentrated in a small fraction of
it, so the grid has to resolve the eclipse rather than the cycle.

The light curve is binned onto the same grid, which is the one approximation
the method makes: two points in the same bin are treated as having the same
phase. A bin is `1 / n_phase` of a cycle, so at 512 that is 0.002.

## Choosing trial periods

Each curve brings its own periods, because in practice they are a periodogram's
top candidates rather than a shared grid. This is a re-ranking: **a period the
period search never proposed cannot be chosen**, so the candidate list bounds
the result no matter how good the bank is. If recovery is poor, check whether
the true period is in the list before blaming the templates.

## A practical note on bank size

Cost is linear in the number of templates: roughly 0.1 ms per template per
object on an A100, whatever the class. A bank of a few dozen templates fits a
light curve in milliseconds; a bank of a thousand takes a tenth of a second.

Fitting an object against every template of every class is usually the wrong
thing to do. The largest class then wins by volume rather than by physics —
if one class holds half the templates, it will win about half the objects. The
useful arrangement is to classify first and fit each object against the bank of
its own class.
