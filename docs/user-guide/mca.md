# Multi-criterion ascent (MCA)

MCA finds periods in sparse, multi-night, multiband light curves, where a
single periodogram often ranks an alias, a harmonic or a noise peak above the
true period. It does not compute a periodogram of its own. It starts from
candidate periods, for example the top peaks of other periodograms, and moves
each one uphill on a score built from four complementary terms, so that the
candidates end on the nearest optimum of that score and can be compared on one
scale.

## Data

Each band is centred on its median and divided by a robust spread, the larger
of half the 5–95% range, 1.4826 times the median absolute deviation, and
0.02 mag, so that bands of different depth and amplitude can share one shape.
The ranks of each band's magnitudes, scaled to [-1, 1], are kept for the phase
dispersion term. Points are grouped into nights wherever the gap between
consecutive points exceeds 0.5 d. Photometric errors are not used.

## Score

Every term is evaluated at a trial frequency $f$ and signed so that larger is
better:

- **power** — the explained variance of a two-harmonic shape shared by all
  bands, with a free gain and offset per band, fitted by alternating least
  squares. Point weights are fixed once per object by a robust Student-t fit at
  the strongest equal-weight two-harmonic frequency on the grid, so isolated
  outliers do not set the period.
- **sigma** — the log of the Student-t residual scale of a shared-shape harmonic
  fit with a free offset and gain per band (minimised). The harmonic order is
  fixed per object, the largest $h \le 4$ with
  $n > 2(2h + 2n_\mathrm{band} + 2)$.
- **pdm** — the phase dispersion of each band's rank-normalised magnitudes, with
  von Mises phase weights ($\kappa = 10$) in place of bins, so the term is
  smooth in $f$ (minimised).
- **loo** — the error predicting each night from the same harmonic model fitted
  to the other nights (minimised). A daily alias $f \pm 1$ c/d fits the
  nights it was fitted to almost as well as the true frequency, but predicts a
  left-out night worse, so this term targets daily aliases.

Each term is z-scored with the mean and standard deviation of its values at
2048 evenly spaced frequencies of the object's own grid, from $1/T$ to `f_max`
in steps of $1/(\mathrm{oversample}\, T)$, where $T$ is the time span. The
score is the sum of the four z-scores. The four terms were chosen from nine
candidates by testing every subset, and equal weights performed as well as any
weighted combination on held-out sky fields.

## Walk

Each candidate period starts one walker. The walker moves in $x = fT$, in units
of one peak width, with a Newton-like step from central finite differences
($\Delta x = 10^{-3}$), capped at half a peak width, and a backtracking line
search that accepts a step only if it raises the score. It stops when the step
falls below `tol` peak widths or after `max_iter` steps. Walkers that end within
half a peak width of a higher-scoring one are merged, and the remaining optima
are returned by score.

## Usage

```python
import periodfind

# One entry per light curve: times (days), magnitudes, band index 0-5,
# and that curve's candidate periods (days), e.g. the top peaks of
# Lomb-Scargle, conditional entropy and AOV.
mca = periodfind.MultiCriterionAscent(top=50)          # device from set_device()
out = mca.calc(times, mags, bands, periods)
best = out[:, mca.columns.index("period_1")]
```

The result has one row per light curve: `n_starts` (distinct candidates used),
`n_optima` (after merging), then the `top` best periods and their scores, named
by `mca.columns` and padded with NaN. A curve with fewer than 8 usable points,
no time span, or no candidate inside $[1/f_\mathrm{max}, T]$ returns a row of
NaN.

`periodfind.cpu.MultiCriterionAscent` runs the walkers of each curve in
parallel on the CPU. `periodfind.gpu.MultiCriterionAscent` runs many curves at
once on the GPU with the same arithmetic in double precision; its periods agree
with the CPU's except where the last digits of the trigonometric functions move
a walker by one step (at most $10^{-3}$ peak widths).

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `f_max` | 288.0 | Highest frequency in cycles per day (288 = 5 minutes) |
| `oversample` | 5 | Grid points per peak width for the bootstrap fit and the z-scoring |
| `tol` | 1e-3 | A walker stops when its step is below this many peak widths |
| `max_iter` | 10000 | Maximum steps per walker |
| `top` | 50 | Number of optima returned per light curve |

## Limitations

MCA can only end on optima that some walker reaches, so the candidate periods
should include the true period's neighbourhood; the top peaks of several
periodograms usually do. When the time span covers only a few cycles of an
integer multiple of the true period, the multiple fits the data as well as the
true period, and the two can be ranked in either order.
