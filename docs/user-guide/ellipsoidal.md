# Ellipsoidal variables

An ellipsoidal variable is a binary in which the visible star is tidally
distorted by its companion, so its projected area and surface brightness change
twice per orbit. periodfind fits these in two ways: `EllipsoidalFit` fits the
BEER model directly, with amplitudes free per band; `TemplateFitSampled` fits a
bank of computed light curves, with the band ratios fixed by the model.

## Model

To leading order in the fractional radius $R/a$, the flux follows the BEER form
(Morris & Naftilan 1993; Faigler & Mazeh 2011),

$$\frac{\Delta F}{F} = -A_\mathrm{ellip} \cos 2\phi - A_\mathrm{refl} \cos \phi + A_\mathrm{beam} \sin \phi, \qquad \phi = \frac{2\pi (t - T_0)}{P_\mathrm{orb}},$$

where $T_0$ is the conjunction at which the companion lies between the visible
star and the observer. The ellipsoidal amplitude is
$A_\mathrm{ellip} \approx \alpha_\mathrm{ellip}\, q\, (R/a)^3 \sin^2 i$, with
$\alpha_\mathrm{ellip} = 0.15 (15 + u)(1 + \tau)/(3 - u)$ depending on the band
through the limb-darkening coefficient $u$ and the gravity-darkening coefficient
$\tau$; the beaming amplitude is $A_\mathrm{beam} = 4 \alpha_\mathrm{beam} K/c$;
and $A_\mathrm{refl}$ measures the light reflected from the irradiated face of
the companion. Terms of order $(R/a)^4$ add further $\cos\phi$ and $\cos 3\phi$
components that make the two minima unequal, while beaming makes the two maxima
unequal.

## Period

The second harmonic dominates the signal, so Lomb-Scargle peaks at
$P_\mathrm{orb}/2$, and the multi-harmonic Fourier periodogram, which selects
the number of harmonics by BIC at each trial period, usually prefers
$P_\mathrm{orb}/2$ with a single harmonic. A Fourier series at $2P$ with
harmonics 1 to $2K$ contains the series at $P$ with harmonics 1 to $K$ as its
even terms, so the fit at the doubled period always fits at least as well, and
the improvement measures the odd harmonics alone. The case for an ellipsoidal
classification therefore rests on detecting those odd harmonics, which are the
unequal minima and maxima; a light curve with a pure $\cos 2\phi$ shape is
equally well described as a sinusoid at $P_\mathrm{orb}/2$, and radial
velocities are needed to separate the two.

## EllipsoidalFit

`EllipsoidalFit` fits the BEER model to each band in flux relative to the band's
weighted mean magnitude, with $T_0$ shared by all bands and
$(A_\mathrm{ellip}, A_\mathrm{beam}, A_\mathrm{refl})$ free per band. At fixed
period and $T_0$ the model is linear in the amplitudes, and its basis is a
rotation of $\{1, \cos\omega t, \sin\omega t, \cos 2\omega t, \sin 2\omega t\}$,
so each band's weighted Gram matrix is built once per period and every trial
$T_0$ costs a 4×4 solve. The fitted subspace repeats when $T_0$ moves by a
quarter cycle, so $T_0$ is scanned over $[0, P/4)$ and refined by a parabola;
of the four equivalent values, the fit reports the one with positive
$A_\mathrm{ellip}$ and positive $A_\mathrm{beam}$, pooled over bands. The
ellipsoidal and reflection terms are symmetric about conjunction, so beaming
alone sets the choice between $T_0$ and $T_0 + P/2$, and that choice carries no
information when beaming is undetected.

With `double_periods=True` (the default) each trial period is taken as a
periodogram peak at $P_\mathrm{orb}/2$ and the fit runs at twice it; the
candidate with the lowest $\chi^2$ is reported. `delta_bic` compares the BEER
fit with a sinusoid at $P_\mathrm{orb}/2$ that has a free phase per band, so it
is positive when the odd harmonics are detected and negative for a pure
$\cos 2\phi$ light curve.

```python
import numpy as np
import periodfind

# peaks: the top periodogram peaks for each curve, in days
fit = periodfind.EllipsoidalFit(n_band=3)
out = fit.calc(times, mags, errs, bands, peaks)
res = {name: out[:, j] for j, name in enumerate(fit.columns)}

candidates = res["delta_bic"] > 10
print(res["period"][candidates], res["A_ellip_1"][candidates])
```

The output has one row per curve: `period`, `t0`, `chi2`, `delta_bic`,
`delta_bic_flat`, and `n_points`, followed by `A_ellip`, `A_ellip_err`,
`A_beam`, `A_beam_err`, `A_refl`, and `A_refl_err` for each band, named in
`fit.columns` with the band index as a suffix. The errors are the formal
least-squares errors at the fitted $T_0$. The ratio of $A_\mathrm{ellip}$ across
bands should follow $\alpha_\mathrm{ellip}(u, \tau)$ for the star's temperature
and gravity, which tests the ellipsoidal interpretation independently of
`delta_bic`; the threshold on `delta_bic` should be set on simulated light
curves with the survey's cadence and noise.

## Template banks

`TemplateFitSampled` fits
$m_b(t) = \mathrm{offset}_b + a\, S_{kb}((t/P - \phi) \bmod 1)$ with a free
offset per band and one amplitude shared across bands (see
[Template Banks](template-banks.md)). For ellipsoidal variables this adds two
constraints to `EllipsoidalFit`: the shared amplitude ties the band ratio of
$A_\mathrm{ellip}$ to the limb and gravity darkening in the template; and a
template computed from a physical model fixes the odd harmonics relative to the
second, which resolves the inclination dependence of the minima and the
half-orbit ambiguity in $T_0$ within the model.

A physical bank comes from ellc (Maxted 2016) or PHOEBE 2 (Prša et al. 2016),
computed over a grid in $q$, $i$, and Roche-lobe filling factor in each band,
converted to magnitudes, and resampled onto the phase grid with phase zero at
the conjunction defined above. For detached systems with small amplitudes, the
BEER expression itself gives a cheaper bank:

```python
import numpy as np
import periodfind


def beer_bank(alpha, refl, beam, n_phase=512):
    """Small-amplitude BEER templates in magnitudes, phase zero at conjunction.

    alpha: A_ellip in each band relative to a reference band, shape (n_band,)
    refl, beam: A_refl / A_ellip and A_beam / A_ellip for each template
    """
    phi = 2 * np.pi * np.arange(n_phase) / n_phase
    k = 2.5 / np.log(10)
    bank = np.empty((len(refl), len(alpha), n_phase))
    for j, (r, s) in enumerate(zip(refl, beam)):
        shape = np.cos(2 * phi) + r * np.cos(phi) - s * np.sin(phi)
        bank[j] = k * np.outer(alpha, shape)
    return bank


r, s = np.meshgrid([0.0, 0.1, 0.2], [0.0, 0.05, 0.1])
bank = beer_bank(alpha, r.ravel(), s.ravel())

tfs = periodfind.TemplateFitSampled(samples=bank)
# both the peak and twice the peak, since only listed periods are tried
trial = [np.array([p, 2 * p]) for p in peaks]
fvu, period, template, shift, amplitude = tfs.calc(times, mags, bands, trial).T
t0 = shift / bank.shape[-1] * period
```

Here `alpha` holds $\alpha_\mathrm{ellip}$ in each band, normalised to one band,
for the temperature and gravity of the stars being fitted. Three properties of
the implementation apply to any ellipsoidal bank: (i) the amplitude is clamped
to $a \ge 0$, so each template must carry its physical sign in magnitudes, with
brighter phases more negative; (ii) the fit folds on $t/P$, so the returned shift
gives $T_0 = (\mathrm{shift}/n_\mathrm{phase})\, P$, modulo $P$, in the time
system of the input; (iii) the score is the pooled fraction of variance
unexplained with no per-point weights, so outlying and low-quality points should
be removed beforehand.

The bank measures parameters for objects already classified as ellipsoidal,
following the advice in [Template Banks](template-banks.md) to fit each object
against the bank of its own class. $A_\mathrm{ellip}$ constrains the product
$q (R/a)^3 \sin^2 i$, so the best template sits somewhere along a degenerate
valley in a physical grid, and separate values of $q$ and $i$ require the
scores of the other templates or radial velocities.

## References

- Faigler, S., & Mazeh, T. 2011, MNRAS, 415, 3921
- Maxted, P. F. L. 2016, A&A, 591, A111
- Morris, S. L., & Naftilan, S. A. 1993, ApJ, 419, 344
- Prša, A., et al. 2016, ApJS, 227, 29
