/// Multi-criterion ascent (MCA) period finder, CPU.
///
/// A period finder for sparse multi-night, multiband light curves. Walkers start
/// at candidate frequencies (for example the peaks of other periodograms) and
/// climb one score, the sum of four terms, each z-scored over 2048 evenly spaced
/// frequencies of the object's own grid and signed so that larger is better:
///
///   power  explained variance of a two-harmonic shape shared by all bands, with a
///          free gain per band (alternating least squares, robust point weights
///          from one bootstrap fit at the strongest grid frequency)
///   sigma  Student-t residual scale of a shared-shape harmonic fit with a free
///          offset and gain per band (fixed harmonic order per object; minimised)
///   pdm    phase dispersion of each band's rank-normalised magnitudes, with von
///          Mises phase weights instead of bins (minimised)
///   loo    prediction error for each night from a fit to the other nights
///          (minimised); this term targets daily aliases
///
/// Each walker moves in x = f T (T the time span) by a Newton-like step from
/// central finite differences (h = 1e-3 / T), at most half a peak width, with
/// backtracking, until the step falls below `tol` peak widths. Walkers that end
/// within half a peak width of a higher one are merged, and the optima are
/// returned by score. This is a direct port of the Python reference
/// implementation used to choose the four terms (combination 89 of nine).
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::fourier::cholesky_solve;

const EPS: f64 = 1e-12;
const NK: usize = 16;
const KAPPA_FOLD: f64 = 10.0;
const N_NORM: usize = 2048;
const N_TERMS: usize = 4;
/// power, sigma, pdm, loo: +1 maximised, -1 minimised.
const SIGN: [f64; N_TERMS] = [1.0, -1.0, -1.0, -1.0];

/// One light curve after filtering, sorting and per-band normalisation.
struct Data {
    t: Vec<f64>,
    y: Vec<f64>,
    rank: Vec<f64>,
    b: Vec<usize>,
    night: Vec<usize>,
    bands: Vec<usize>,
    /// Point indices of each band (indexed by band), built once.
    members: Vec<Vec<usize>>,
    nn: usize,
    span: f64,
}

/// A fixed set of points and the bands present in it, built once per object.
struct Subset {
    idx: Vec<usize>,
    bands: Vec<usize>,
}

impl Subset {
    fn new(b: &[usize], idx: Vec<usize>) -> Subset {
        let mut bands: Vec<usize> = idx.iter().map(|&i| b[i]).collect();
        bands.sort_unstable();
        bands.dedup();
        Subset { idx, bands }
    }
}

/// Reusable per-point workspace for `fit_h`; every entry used is overwritten
/// before it is read.
struct Work {
    w: Vec<f64>,
    residual: Vec<f64>,
    shape: Vec<f64>,
    yc: Vec<f64>,
}

impl Work {
    fn new(n: usize) -> Work {
        Work {
            w: vec![0.0; n],
            residual: vec![0.0; n],
            shape: vec![0.0; n],
            yc: vec![0.0; n],
        }
    }
}

/// The result for one light curve.
#[derive(Debug, Clone)]
pub struct McaResult {
    pub n_starts: usize,
    pub n_optima: usize,
    /// Periods of the optima, best first (days).
    pub periods: Vec<f64>,
    pub scores: Vec<f64>,
}

impl McaResult {
    fn nan() -> McaResult {
        McaResult {
            n_starts: 0,
            n_optima: 0,
            periods: vec![],
            scores: vec![],
        }
    }
}

fn quantile_sorted(s: &[f64], q: f64) -> f64 {
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    s[lo] + (s[hi] - s[lo]) * frac
}

fn setup(t: &[f64], m: &[f64], band: &[u8]) -> Option<Data> {
    let mut idx: Vec<usize> = (0..t.len())
        .filter(|&i| t[i].is_finite() && m[i].is_finite() && (band[i] as usize) < 6)
        .collect();
    if idx.len() < 8 {
        return None;
    }
    idx.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let t0 = t[idx[0]];
    let tt: Vec<f64> = idx.iter().map(|&i| t[i] - t0).collect();
    let span = tt[tt.len() - 1];
    if !(span > 0.0) {
        return None;
    }
    let b: Vec<usize> = idx.iter().map(|&i| band[i] as usize).collect();
    let mut y: Vec<f64> = idx.iter().map(|&i| m[i]).collect();
    let mut rank = vec![0.0; y.len()];
    let mut bands = b.clone();
    bands.sort_unstable();
    bands.dedup();
    for &j in &bands {
        let members: Vec<usize> = (0..y.len()).filter(|&i| b[i] == j).collect();
        let v: Vec<f64> = members.iter().map(|&i| y[i]).collect();
        let mut s = v.clone();
        s.sort_by(f64::total_cmp);
        let med = quantile_sorted(&s, 0.5);
        let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        let scale = (0.5 * (quantile_sorted(&s, 0.95) - quantile_sorted(&s, 0.05)))
            .max(1.4826 * quantile_sorted(&dev, 0.5))
            .max(0.02);
        for &i in &members {
            y[i] = (y[i] - med) / scale;
        }
        // Average ranks of ties, scaled to [-1, 1].
        let n = v.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &c| v[a].total_cmp(&v[c]));
        let mut k = 0;
        while k < n {
            let mut e = k + 1;
            while e < n && v[order[e]] == v[order[k]] {
                e += 1;
            }
            let rr = (k + e - 1) as f64 / 2.0;
            let val = 2.0 * (rr + 0.5) / n as f64 - 1.0;
            for q in k..e {
                rank[members[order[q]]] = val;
            }
            k = e;
        }
    }
    let mut night = vec![0usize; tt.len()];
    for i in 1..tt.len() {
        night[i] = night[i - 1] + usize::from(tt[i] - tt[i - 1] > 0.5);
    }
    let nn = night[night.len() - 1] + 1;
    let mut members = vec![Vec::new(); 6];
    for (i, &j) in b.iter().enumerate() {
        members[j].push(i);
    }
    Some(Data {
        t: tt,
        y,
        rank,
        b,
        night,
        bands,
        members,
        nn,
        span,
    })
}

/// Harmonic basis [cos φ .. cos hφ, sin φ .. sin hφ] at one point.
#[inline]
fn basis(f: f64, t: f64, h: usize, out: &mut [f64; 8]) {
    let ph = 2.0 * PI * f * t;
    for k in 1..=h {
        let (s, c) = (ph * k as f64).sin_cos();
        out[k - 1] = c;
        out[h + k - 1] = s;
    }
}

struct HFit {
    beta: [f64; 8],
    a: [f64; 6],
    mu: [f64; 6],
    sigma: f64,
}

/// Robust (Student-t, 4 dof) shared-shape harmonic fit of order h on the points
/// of `sub`, with a free offset and non-negative gain per band. `x_all` holds the
/// order-h basis of every point at the trial frequency, computed once by the
/// caller; `ws` is workspace. On return `ws.residual[..n]` holds the residuals.
fn fit_h(
    d: &Data,
    x_all: &[[f64; 8]],
    h: usize,
    sub: &Subset,
    iterations: usize,
    ws: &mut Work,
) -> Option<HFit> {
    let idx = &sub.idx;
    let n = idx.len();
    let p = 2 * h;
    let x = |q: usize| &x_all[idx[q]];
    let y = |q: usize| d.y[idx[q]];
    let b = |q: usize| d.b[idx[q]];
    let bands = &sub.bands;

    let mean_y = (0..n).map(y).sum::<f64>() / n as f64;
    let std_y = ((0..n)
        .map(|q| (y(q) - mean_y) * (y(q) - mean_y))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let mut sigma = std_y.max(0.05);
    let w = &mut ws.w[..n];
    w.fill(1.0);
    let residual = &mut ws.residual[..n];
    let shape = &mut ws.shape[..n];
    let yc = &mut ws.yc[..n];
    let mut a = [1.0f64; 6];
    let mut mu = [0.0f64; 6];
    let mut beta = [0.0f64; 8];

    for _ in 0..iterations {
        let mut mx = [[0.0f64; 8]; 6];
        let mut my = [0.0f64; 6];
        for &j in bands.iter() {
            let (mut tot, mut sy) = (0.0, 0.0);
            let mut sx = [0.0f64; 8];
            for q in 0..n {
                if b(q) == j {
                    tot += w[q];
                    sy += w[q] * y(q);
                    for r in 0..p {
                        sx[r] += w[q] * x(q)[r];
                    }
                }
            }
            let tot = tot.max(EPS);
            for r in 0..p {
                mx[j][r] = sx[r] / tot;
            }
            my[j] = sy / tot;
        }
        let mut g = [0.0f64; 64];
        let mut rhs = [0.0f64; 8];
        let mut wsum = 0.0;
        for q in 0..n {
            let j = b(q);
            yc[q] = y(q) - my[j];
            let mut dsg = [0.0f64; 8];
            for r in 0..p {
                dsg[r] = (x(q)[r] - mx[j][r]) * a[j];
            }
            for r in 0..p {
                rhs[r] += dsg[r] * w[q] * yc[q];
                for s in 0..p {
                    g[r * p + s] += w[q] * dsg[r] * dsg[s];
                }
            }
            wsum += w[q];
        }
        let ridge = 1e-7 * wsum.max(1.0);
        for r in 0..p {
            g[r * p + r] += ridge;
        }
        if !cholesky_solve(&mut g[..p * p], &mut rhs[..p], p) {
            return None;
        }
        beta[..p].copy_from_slice(&rhs[..p]);
        for q in 0..n {
            shape[q] = (0..p).map(|r| x(q)[r] * beta[r]).sum();
        }
        for &j in bands.iter() {
            let mxb: f64 = (0..p).map(|r| mx[j][r] * beta[r]).sum();
            let (mut num, mut den) = (0.0, 0.0);
            for q in 0..n {
                if b(q) == j {
                    let dx = shape[q] - mxb;
                    num += w[q] * dx * yc[q];
                    den += w[q] * dx * dx;
                }
            }
            a[j] = (num / den.max(EPS)).max(0.0);
            mu[j] = my[j] - a[j] * mxb;
        }
        let norm = (bands.iter().map(|&j| a[j] * a[j]).sum::<f64>() / bands.len() as f64)
            .sqrt()
            .max(1e-8);
        for &j in bands.iter() {
            a[j] /= norm;
        }
        for r in 0..p {
            beta[r] *= norm;
        }
        for q in 0..n {
            shape[q] *= norm;
            residual[q] = y(q) - mu[b(q)] - a[b(q)] * shape[q];
        }
        for _ in 0..4 {
            let s2 = (sigma * sigma).max(1e-8);
            let m: f64 = residual
                .iter()
                .map(|r| 5.0 * r * r / (4.0 + r * r / s2))
                .sum::<f64>()
                / n as f64;
            sigma = m.sqrt().max(1e-4);
        }
        for q in 0..n {
            w[q] = 5.0 / (4.0 + residual[q] * residual[q] / (sigma * sigma));
        }
    }
    Some(HFit { beta, a, mu, sigma })
}

/// The order-h basis of every point at frequency f.
fn basis_all(d: &Data, f: f64, h: usize) -> Vec<[f64; 8]> {
    let mut x = vec![[0.0f64; 8]; d.t.len()];
    for (i, xi) in x.iter_mut().enumerate() {
        basis(f, d.t[i], h, xi);
    }
    x
}

/// Explained variance of a two-harmonic shape shared by all bands with a free
/// non-negative gain per band, fitted by four rounds of alternating least
/// squares with fixed point weights `w`.
fn power_term(d: &Data, f: f64, w: &[f64]) -> f64 {
    let n = d.t.len();
    // Basis order {cos φ, cos 2φ, sin φ, sin 2φ}.
    let mut xc = vec![[0.0f64; 4]; n];
    for q in 0..n {
        let mut u = [0.0f64; 8];
        basis(f, d.t[q], 2, &mut u);
        xc[q] = [u[0], u[1], u[2], u[3]];
    }
    let mut yc = d.y.clone();
    for &j in &d.bands {
        let (mut tot, mut sy) = (0.0, 0.0);
        let mut sx = [0.0f64; 4];
        for q in 0..n {
            if d.b[q] == j {
                tot += w[q];
                sy += w[q] * yc[q];
                for r in 0..4 {
                    sx[r] += w[q] * xc[q][r];
                }
            }
        }
        for q in 0..n {
            if d.b[q] == j {
                for r in 0..4 {
                    xc[q][r] -= sx[r] / tot;
                }
                yc[q] -= sy / tot;
            }
        }
    }
    let wsum: f64 = w.iter().sum();
    let mut a = [1.0f64; 6];
    let mut beta = [0.0f64; 4];
    let mut wave = vec![0.0f64; n];
    for _ in 0..4 {
        let mut g = [0.0f64; 16];
        let mut rhs = [0.0f64; 4];
        for q in 0..n {
            let aj = a[d.b[q]];
            let dsg = [xc[q][0] * aj, xc[q][1] * aj, xc[q][2] * aj, xc[q][3] * aj];
            for r in 0..4 {
                rhs[r] += dsg[r] * w[q] * yc[q];
                for s in 0..4 {
                    g[r * 4 + s] += w[q] * dsg[r] * dsg[s];
                }
            }
        }
        for r in 0..4 {
            g[r * 4 + r] += 1e-7 * wsum.max(1.0);
        }
        if !cholesky_solve(&mut g, &mut rhs, 4) {
            return f64::NAN;
        }
        beta = rhs;
        for q in 0..n {
            wave[q] = (0..4).map(|r| xc[q][r] * beta[r]).sum();
        }
        for &j in &d.bands {
            let (mut num, mut den) = (0.0, 0.0);
            for q in 0..n {
                if d.b[q] == j {
                    num += w[q] * wave[q] * yc[q];
                    den += w[q] * wave[q] * wave[q];
                }
            }
            let ratio = num / (den + 1e-8);
            a[j] = 0.5 * (ratio + (ratio * ratio + 1e-8).sqrt());
        }
        let norm = (d.bands.iter().map(|&j| a[j] * a[j]).sum::<f64>() / d.bands.len() as f64
            + 1e-16)
            .sqrt();
        for &j in &d.bands {
            a[j] /= norm;
        }
        for r in 0..4 {
            beta[r] *= norm;
        }
    }
    let (mut sse, mut tot) = (0.0, 0.0);
    for q in 0..n {
        let wv: f64 = (0..4).map(|r| xc[q][r] * beta[r]).sum::<f64>() * a[d.b[q]];
        sse += w[q] * (yc[q] - wv) * (yc[q] - wv);
        tot += w[q] * yc[q] * yc[q];
    }
    1.0 - sse / (tot + EPS)
}

/// Fourier coefficients of the von Mises phase kernel, K(x) = c0 + 2 sum c_k cos kx.
fn kernel_coefficients() -> ([f64; NK + 1], f64) {
    let mut ck = [0.0f64; NK + 1];
    let n = 1024;
    for k in 0..=NK {
        let mut s = 0.0;
        for i in 0..n {
            let th = 2.0 * PI * i as f64 / n as f64;
            s += (KAPPA_FOLD * (th.cos() - 1.0)).exp() * (k as f64 * th).cos();
        }
        ck[k] = s / n as f64;
    }
    let k0 = ck[0] + 2.0 * ck[1..].iter().sum::<f64>();
    (ck, k0)
}

/// Kernel phase dispersion of the rank-normalised magnitudes, averaged over
/// bands with at least 10 points; NaN if there are none.
fn pdm_term(d: &Data, f: f64, ck: &[f64; NK + 1], k0: f64) -> f64 {
    let (mut total, mut nb) = (0.0, 0usize);
    for &j in &d.bands {
        let members = &d.members[j];
        let n = members.len();
        if n < 10 {
            continue;
        }
        let v: Vec<f64> = members.iter().map(|&i| d.rank[i]).collect();
        let mut cs = vec![[0.0f64; 2 * NK]; n];
        let (mut c1, mut s1, mut cv, mut sv) =
            ([0.0f64; NK], [0.0f64; NK], [0.0f64; NK], [0.0f64; NK]);
        for (q, &i) in members.iter().enumerate() {
            let ph = f * d.t[i];
            for k in 1..=NK {
                let (s, c) = (2.0 * PI * ph * k as f64).sin_cos();
                cs[q][k - 1] = c;
                cs[q][NK + k - 1] = s;
                c1[k - 1] += c;
                s1[k - 1] += s;
                cv[k - 1] += c * v[q];
                sv[k - 1] += s * v[q];
            }
        }
        let sum_v: f64 = v.iter().sum();
        let mean_v = sum_v / n as f64;
        let var_v = v.iter().map(|x| (x - mean_v) * (x - mean_v)).sum::<f64>() / n as f64;
        let mut acc = 0.0;
        for q in 0..n {
            let (mut s0, mut sv1) = (ck[0] * n as f64, ck[0] * sum_v);
            let (mut t0, mut t1) = (0.0, 0.0);
            for k in 0..NK {
                let (c, s) = (cs[q][k], cs[q][NK + k]);
                t0 += ck[k + 1] * (c * c1[k] + s * s1[k]);
                t1 += ck[k + 1] * (c * cv[k] + s * sv[k]);
            }
            s0 += 2.0 * t0 - k0;
            sv1 += 2.0 * t1 - k0 * v[q];
            let mean = sv1 / s0.max(EPS);
            acc += (v[q] - mean) * (v[q] - mean);
        }
        total += acc / n as f64 / var_v.max(EPS);
        nb += 1;
    }
    if nb == 0 {
        f64::NAN
    } else {
        total / nb as f64
    }
}

/// Mean over nights of the robust prediction error for that night from a fit
/// to the other nights.
fn loo_term(
    d: &Data,
    x_all: &[[f64; 8]],
    h: usize,
    nights: &[(Subset, Vec<usize>)],
    ws: &mut Work,
) -> f64 {
    if nights.is_empty() {
        return f64::NAN;
    }
    let mut err = 0.0;
    for (train, test) in nights {
        let Some(fit) = fit_h(d, x_all, h, train, 2, ws) else {
            return f64::NAN;
        };
        let mut acc = 0.0;
        for &i in test {
            let u = &x_all[i];
            let shape: f64 = (0..2 * h).map(|r| u[r] * fit.beta[r]).sum();
            let r = d.y[i] - (fit.mu[d.b[i]] + fit.a[d.b[i]] * shape);
            acc += (r * r + 1e-4).sqrt() - 0.01;
        }
        err += acc / test.len() as f64;
    }
    err / nights.len() as f64
}

struct Objective {
    d: Data,
    w: Vec<f64>,
    h: usize,
    all: Subset,
    nights: Vec<(Subset, Vec<usize>)>,
    ck: [f64; NK + 1],
    k0: f64,
    location: [f64; N_TERMS],
    scale: [f64; N_TERMS],
    used: [bool; N_TERMS],
}

impl Objective {
    fn new(d: Data, grid: &[f64]) -> Option<Objective> {
        let n = d.t.len();
        // Bootstrap: the strongest equal-weight two-harmonic frequency on the grid
        // sets fixed robust point weights for the power term.
        let ones = vec![1.0f64; n];
        let ev: Vec<f64> = grid.par_iter().map(|&f| power_term(&d, f, &ones)).collect();
        let mut first = 0usize;
        for (i, v) in ev.iter().enumerate() {
            if *v > ev[first] || (ev[first].is_nan() && !v.is_nan()) {
                first = i;
            }
        }
        let all = Subset::new(&d.b, (0..n).collect());
        let mut ws = Work::new(n);
        let x2 = basis_all(&d, grid[first], 2);
        let init = fit_h(&d, &x2, 2, &all, 5, &mut ws)?;
        let s = init.sigma.max(1e-4);
        let w: Vec<f64> = ws.residual[..n]
            .iter()
            .map(|r| 5.0 / (4.0 + (r / s) * (r / s)))
            .collect();

        let nb = d.bands.len();
        let h = (1..=4)
            .rev()
            .find(|&h| n > 2 * (2 * h + 2 * nb + 2))
            .unwrap_or(1);
        let mut nights = Vec::new();
        for night in 0..d.nn {
            let train: Vec<usize> = (0..n).filter(|&i| d.night[i] != night).collect();
            if train.len() < 8 {
                continue;
            }
            let mut seen = [false; 6];
            for &i in &train {
                seen[d.b[i]] = true;
            }
            let test: Vec<usize> = (0..n)
                .filter(|&i| d.night[i] == night && seen[d.b[i]])
                .collect();
            if !test.is_empty() {
                nights.push((Subset::new(&d.b, train), test));
            }
        }
        let (ck, k0) = kernel_coefficients();
        let mut obj = Objective {
            d,
            w,
            h,
            all,
            nights,
            ck,
            k0,
            location: [0.0; N_TERMS],
            scale: [1.0; N_TERMS],
            used: [false; N_TERMS],
        };

        // Location and scale of each term over evenly spaced grid frequencies.
        let m = N_NORM.min(grid.len());
        let mut pick: Vec<usize> = (0..m)
            .map(|i| {
                if m == 1 {
                    0
                } else {
                    (i as f64 * (grid.len() - 1) as f64 / (m - 1) as f64) as usize
                }
            })
            .collect();
        pick.dedup();
        let vals: Vec<[f64; N_TERMS]> = pick.par_iter().map(|&i| obj.components(grid[i])).collect();
        for j in 0..N_TERMS {
            let col: Vec<f64> = vals.iter().map(|v| v[j]).collect();
            if col.iter().all(|v| v.is_finite()) {
                let mean = col.iter().sum::<f64>() / col.len() as f64;
                let sd = (col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
                    / col.len() as f64)
                    .sqrt();
                obj.used[j] = true;
                obj.location[j] = mean;
                obj.scale[j] = if sd > 1e-10 { sd } else { 1.0 };
            }
        }
        if !obj.used.iter().any(|&u| u) {
            return None;
        }
        Some(obj)
    }

    /// The four terms at frequency f, signed so that larger is better.
    fn components(&self, f: f64) -> [f64; N_TERMS] {
        let power = power_term(&self.d, f, &self.w);
        // Each point's order-h basis at f is the same for the full fit and for
        // every leave-one-night-out fit, so it is computed once here.
        let x_all = basis_all(&self.d, f, self.h);
        let mut ws = Work::new(self.d.t.len());
        let sigma = fit_h(&self.d, &x_all, self.h, &self.all, 3, &mut ws)
            .map_or(f64::NAN, |fit| (fit.sigma + EPS).ln());
        let pdm = pdm_term(&self.d, f, &self.ck, self.k0);
        let loo = loo_term(&self.d, &x_all, self.h, &self.nights, &mut ws);
        let raw = [power, sigma, pdm, loo];
        let mut out = [0.0; N_TERMS];
        for j in 0..N_TERMS {
            out[j] = raw[j] * SIGN[j];
        }
        out
    }

    fn score(&self, f: f64) -> f64 {
        let c = self.components(f);
        let mut s = 0.0;
        for j in 0..N_TERMS {
            if self.used[j] {
                let z = (c[j] - self.location[j]) / self.scale[j];
                s += if z.is_nan() {
                    0.0
                } else if z.is_infinite() {
                    z.signum() * f64::MAX
                } else {
                    z
                };
            }
        }
        s
    }
}

/// One walker in x = f T; returns its final x and score.
fn ascend(
    obj: &Objective,
    start: f64,
    t_span: f64,
    lo: f64,
    hi: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64) {
    let (low, high) = (lo * t_span, hi * t_span);
    let mut x = (start * t_span).clamp(low, high);
    let mut score = obj.score(x / t_span);
    let delta = 1e-3;
    for _ in 0..max_iter {
        let plus = (x + delta).min(high);
        let minus = (x - delta).max(low);
        let sp = obj.score(plus / t_span);
        let sm = obj.score(minus / t_span);
        let (hp, hm) = (plus - x, x - minus);
        let right = if hp > 0.0 { (sp - score) / hp } else { 0.0 };
        let left = if hm > 0.0 { (score - sm) / hm } else { 0.0 };
        let both = hp > 0.0 && hm > 0.0;
        let g = if hm == 0.0 {
            right
        } else if hp == 0.0 {
            left
        } else {
            (hm * right + hp * left) / (hp + hm).max(EPS)
        };
        let curvature = if both {
            2.0 * (right - left) / (hp + hm).max(EPS)
        } else {
            0.0
        };
        if (g * t_span).abs() < 1e-10 && curvature <= 1e-8 {
            break;
        }
        let mut step = (g / curvature.abs().max(1e-3)).clamp(-0.5, 0.5);
        if (g * t_span).abs() < 1e-10 && curvature > 1e-8 {
            step = if sp >= sm { 0.05 } else { -0.05 };
        }
        let mut moved = false;
        loop {
            let trial = (x + step).clamp(low, high);
            if (trial - x).abs() < tol {
                break;
            }
            let cand = obj.score(trial / t_span);
            if cand >= score + 1e-4 * g * (trial - x) {
                let disp = trial - x;
                x = trial;
                score = cand;
                moved = disp.abs() >= tol;
                break;
            }
            step *= 0.5;
        }
        if !moved {
            break;
        }
    }
    (x / t_span, score)
}

/// Score-ordered merge; kept optima remain more than half a peak width apart.
fn merge(freq: &[f64], score: &[f64], t_span: f64) -> Vec<usize> {
    let radius = 0.5 / t_span;
    let mut order: Vec<usize> = (0..freq.len()).collect();
    order.sort_by(|&a, &b| {
        score[b]
            .total_cmp(&score[a])
            .then(freq[a].total_cmp(&freq[b]))
    });
    let mut bins: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        let cell = (freq[i] / radius).floor() as i64;
        let mut close = false;
        for c in [cell - 1, cell, cell + 1] {
            if let Some(&kk) = bins.get(&c) {
                if (freq[i] - freq[keep[kk]]).abs() <= radius {
                    close = true;
                }
            }
        }
        if !close {
            bins.insert(cell, keep.len());
            keep.push(i);
        }
    }
    keep
}

/// Run MCA on one light curve from the given candidate periods.
#[allow(clippy::too_many_arguments)]
pub fn mca_one(
    times: &[f64],
    mags: &[f64],
    bands: &[u8],
    start_periods: &[f64],
    f_max: f64,
    oversample: f64,
    tol: f64,
    max_iter: usize,
    top: usize,
) -> McaResult {
    let Some(d) = setup(times, mags, bands) else {
        return McaResult::nan();
    };
    let t_span = d.span;
    let lo = 1.0 / t_span;
    if f_max < lo {
        return McaResult::nan();
    }
    let n_grid = ((f_max - lo) * oversample * t_span).floor() as usize + 1;
    let grid: Vec<f64> = (0..n_grid)
        .map(|i| lo + i as f64 / (oversample * t_span))
        .collect();
    let Some(obj) = Objective::new(d, &grid) else {
        return McaResult::nan();
    };
    let mut starts: Vec<f64> = start_periods
        .iter()
        .filter(|p| p.is_finite() && **p > 0.0)
        .map(|p| 1.0 / p)
        .filter(|f| *f >= lo && *f <= f_max)
        .collect();
    starts.sort_by(f64::total_cmp);
    starts.dedup();
    if starts.is_empty() {
        return McaResult::nan();
    }
    let walked: Vec<(f64, f64)> = starts
        .par_iter()
        .map(|&f| ascend(&obj, f, t_span, lo, f_max, tol, max_iter))
        .collect();
    let freq: Vec<f64> = walked.iter().map(|w| w.0).collect();
    let score: Vec<f64> = walked.iter().map(|w| w.1).collect();
    let keep = merge(&freq, &score, t_span);
    let mut order = keep.clone();
    order.sort_by(|&a, &b| {
        score[b]
            .total_cmp(&score[a])
            .then(freq[a].total_cmp(&freq[b]))
    });
    let take = order.len().min(top);
    McaResult {
        n_starts: starts.len(),
        n_optima: keep.len(),
        periods: order[..take].iter().map(|&i| 1.0 / freq[i]).collect(),
        scores: order[..take].iter().map(|&i| score[i]).collect(),
    }
}

/// Run MCA on a batch of light curves, one object at a time (walkers run in
/// parallel within an object).
#[allow(clippy::too_many_arguments)]
pub fn mca_batched(
    times: &[Vec<f64>],
    mags: &[Vec<f64>],
    bands: &[Vec<u8>],
    start_periods: &[Vec<f64>],
    f_max: f64,
    oversample: f64,
    tol: f64,
    max_iter: usize,
    top: usize,
) -> Vec<McaResult> {
    (0..times.len())
        .map(|i| {
            mca_one(
                &times[i],
                &mags[i],
                &bands[i],
                &start_periods[i],
                f_max,
                oversample,
                tol,
                max_iter,
                top,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sparse multiband sinusoid: 14 nights, 1-3 bands a night, a few points each.
    fn curve(period: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut t, mut m, mut b) = (vec![], vec![], vec![]);
        for night in 0..14 {
            let base = 60000.0 + (night * 4) as f64 + (u() * 2.0).floor();
            for band in 0..3u8 {
                if u() < 0.6 {
                    for _ in 0..6 {
                        let tt = base + 0.1 + 0.25 * u();
                        t.push(tt);
                        b.push(band);
                        let noise = 0.03 * (u() - 0.5);
                        m.push(
                            18.0 + band as f64 * 0.3 + 0.3 * (2.0 * PI * tt / period).sin() + noise,
                        );
                    }
                }
            }
        }
        (t, m, b)
    }

    #[test]
    fn finds_sinusoid_from_nearby_start() {
        let period = 0.31;
        let (t, m, b) = curve(period, 1);
        let r = mca_one(
            &t,
            &m,
            &b,
            &[period * 1.003, 0.21, 0.47],
            40.0,
            5.0,
            1e-3,
            10000,
            5,
        );
        assert!(r.n_optima >= 1);
        assert!(
            (r.periods[0] / period - 1.0).abs() < 1e-3,
            "best {}",
            r.periods[0]
        );
    }

    #[test]
    fn too_few_points_is_empty() {
        let r = mca_one(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[0, 0, 0],
            &[0.5],
            10.0,
            5.0,
            1e-3,
            100,
            5,
        );
        assert_eq!(r.n_optima, 0);
        assert!(r.periods.is_empty());
    }

    #[test]
    fn duplicate_starts_are_merged() {
        let (t, m, b) = curve(0.31, 2);
        let r = mca_one(&t, &m, &b, &[0.31, 0.31, 0.31], 40.0, 5.0, 1e-3, 10000, 5);
        assert_eq!(r.n_starts, 1);
    }
}
