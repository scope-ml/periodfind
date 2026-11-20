// Copyright 2026. Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

// Multi-criterion ascent (MCA) period finder, CUDA. A line-for-line port of
// periodfind's rust/src/mca.rs (same terms, same order of operations), with
// three kernels: the equal-weight power term over each object's full grid
// (bootstrap), the four terms at the 2048 scaling frequencies, and one thread
// per walker for the ascent. Host code does the per-object setup, the
// bootstrap fit, the term scaling and the merge, using the same functions.
//
// Output layout matches periodfind's CPU MultiCriterionAscent: per light curve
// n_starts, n_optima, then the `top` best periods and their scores (NaN padded).
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "mca.h"

#define HD __host__ __device__
#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                            \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

constexpr double EPS = 1e-12;
constexpr int NK = 16;
constexpr double KAPPA_FOLD = 10.0;
constexpr int N_NORM = 2048;
constexpr int N_TERMS = 4;
constexpr double PI_D = 3.14159265358979323846;  // == Rust std::f64::consts::PI

struct Meta {
    int n, nb, h, n_nights, pt_off, night_off;
    int bands[6];
    int mem_off[6], mem_cnt[6];
    double span;
    double loc[N_TERMS], scale[N_TERMS];
    int used[N_TERMS];
};

// Pointers into concatenated per-object arrays (device or host copies).
struct View {
    const double *t, *y, *rank, *w;
    const int *b, *members;
    const int *tr_idx, *tr_off, *tr_cnt, *tr_nb, *tr_bands;  // per night (global night index)
    const int *te_idx, *te_off, *te_cnt;
    double ck[NK + 1];
    double k0;
};

struct HFit {
    double beta[8], a[6], mu[6], sigma;
};

HD inline void basis(double f, double t, int h, double* out) {
    double ph = 2.0 * PI_D * f * t;
    for (int k = 1; k <= h; k++) {
        double s = sin(ph * (double)k), c = cos(ph * (double)k);
        out[k - 1] = c;
        out[h + k - 1] = s;
    }
}

// Same algorithm as periodfind fourier.rs cholesky_solve.
HD inline bool cholesky_solve(double* a, double* b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = a[i * n + j];
            for (int k = 0; k < j; k++) s -= a[i * n + k] * a[j * n + k];
            if (i == j) {
                if (s <= 0.0) return false;
                a[i * n + j] = sqrt(s);
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= a[i * n + k] * b[k];
        b[i] = s / a[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= a[k * n + i] * b[k];
        b[i] = s / a[i * n + i];
    }
    return true;
}

// Robust shared-shape harmonic fit (mca.rs fit_h). idx == nullptr means all points.
// x_all: n x 8 basis; ws: 4n workspace (w, residual, shape, yc).
HD bool fit_h(const View& V, const Meta& M, const double* x_all, int h, const int* idx, int n,
              const int* bands, int nbands, int iterations, double* ws, HFit* out) {
    const int p = 2 * h;
    double *w = ws, *residual = ws + M.n, *shape = ws + 2 * M.n, *yc = ws + 3 * M.n;
#define IX(q) (idx ? idx[(q)] : (q))
#define XQ(q) (x_all + 8 * IX(q))
#define YQ(q) (V.y[M.pt_off + IX(q)])
#define BQ(q) (V.b[M.pt_off + IX(q)])
    double mean_y = 0.0;
    for (int q = 0; q < n; q++) mean_y += YQ(q);
    mean_y /= (double)n;
    double vs = 0.0;
    for (int q = 0; q < n; q++) vs += (YQ(q) - mean_y) * (YQ(q) - mean_y);
    double sigma = fmax(sqrt(vs / (double)n), 0.05);
    for (int q = 0; q < n; q++) w[q] = 1.0;
    double a[6], mu[6], beta[8];
    for (int j = 0; j < 6; j++) { a[j] = 1.0; mu[j] = 0.0; }
    for (int r = 0; r < 8; r++) beta[r] = 0.0;
    for (int it = 0; it < iterations; it++) {
        double mx[6][8], my[6];
        for (int jj = 0; jj < nbands; jj++) {
            int j = bands[jj];
            double tot = 0.0, sy = 0.0, sx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int q = 0; q < n; q++) {
                if (BQ(q) == j) {
                    tot += w[q];
                    sy += w[q] * YQ(q);
                    for (int r = 0; r < p; r++) sx[r] += w[q] * XQ(q)[r];
                }
            }
            tot = fmax(tot, EPS);
            for (int r = 0; r < p; r++) mx[j][r] = sx[r] / tot;
            my[j] = sy / tot;
        }
        double g[64], rhs[8];
        for (int r = 0; r < 64; r++) g[r] = 0.0;
        for (int r = 0; r < 8; r++) rhs[r] = 0.0;
        double wsum = 0.0;
        for (int q = 0; q < n; q++) {
            int j = BQ(q);
            yc[q] = YQ(q) - my[j];
            double dsg[8];
            for (int r = 0; r < p; r++) dsg[r] = (XQ(q)[r] - mx[j][r]) * a[j];
            for (int r = 0; r < p; r++) {
                rhs[r] += dsg[r] * w[q] * yc[q];
                for (int s = 0; s < p; s++) g[r * p + s] += w[q] * dsg[r] * dsg[s];
            }
            wsum += w[q];
        }
        double ridge = 1e-7 * fmax(wsum, 1.0);
        for (int r = 0; r < p; r++) g[r * p + r] += ridge;
        if (!cholesky_solve(g, rhs, p)) return false;
        for (int r = 0; r < p; r++) beta[r] = rhs[r];
        for (int q = 0; q < n; q++) {
            double s = 0.0;
            for (int r = 0; r < p; r++) s += XQ(q)[r] * beta[r];
            shape[q] = s;
        }
        for (int jj = 0; jj < nbands; jj++) {
            int j = bands[jj];
            double mxb = 0.0;
            for (int r = 0; r < p; r++) mxb += mx[j][r] * beta[r];
            double num = 0.0, den = 0.0;
            for (int q = 0; q < n; q++) {
                if (BQ(q) == j) {
                    double dx = shape[q] - mxb;
                    num += w[q] * dx * yc[q];
                    den += w[q] * dx * dx;
                }
            }
            a[j] = fmax(num / fmax(den, EPS), 0.0);
            mu[j] = my[j] - a[j] * mxb;
        }
        double ss = 0.0;
        for (int jj = 0; jj < nbands; jj++) ss += a[bands[jj]] * a[bands[jj]];
        double norm = fmax(sqrt(ss / (double)nbands), 1e-8);
        for (int jj = 0; jj < nbands; jj++) a[bands[jj]] /= norm;
        for (int r = 0; r < p; r++) beta[r] *= norm;
        for (int q = 0; q < n; q++) {
            shape[q] *= norm;
            residual[q] = YQ(q) - mu[BQ(q)] - a[BQ(q)] * shape[q];
        }
        for (int k = 0; k < 4; k++) {
            double s2 = fmax(sigma * sigma, 1e-8), m = 0.0;
            for (int q = 0; q < n; q++) {
                double r = residual[q];
                m += 5.0 * r * r / (4.0 + r * r / s2);
            }
            m /= (double)n;
            sigma = fmax(sqrt(m), 1e-4);
        }
        for (int q = 0; q < n; q++) w[q] = 5.0 / (4.0 + residual[q] * residual[q] / (sigma * sigma));
    }
#undef IX
#undef XQ
#undef YQ
#undef BQ
    for (int r = 0; r < 8; r++) out->beta[r] = beta[r];
    for (int j = 0; j < 6; j++) { out->a[j] = a[j]; out->mu[j] = mu[j]; }
    out->sigma = sigma;
    return true;
}

// mca.rs power_term, without stored per-point arrays: the basis and the centred
// values are recomputed from the same expressions wherever the Rust code reads
// its stored copies. w == nullptr means equal weights of 1.
HD double power_term(const View& V, const Meta& M, double f, const double* w) {
    const int n = M.n;
    double mxc[6][4], myc[6];
    for (int jj = 0; jj < M.nb; jj++) {
        int j = M.bands[jj];
        double tot = 0.0, sy = 0.0, sx[4] = {0, 0, 0, 0};
        for (int q = 0; q < n; q++) {
            if (V.b[M.pt_off + q] == j) {
                double wq = w ? w[M.pt_off + q] : 1.0;
                double u[8];
                basis(f, V.t[M.pt_off + q], 2, u);
                tot += wq;
                sy += wq * V.y[M.pt_off + q];
                for (int r = 0; r < 4; r++) sx[r] += wq * u[r];
            }
        }
        for (int r = 0; r < 4; r++) mxc[j][r] = sx[r] / tot;
        myc[j] = sy / tot;
    }
    double wsum = 0.0;
    for (int q = 0; q < n; q++) wsum += w ? w[M.pt_off + q] : 1.0;
    double a[6] = {1, 1, 1, 1, 1, 1}, beta[4] = {0, 0, 0, 0};
#define XC(q, out)                                                       \
    do {                                                                 \
        double u_[8];                                                    \
        basis(f, V.t[M.pt_off + (q)], 2, u_);                            \
        int j_ = V.b[M.pt_off + (q)];                                    \
        for (int r_ = 0; r_ < 4; r_++) (out)[r_] = u_[r_] - mxc[j_][r_]; \
    } while (0)
    for (int it = 0; it < 4; it++) {
        double g[16], rhs[4];
        for (int r = 0; r < 16; r++) g[r] = 0.0;
        for (int r = 0; r < 4; r++) rhs[r] = 0.0;
        for (int q = 0; q < n; q++) {
            double xc[4];
            XC(q, xc);
            int j = V.b[M.pt_off + q];
            double wq = w ? w[M.pt_off + q] : 1.0, ycq = V.y[M.pt_off + q] - myc[j], aj = a[j];
            double dsg[4] = {xc[0] * aj, xc[1] * aj, xc[2] * aj, xc[3] * aj};
            for (int r = 0; r < 4; r++) {
                rhs[r] += dsg[r] * wq * ycq;
                for (int s = 0; s < 4; s++) g[r * 4 + s] += wq * dsg[r] * dsg[s];
            }
        }
        for (int r = 0; r < 4; r++) g[r * 4 + r] += 1e-7 * fmax(wsum, 1.0);
        if (!cholesky_solve(g, rhs, 4)) return NAN;
        for (int r = 0; r < 4; r++) beta[r] = rhs[r];
        for (int jj = 0; jj < M.nb; jj++) {
            int j = M.bands[jj];
            double num = 0.0, den = 0.0;
            for (int q = 0; q < n; q++) {
                if (V.b[M.pt_off + q] == j) {
                    double xc[4];
                    XC(q, xc);
                    double wave = 0.0;
                    for (int r = 0; r < 4; r++) wave += xc[r] * beta[r];
                    double wq = w ? w[M.pt_off + q] : 1.0, ycq = V.y[M.pt_off + q] - myc[j];
                    num += wq * wave * ycq;
                    den += wq * wave * wave;
                }
            }
            double ratio = num / (den + 1e-8);
            a[j] = 0.5 * (ratio + sqrt(ratio * ratio + 1e-8));
        }
        double ss = 0.0;
        for (int jj = 0; jj < M.nb; jj++) ss += a[M.bands[jj]] * a[M.bands[jj]];
        double norm = sqrt(ss / (double)M.nb + 1e-16);
        for (int jj = 0; jj < M.nb; jj++) a[M.bands[jj]] /= norm;
        for (int r = 0; r < 4; r++) beta[r] *= norm;
    }
    double sse = 0.0, tot = 0.0;
    for (int q = 0; q < n; q++) {
        double xc[4];
        XC(q, xc);
        int j = V.b[M.pt_off + q];
        double wv = 0.0;
        for (int r = 0; r < 4; r++) wv += xc[r] * beta[r];
        wv *= a[j];
        double wq = w ? w[M.pt_off + q] : 1.0, ycq = V.y[M.pt_off + q] - myc[j];
        sse += wq * (ycq - wv) * (ycq - wv);
        tot += wq * ycq * ycq;
    }
#undef XC
    return 1.0 - sse / (tot + EPS);
}

// mca.rs pdm_term; the per-point cos/sin are recomputed from the same expression
// in the second pass instead of being stored.
HD double pdm_term(const View& V, const Meta& M, double f) {
    double total = 0.0;
    int nbu = 0;
    for (int jj = 0; jj < M.nb; jj++) {
        int j = M.bands[jj];
        int n = M.mem_cnt[j];
        if (n < 10) continue;
        const int* mem = V.members + M.mem_off[j];
        double c1[NK], s1[NK], cv[NK], sv[NK];
        for (int k = 0; k < NK; k++) c1[k] = s1[k] = cv[k] = sv[k] = 0.0;
        double sum_v = 0.0;
        for (int q = 0; q < n; q++) {
            int i = M.pt_off + mem[q];
            double ph = f * V.t[i], v = V.rank[i];
            for (int k = 1; k <= NK; k++) {
                double arg = 2.0 * PI_D * ph * (double)k, s = sin(arg), c = cos(arg);
                c1[k - 1] += c;
                s1[k - 1] += s;
                cv[k - 1] += c * v;
                sv[k - 1] += s * v;
            }
        }
        for (int q = 0; q < n; q++) sum_v += V.rank[M.pt_off + mem[q]];
        double mean_v = sum_v / (double)n, var_v = 0.0;
        for (int q = 0; q < n; q++) {
            double d = V.rank[M.pt_off + mem[q]] - mean_v;
            var_v += d * d;
        }
        var_v /= (double)n;
        double acc = 0.0;
        for (int q = 0; q < n; q++) {
            int i = M.pt_off + mem[q];
            double ph = f * V.t[i], v = V.rank[i];
            double s0 = V.ck[0] * (double)n, sv1 = V.ck[0] * sum_v, t0 = 0.0, t1 = 0.0;
            for (int k = 1; k <= NK; k++) {
                double arg = 2.0 * PI_D * ph * (double)k, s = sin(arg), c = cos(arg);
                t0 += V.ck[k] * (c * c1[k - 1] + s * s1[k - 1]);
                t1 += V.ck[k] * (c * cv[k - 1] + s * sv[k - 1]);
            }
            s0 += 2.0 * t0 - V.k0;
            sv1 += 2.0 * t1 - V.k0 * v;
            double mean = sv1 / fmax(s0, EPS);
            acc += (v - mean) * (v - mean);
        }
        total += acc / (double)n / fmax(var_v, EPS);
        nbu++;
    }
    return nbu == 0 ? NAN : total / (double)nbu;
}

HD void basis_all(const View& V, const Meta& M, double f, int h, double* x_all) {
    for (int i = 0; i < M.n; i++) basis(f, V.t[M.pt_off + i], h, x_all + 8 * i);
}

HD double loo_term(const View& V, const Meta& M, const double* x_all, double* ws) {
    if (M.n_nights == 0) return NAN;
    double err = 0.0;
    for (int k = 0; k < M.n_nights; k++) {
        int g = M.night_off + k;
        HFit fit;
        if (!fit_h(V, M, x_all, M.h, V.tr_idx + V.tr_off[g], V.tr_cnt[g], V.tr_bands + 6 * g, V.tr_nb[g], 2,
                   ws, &fit))
            return NAN;
        double acc = 0.0;
        const int* te = V.te_idx + V.te_off[g];
        for (int q = 0; q < V.te_cnt[g]; q++) {
            int i = te[q];
            const double* u = x_all + 8 * i;
            double shape = 0.0;
            for (int r = 0; r < 2 * M.h; r++) shape += u[r] * fit.beta[r];
            int bi = V.b[M.pt_off + i];
            double r = V.y[M.pt_off + i] - (fit.mu[bi] + fit.a[bi] * shape);
            acc += sqrt(r * r + 1e-4) - 0.01;
        }
        err += acc / (double)V.te_cnt[g];
    }
    return err / (double)M.n_nights;
}

// The four terms, signed so that larger is better. scratch: 12 * n doubles.
HD void components(const View& V, const Meta& M, double f, double* scratch, double* out) {
    const double SIGN[N_TERMS] = {1.0, -1.0, -1.0, -1.0};
    double power = power_term(V, M, f, V.w);
    double* x_all = scratch;
    double* ws = scratch + 8 * M.n;
    basis_all(V, M, f, M.h, x_all);
    HFit fit;
    double sigma = fit_h(V, M, x_all, M.h, nullptr, M.n, M.bands, M.nb, 3, ws, &fit) ? log(fit.sigma + EPS) : NAN;
    double pdm = pdm_term(V, M, f);
    double loo = loo_term(V, M, x_all, ws);
    double raw[N_TERMS] = {power, sigma, pdm, loo};
    for (int j = 0; j < N_TERMS; j++) out[j] = raw[j] * SIGN[j];
}

HD double score(const View& V, const Meta& M, double f, double* scratch) {
    double c[N_TERMS];
    components(V, M, f, scratch, c);
    double s = 0.0;
    for (int j = 0; j < N_TERMS; j++) {
        if (M.used[j]) {
            double z = (c[j] - M.loc[j]) / M.scale[j];
            s += isnan(z) ? 0.0 : (isinf(z) ? (z > 0 ? DBL_MAX : -DBL_MAX) : z);
        }
    }
    return s;
}

__global__ void boot_kernel(View V, const Meta* metas, const int* item_obj, const double* item_f, int n_items,
                            double* ev) {
    for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < n_items; it += gridDim.x * blockDim.x)
        ev[it] = power_term(V, metas[item_obj[it]], item_f[it], nullptr);
}

__global__ void norm_kernel(View V, const Meta* metas, const int* item_obj, const double* item_f, int n_items,
                            const long long* scratch_off, double* scratch, double* out) {
    int it = blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    components(V, metas[item_obj[it]], item_f[it], scratch + scratch_off[it], out + N_TERMS * it);
}

__global__ void ascent_kernel(View V, const Meta* metas, const int* item_obj, const double* item_f, int n_items,
                              const long long* scratch_off, double* scratch, double* out_f, double* out_s,
                              double F_MAX, double TOL, int MAX_ITER) {
    int it = blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_items) return;
    const Meta& M = metas[item_obj[it]];
    double* sc = scratch + scratch_off[it];
    double t_span = M.span, lo = 1.0 / t_span;
    double low = lo * t_span, high = F_MAX * t_span;
    double x = fmin(fmax(item_f[it] * t_span, low), high);
    double sco = score(V, M, x / t_span, sc);
    const double delta = 1e-3;
    for (int k = 0; k < MAX_ITER; k++) {
        double plus = fmin(x + delta, high), minus = fmax(x - delta, low);
        double sp = score(V, M, plus / t_span, sc), sm = score(V, M, minus / t_span, sc);
        double hp = plus - x, hm = x - minus;
        double right = hp > 0.0 ? (sp - sco) / hp : 0.0;
        double left = hm > 0.0 ? (sco - sm) / hm : 0.0;
        bool both = hp > 0.0 && hm > 0.0;
        double g = hm == 0.0 ? right : (hp == 0.0 ? left : (hm * right + hp * left) / fmax(hp + hm, EPS));
        double curv = both ? 2.0 * (right - left) / fmax(hp + hm, EPS) : 0.0;
        if (fabs(g * t_span) < 1e-10 && curv <= 1e-8) break;
        double step = fmin(fmax(g / fmax(fabs(curv), 1e-3), -0.5), 0.5);
        if (fabs(g * t_span) < 1e-10 && curv > 1e-8) step = sp >= sm ? 0.05 : -0.05;
        bool moved = false;
        while (true) {
            double trial = fmin(fmax(x + step, low), high);
            if (fabs(trial - x) < TOL) break;
            double cand = score(V, M, trial / t_span, sc);
            if (cand >= sco + 1e-4 * g * (trial - x)) {
                double disp = trial - x;
                x = trial;
                sco = cand;
                moved = fabs(disp) >= TOL;
                break;
            }
            step *= 0.5;
        }
        if (!moved) break;
    }
    out_f[it] = x / t_span;
    out_s[it] = sco;
}

// ---------------------------------------------------------------- host side
static double quantile_sorted(const std::vector<double>& s, double q) {
    size_t n = s.size();
    if (n == 1) return s[0];
    double pos = q * (double)(n - 1);
    size_t lo = (size_t)floor(pos), hi = std::min(lo + 1, n - 1);
    double frac = pos - (double)lo;
    return s[lo] + (s[hi] - s[lo]) * frac;
}

struct HostObj {
    bool ok = false;
    std::vector<double> t, y, rank;
    std::vector<int> b, night, bands;
    std::vector<std::vector<int>> members = std::vector<std::vector<int>>(6);
    int nn = 0;
    double span = 0;
    std::vector<double> starts, grid;
};

static HostObj setup(const std::vector<double>& t, const std::vector<double>& m, const std::vector<uint8_t>& band) {
    HostObj o;
    std::vector<int> idx;
    for (size_t i = 0; i < t.size(); i++)
        if (std::isfinite(t[i]) && std::isfinite(m[i]) && band[i] < 6) idx.push_back((int)i);
    if (idx.size() < 8) return o;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int c) { return t[a] < t[c]; });
    double t0 = t[idx[0]];
    for (int i : idx) o.t.push_back(t[i] - t0);
    o.span = o.t.back();
    if (!(o.span > 0.0)) return o;
    for (int i : idx) { o.b.push_back(band[i]); o.y.push_back(m[i]); }
    o.rank.assign(o.y.size(), 0.0);
    o.bands = o.b;
    std::sort(o.bands.begin(), o.bands.end());
    o.bands.erase(std::unique(o.bands.begin(), o.bands.end()), o.bands.end());
    for (int j : o.bands) {
        std::vector<int> mem;
        for (size_t i = 0; i < o.y.size(); i++)
            if (o.b[i] == j) mem.push_back((int)i);
        std::vector<double> v;
        for (int i : mem) v.push_back(o.y[i]);
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        double med = quantile_sorted(s, 0.5);
        std::vector<double> dev;
        for (double x : v) dev.push_back(fabs(x - med));
        std::sort(dev.begin(), dev.end());
        double scale = std::max(std::max(0.5 * (quantile_sorted(s, 0.95) - quantile_sorted(s, 0.05)),
                                         1.4826 * quantile_sorted(dev, 0.5)),
                                0.02);
        for (int i : mem) o.y[i] = (o.y[i] - med) / scale;
        size_t n = v.size();
        std::vector<int> order(n);
        for (size_t i = 0; i < n; i++) order[i] = (int)i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return v[a] < v[c]; });
        size_t k = 0;
        while (k < n) {
            size_t e = k + 1;
            while (e < n && v[order[e]] == v[order[k]]) e++;
            double rr = (double)(k + e - 1) / 2.0;
            double val = 2.0 * (rr + 0.5) / (double)n - 1.0;
            for (size_t q = k; q < e; q++) o.rank[mem[order[q]]] = val;
            k = e;
        }
    }
    o.night.assign(o.t.size(), 0);
    for (size_t i = 1; i < o.t.size(); i++) o.night[i] = o.night[i - 1] + (o.t[i] - o.t[i - 1] > 0.5 ? 1 : 0);
    o.nn = o.night.back() + 1;
    for (size_t i = 0; i < o.b.size(); i++) o.members[o.b[i]].push_back((int)i);
    o.ok = true;
    return o;
}

MultiCriterionAscent::MultiCriterionAscent(double f_max, double oversample, double tol, int max_iter, int top)
    : f_max_(f_max), oversample_(oversample), tol_(tol), max_iter_(max_iter), top_(top) {}

void MultiCriterionAscent::CalcMCABatched(const std::vector<double*>& times, const std::vector<double*>& mags,
                                          const std::vector<uint8_t*>& bands,
                                          const std::vector<size_t>& lengths,
                                          const std::vector<double*>& periods,
                                          const std::vector<size_t>& n_periods, double* out) const {
    const double F_MAX = f_max_, OVERSAMPLE = oversample_, TOL = tol_;
    const int MAX_ITER = max_iter_, TOP = top_;
    const int n_obj = (int)times.size();
    std::vector<HostObj> objs(n_obj);
    std::vector<std::vector<double>> cands(n_obj);
    for (int o = 0; o < n_obj; o++) {
        std::vector<double> t(times[o], times[o] + lengths[o]), m(mags[o], mags[o] + lengths[o]);
        std::vector<uint8_t> b(bands[o], bands[o] + lengths[o]);
        cands[o].assign(periods[o], periods[o] + n_periods[o]);
        objs[o] = setup(t, m, b);
    }

    // Kernel coefficients (as mca.rs kernel_coefficients).
    View H{};
    for (int k = 0; k <= NK; k++) {
        double s = 0.0;
        for (int i = 0; i < 1024; i++) {
            double th = 2.0 * PI_D * (double)i / 1024.0;
            s += exp(KAPPA_FOLD * (cos(th) - 1.0)) * cos((double)k * th);
        }
        H.ck[k] = s / 1024.0;
    }
    double sck = 0.0;
    for (int k = 1; k <= NK; k++) sck += H.ck[k];
    H.k0 = H.ck[0] + 2.0 * sck;

    // Concatenate points, members and night subsets.
    std::vector<Meta> metas(n_obj);
    std::vector<double> T, Y, R, W;
    std::vector<int> B, MEM, TRI, TRO, TRC, TRN, TRB, TEI, TEO, TEC;
    int max_n = 1;
    for (int o = 0; o < n_obj; o++) {
        Meta& M = metas[o];
        memset(&M, 0, sizeof(Meta));
        HostObj& O = objs[o];
        if (!O.ok) continue;
        int n = (int)O.t.size();
        max_n = std::max(max_n, n);
        M.n = n;
        M.nb = (int)O.bands.size();
        for (int jj = 0; jj < M.nb; jj++) M.bands[jj] = O.bands[jj];
        M.pt_off = (int)T.size();
        M.span = O.span;
        T.insert(T.end(), O.t.begin(), O.t.end());
        Y.insert(Y.end(), O.y.begin(), O.y.end());
        R.insert(R.end(), O.rank.begin(), O.rank.end());
        B.insert(B.end(), O.b.begin(), O.b.end());
        W.resize(T.size(), 0.0);
        for (int j = 0; j < 6; j++) {
            M.mem_off[j] = (int)MEM.size();
            M.mem_cnt[j] = (int)O.members[j].size();
            MEM.insert(MEM.end(), O.members[j].begin(), O.members[j].end());
        }
        int h = 1;
        for (int hh = 4; hh >= 1; hh--)
            if (n > 2 * (2 * hh + 2 * M.nb + 2)) { h = hh; break; }
        M.h = h;
        M.night_off = (int)TRO.size();
        for (int night = 0; night < O.nn; night++) {
            std::vector<int> train, test;
            bool seen[6] = {false, false, false, false, false, false};
            for (int i = 0; i < n; i++)
                if (O.night[i] != night) train.push_back(i);
            if (train.size() < 8) continue;
            for (int i : train) seen[O.b[i]] = true;
            for (int i = 0; i < n; i++)
                if (O.night[i] == night && seen[O.b[i]]) test.push_back(i);
            if (test.empty()) continue;
            std::vector<int> tb;
            for (int i : train) tb.push_back(O.b[i]);
            std::sort(tb.begin(), tb.end());
            tb.erase(std::unique(tb.begin(), tb.end()), tb.end());
            TRO.push_back((int)TRI.size());
            TRC.push_back((int)train.size());
            TRI.insert(TRI.end(), train.begin(), train.end());
            TRN.push_back((int)tb.size());
            for (int s = 0; s < 6; s++) TRB.push_back(s < (int)tb.size() ? tb[s] : 0);
            TEO.push_back((int)TEI.size());
            TEC.push_back((int)test.size());
            TEI.insert(TEI.end(), test.begin(), test.end());
            M.n_nights++;
        }
        double lo = 1.0 / O.span;
        long long ng = (long long)floor((F_MAX - lo) * OVERSAMPLE * O.span) + 1;
        O.grid.resize(ng);
        for (long long i = 0; i < ng; i++) O.grid[i] = lo + (double)i / (OVERSAMPLE * O.span);
        for (double p : cands[o])
            if (std::isfinite(p) && p > 0.0) {
                double f = 1.0 / p;
                if (f >= lo && f <= F_MAX) O.starts.push_back(f);
            }
        std::sort(O.starts.begin(), O.starts.end());
        O.starts.erase(std::unique(O.starts.begin(), O.starts.end()), O.starts.end());
    }

    auto up_d = [](const std::vector<double>& v) {
        double* d;
        CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * 8));
        if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 8, cudaMemcpyHostToDevice));
        return d;
    };
    auto up_i = [](const std::vector<int>& v) {
        int* d;
        CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * 4));
        if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return d;
    };
    View D = H;
    D.t = up_d(T); D.y = up_d(Y); D.rank = up_d(R); D.b = up_i(B); D.members = up_i(MEM);
    D.tr_idx = up_i(TRI); D.tr_off = up_i(TRO); D.tr_cnt = up_i(TRC); D.tr_nb = up_i(TRN); D.tr_bands = up_i(TRB);
    D.te_idx = up_i(TEI); D.te_off = up_i(TEO); D.te_cnt = up_i(TEC);
    H.t = T.data(); H.y = Y.data(); H.rank = R.data(); H.b = B.data(); H.members = MEM.data();
    H.tr_idx = TRI.data(); H.tr_off = TRO.data(); H.tr_cnt = TRC.data(); H.tr_nb = TRN.data(); H.tr_bands = TRB.data();
    H.te_idx = TEI.data(); H.te_off = TEO.data(); H.te_cnt = TEC.data();
    Meta* d_meta;
    CUDA_CHECK(cudaMalloc(&d_meta, n_obj * sizeof(Meta)));

    // 1. Bootstrap: equal-weight power over every object's full grid.
    std::vector<int> io;
    std::vector<double> iff;
    for (int o = 0; o < n_obj; o++)
        if (objs[o].ok)
            for (double f : objs[o].grid) { io.push_back(o); iff.push_back(f); }
    CUDA_CHECK(cudaMemcpy(d_meta, metas.data(), n_obj * sizeof(Meta), cudaMemcpyHostToDevice));
    int *d_io = up_i(io);
    double *d_if = up_d(iff), *d_ev;
    CUDA_CHECK(cudaMalloc(&d_ev, iff.size() * 8));
    boot_kernel<<<4096, 128>>>(D, d_meta, d_io, d_if, (int)iff.size(), d_ev);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> ev(iff.size());
    CUDA_CHECK(cudaMemcpy(ev.data(), d_ev, ev.size() * 8, cudaMemcpyDeviceToHost));
    cudaFree(d_io); cudaFree(d_if); cudaFree(d_ev);

    // Host: robust weights from the bootstrap fit (same functions as the device).
    std::vector<char> alive(n_obj, 0);
    size_t pos = 0;
    for (int o = 0; o < n_obj; o++) {
        if (!objs[o].ok) continue;
        Meta& M = metas[o];
        size_t ng = objs[o].grid.size(), first = 0;
        for (size_t i = 0; i < ng; i++) {
            double v = ev[pos + i], vf = ev[pos + first];
            if (v > vf || (std::isnan(vf) && !std::isnan(v))) first = i;
        }
        pos += ng;
        std::vector<double> x2(8 * M.n), ws(4 * M.n);
        basis_all(H, M, objs[o].grid[first], 2, x2.data());
        HFit init;
        if (!fit_h(H, M, x2.data(), 2, nullptr, M.n, M.bands, M.nb, 5, ws.data(), &init)) continue;
        double s = fmax(init.sigma, 1e-4);
        for (int q = 0; q < M.n; q++) {
            double r = ws[M.n + q];
            W[M.pt_off + q] = 5.0 / (4.0 + (r / s) * (r / s));
        }
        alive[o] = 1;
    }
    D.w = up_d(W);
    H.w = W.data();

    // Scratch for norm/ascent threads: 12 * n doubles each, in chunks.
    const long long budget = 768LL * 1024 * 1024 / 8;  // doubles
    auto run_chunks = [&](bool ascent, std::vector<int>& objv, std::vector<double>& fv, std::vector<double>& outA,
                          std::vector<double>& outB) {
        size_t total = objv.size(), start = 0;
        while (start < total) {
            std::vector<long long> off;
            long long acc = 0;
            size_t end = start;
            while (end < total && acc + 12LL * metas[objv[end]].n <= budget) {
                off.push_back(acc);
                acc += 12LL * metas[objv[end]].n;
                end++;
            }
            int cnt = (int)(end - start);
            std::vector<int> o2(objv.begin() + start, objv.begin() + end);
            std::vector<double> f2(fv.begin() + start, fv.begin() + end);
            int* d_o = up_i(o2);
            double* d_f = up_d(f2);
            long long* d_off;
            CUDA_CHECK(cudaMalloc(&d_off, off.size() * 8));
            CUDA_CHECK(cudaMemcpy(d_off, off.data(), off.size() * 8, cudaMemcpyHostToDevice));
            double* d_sc;
            CUDA_CHECK(cudaMalloc(&d_sc, std::max(1LL, acc) * 8));
            if (ascent) {
                double *d_a, *d_b;
                CUDA_CHECK(cudaMalloc(&d_a, cnt * 8));
                CUDA_CHECK(cudaMalloc(&d_b, cnt * 8));
                ascent_kernel<<<(cnt + 63) / 64, 64>>>(D, d_meta, d_o, d_f, cnt, d_off, d_sc, d_a, d_b, F_MAX, TOL, MAX_ITER);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(outA.data() + start, d_a, cnt * 8, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(outB.data() + start, d_b, cnt * 8, cudaMemcpyDeviceToHost));
                cudaFree(d_a); cudaFree(d_b);
            } else {
                double* d_out;
                CUDA_CHECK(cudaMalloc(&d_out, (size_t)cnt * N_TERMS * 8));
                norm_kernel<<<(cnt + 63) / 64, 64>>>(D, d_meta, d_o, d_f, cnt, d_off, d_sc, d_out);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(outA.data() + start * N_TERMS, d_out, (size_t)cnt * N_TERMS * 8,
                                      cudaMemcpyDeviceToHost));
                cudaFree(d_out);
            }
            cudaFree(d_o); cudaFree(d_f); cudaFree(d_off); cudaFree(d_sc);
            start = end;
        }
    };

    // 2. Term scaling at up to 2048 evenly spaced grid frequencies.
    std::vector<int> no;
    std::vector<double> nf;
    std::vector<int> nstart(n_obj, 0), ncount(n_obj, 0);
    for (int o = 0; o < n_obj; o++) {
        if (!alive[o]) continue;
        const std::vector<double>& g = objs[o].grid;
        int m = (int)std::min<size_t>(N_NORM, g.size());
        std::vector<size_t> pick;
        for (int i = 0; i < m; i++) {
            size_t p = m == 1 ? 0 : (size_t)((double)i * (double)(g.size() - 1) / (double)(m - 1));
            if (pick.empty() || pick.back() != p) pick.push_back(p);
        }
        nstart[o] = (int)no.size();
        ncount[o] = (int)pick.size();
        for (size_t p : pick) { no.push_back(o); nf.push_back(g[p]); }
    }
    std::vector<double> nval(no.size() * N_TERMS), dummy;
    CUDA_CHECK(cudaMemcpy(d_meta, metas.data(), n_obj * sizeof(Meta), cudaMemcpyHostToDevice));
    run_chunks(false, no, nf, nval, dummy);
    for (int o = 0; o < n_obj; o++) {
        if (!alive[o]) continue;
        Meta& M = metas[o];
        bool any = false;
        for (int j = 0; j < N_TERMS; j++) {
            bool fin = true;
            double s = 0.0;
            for (int i = 0; i < ncount[o]; i++) {
                double v = nval[(size_t)(nstart[o] + i) * N_TERMS + j];
                fin = fin && std::isfinite(v);
                s += v;
            }
            M.loc[j] = 0.0;
            M.scale[j] = 1.0;
            M.used[j] = 0;
            if (fin) {
                double mean = s / (double)ncount[o], ss = 0.0;
                for (int i = 0; i < ncount[o]; i++) {
                    double d = nval[(size_t)(nstart[o] + i) * N_TERMS + j] - mean;
                    ss += d * d;
                }
                double sd = sqrt(ss / (double)ncount[o]);
                M.used[j] = 1;
                M.loc[j] = mean;
                M.scale[j] = sd > 1e-10 ? sd : 1.0;
                any = true;
            }
        }
        if (!any || objs[o].starts.empty()) alive[o] = 0;
    }

    // 3. Ascent: one thread per walker.
    std::vector<int> wo;
    std::vector<double> wf;
    for (int o = 0; o < n_obj; o++)
        if (alive[o])
            for (double f : objs[o].starts) { wo.push_back(o); wf.push_back(f); }
    std::vector<double> rf(wo.size()), rs(wo.size());
    CUDA_CHECK(cudaMemcpy(d_meta, metas.data(), n_obj * sizeof(Meta), cudaMemcpyHostToDevice));
    run_chunks(true, wo, wf, rf, rs);

    // 4. Merge and write (same layout as calc_mca_batched).
    const int ncol = 2 + 2 * TOP;
    for (size_t i = 0; i < (size_t)n_obj * ncol; i++) out[i] = NAN;
    size_t w0 = 0;
    for (int o = 0; o < n_obj; o++) {
        double* row = out + (size_t)o * ncol;
        if (!alive[o]) { row[0] = 0; row[1] = 0; continue; }
        size_t k = objs[o].starts.size();
        std::vector<double> fr(rf.begin() + w0, rf.begin() + w0 + k), sc(rs.begin() + w0, rs.begin() + w0 + k);
        w0 += k;
        double radius = 0.5 / objs[o].span;
        std::vector<int> order(k);
        for (size_t i = 0; i < k; i++) order[i] = (int)i;
        auto cmp = [&](int a, int c) { return sc[a] != sc[c] ? sc[a] > sc[c] : fr[a] < fr[c]; };
        std::stable_sort(order.begin(), order.end(), cmp);
        std::unordered_map<long long, int> bins;
        std::vector<int> keep;
        for (int i : order) {
            long long cell = (long long)floor(fr[i] / radius);
            bool close = false;
            for (long long c = cell - 1; c <= cell + 1; c++) {
                auto it = bins.find(c);
                if (it != bins.end() && fabs(fr[i] - fr[keep[it->second]]) <= radius) close = true;
            }
            if (!close) {
                bins[cell] = (int)keep.size();
                keep.push_back(i);
            }
        }
        std::stable_sort(keep.begin(), keep.end(), cmp);
        row[0] = (double)k;
        row[1] = (double)keep.size();
        for (int q = 0; q < (int)std::min<size_t>(TOP, keep.size()); q++) {
            row[2 + q] = 1.0 / fr[keep[q]];
            row[2 + TOP + q] = sc[keep[q]];
        }
    }
    // Release device memory.
    cudaFree((void*)D.t); cudaFree((void*)D.y); cudaFree((void*)D.rank); cudaFree((void*)D.b);
    cudaFree((void*)D.members); cudaFree((void*)D.tr_idx); cudaFree((void*)D.tr_off); cudaFree((void*)D.tr_cnt);
    cudaFree((void*)D.tr_nb); cudaFree((void*)D.tr_bands); cudaFree((void*)D.te_idx); cudaFree((void*)D.te_off);
    cudaFree((void*)D.te_cnt); cudaFree((void*)D.w); cudaFree(d_meta);
}
