#include "tfs.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "dblatomic.cuh"
#include "errchk.cuh"

// Sampled template fitting, in three passes.
//
// The accumulators are the reason for the split. At 820 templates and a 512
// point grid, num and den hold 420k doubles per trial period, which is far
// past shared memory, so the templates cannot all live in one block.
//
//   fold   one block per (curve, period). Bins the curve per band and
//          transforms the per bin counts c and summed magnitudes s. The
//          result is reused by every template, so it is computed once.
//   score  one block per (curve, period, template). Three inverse transforms
//          per band give num and den at all n_phase shifts, and the block
//          reduces to its own best shift.
//   reduce one block per curve, over its periods and templates.
//
// Double precision throughout. The fit is an argmin over nearly tied columns
// and folds a long baseline at short periods.

#define TFS_THREADS 256
#define TFS_MAX_PHASE 1024
#define TFS_MAX_BAND 8

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

// In place iterative radix-2 FFT across the whole block, on shared memory.
// n must be a power of two. Deliberately the same few lines as the Rust
// implementation, so the two can be held to the same answers.
__device__ void FFTShared(double* re, double* im, int n, int inverse) {
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    int log2n = 0;
    while ((1 << log2n) < n)
        ++log2n;

    // bit reversal permutation; the i < j guard means each pair moves once
    for (int i = tid; i < n; i += nt) {
        int j = (int)(__brev((unsigned int)i) >> (32 - log2n));
        if (i < j) {
            double tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            double ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }
    __syncthreads();

    const double sign = inverse ? 1.0 : -1.0;
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const double ang = sign * 2.0 * M_PI / (double)len;
        for (int k = tid; k < n / 2; k += nt) {
            const int blk = (k / half) * len;
            const int j = k % half;
            double wr, wi;
            sincos(ang * (double)j, &wi, &wr);
            const int a = blk + j;
            const int b = a + half;
            const double ur = re[a], ui = im[a];
            const double vr = re[b] * wr - im[b] * wi;
            const double vi = re[b] * wi + im[b] * wr;
            re[a] = ur + vr;
            im[a] = ui + vi;
            re[b] = ur - vr;
            im[b] = ui - vi;
        }
        __syncthreads();
    }

    if (inverse) {
        const double inv = 1.0 / (double)n;
        for (int i = tid; i < n; i += nt) {
            re[i] *= inv;
            im[i] *= inv;
        }
        __syncthreads();
    }
}

// Serial host transform, for the bank. Same algorithm.
static void FFTHost(double* re, double* im, int n, int inverse) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j) {
            double t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
    const double sign = inverse ? 1.0 : -1.0;
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * M_PI / (double)len;
        for (int i = 0; i < n; i += len) {
            double wr = 1.0, wi = 0.0;
            const double cr = cos(ang), ci = sin(ang);
            for (int k = 0; k < len / 2; ++k) {
                const int a = i + k, b = a + len / 2;
                const double ur = re[a], ui = im[a];
                const double vr = re[b] * wr - im[b] * wi;
                const double vi = re[b] * wi + im[b] * wr;
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
                const double nwr = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = nwr;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; ++i) {
            re[i] /= n;
            im[i] /= n;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1: fold
// ---------------------------------------------------------------------------

__global__ void FoldKernel(const double* __restrict__ times,
                           const double* __restrict__ mags,
                           const uint8_t* __restrict__ bands,
                           const size_t* __restrict__ offsets,
                           const size_t* __restrict__ lengths,
                           const double* __restrict__ job_period,
                           const size_t* __restrict__ job_curve,
                           size_t n_band,
                           size_t n_phase,
                           size_t min_points,
                           double* __restrict__ out_fc,
                           double* __restrict__ out_fs,
                           double* __restrict__ out_stat,
                           double* __restrict__ out_tss) {
    const size_t job = blockIdx.x;
    const size_t curve = job_curve[job];
    const double period = job_period[job];
    const int tid = threadIdx.x;

    extern __shared__ double sh[];
    double* cre = sh;
    double* cim = cre + n_phase;
    double* sre = cim + n_phase;
    double* sim = sre + n_phase;

    const size_t off = offsets[curve];
    const size_t len = lengths[curve];

    double tss_total = 0.0;
    for (size_t b = 0; b < n_band; ++b) {
        for (size_t i = tid; i < n_phase; i += blockDim.x) {
            cre[i] = 0.0;
            cim[i] = 0.0;
            sre[i] = 0.0;
            sim[i] = 0.0;
        }
        __syncthreads();

        // bin this band. Counts are small integers and the sums are per bin,
        // so shared atomics are exact enough here; the score never sees the
        // order these landed in.
        for (size_t i = tid; i < len; i += blockDim.x) {
            if ((size_t)bands[off + i] != b)
                continue;
            double ph = times[off + i] / period;
            ph = ph - floor(ph);
            int k = (int)(ph * (double)n_phase);
            if (k >= (int)n_phase)
                k = (int)n_phase - 1;
            PfAtomicAdd(&cre[k], 1.0);
            PfAtomicAdd(&sre[k], mags[off + i]);
        }
        __syncthreads();

        // per band n, mean and total sum of squares, from the bins
        if (tid == 0) {
            double n = 0.0, sum = 0.0;
            for (size_t k = 0; k < n_phase; ++k) {
                n += cre[k];
                sum += sre[k];
            }
            out_stat[(job * n_band + b) * 3 + 0] = n;
            out_stat[(job * n_band + b) * 3 + 1] = n > 0.0 ? sum / n : 0.0;
        }
        __syncthreads();

        // sum of squares needs the points themselves, not the bins
        double local = 0.0;
        for (size_t i = tid; i < len; i += blockDim.x) {
            if ((size_t)bands[off + i] != b)
                continue;
            const double y = mags[off + i];
            local += y * y;
        }
        __shared__ double red[TFS_THREADS];
        red[tid] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            const double n = out_stat[(job * n_band + b) * 3 + 0];
            const double mean = out_stat[(job * n_band + b) * 3 + 1];
            double tss = red[0] - n * mean * mean;
            if (n < (double)min_points || tss <= 0.0)
                tss = 0.0;
            out_stat[(job * n_band + b) * 3 + 2] = tss;
        }
        __syncthreads();

        FFTShared(cre, cim, (int)n_phase, 0);
        FFTShared(sre, sim, (int)n_phase, 0);

        const size_t o = (job * n_band + b) * n_phase * 2;
        for (size_t i = tid; i < n_phase; i += blockDim.x) {
            out_fc[o + 2 * i + 0] = cre[i];
            out_fc[o + 2 * i + 1] = cim[i];
            out_fs[o + 2 * i + 0] = sre[i];
            out_fs[o + 2 * i + 1] = sim[i];
        }
        __syncthreads();
        tss_total += out_stat[(job * n_band + b) * 3 + 2];
    }
    if (tid == 0)
        out_tss[job] = tss_total;
}

// ---------------------------------------------------------------------------
// Pass 2: score
// ---------------------------------------------------------------------------

__global__ void ScoreKernel(const double* __restrict__ fc,
                            const double* __restrict__ fs,
                            const double* __restrict__ stat,
                            const double* __restrict__ tss_all,
                            const double* __restrict__ bank_fs,
                            const double* __restrict__ bank_fs2,
                            size_t n_template,
                            size_t n_band,
                            size_t n_phase,
                            double* __restrict__ best) {
    // job on x and template on y, not the other way round: gridDim.y is
    // capped at 65,535 and the job count is curves times candidate periods,
    // which passes that at a few hundred curves. gridDim.x allows 2^31-1.
    const size_t job = blockIdx.x;
    const size_t k = blockIdx.y;  // template
    const int tid = threadIdx.x;

    extern __shared__ double sh[];
    double* wre = sh;
    double* wim = wre + n_phase;
    double* cs = wim + n_phase;
    double* num = cs + n_phase;
    double* den = num + n_phase;

    const double tss = tss_all[job];
    if (tss <= 0.0) {
        if (tid == 0) {
            best[(job * n_template + k) * 3 + 0] = DBL_MAX;
            best[(job * n_template + k) * 3 + 1] = -1.0;
            best[(job * n_template + k) * 3 + 2] = 0.0;
        }
        return;
    }

    for (size_t i = tid; i < n_phase; i += blockDim.x) {
        num[i] = 0.0;
        den[i] = 0.0;
    }
    __syncthreads();

    for (size_t b = 0; b < n_band; ++b) {
        const double n = stat[(job * n_band + b) * 3 + 0];
        const double mean = stat[(job * n_band + b) * 3 + 1];
        const double tb = stat[(job * n_band + b) * 3 + 2];
        if (tb <= 0.0 || n <= 0.0)
            continue;

        const size_t fo = (job * n_band + b) * n_phase * 2;
        const size_t bo = (k * n_band + b) * n_phase * 2;

        // cS = ifft(Fc * conj(FS))
        for (size_t i = tid; i < n_phase; i += blockDim.x) {
            const double xr = fc[fo + 2 * i], xi = fc[fo + 2 * i + 1];
            const double yr = bank_fs[bo + 2 * i], yi = bank_fs[bo + 2 * i + 1];
            wre[i] = xr * yr + xi * yi;
            wim[i] = xi * yr - xr * yi;
        }
        __syncthreads();
        FFTShared(wre, wim, (int)n_phase, 1);
        for (size_t i = tid; i < n_phase; i += blockDim.x)
            cs[i] = wre[i];
        __syncthreads();

        // sS = ifft(Fs * conj(FS)), giving num
        for (size_t i = tid; i < n_phase; i += blockDim.x) {
            const double xr = fs[fo + 2 * i], xi = fs[fo + 2 * i + 1];
            const double yr = bank_fs[bo + 2 * i], yi = bank_fs[bo + 2 * i + 1];
            wre[i] = xr * yr + xi * yi;
            wim[i] = xi * yr - xr * yi;
        }
        __syncthreads();
        FFTShared(wre, wim, (int)n_phase, 1);
        for (size_t i = tid; i < n_phase; i += blockDim.x)
            num[i] += wre[i] - mean * cs[i];
        __syncthreads();

        // cS2 = ifft(Fc * conj(FS2)), giving den
        for (size_t i = tid; i < n_phase; i += blockDim.x) {
            const double xr = fc[fo + 2 * i], xi = fc[fo + 2 * i + 1];
            const double yr = bank_fs2[bo + 2 * i],
                         yi = bank_fs2[bo + 2 * i + 1];
            wre[i] = xr * yr + xi * yi;
            wim[i] = xi * yr - xr * yi;
        }
        __syncthreads();
        FFTShared(wre, wim, (int)n_phase, 1);
        for (size_t i = tid; i < n_phase; i += blockDim.x)
            den[i] += wre[i] - cs[i] * cs[i] / n;
        __syncthreads();
    }

    // best shift for this template
    double loc_f = DBL_MAX, loc_a = 0.0;
    int loc_i = -1;
    for (size_t i = tid; i < n_phase; i += blockDim.x) {
        const double nu = num[i], de = den[i];
        const double a = de > 1e-14 ? fmax(nu / de, 0.0) : 0.0;
        const double rss = tss - 2.0 * a * nu + a * a * de;
        const double fvu = rss / tss;
        if (fvu < loc_f) {
            loc_f = fvu;
            loc_i = (int)i;
            loc_a = a;
        }
    }
    __shared__ double rf[TFS_THREADS];
    __shared__ double ra[TFS_THREADS];
    __shared__ int ri[TFS_THREADS];
    rf[tid] = loc_f;
    ra[tid] = loc_a;
    ri[tid] = loc_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && rf[tid + s] < rf[tid]) {
            rf[tid] = rf[tid + s];
            ra[tid] = ra[tid + s];
            ri[tid] = ri[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        best[(job * n_template + k) * 3 + 0] = rf[0];
        best[(job * n_template + k) * 3 + 1] = (double)ri[0];
        best[(job * n_template + k) * 3 + 2] = ra[0];
    }
}

// ---------------------------------------------------------------------------
// Pass 3: reduce
// ---------------------------------------------------------------------------

__global__ void ReduceKernel(const double* __restrict__ best,
                             const double* __restrict__ job_period,
                             const size_t* __restrict__ curve_job_off,
                             const size_t* __restrict__ curve_n_jobs,
                             size_t n_template,
                             double* __restrict__ out) {
    const size_t curve = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t j0 = curve_job_off[curve];
    const size_t nj = curve_n_jobs[curve];
    const size_t total = nj * n_template;

    double loc_f = DBL_MAX, loc_p = 0.0, loc_a = 0.0;
    int loc_t = -1, loc_s = -1;
    for (size_t idx = tid; idx < total; idx += blockDim.x) {
        const size_t j = j0 + idx / n_template;
        const size_t k = idx % n_template;
        const double f = best[(j * n_template + k) * 3 + 0];
        if (f < loc_f) {
            loc_f = f;
            loc_t = (int)k;
            loc_s = (int)best[(j * n_template + k) * 3 + 1];
            loc_a = best[(j * n_template + k) * 3 + 2];
            loc_p = job_period[j];
        }
    }
    __shared__ double rf[TFS_THREADS], rp[TFS_THREADS], ra[TFS_THREADS];
    __shared__ int rt[TFS_THREADS], rs[TFS_THREADS];
    rf[tid] = loc_f;
    rp[tid] = loc_p;
    ra[tid] = loc_a;
    rt[tid] = loc_t;
    rs[tid] = loc_s;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && rf[tid + s] < rf[tid]) {
            rf[tid] = rf[tid + s];
            rp[tid] = rp[tid + s];
            ra[tid] = ra[tid + s];
            rt[tid] = rt[tid + s];
            rs[tid] = rs[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const bool ok = rf[0] != DBL_MAX && rt[0] >= 0;
        out[curve * 5 + 0] = ok ? rf[0] : NAN;
        out[curve * 5 + 1] = ok ? rp[0] : NAN;
        out[curve * 5 + 2] = ok ? (double)rt[0] : -1.0;
        out[curve * 5 + 3] = ok ? (double)rs[0] : -1.0;
        out[curve * 5 + 4] = ok ? ra[0] : NAN;
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

TemplateFitSampled::TemplateFitSampled(const double* samples,
                                       size_t n_template,
                                       size_t n_band,
                                       size_t n_phase)
    : n_template_(n_template), n_band_(n_band), n_phase_(n_phase) {
    if ((n_phase & (n_phase - 1)) != 0 || n_phase < 4
        || n_phase > TFS_MAX_PHASE) {
        fprintf(stderr, "tfs: n_phase %zu must be a power of two in [4, %d]\n",
                n_phase, TFS_MAX_PHASE);
        exit(1);
    }
    if (n_band > TFS_MAX_BAND) {
        fprintf(stderr, "tfs: n_band %zu exceeds %d\n", n_band, TFS_MAX_BAND);
        exit(1);
    }

    const size_t nc = n_template * n_band * n_phase;
    std::vector<double> fs(nc * 2, 0.0), fs2(nc * 2, 0.0);
    std::vector<double> re(n_phase), im(n_phase);
    for (size_t kb = 0; kb < n_template * n_band; ++kb) {
        const size_t o = kb * n_phase;
        for (size_t p = 0; p < n_phase; ++p) {
            re[p] = samples[o + p];
            im[p] = 0.0;
        }
        FFTHost(re.data(), im.data(), (int)n_phase, 0);
        for (size_t p = 0; p < n_phase; ++p) {
            fs[(o + p) * 2 + 0] = re[p];
            fs[(o + p) * 2 + 1] = im[p];
        }
        for (size_t p = 0; p < n_phase; ++p) {
            const double s = samples[o + p];
            re[p] = s * s;
            im[p] = 0.0;
        }
        FFTHost(re.data(), im.data(), (int)n_phase, 0);
        for (size_t p = 0; p < n_phase; ++p) {
            fs2[(o + p) * 2 + 0] = re[p];
            fs2[(o + p) * 2 + 1] = im[p];
        }
    }

    gpuErrchk(cudaMalloc(&dev_fs_, fs.size() * sizeof(double)));
    gpuErrchk(cudaMalloc(&dev_fs2_, fs2.size() * sizeof(double)));
    gpuErrchk(cudaMemcpy(dev_fs_, fs.data(), fs.size() * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_fs2_, fs2.data(), fs2.size() * sizeof(double),
                         cudaMemcpyHostToDevice));
}

TemplateFitSampled::~TemplateFitSampled() {
    cudaFree(dev_fs_);
    cudaFree(dev_fs2_);
}

void TemplateFitSampled::CalcTFSBatched(const std::vector<double*>& times,
                                        const std::vector<double*>& mags,
                                        const std::vector<uint8_t*>& bands,
                                        const std::vector<size_t>& lengths,
                                        const std::vector<double*>& periods,
                                        const std::vector<size_t>& n_periods,
                                        size_t min_points,
                                        double* out) const {
    const size_t n_curve = times.size();

    std::vector<double> h_t, h_m;
    std::vector<uint8_t> h_b;
    std::vector<size_t> h_off(n_curve), h_len(n_curve);
    std::vector<double> h_jobper;
    std::vector<size_t> h_jobcur, h_cjoff(n_curve), h_cjn(n_curve);

    for (size_t i = 0; i < n_curve; ++i) {
        h_off[i] = h_t.size();
        h_len[i] = lengths[i];
        h_t.insert(h_t.end(), times[i], times[i] + lengths[i]);
        h_m.insert(h_m.end(), mags[i], mags[i] + lengths[i]);
        h_b.insert(h_b.end(), bands[i], bands[i] + lengths[i]);
        h_cjoff[i] = h_jobper.size();
        h_cjn[i] = n_periods[i];
        for (size_t p = 0; p < n_periods[i]; ++p) {
            h_jobper.push_back(periods[i][p]);
            h_jobcur.push_back(i);
        }
    }
    const size_t n_job = h_jobper.size();

    double *d_t, *d_m, *d_jobper, *d_fc, *d_fs, *d_stat, *d_tss, *d_best,
        *d_out;
    uint8_t* d_b;
    size_t *d_off, *d_len, *d_jobcur, *d_cjoff, *d_cjn;

    gpuErrchk(cudaMalloc(&d_t, h_t.size() * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_m, h_m.size() * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_b, h_b.size() * sizeof(uint8_t)));
    gpuErrchk(cudaMalloc(&d_off, n_curve * sizeof(size_t)));
    gpuErrchk(cudaMalloc(&d_len, n_curve * sizeof(size_t)));
    gpuErrchk(cudaMalloc(&d_jobper, n_job * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_jobcur, n_job * sizeof(size_t)));
    gpuErrchk(cudaMalloc(&d_cjoff, n_curve * sizeof(size_t)));
    gpuErrchk(cudaMalloc(&d_cjn, n_curve * sizeof(size_t)));
    gpuErrchk(
        cudaMalloc(&d_fc, n_job * n_band_ * n_phase_ * 2 * sizeof(double)));
    gpuErrchk(
        cudaMalloc(&d_fs, n_job * n_band_ * n_phase_ * 2 * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_stat, n_job * n_band_ * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_tss, n_job * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_best, n_job * n_template_ * 3 * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_out, n_curve * 5 * sizeof(double)));

    gpuErrchk(cudaMemcpy(d_t, h_t.data(), h_t.size() * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_m, h_m.data(), h_m.size() * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_off, h_off.data(), n_curve * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_len, h_len.data(), n_curve * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_jobper, h_jobper.data(), n_job * sizeof(double),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_jobcur, h_jobcur.data(), n_job * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cjoff, h_cjoff.data(), n_curve * sizeof(size_t),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cjn, h_cjn.data(), n_curve * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    const size_t sh_fold = 4 * n_phase_ * sizeof(double);
    FoldKernel<<<n_job, TFS_THREADS, sh_fold>>>(
        d_t, d_m, d_b, d_off, d_len, d_jobper, d_jobcur, n_band_, n_phase_,
        min_points, d_fc, d_fs, d_stat, d_tss);
    gpuErrchk(cudaPeekAtLastError());

    const size_t sh_score = 5 * n_phase_ * sizeof(double);
    dim3 grid((unsigned int)n_job, (unsigned int)n_template_);
    ScoreKernel<<<grid, TFS_THREADS, sh_score>>>(d_fc, d_fs, d_stat, d_tss,
                                                 dev_fs_, dev_fs2_, n_template_,
                                                 n_band_, n_phase_, d_best);
    gpuErrchk(cudaPeekAtLastError());

    ReduceKernel<<<n_curve, TFS_THREADS>>>(d_best, d_jobper, d_cjoff, d_cjn,
                                           n_template_, d_out);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(out, d_out, n_curve * 5 * sizeof(double),
                         cudaMemcpyDeviceToHost));

    cudaFree(d_t);
    cudaFree(d_m);
    cudaFree(d_b);
    cudaFree(d_off);
    cudaFree(d_len);
    cudaFree(d_jobper);
    cudaFree(d_jobcur);
    cudaFree(d_cjoff);
    cudaFree(d_cjn);
    cudaFree(d_fc);
    cudaFree(d_fs);
    cudaFree(d_stat);
    cudaFree(d_tss);
    cudaFree(d_best);
    cudaFree(d_out);
}
