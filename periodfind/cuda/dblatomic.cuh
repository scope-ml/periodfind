// Copyright 2026. Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file.

#ifndef PERIODFIND_CUDA_DBLATOMIC_CUH_
#define PERIODFIND_CUDA_DBLATOMIC_CUH_

// atomicAdd on double is native only from compute capability 6.0, and the arch
// list in setup.py starts at compute_50, so the older targets need the
// documented compare-and-swap fallback.
//
// Accumulating in float instead, as the older kernels here do, is not an option
// for the template fitters: the score is an argmin over tens of thousands of
// nearly tied columns, and in single precision a different column wins. Pascal
// onwards takes the native path, so no GPU in practice pays for this.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline double PfAtomicAdd(double* addr, double val) {
    unsigned long long int* p = (unsigned long long int*)addr;
    unsigned long long int old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            p, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double PfAtomicAdd(double* addr, double val) {
    return atomicAdd(addr, val);
}
#endif

#endif  // PERIODFIND_CUDA_DBLATOMIC_CUH_
