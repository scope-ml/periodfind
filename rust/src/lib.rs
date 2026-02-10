use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;

mod aov;
mod ce;
mod fold;
mod fourier;
mod fpw;
mod ls;

/// Compute batched Conditional Entropy periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_ce_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();

    let mut output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));

    for curve_idx in 0..n_curves {
        let times_slice = times_list[curve_idx].as_slice()?;
        let mags_slice = mags_list[curve_idx].as_slice()?;

        let result = ce::calc_ce(
            times_slice,
            mags_slice,
            periods_slice,
            period_dts_slice,
            num_phase,
            num_mag,
            phase_overlap,
            mag_overlap,
        );

        for period_idx in 0..n_periods {
            for pdt_idx in 0..n_pdts {
                output[[curve_idx, period_idx, pdt_idx]] = result[period_idx * n_pdts + pdt_idx];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

/// Compute batched Analysis of Variance periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_aov_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    num_overlap: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();

    let mut output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));

    for curve_idx in 0..n_curves {
        let times_slice = times_list[curve_idx].as_slice()?;
        let mags_slice = mags_list[curve_idx].as_slice()?;

        let result = aov::calc_aov(
            times_slice,
            mags_slice,
            periods_slice,
            period_dts_slice,
            num_bins,
            num_overlap,
        );

        for period_idx in 0..n_periods {
            for pdt_idx in 0..n_pdts {
                output[[curve_idx, period_idx, pdt_idx]] = result[period_idx * n_pdts + pdt_idx];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

/// Compute batched Lomb-Scargle periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_ls_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();

    let mut output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));

    for curve_idx in 0..n_curves {
        let times_slice = times_list[curve_idx].as_slice()?;
        let mags_slice = mags_list[curve_idx].as_slice()?;

        let result = ls::calc_ls(times_slice, mags_slice, periods_slice, period_dts_slice);

        for period_idx in 0..n_periods {
            for pdt_idx in 0..n_pdts {
                output[[curve_idx, period_idx, pdt_idx]] = result[period_idx * n_pdts + pdt_idx];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

/// Compute batched FPW periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_fpw_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();

    let mut output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));

    for curve_idx in 0..n_curves {
        let times_slice = times_list[curve_idx].as_slice()?;
        let mags_slice = mags_list[curve_idx].as_slice()?;
        let errs_slice = errs_list[curve_idx].as_slice()?;

        let result = fpw::calc_fpw(
            times_slice,
            mags_slice,
            errs_slice,
            periods_slice,
            period_dts_slice,
            num_bins,
        );

        for period_idx in 0..n_periods {
            for pdt_idx in 0..n_pdts {
                output[[curve_idx, period_idx, pdt_idx]] = result[period_idx * n_pdts + pdt_idx];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

/// Compute batched Fourier decomposition (feature extraction).
///
/// Returns a 2D numpy array of shape (n_curves, 14).
/// Features: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
#[pyfunction]
fn calc_fourier_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let periods_slice = periods.as_slice()?;
    let n_curves = times_list.len();

    // Extract slices while the GIL is held
    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    // Release the GIL for the Rayon-parallel computation
    let flat = py.allow_threads(|| {
        fourier::calc_fourier_batch(&times_vecs, &mags_vecs, &errs_vecs, periods_slice)
    });

    // Reshape into a 2D array
    let n_feat = fourier::NUM_FEATURES;
    let mut output = Array2::<f32>::zeros((n_curves, n_feat));
    for i in 0..n_curves {
        for j in 0..n_feat {
            output[[i, j]] = flat[i * n_feat + j];
        }
    }

    Ok(output.into_pyarray(py).into())
}

/// Native CPU implementations of period-finding algorithms.
#[pymodule]
fn periodfind_cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_ce_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_aov_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ls_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fpw_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fourier_batched, m)?)?;
    Ok(())
}
