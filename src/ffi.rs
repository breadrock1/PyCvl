use crate::converter::*;

use cvlcore::core::bounds::*;
use cvlcore::core::cvl::*;
use cvlcore::core::mat::*;

use numpy::{Ix2, Ix3};
use numpy::{PyArray, PyReadonlyArray2, PyReadonlyArray3};

use pyo3::prelude::{pyclass, pymethods, pymodule, pyfunction, wrap_pyfunction};
use pyo3::prelude::{Py, Python, PyModule, PyResult};
use pyo3::exceptions::PyException;

use std::rc::Rc;


#[pyclass(subclass)]
pub struct PyColorBounds {
    ch1: i32,
    ch2: i32,
    ch3: i32,
    ch4: i32,
}

#[pymethods]
impl PyColorBounds {
    #[new]
    fn new(ch1: i32, ch2: i32, ch3: i32, ch4: i32) -> PyColorBounds {
        PyColorBounds { ch1, ch2, ch3, ch4 }
    }
}

#[pyfunction]
fn median(_py: Python<'_>, frame: PyReadonlyArray2<u8>) -> PyResult<f64> {
    let mat = convert_array_2_to_mat(frame);
    Ok(calculate_mat_median(&mat).unwrap_or(0.0))
}

#[pyfunction]
fn grayscale(py: Python<'_>, frame: PyReadonlyArray3<u8>) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_3_to_mat(frame);
    println!("{} {} {}", &mat.rows(), &mat.columns(), &mat.channels());
    println!("{}", &mat.to_slice().unwrap().len());
    match gen_grayscale_frame(&mat) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

#[pyfunction]
fn threshold(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    thresh: f64,
    maxval: f64
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_threshold_frame(&mat, thresh, maxval) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

#[pyfunction]
fn canny(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    low: f64,
    high: f64,
    size: i32,
    is_l2: bool
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_canny_frame(&mat, low, high, size, is_l2) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

#[pyfunction]
fn canny_sigma(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    size: i32,
    sigma: f64,
    is_l2: bool
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_canny_frame_by_sigma(&mat, size, sigma, is_l2) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

#[pyfunction]
fn difference(py: Python<'_>, frames: Vec<PyReadonlyArray2<u8>>) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let cvlmats: Vec<Rc<CvlMat>> = frames
        .into_iter()
        .map(|f| convert_array_2_to_mat(f))
        .map(Rc::new)
        .collect();

    match gen_abs_frame(&cvlmats) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

#[pyfunction]
fn difference_reduce(py: Python<'_>, frames: Vec<PyReadonlyArray2<u8>>) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let cvlmats: Vec<Rc<CvlMat>> = frames
        .into_iter()
        .map(|f| convert_array_2_to_mat(f))
        .map(Rc::new)
        .collect();

    match gen_abs_frame_reduce(&cvlmats) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

// #[pyfunction]
// fn difference_reduce<'a>(py: Python<'a>, frames: PyList) -> PyResult<&'a PyByteArray> {
//
// }

#[pyfunction]
fn vibration(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    neighbours: i32,
    window_size: i32,
    color_bounds: Py<PyColorBounds>,
) -> PyResult<Py<PyArray<f64, Ix3>>> {
    let ref_bounds = color_bounds.as_ref(py).borrow();
    let bounds = ColorBounds::new(ref_bounds.ch1, ref_bounds.ch2, ref_bounds.ch3, ref_bounds.ch4);
    let mat = convert_array_2_to_mat(frame);
    match compute_vibration(&mat, neighbours, window_size, &bounds) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_3(py, result),
    }
}

#[pymodule]
fn pycvl(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyColorBounds>()?;
    module.add_function(wrap_pyfunction!(canny, module)?)?;
    module.add_function(wrap_pyfunction!(median, module)?)?;
    module.add_function(wrap_pyfunction!(grayscale, module)?)?;
    module.add_function(wrap_pyfunction!(threshold, module)?)?;
    module.add_function(wrap_pyfunction!(vibration, module)?)?;
    module.add_function(wrap_pyfunction!(difference, module)?)?;
    module.add_function(wrap_pyfunction!(canny_sigma, module)?)?;
    module.add_function(wrap_pyfunction!(difference_reduce, module)?)?;
    Ok(())
}
