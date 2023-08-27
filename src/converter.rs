use crate::helper::{PyMatrix2, PyMatrix3F};

use cvlcore::core::mat::*;

use numpy::ndarray::{Array2, Array3, Dimension};
use numpy::{PyArray, ToPyArray};

use pyo3::exceptions::PyException;
use pyo3::prelude::{Py, PyResult, Python};

pub fn convert_mat_to_pyarray_2(py: Python<'_>, frame: CvlMat) -> PyResult<Py<PyMatrix2>> {
    let rows = frame.rows() as usize;
    let cols = frame.columns() as usize;
    let vec_data = frame.to_slice().unwrap().to_vec();
    match Array2::from_shape_vec((rows, cols), vec_data) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(array) => Ok(array.to_pyarray(py).to_owned()),
    }
}

pub fn convert_mat_to_pyarray_3(py: Python<'_>, frame: CvlMat) -> PyResult<Py<PyMatrix3F>> {
    let rows = frame.rows() as usize;
    let cols = frame.columns() as usize;
    let chls = frame.channels() as usize;
    match Array3::<f64>::from_shape_vec((rows, cols, chls), frame.to_scalar_vec()) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(array) => Ok(array.to_pyarray(py).to_owned()),
    }
}

pub fn convert_array_to_mat<T: Dimension>(frame: &PyArray<u8, T>, cv_type: i32) -> CvlMat {
    let shape_slice = frame.shape();
    let array_data = unsafe { frame.as_slice().unwrap() };
    CvlMat::new_with_data(
        shape_slice[0] as i32,
        shape_slice[1] as i32,
        cv_type,
        array_data,
    )
}
