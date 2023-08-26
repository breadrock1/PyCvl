use cvlcore::core::mat::*;

use numpy::ndarray::{Array2, Array3};
use numpy::{Ix2, Ix3, ToPyArray};
use numpy::{PyArray, PyReadonlyArray2, PyReadonlyArray3};

use pyo3::exceptions::PyException;
use pyo3::prelude::{Py, PyResult, Python};

pub fn convert_mat_to_pyarray_2(py: Python<'_>, frame: CvlMat) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let rows = frame.rows() as usize;
    let cols = frame.columns() as usize;
    let vec_data = frame.to_slice().unwrap().to_vec();
    match Array2::from_shape_vec((rows, cols), vec_data) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(array) => Ok(array.to_pyarray(py).to_owned()),
    }
}

pub fn convert_mat_to_pyarray_3(py: Python<'_>, frame: CvlMat) -> PyResult<Py<PyArray<f64, Ix3>>> {
    let rows = frame.rows() as usize;
    let cols = frame.columns() as usize;
    let chls = frame.channels() as usize;
    match Array3::<f64>::from_shape_vec((rows, cols, chls), frame.to_scalar_vec()) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(array) => Ok(array.to_pyarray(py).to_owned()),
    }
}

pub fn convert_array_2_to_mat(frame: PyReadonlyArray2<u8>) -> CvlMat {
    let shape_slice = frame.shape();
    let array_data = frame.as_slice().unwrap();
    CvlMat::new_with_data(shape_slice[0] as i32, shape_slice[1] as i32, 0, array_data)
}

pub fn convert_array_3_to_mat(frame: PyReadonlyArray3<u8>) -> CvlMat {
    let shape_slice = frame.shape();
    let array_data = frame.as_slice().unwrap();
    CvlMat::new_with_data(shape_slice[0] as i32, shape_slice[1] as i32, 16, array_data)
}
