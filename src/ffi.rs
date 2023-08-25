use crate::converter::*;

use cvlcore::core::bounds::*;
use cvlcore::core::cvl::*;
use cvlcore::core::mat::*;

use numpy::{Ix2, Ix3};
use numpy::{PyArray, PyReadonlyArray2, PyReadonlyArray3};

use pyo3::exceptions::PyException;
use pyo3::prelude::{pyclass, pyfunction, pymethods, pymodule, wrap_pyfunction};
use pyo3::prelude::{Py, PyModule, PyResult, Python};

use std::rc::Rc;

/// There is helper structure to store channels values to setting up colors to pixels
/// while generating vibration-image. For example we have vibration-image map where
/// each pixel value is count of neighbours pixels:
///
///     0 1 1    0 1 1
///     1 0 1 -> 1 6 1
///     1 1 0    1 1 0
///
/// Then we must set color on each modified pixel by each PyColorBounds channel. For example,
/// we have PyColorBounds with values: 5, 6, 7, 8. Then we find min channel value which more or
/// equals to current pixel (6).
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

/// This method returns arithmetic mean (average) of all elements in array.
/// In mathematics and statistics, the arithmetic mean / arithmetic average is the sum of a
/// collection of numbers divided by the count of numbers in the collection. The collection
/// is often a set of results from an experiment, an observational study, or a survey. The
/// term "arithmetic mean" is preferred in some mathematics and statistics contexts because
/// it helps distinguish it from other types of means, such as geometric and harmonic.
///
/// ## Parameters:
/// * frame: (&PyReadonlyArray3<u8>) the passed stream frame (ndarray) to transform.
///
/// ## Results:
/// Returns `Ok(PyResult<f64>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) if failed while median value of passed frame.
#[pyfunction]
fn median(_py: Python<'_>, frame: PyReadonlyArray2<u8>) -> PyResult<f64> {
    let mat = convert_array_2_to_mat(frame);
    Ok(calculate_mat_median(&mat).unwrap_or(0.0))
}

/// Transformations within RGB space like adding/removing the alpha channel, reversing the
/// channel order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as
/// conversion to/from grayscale.
///
/// ![grayscale](/resources/grayscale.jpg "Example of Grayscale image")
///
/// ## Parameters:
/// * frame: (&PyReadonlyArray3<u8>) the passed stream frame (ndarray) to transform.
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to grayscale image.
#[pyfunction]
fn grayscale(py: Python<'_>, frame: PyReadonlyArray3<u8>) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_3_to_mat(frame);
    match gen_grayscale_frame(&mat) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

/// This method returns threshold image from passed bgr-image by passed black/white bounds
/// values. The simplest thresholding methods replace each pixel in an image with a black
/// pixel if the image intensity less than a fixed value called the threshold if the pixel
/// intensity is greater than that threshold. This function is necessary for further image
/// transformation to generate pixels vibration image.
///
/// ## Parameters
/// * frame: (&PyReadonlyArray3<u8>) the passed stream frame (ndarray) to transform.
/// * tresh: (f64) the black bound-value to swap pixel value to 1.
/// * maxval: (f64) the white bound-value to swap pixel value to 0.
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to threshold image.
#[pyfunction]
fn threshold(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    thresh: f64,
    maxval: f64,
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_threshold_frame(&mat, thresh, maxval) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ![canny](/resources/canny.jpg "Example of Canny image")
///
/// ## Parameters:
/// * frame: (&PyReadonlyArray3<u8>) the passed stream frame (ndarray) to transform.
/// * low: (f64) the first threshold for the hysteresis procedure.
/// * high: (f64) the second threshold for the hysteresis procedure.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to canny image.
#[pyfunction]
fn canny(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    low: f64,
    high: f64,
    size: i32,
    is_l2: bool,
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_canny_frame(&mat, low, high, size, is_l2) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

/// This method returns canny image from passed grayscale image by passed parameters.
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
/// Canny also produced a computational theory of edge detection explaining why the
/// technique works.
///
/// ![canny](/resources/canny.jpg "Example of Canny image")
///
/// ## Parameters:
/// * frame: (&PyReadonlyArray3<u8>) the passed stream frame (ndarray) to transform.
/// * size: (i32) the aperture size of Sobel operator to generate Canny view.
/// * sigma: (f64) the value to vary the percentage thresholds that are determined based on simple statistics.
/// * is_l2: (bool) the specifies the equation for finding gradient magnitude.
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to canny image.
#[pyfunction]
fn canny_sigma(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    size: i32,
    sigma: f64,
    is_l2: bool,
) -> PyResult<Py<PyArray<u8, Ix2>>> {
    let mat = convert_array_2_to_mat(frame);
    match gen_canny_frame_by_sigma(&mat, size, sigma, is_l2) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_2(py, result),
    }
}

/// This recursive method returns result-image of opencv::absdiff() method by passed
/// list of followed one by one frames of video stream. A result-image presents matrix
/// Absolute difference between two 2D-arrays when they have the same size and type
/// which used for removing from further analysis static pixels.
///
/// ![difference](/resources/difference.jpg "Example of Difference image")
///
/// For example, we have both matrix:
///
///     0 1 0       0 1 0      0 0 0
///     1 0 1  and  1 1 1  =>  0 1 0
///     0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&Vec<PyReadonlyArray2<u8>>) a list of frames to get difference-image;
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to difference-image.
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

/// This method returns reduced result-image of opencv::absdiff() method by passed
/// list of followed one by one frames of video stream. A result-image presents matrix
/// Absolute difference between two 2D-arrays when they have the same size and type
/// which used for removing from further analysis static pixels.
///
/// ![difference](/resources/difference.jpg "Example of Difference image")
///
/// For example, we have both matrix:
///
///     0 1 0       0 1 0      0 0 0
///     1 0 1  and  1 1 1  =>  0 1 0
///     0 1 0       0 1 0      0 0 0
///
/// ## Parameters:
/// * frame_images: (&Vec<PyReadonlyArray2<u8>>) a list of frames to get difference-image;
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<u8, Ix2>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to difference-image.
#[pyfunction]
fn difference_reduce(
    py: Python<'_>,
    frames: Vec<PyReadonlyArray2<u8>>,
) -> PyResult<Py<PyArray<u8, Ix2>>> {
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

/// This method returns image with vibrating pixels (colored by bounds values) by passed image.
/// The main algorithm iterates over each pixel of Canny-image and calculate amount of nonzero
/// pixels around current pixel. A target computed value replaced instead pixel value.
/// The vibration image is network procedure which used for anxiety triggering by dispersion
/// of statistic values.
///
/// ![vibration](/resources/vibration.jpg "Example of Vibration image")
///
/// ## Parameters:
/// * frame: (&PyReadonlyArray2<u8>) the passed stream frame (ndarray) to transform.
/// * color_bounds: (&ColorBounds) a object with channels values to set color for pixels.
/// * neighbours: (i32) a neighbours count value to filter noise of vibration.
/// * window_size: (i32) a offset from central pixel to compute non-null pixel neighbours.
///
/// ## Returns:
/// Returns `Ok(Py<PyArray<f64, Ix3>>)` on success, otherwise returns an error.
///
/// ## Errors:
/// Returns [`PyException`](PyException) from [`ProcessError`](ProcessError) if failed
/// while trying to transform passed image to vibration-image.
#[pyfunction]
fn vibration(
    py: Python<'_>,
    frame: PyReadonlyArray2<u8>,
    color_bounds: Py<PyColorBounds>,
    neighbours: i32,
    window_size: i32,
) -> PyResult<Py<PyArray<f64, Ix3>>> {
    let ref_bounds = color_bounds.as_ref(py).borrow();
    let bounds = ColorBounds::new(
        ref_bounds.ch1,
        ref_bounds.ch2,
        ref_bounds.ch3,
        ref_bounds.ch4,
    );
    let mat = convert_array_2_to_mat(frame);
    match compute_vibration(&mat, neighbours, window_size, &bounds) {
        Err(err) => Err(PyException::new_err(err.to_string())),
        Ok(result) => convert_mat_to_pyarray_3(py, result),
    }
}

// #[pyfunction]
// fn difference_reduce<'a>(py: Python<'a>, frames: PyList) -> PyResult<&'a PyByteArray> {
//
// }

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
