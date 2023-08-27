use pyo3::{pyclass, pymethods};

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
    pub ch1: i32,
    pub ch2: i32,
    pub ch3: i32,
    pub ch4: i32,
}

#[pymethods]
impl PyColorBounds {
    /// There is main constructor (__init__(self)) of PyColorBounds class.
    ///
    /// ## Parameters:
    /// * ch1: (i32) a passed stream frame (ndarray) to transform.
    /// * ch2: (i32) a passed stream frame (ndarray) to transform.
    /// * ch3: (i32) a passed stream frame (ndarray) to transform.
    /// * ch4: (i32) a passed stream frame (ndarray) to transform.
    #[new]
    fn new(ch1: i32, ch2: i32, ch3: i32, ch4: i32) -> PyColorBounds {
        PyColorBounds { ch1, ch2, ch3, ch4 }
    }
}
