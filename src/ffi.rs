use crate::bounds::PyColorBounds;
use crate::helper::PyCvlHelper;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[pymodule]
fn pycvl(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyColorBounds>()?;
    module.add_class::<PyCvlHelper>()?;
    Ok(())
}
