//! Feature standardization, mirroring `transforms.ZScaler`.
//!
//! Applies `(x - mu) / std` per feature. The Python path scales in float64 (NumPy)
//! and then hands the array to JAX, which truncates to float32; we reproduce that
//! by scaling in `f64` and casting the result to `f32`.

use ndarray::{Array3, ArrayView3};

use crate::config::Scaler;

/// Standardize features and cast to `f32`.
///
/// `x` has shape `(locations, time, 4)`; `mu`/`std` are per-feature 4-vectors.
pub fn apply(scaler: &Scaler, x: ArrayView3<f64>) -> Array3<f32> {
    let n_features = x.shape()[2];
    let mut out = Array3::<f32>::zeros(x.raw_dim());
    for (idx, &value) in x.indexed_iter() {
        let f = idx.2 % n_features;
        let std = scaler.std[f];
        let scaled = (value - scaler.mu[f]) / std;
        out[[idx.0, idx.1, idx.2]] = scaled as f32;
    }
    out
}
