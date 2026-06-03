//! Primitive neural-network ops over `(location, time, feature)` arrays.
//!
//! Batch dimension is the location axis; there is no extra batch axis at inference.
//! All arithmetic is in `f32` to match JAX's default precision.

use ndarray::{Array1, Array2, Array3, s};

use crate::weights::Cell;

/// Elementwise ReLU.
pub fn relu3(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Numerically stable softplus, `ln(1 + exp(x))`.
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-(x.abs())).exp().ln_1p()
}

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply a dense layer (`x @ kernel + bias`) over the last axis of a 3-D array.
///
/// `x` is `(loc, time, in)`, `kernel` is `(in, out)`, `bias` is `(out,)`.
pub fn dense3(x: &Array3<f32>, kernel: &Array2<f32>, bias: &Array1<f32>) -> Array3<f32> {
    let (n_loc, t, _) = x.dim();
    let out = kernel.ncols();
    let mut result = Array3::<f32>::zeros((n_loc, t, out));
    for l in 0..n_loc {
        let xl = x.slice(s![l, .., ..]); // (time, in)
        let mut yl = xl.dot(kernel); // (time, out)
        for mut row in yl.rows_mut() {
            row += bias;
        }
        result.slice_mut(s![l, .., ..]).assign(&yl);
    }
    result
}

/// Concatenate two `(loc, time, *)` arrays along the feature axis.
pub fn concat_features(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    ndarray::concatenate(ndarray::Axis(2), &[a.view(), b.view()]).expect("feature concat shapes")
}

/// Concatenate two `(loc, *, feat)` arrays along the time axis.
pub fn concat_time(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
    ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view()]).expect("time concat shapes")
}

/// Run a `SimpleCell` RNN over a sequence, returning the full state sequence.
///
/// `inputs` is `(loc, time, in)`. The carry starts from `initial` (or zeros) and
/// each step computes `h = tanh(x @ Wi + bi + h @ Wh)`. Returns `(loc, time, hidden)`.
pub fn run_rnn(cell: &Cell, inputs: &Array3<f32>, initial: Option<&Array2<f32>>) -> Array3<f32> {
    let (n_loc, t, _) = inputs.dim();
    let hidden = cell.i_kernel.ncols();
    let mut h: Array2<f32> = match initial {
        Some(c) => c.clone(),
        None => Array2::<f32>::zeros((n_loc, hidden)),
    };
    let mut states = Array3::<f32>::zeros((n_loc, t, hidden));
    for step in 0..t {
        let x_t = inputs.slice(s![.., step, ..]); // (loc, in)
        let mut pre = x_t.dot(&cell.i_kernel); // (loc, hidden)
        for mut row in pre.rows_mut() {
            row += &cell.i_bias;
        }
        pre = pre + h.dot(&cell.h_kernel);
        h = pre.mapv(|v| v.tanh());
        states.slice_mut(s![.., step, ..]).assign(&h);
    }
    states
}

/// Take the last time step of a `(loc, time, feat)` array as `(loc, feat)`.
pub fn last_step(states: &Array3<f32>) -> Array2<f32> {
    let t = states.dim().1;
    states.slice(s![.., t - 1, ..]).to_owned()
}
