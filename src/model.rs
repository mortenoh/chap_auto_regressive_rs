//! The `base` `ARModel2` forward pass (deterministic; dropout is inference-off).
//!
//! Reproduces `rnn_model.ARModel2.__call__`: per-location preprocessing, the
//! auto-regressive join, an encoder RNN over the context, a decoder RNN over the
//! forecast horizon seeded by the encoder's final state, and a two-layer head that
//! emits the two-channel `eta` consumed by the negative-binomial sampler.

use ndarray::{Array2, Array3, s};

use crate::layers::{concat_features, concat_time, dense3, last_step, relu3, run_rnn};
use crate::weights::Params;

/// Build the per-location embedding broadcast over time, shape `(loc, time, emb)`.
fn embedding_block(params: &Params, n_loc: usize, t: usize) -> Array3<f32> {
    let emb = params.embedding.ncols();
    let mut block = Array3::<f32>::zeros((n_loc, t, emb));
    for l in 0..n_loc {
        let row = params.embedding.row(l);
        for ti in 0..t {
            block.slice_mut(s![l, ti, ..]).assign(&row);
        }
    }
    block
}

/// The `Preprocess` stage: embed locations, project features (dropout is identity).
fn preprocess(params: &Params, x: &Array3<f32>) -> Array3<f32> {
    let (n_loc, t, _) = x.dim();
    let emb = embedding_block(params, n_loc, t);
    let joined = concat_features(x, &emb); // (loc, time, 4 + emb)
    let h = relu3(&dense3(&joined, &params.pre_d0_kernel, &params.pre_d0_bias));
    dense3(&h, &params.pre_d1_kernel, &params.pre_d1_bias) // (loc, time, 2)
}

/// The auto-regressive join (`ARAdder`): prepend the lagged target onto the
/// processed features over the context window. `pre` is `(loc, time, 2)`, `ar_y`
/// is `(loc, context)`; returns `(loc, context, 3)`.
fn ar_join(pre: &Array3<f32>, ar_y: &Array2<f32>) -> Array3<f32> {
    let (n_loc, context) = ar_y.dim();
    // y[..., None]
    let y_block = ar_y
        .clone()
        .into_shape_with_order((n_loc, context, 1))
        .expect("ar_y reshape");
    // x[..., 1 : context + 1, :]
    let feature_block = pre.slice(s![.., 1..context + 1, ..]).to_owned();
    concat_features(&y_block, &feature_block)
}

/// Run the full forward pass, returning `eta` of shape `(loc, context+future-1, 2)`.
///
/// `scaled_x` is the standardized feature window `(loc, context+future, 4)`;
/// `ar_y` is the interpolated context target `(loc, context)`.
pub fn forward(params: &Params, scaled_x: &Array3<f32>, ar_y: &Array2<f32>) -> Array3<f32> {
    let context = ar_y.dim().1;
    let pre = preprocess(params, scaled_x);

    let prev_x = ar_join(&pre, ar_y);
    let states = run_rnn(&params.cell_pre, &prev_x, None); // (loc, context, 4)

    // Decoder reads the processed features beyond the context (no observed cases),
    // starting from the encoder's final hidden state.
    let decoder_input = pre.slice(s![.., context + 1.., ..]).to_owned(); // (loc, future-1, 2)
    let initial = last_step(&states);
    let new_states = run_rnn(&params.cell_post, &decoder_input, Some(&initial));

    let combined = concat_time(&states, &new_states); // (loc, context+future-1, 4)
    let h = relu3(&dense3(
        &combined,
        &params.head_d0_kernel,
        &params.head_d0_bias,
    ));
    dense3(&h, &params.head_d1_kernel, &params.head_d1_bias) // (loc, ..., 2)
}

/// Slice the forecast-horizon outputs from `eta`: `eta[:, context-1:, :]`, shape
/// `(loc, future, 2)`.
pub fn forecast_slice(eta: &Array3<f32>, context: usize) -> Array3<f32> {
    eta.slice(s![.., context - 1.., ..]).to_owned()
}
