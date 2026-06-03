//! Load the exported network parameters from `weights.safetensors`.
//!
//! The keys are the flattened Flax pytree paths produced by `export_weights.py`.
//! Only the `base` architecture is loaded here.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;

/// A `SimpleCell` (Elman RNN) parameter set: `tanh(i(x) + h(h_prev))`.
pub struct Cell {
    /// Input kernel `(in, hidden)`.
    pub i_kernel: Array2<f32>,
    /// Input bias `(hidden,)`.
    pub i_bias: Array1<f32>,
    /// Hidden kernel `(hidden, hidden)`, no bias.
    pub h_kernel: Array2<f32>,
}

/// All parameters of the `base` `ARModel2` network.
pub struct Params {
    /// Per-location embedding table, `(n_locations, embedding_dim)`.
    pub embedding: Array2<f32>,
    /// Preprocess first dense kernel, `(features + embedding_dim, hidden)`.
    pub pre_d0_kernel: Array2<f32>,
    /// Preprocess first dense bias.
    pub pre_d0_bias: Array1<f32>,
    /// Preprocess output dense kernel, `(hidden, 2)`.
    pub pre_d1_kernel: Array2<f32>,
    /// Preprocess output dense bias.
    pub pre_d1_bias: Array1<f32>,
    /// Encoder RNN cell (reads the context with the lagged target joined in).
    pub cell_pre: Cell,
    /// Decoder RNN cell (rolls across the forecast horizon).
    pub cell_post: Cell,
    /// Head first dense kernel, `(hidden, 6)`.
    pub head_d0_kernel: Array2<f32>,
    /// Head first dense bias.
    pub head_d0_bias: Array1<f32>,
    /// Head output dense kernel, `(6, 2)` -> the two `eta` channels.
    pub head_d1_kernel: Array2<f32>,
    /// Head output dense bias.
    pub head_d1_bias: Array1<f32>,
}

/// Decoded tensors keyed by their flattened path.
struct Tensors {
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        let mut map = HashMap::new();
        for (name, view) in st.tensors() {
            if view.dtype() != safetensors::Dtype::F32 {
                bail!("tensor {name:?} is not float32");
            }
            let data: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            map.insert(name, (view.shape().to_vec(), data));
        }
        Ok(Tensors { map })
    }

    fn take(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor {name:?}"))
    }

    fn arr1(&self, name: &str) -> Result<Array1<f32>> {
        let (shape, data) = self.take(name)?;
        if shape.len() != 1 {
            bail!("tensor {name:?} expected rank 1, got shape {shape:?}");
        }
        Ok(Array1::from(data.clone()))
    }

    fn arr2(&self, name: &str) -> Result<Array2<f32>> {
        let (shape, data) = self.take(name)?;
        if shape.len() != 2 {
            bail!("tensor {name:?} expected rank 2, got shape {shape:?}");
        }
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .with_context(|| format!("reshaping tensor {name:?}"))
    }
}

impl Params {
    /// Load the `base` parameters from a weights directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let t = Tensors::load(&dir.join("weights.safetensors"))?;
        Ok(Params {
            embedding: t.arr2("params/preprocess/Embed_0/embedding")?,
            pre_d0_kernel: t.arr2("params/preprocess/Dense_0/kernel")?,
            pre_d0_bias: t.arr1("params/preprocess/Dense_0/bias")?,
            pre_d1_kernel: t.arr2("params/preprocess/Dense_1/kernel")?,
            pre_d1_bias: t.arr1("params/preprocess/Dense_1/bias")?,
            cell_pre: Cell {
                i_kernel: t.arr2("params/cell_pre/i/kernel")?,
                i_bias: t.arr1("params/cell_pre/i/bias")?,
                h_kernel: t.arr2("params/cell_pre/h/kernel")?,
            },
            cell_post: Cell {
                i_kernel: t.arr2("params/cell_post/i/kernel")?,
                i_bias: t.arr1("params/cell_post/i/bias")?,
                h_kernel: t.arr2("params/cell_post/h/kernel")?,
            },
            head_d0_kernel: t.arr2("params/Dense_0/kernel")?,
            head_d0_bias: t.arr1("params/Dense_0/bias")?,
            head_d1_kernel: t.arr2("params/Dense_1/kernel")?,
            head_d1_bias: t.arr1("params/Dense_1/bias")?,
        })
    }
}
