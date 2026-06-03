//! Row-wise linear interpolation over NaNs, mirroring `data_loader.interpolate_nans`.
//!
//! The past target is fed back into the network as the auto-regressive input, so
//! gaps in surveillance (NaNs) are filled by linear interpolation within each
//! location's row. A row with no observations at all is filled with zeros.

use ndarray::Array2;

/// Return a copy of `y` (`(locations, periods)`) with NaNs linearly interpolated
/// within each row; an all-NaN row becomes all zeros.
pub fn interpolate_nans(y: &Array2<f64>) -> Array2<f64> {
    let mut out = y.clone();
    for mut row in out.rows_mut() {
        let n = row.len();
        let observed: Vec<usize> = (0..n).filter(|&i| !row[i].is_nan()).collect();
        if observed.is_empty() {
            row.fill(0.0);
            continue;
        }
        for i in 0..n {
            if !row[i].is_nan() {
                continue;
            }
            row[i] = interp_at(i, &observed, &row);
        }
    }
    out
}

/// Linear interpolation at index `i` given the sorted `observed` indices, matching
/// `numpy.interp` (clamped to the endpoints outside the observed range).
fn interp_at(i: usize, observed: &[usize], row: &ndarray::ArrayViewMut1<f64>) -> f64 {
    let first = observed[0];
    let last = *observed.last().unwrap();
    if i <= first {
        return row[first];
    }
    if i >= last {
        return row[last];
    }
    // Find the observed points bracketing i.
    let pos = observed.partition_point(|&o| o < i);
    let lo = observed[pos - 1];
    let hi = observed[pos];
    let t = (i - lo) as f64 / (hi - lo) as f64;
    row[lo] + t * (row[hi] - row[lo])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fills_interior_gap_linearly() {
        let y = array![[0.0, f64::NAN, 4.0]];
        let out = interpolate_nans(&y);
        assert_eq!(out[[0, 1]], 2.0);
    }

    #[test]
    fn clamps_at_edges_and_zeros_all_nan() {
        let y = array![[f64::NAN, 3.0, f64::NAN], [f64::NAN, f64::NAN, f64::NAN]];
        let out = interpolate_nans(&y);
        assert_eq!(out[[0, 0]], 3.0);
        assert_eq!(out[[0, 2]], 3.0);
        assert_eq!(out.row(1).to_vec(), vec![0.0, 0.0, 0.0]);
    }
}
