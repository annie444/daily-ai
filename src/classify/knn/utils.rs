use ndarray::prelude::*;
use ndarray_rand::rand_distr::num_traits::Zero;

use crate::classify::linalg::row_norms;

pub fn pairwize_euclidean_distances(
    x: &Array2<f64>,                      // x = (n_samples_x, n_features)
    y: Option<&Array2<f64>>,              // y = (n_samples_y, n_features)
    x_norm_squared: Option<&Array1<f64>>, // x_norm_squared = (n_samples_x,)
    y_norm_squared: Option<&Array1<f64>>, // y_norm_squared = (n_samples_y,)
    squared: bool,
) -> Array2<f64> {
    let y_ref = match y {
        Some(arr) => arr,
        None => x,
    };
    euclidean_distances(x, y_ref, x_norm_squared, y_norm_squared, squared)
}

pub fn kth_by_column<Num>(x: &Array2<Num>, k: usize) -> Array1<Num>
where
    Num: PartialOrd + Copy + Zero,
{
    let n_cols = x.dim().1; // x = (n_rows, n_cols)

    let mut result = Array1::<Num>::zeros(n_cols);

    for col in 0..n_cols {
        // Copy the column so we can mutate it
        let mut column: Vec<Num> = x.column(col).to_vec();

        // nth_element equivalent → puts k-th smallest at `column[k]`
        // and partially orders the vector
        column.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());

        result[col] = column[k];
    }

    result
}

pub fn euclidean_distances(
    a: &Array2<f64>,                      // a = (n_a, n_features)
    b: &Array2<f64>,                      // b = (n_b, n_features)
    a_norm_squared: Option<&Array1<f64>>, // a_norm_squared = (n_a,)
    b_norm_squared: Option<&Array1<f64>>, // b_norm_squared = (n_b,)
    squared: bool,
) -> Array2<f64> {
    let aa: Array2<f64> = match a_norm_squared {
        Some(norms) => norms
            .to_owned()
            .to_shape((a.nrows(), 1))
            .unwrap()
            .to_owned(),
        None => row_norms(a, true)
            .to_shape((a.nrows(), 1))
            .unwrap()
            .to_owned(),
    };
    let bb: Array2<f64> = match b_norm_squared {
        Some(norms) => norms.to_shape((1, b.nrows())).unwrap().to_owned(),
        None => row_norms(b, true)
            .to_shape((1, b.nrows()))
            .unwrap()
            .to_owned(),
    };
    let mut distances: Array2<f64> = -2.0 * a.dot(&b.t());
    distances += &aa;
    distances += &bb;
    distances.mapv_inplace(|d| if d > 0.0 { d } else { 0.0 });
    if !squared {
        distances.mapv_inplace(|d| d.sqrt());
    }
    distances
}

/// Equivalent to: np.searchsorted(np.cumsum(w * d), rand_vals)
pub fn searchsorted_weighted(
    sample_weight: &Array1<f64>,   // (n_samples,)
    closest_dist_sq: &Array2<f64>, // (n_samples, n_samples)
    rand_vals: &Array1<f64>,       // (n_trials,)
) -> Vec<usize> {
    let n = sample_weight.len();
    assert_eq!(
        closest_dist_sq.nrows(),
        n,
        "The number of rows in closest_dist_sq must match the length of sample_weight"
    );
    assert_eq!(
        closest_dist_sq.ncols(),
        n,
        "The number of columns in closest_dist_sq must match the length of sample_weight"
    );

    // 1. weighted = sample_weight * closest_dist_sq
    let mut weighted = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let w = sample_weight[i];
        for j in 0..n {
            weighted[(i, j)] = w * closest_dist_sq[(i, j)];
        }
    }

    // 2. Flatten to 1D row-major order
    let flat: Vec<f64> = weighted.iter().copied().collect();

    // 3. Compute cumulative sum (np.cumsum)
    let mut cumsum = Vec::with_capacity(flat.len());
    let mut acc = 0.0;
    for v in flat {
        acc += v;
        cumsum.push(acc);
    }

    // 4. searchsorted on cumsum (left side)
    let mut results = Vec::with_capacity(rand_vals.len());
    for &rv in rand_vals {
        match cumsum.binary_search_by(|x| x.partial_cmp(&rv).unwrap()) {
            Ok(idx) => results.push(idx),
            Err(idx) => results.push(idx),
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_distances_self_matches_manual() {
        let a = array![[0.0, 0.0], [3.0, 4.0]];
        let dists = euclidean_distances(&a, &a, None, None, true);
        let expected = array![[0.0, 25.0], [25.0, 0.0]];
        assert_eq!(dists, expected);
    }

    #[test]
    fn pairwise_euclidean_distances_with_optional_y() {
        let x = array![[0.0], [2.0]];
        let y = array![[1.0]];
        let dists = pairwize_euclidean_distances(&x, Some(&y), None, None, false);
        let expected = array![[1.0], [1.0]]; // distances: |0-1| and |2-1|
        assert_eq!(dists, expected);
    }

    #[test]
    fn kth_by_column_selects_kth_smallest() {
        let x = array![[3, 1, 2], [4, 5, 0]];
        let kth = kth_by_column(&x, 1);
        // For each column after partial ordering: col0 -> [3,4], col1 -> [1,5], col2 -> [0,2]
        assert_eq!(kth, arr1(&[4, 5, 2]));
    }

    #[test]
    fn searchsorted_weighted_matches_cumsum_behavior() {
        let sample_weight = arr1(&[1.0, 2.0]);
        let closest_dist_sq = array![[1.0, 3.0], [2.0, 4.0]];
        // weighted = [[1,3],[4,8]] => flat cumsum = [1,4,8,16]
        let rand_vals = arr1(&[0.5, 1.0, 5.0, 15.0]);

        let result = searchsorted_weighted(&sample_weight, &closest_dist_sq, &rand_vals);

        assert_eq!(result, vec![0, 0, 2, 3]);
    }
}
