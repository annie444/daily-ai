use ndarray::prelude::*;
use ndarray_linalg::*;

use crate::AppResult;

#[tracing::instrument(name = "Performing PCA", level = "info", skip(data_norm, n_components))]
pub fn pca_reduce(data_norm: &Array2<f64>, n_components: usize) -> AppResult<Array2<f64>> {
    let mean: Array1<f64> = data_norm.mean_axis(Axis(1)).unwrap();
    let mut centered: Array2<f64> = data_norm.clone();
    for mut col in centered.axis_iter_mut(Axis(1)) {
        col -= &mean;
    }
    let (_, _, v) = centered.svd(false, true)?;
    let v: Array2<f64> = v.unwrap().t().to_owned();
    let components: Array2<f64> = v.slice(s![.., 0..n_components]).to_owned();
    let reduced: Array2<f64> = centered.dot(&components);
    Ok(reduced)
}
