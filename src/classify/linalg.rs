use std::cmp::Ordering;
use std::collections::HashMap;

use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use ndarray::{OwnedRepr, RemoveAxis, prelude::*};
use tracing::{debug, warn};

use crate::AppResult;
use crate::safari::SafariHistoryItem;

/// Compute full pairwise Euclidean distance matrix.
/// X: (n_samples, n_features)
fn pairwise_distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();

    // Gram matrix: G[i,j] = x_i · x_j
    let gram: Array2<f64> = x.dot(&x.t());

    // Squared norms ||x_i||^2 for each row
    let norms2: Array1<f64> = x
        .axis_iter(Axis(0))
        .map(|row| row.mapv(|v| v * v).sum())
        .collect();

    let mut dists2 = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let val = norms2[i] + norms2[j] - 2.0 * gram[(i, j)];
            // numerical safety: distances can't be negative
            dists2[(i, j)] = if val > 0.0 { val } else { 0.0 };
        }
    }

    // sqrt to get Euclidean distances
    dists2.mapv(|v| v.sqrt())
}

pub fn row_norms<D>(
    x: &ArrayBase<OwnedRepr<f64>, D>,
    squared: bool,
) -> ArrayBase<OwnedRepr<f64>, D::Smaller>
where
    D: Dimension + RemoveAxis,
{
    let prod: ArrayBase<OwnedRepr<f64>, D> = x * x;
    let sum: ArrayBase<OwnedRepr<f64>, D::Smaller> = prod.sum_axis(Axis(1));
    if !squared { sum.sqrt() } else { sum }
}

/// Compute pairwise distances between all rows in the data.
/// For each row, find the distances to its k nearest neighbors.
/// Return an array of shape (n_samples, k) with the k distances for each sample.
/// Assumes k < n_samples.
///
/// # Arguments
/// * `x_reduced` - Input data of shape (n_samples, n_features)
/// * `k` - Number of nearest neighbors to consider
/// # Returns
/// * `Array2<f64>` - Array of shape (n_samples, k) with k nearest neighbor distances for each sample
/// # Panics
/// * If k >= n_samples
#[tracing::instrument(name = "Finding clusters of URLs", level = "info", skip(x_reduced))]
pub fn knn_k_distances(x_reduced: &Array2<f64>, k: usize) -> Array2<f64> {
    let n = x_reduced.nrows();
    assert!(k < n, "k must be < number of samples");

    let dists = pairwise_distances(x_reduced);
    let mut kdists = Array2::<f64>::zeros((n, k));

    for i in 0..n {
        // extract row i as Vec
        let mut row: Vec<f64> = dists.row(i).to_vec();

        // ignore self-distance
        row[i] = f64::INFINITY;

        // sort ascending
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // take first k neighbors
        for m in 0..k {
            kdists[(i, m)] = row[m];
        }
    }

    kdists
}

/// Extract the k-th nearest neighbor distances from the k-distances array,
/// sort them in ascending order, and return as a Vec.
///
/// # Arguments
/// * `kdists` - Array of shape (n_samples, k) with k nearest neighbor distances
///
/// # Returns
/// * `Vec<f64>` - Sorted vector of k-th nearest neighbor distances
fn kth_distances_for_elbow(kdists: &Array2<f64>) -> Vec<f64> {
    let last_col = kdists.ncols().saturating_sub(1);
    let mut d: Vec<f64> = kdists.column(last_col).to_vec();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

/// Find the value at the elbow of the k-distances array.
#[tracing::instrument(name = "Filtering browsing history", level = "info", skip(kd))]
fn elbow_kneedle(kd: &[f64]) -> f64 {
    let n = kd.len();
    let x1 = 0.0;
    let y1 = kd[0];
    let x2 = (n - 1) as f64;
    let y2 = kd[n - 1];

    let ab_x = x2 - x1; // = n - 1
    let ab_y = y2 - y1;

    let ab_norm = (ab_x * ab_x + ab_y * ab_y).sqrt();

    let mut max_dist = -f64::INFINITY;
    let mut max_i = 0;

    for (i, py) in kd.iter().enumerate() {
        let px = i as f64;

        // cross product magnitude in 2D
        let cross = (px - x1) * ab_y - (py - y1) * ab_x;

        let dist = cross.abs() / ab_norm;

        if dist > max_dist {
            max_dist = dist;
            max_i = i;
        }
    }

    kd[max_i]
}

#[tracing::instrument(name = "Selecting eps", level = "info", skip(x_reduced))]
pub fn select_eps_from_k_dists(x_reduced: &Array2<f64>, k: usize) -> f64 {
    let kdists = knn_k_distances(x_reduced, k);
    let kth_dists = kth_distances_for_elbow(&kdists);
    let eps = elbow_kneedle(&kth_dists);
    debug!("Selected eps from k-distances: {}", eps);
    eps
}

/// Cluster embeddings with DBSCAN and return a vector of Option<usize> labels.
#[tracing::instrument(name = "Transforming links", level = "info", skip(data))]
pub fn cluster_embeddings(data: &Array2<f64>, eps: f64, min_size: usize) -> AppResult<Vec<i32>> {
    let params = HdbscanHyperParams::builder()
        .min_cluster_size(min_size)
        .epsilon(eps)
        .dist_metric(DistanceMetric::Euclidean)
        .nn_algorithm(NnAlgorithm::Auto)
        .build();
    let data = data
        .axis_iter(Axis(0))
        .map(|row| row.as_slice().unwrap().to_vec())
        .collect::<Vec<Vec<f64>>>();
    let hdbscan = Hdbscan::new(&data, params);
    Ok(hdbscan.cluster()?)
}

#[tracing::instrument(name = "Grouping links", level = "info", skip(urls, labels))]
pub fn group_by_cluster(
    urls: &[(SafariHistoryItem, Vec<f32>)],
    labels: Vec<i32>,
) -> HashMap<usize, Vec<SafariHistoryItem>> {
    let mut map: HashMap<usize, Vec<SafariHistoryItem>> = HashMap::new();

    for (i, label) in labels.into_iter().enumerate() {
        map.entry(label as usize)
            .or_default()
            .push(urls[i].0.clone());
    }

    map
}

#[tracing::instrument(name = "Normalizing links", level = "info", skip(data))]
pub fn normalize_embedding(data: Array2<f64>) -> Array2<f64> {
    let (norm, _) = ndarray_linalg::norm::normalize(data, ndarray_linalg::norm::NormalizeAxis::Row);
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::OffsetDateTime;

    #[test]
    fn row_norms_squared_and_unsquared() {
        let x = array![[3.0, 4.0], [1.0, 2.0]]; // norms: 5 and sqrt(5)
        let squared = row_norms(&x, true);
        let unsquared = row_norms(&x, false);
        assert_eq!(squared, arr1(&[25.0, 5.0]));
        assert!((unsquared[0] - 5.0).abs() < 1e-10);
        assert!((unsquared[1] - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn knn_k_distances_matches_expected() {
        let x = array![[0.0], [1.0], [3.0]]; // 1D points
        let kdists = knn_k_distances(&x, 2);
        // distances from each point to its 2 nearest neighbors
        let expected = array![
            [1.0, 3.0], // from 0 -> 1, 3
            [1.0, 2.0], // from 1 -> 0, 3
            [2.0, 3.0]  // from 3 -> 1, 0
        ];
        for (a, e) in kdists.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-10, "expected {e}, got {a}");
        }
    }

    #[test]
    fn select_eps_from_k_dists_returns_elbow() {
        // Points along a line with widening gaps to make an elbow
        let x = array![[0.0], [10.0], [20.0], [40.0]];
        let eps = select_eps_from_k_dists(&x, 2);
        // Expected elbow corresponds to the 2nd-neighbor distances: [20,10,20,30] -> elbow near 20.
        assert_eq!(eps, 20.0);
    }

    #[test]
    fn group_by_cluster_groups_urls() {
        let urls = vec![
            (
                SafariHistoryItem {
                    url: "a".into(),
                    title: None,
                    visit_count: 1,
                    last_visited: OffsetDateTime::UNIX_EPOCH,
                },
                vec![0.0_f32],
            ),
            (
                SafariHistoryItem {
                    url: "b".into(),
                    title: None,
                    visit_count: 1,
                    last_visited: OffsetDateTime::UNIX_EPOCH,
                },
                vec![1.0_f32],
            ),
        ];
        let labels = vec![0, 1];

        let grouped = group_by_cluster(&urls, labels);

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped.get(&0).unwrap()[0].url, "a");
        assert_eq!(grouped.get(&1).unwrap()[0].url, "b");
    }

    #[test]
    fn normalize_embedding_rows_to_unit_norm() {
        let data = array![[3.0, 4.0], [0.0, 5.0]];
        let normed = normalize_embedding(data);
        for row in normed.axis_iter(Axis(0)) {
            let norm: f64 = row.mapv(|v| v * v).sum().sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }
}
