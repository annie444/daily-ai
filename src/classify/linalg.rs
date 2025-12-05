use std::collections::HashMap;

use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use ndarray::{OwnedRepr, RemoveAxis, prelude::*};
use tracing::warn;

use crate::AppResult;
use crate::safari::SafariHistoryItem;

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

/// Find the value at the elbow of the k-distances array.
#[tracing::instrument(name = "Filtering browsing history", level = "info", skip(kd))]
pub fn elbow_kneedle(kd: ArrayView1<f64>) -> f64 {
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
