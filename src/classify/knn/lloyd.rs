use ndarray::prelude::*;

use crate::classify::linalg::row_norms;

static CHUNK_SIZE: usize = 256;

/// Compute the inertia (sum of squared distances) for the current labels.
fn inertia_dense(
    x: &Array2<f64>,             // x = (n_samples, n_features)
    sample_weight: &Array1<f64>, // sample_weight = (n_samples,)
    centers: &Array2<f64>,       // centers = (n_clusters, n_features)
    labels: &Array1<usize>,      // labels = (n_samples,)
) -> f64 {
    let mut inertia = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        // row = (n_features,)
        let row = x.row(i);
        // center = (n_features,)
        let center = centers.row(label);
        let diff = &row - &center;
        let sq_dist = diff.mapv(|v| v * v).sum();
        inertia += sq_dist * sample_weight[i];
    }
    inertia
}

fn update_chunk_dense(
    x_chunk: &Array2<f64>,               // x_chunk = (chunk_size, n_features)
    sample_weight_chunk: &Array1<f64>,   // sample_weight_chunk = (chunk_size,)
    centers_old: &Array2<f64>,           // centers_old = (n_clusters, n_features)
    centers_squared_norms: &Array1<f64>, // centers_squared_norms = (n_clusters,)
    update_centers: bool,
) -> (Array1<usize>, Array2<f64>, Array1<f64>) {
    let n_samples = x_chunk.nrows();
    let n_features = x_chunk.ncols();
    let n_clusters = centers_old.nrows();

    // pairwise = (chunk_size, n_clusters)
    let mut pairwise = x_chunk.dot(&centers_old.t());
    pairwise.mapv_inplace(|v| -2.0 * v);

    // x_sq = (chunk_size, 1) broadcast to (chunk_size, n_clusters)
    let x_sq = row_norms(x_chunk, true)
        .to_shape((n_samples, 1))
        .expect("reshape x norms")
        .to_owned();
    pairwise += &x_sq.broadcast((n_samples, n_clusters)).unwrap();

    // centers_sq = (1, n_clusters) broadcast to (chunk_size, n_clusters)
    let centers_sq = centers_squared_norms
        .clone()
        .to_shape((1, n_clusters))
        .expect("reshape center norms")
        .to_owned();
    pairwise += &centers_sq.broadcast((n_samples, n_clusters)).unwrap();

    let mut labels_chunk = Array1::<usize>::zeros(n_samples);
    let mut centers_new_chunk = Array2::<f64>::zeros((n_clusters, n_features));
    let mut weight_in_clusters_chunk = Array1::<f64>::zeros(n_clusters);

    for i in 0..n_samples {
        // distances_row = (n_clusters,)
        let distances_row = pairwise.row(i);
        let (label, _) = distances_row
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        labels_chunk[i] = label;

        if update_centers {
            let weight = sample_weight_chunk[i];
            weight_in_clusters_chunk[label] += weight;
            // accumulates weighted sum for the cluster
            for k in 0..n_features {
                centers_new_chunk[(label, k)] += x_chunk[(i, k)] * weight;
            }
        }
    }

    (labels_chunk, centers_new_chunk, weight_in_clusters_chunk)
}

/// Single Lloyd iteration split into chunks to limit temporary allocations.
fn lloyd_iter_chunked_dense(
    x: &Array2<f64>,             // x = (n_samples, n_features)
    sample_weight: &Array1<f64>, // sample_weight = (n_samples,)
    centers_old: &Array2<f64>,   // centers_old = (n_clusters, n_features)
    update_centers: bool,
) -> (Array2<f64>, Array1<f64>, Array1<usize>, Array1<f64>) {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_clusters = centers_old.nrows();

    if n_samples == 0 {
        return (
            centers_old.clone(),
            Array1::<f64>::zeros(n_clusters),
            Array1::<usize>::zeros(0),
            Array1::<f64>::zeros(n_clusters),
        );
    }

    let n_samples_chunk = n_samples.min(CHUNK_SIZE);
    let mut n_chunks = n_samples / n_samples_chunk;
    let n_samples_rem = n_samples % n_samples_chunk;
    if n_samples != n_chunks * n_samples_chunk {
        n_chunks += 1;
    }

    let centers_squared_norms = row_norms(centers_old, true);
    let mut centers_new = Array2::<f64>::zeros((n_clusters, n_features));
    let mut weight_in_clusters = Array1::<f64>::zeros(n_clusters);
    let mut labels = Array1::<usize>::zeros(n_samples);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * n_samples_chunk;
        let end = if chunk_idx == n_chunks - 1 && n_samples_rem > 0 {
            start + n_samples_rem
        } else {
            start + n_samples_chunk
        };

        // x_chunk = (end - start, n_features)
        let x_chunk = x.slice(s![start..end, ..]).to_owned();
        // sample_weight_chunk = (end - start,)
        let sample_weight_chunk = sample_weight.slice(s![start..end]).to_owned();

        let (labels_chunk, centers_new_chunk, weight_chunk) = update_chunk_dense(
            &x_chunk,
            &sample_weight_chunk,
            centers_old,
            &centers_squared_norms,
            update_centers,
        );

        // labels = (n_samples,)
        labels.slice_mut(s![start..end]).assign(&labels_chunk);

        if update_centers {
            centers_new += &centers_new_chunk;
            weight_in_clusters += &weight_chunk;
        }
    }

    let mut center_shift = Array1::<f64>::zeros(n_clusters);

    if update_centers {
        for cluster in 0..n_clusters {
            let weight = weight_in_clusters[cluster];
            if weight > 0.0 {
                // centers_new row = (n_features,)
                for k in 0..n_features {
                    centers_new[(cluster, k)] /= weight;
                }
            } else {
                // keep previous center if cluster is empty
                centers_new
                    .row_mut(cluster)
                    .assign(&centers_old.row(cluster));
            }
        }

        let diff = centers_old - &centers_new; // (n_clusters, n_features)
        center_shift = row_norms(&diff, false); // (n_clusters,)
    } else {
        centers_new = centers_old.clone();
    }

    (centers_new, weight_in_clusters, labels, center_shift)
}

/// Run a single K-Means using Lloyd's algorithm.
/// Returns (labels, inertia, centers, n_iter)
pub fn kmeans_single_lloyd(
    x: &Array2<f64>,             // x = (n_samples, n_features)
    sample_weight: &Array1<f64>, // sample_weight = (n_samples,)
    centers_init: &Array2<f64>,  // centers_init = (n_clusters, n_features)
    max_iter: usize,
    tol: f64,
) -> (Array1<usize>, f64, Array2<f64>, usize) {
    let n_samples = x.nrows();

    // Buffers reused across iterations
    let mut centers = centers_init.clone();
    let mut labels = Array1::<usize>::zeros(n_samples);
    let mut labels_old = Array1::<usize>::from_elem(n_samples, usize::MAX);
    let mut strict_convergence = false;
    let mut iterations = 0;

    for i in 0..max_iter {
        let (centers_new, _weight_in_clusters, new_labels, center_shift) =
            lloyd_iter_chunked_dense(x, sample_weight, &centers, true);

        iterations = i + 1;

        if new_labels == labels_old {
            centers = centers_new;
            labels = new_labels;
            strict_convergence = true;
            break;
        }

        let center_shift_tot: f64 = center_shift.iter().map(|v| v * v).sum();

        centers = centers_new;
        labels = new_labels.clone();
        labels_old = new_labels;

        if center_shift_tot <= tol {
            break;
        }
    }

    if !strict_convergence {
        // Ensure labels reflect final centers
        let (_, _, refreshed_labels, _) =
            lloyd_iter_chunked_dense(x, sample_weight, &centers, false);
        labels = refreshed_labels;
    }

    let inertia = inertia_dense(x, sample_weight, &centers, &labels);

    (labels, inertia, centers, iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, array};

    use super::*;

    fn assert_all_close_1d(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64) {
        assert_eq!(actual.len(), expected.len(), "1D shapes differ");
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= tol, "expected {e}, got {a}, tol {tol}");
        }
    }

    fn assert_all_close_2d(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        assert_eq!(actual.dim(), expected.dim(), "2D shapes differ");
        for ((a, e), idx) in actual
            .iter()
            .zip(expected.iter())
            .zip(actual.indexed_iter().map(|(idx, _)| idx))
        {
            assert!(
                (a - e).abs() <= tol,
                "expected {e}, got {a} at {:?}, tol {tol}",
                idx
            );
        }
    }

    #[test]
    fn kmeans_lloyd_matches_two_cluster_example() {
        let x = array![
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 0.0],
            [10.0, 2.0],
            [10.0, 4.0],
            [10.0, 0.0]
        ]; // x = (6, 2)
        let sample_weight = Array1::<f64>::ones(x.nrows()); // (6,)
        let centers_init = array![[1.0, 2.0], [10.0, 2.0]]; // (2, 2)

        let (labels, inertia, centers, n_iter) =
            kmeans_single_lloyd(&x, &sample_weight, &centers_init, 20, 1e-6);

        assert!(n_iter > 0);
        assert_eq!(labels.to_vec(), vec![0, 0, 0, 1, 1, 1]);
        let expected_centers = array![[1.0, 2.0], [10.0, 2.0]];
        assert_all_close_2d(&centers, &expected_centers, 1e-8);
        assert!((inertia - 16.0).abs() < 1e-8, "inertia={inertia}");
    }

    #[test]
    fn kmeans_respects_sample_weights() {
        let x = array![[0.0], [2.0], [10.0]]; // x = (3, 1)
        let sample_weight = arr1(&[1.0, 3.0, 1.0]); // (3,)
        let centers_init = array![[0.0], [10.0]]; // (2, 1)

        let (labels, inertia, centers, _) =
            kmeans_single_lloyd(&x, &sample_weight, &centers_init, 20, 1e-8);

        assert_eq!(labels.to_vec(), vec![0, 0, 1]);
        let expected_centers = array![[1.5], [10.0]];
        assert_all_close_2d(&centers, &expected_centers, 1e-8);
        assert!((inertia - 3.0).abs() < 1e-8, "inertia={inertia}");
    }

    #[test]
    fn lloyd_iter_labels_without_updating_centers() {
        let x = array![[0.0], [9.0], [10.0], [11.0]]; // (4, 1)
        let sample_weight = Array1::<f64>::ones(x.nrows()); // (4,)
        let centers_old = array![[0.0], [10.0]]; // (2, 1)

        let (centers_new, weight_in_clusters, labels, center_shift) =
            lloyd_iter_chunked_dense(&x, &sample_weight, &centers_old, false);

        assert_eq!(labels.to_vec(), vec![0, 1, 1, 1]);
        assert_eq!(centers_new, centers_old);
        assert!(weight_in_clusters.iter().all(|w| *w == 0.0));
        assert!(center_shift.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn chunked_iteration_handles_multiple_chunks() {
        // Build 270 samples to force two chunks when CHUNK_SIZE=256.
        let mut data = Vec::with_capacity(270);
        data.extend(vec![0.0; 135]);
        data.extend(vec![10.0; 135]);
        let x = Array2::from_shape_vec((270, 1), data).unwrap(); // x = (270, 1)
        let sample_weight = Array1::<f64>::ones(x.nrows()); // (270,)
        let centers_init = array![[0.0], [10.0]]; // (2, 1)

        let (labels, _inertia, centers, _) =
            kmeans_single_lloyd(&x, &sample_weight, &centers_init, 30, 1e-8);

        // Expect two equal-sized clusters centered near 0 and 10.
        let expected_centers = array![[0.0], [10.0]];
        assert_all_close_2d(&centers, &expected_centers, 1e-8);

        let count_cluster0 = labels.iter().filter(|&&l| l == 0).count();
        let count_cluster1 = labels.iter().filter(|&&l| l == 1).count();
        assert_eq!((count_cluster0, count_cluster1), (135, 135));
    }
}
