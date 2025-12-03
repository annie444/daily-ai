#![allow(dead_code)]

mod lloyd;
mod utils;

use std::cmp::Ordering;

use ndarray::prelude::*;
use ndarray_rand::{
    RandomExt, rand,
    rand_distr::{Distribution, Uniform},
};
use tracing::warn;

use crate::AppResult;
use crate::classify::linalg::row_norms;

static DEFAILT_K: usize = 8;
static DEFAULT_N_INIT: usize = 0;
static DEFAUTL_MAX_ITER: usize = 300;
static DEFAULT_TOLERACE: f64 = 1e-4;

pub enum KnnInit {
    Random(usize),
    KMeansPlusPlus(usize),
}

fn kmeans_plus_plus<D>(
    x: &Array2<f64>, // x = (n_samples, n_features)
    n_clusters: usize,
    sample_weight: &Array1<f64>,   // sample_weight = (n_samples,)
    x_squared_norms: &Array1<f64>, // x_squared_norms = (n_samples,)
    random_state: D,
    n_local_trials: Option<usize>,
) -> (Array2<f64>, Vec<isize>)
where
    D: Distribution<f64> + Copy,
{
    // Placeholder for k-means++ initialization logic
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let n_local_trials = n_local_trials.unwrap_or(2 + (n_clusters as f64).ln() as usize);
    let mut centers: Array2<f64> = Array2::<f64>::zeros((n_clusters, n_features));
    let center_id = (n_samples as f64 * random_state.sample(&mut rand::rng())).round() as usize;
    let mut indices: Vec<isize> = vec![-1; n_clusters];
    centers.row_mut(0).assign(&x.row(center_id));
    indices[0] = center_id as isize;
    let mut closest_dist_sq: Array2<f64> = utils::euclidean_distances(
        &centers.slice(s![0..1, ..]).to_owned(),
        x,
        None,
        Some(x_squared_norms),
        true,
    );
    let mut current_pot = closest_dist_sq.dot(sample_weight);
    #[allow(clippy::needless_range_loop)]
    for c in 1..n_clusters {
        let rand_vals = Array1::<f64>::random(n_local_trials, random_state) * current_pot;
        let mut candidate_ids =
            utils::searchsorted_weighted(sample_weight, &closest_dist_sq, &rand_vals);
        let amax = closest_dist_sq.len() - 1;
        candidate_ids.iter_mut().for_each(|id| {
            if *id >= amax {
                *id = amax;
            }
        });
        let mut distance_to_candidates: Array2<f64> = utils::euclidean_distances(
            &x.select(Axis(0), candidate_ids.as_slice()).to_owned(),
            x,
            None,
            Some(x_squared_norms),
            true,
        );
        distance_to_candidates
            .iter_mut()
            .zip(closest_dist_sq.iter())
            .for_each(|(dist, &closest)| {
                if closest < *dist {
                    *dist = closest;
                }
            });
        let candidates_pot = distance_to_candidates.dot(
            &sample_weight
                .to_shape((sample_weight.len(), 1))
                .unwrap()
                .to_owned(),
        );
        let mut best_candidate = candidates_pot
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();
        current_pot = candidates_pot.row(best_candidate).to_owned();
        closest_dist_sq = distance_to_candidates
            .row(best_candidate)
            .to_shape((distance_to_candidates.nrows(), 1))
            .unwrap()
            .to_owned();
        best_candidate = candidate_ids[best_candidate];
        centers.row_mut(c).assign(&x.row(best_candidate));
        indices[c] = best_candidate as isize;
    }
    (centers, indices)
}

impl KnnInit {
    fn value(&self) -> usize {
        match self {
            KnnInit::Random(n) => *n,
            KnnInit::KMeansPlusPlus(n) => *n,
        }
    }

    pub fn set_n_init(&mut self, n_init: usize) {
        match self {
            KnnInit::Random(n) => *n = n_init,
            KnnInit::KMeansPlusPlus(n) => *n = n_init,
        }
    }

    pub fn n_init(&self) -> usize {
        if self.value() > 0 {
            self.value()
        } else {
            match self {
                KnnInit::Random(_) => 10,
                KnnInit::KMeansPlusPlus(_) => 1,
            }
        }
    }

    fn init_centroids<D>(
        &self,
        x: &Array2<f64>,               // x = (n_samples, n_features)
        x_squared_norms: &Array1<f64>, // x_squared_norms = (n_samples,)
        random_state: D,
        sample_weight: &Array1<f64>, // sample_weight = (n_samples,)
        n_clusters: usize,
    ) -> Array2<f64>
    where
        D: Distribution<f64> + Copy,
    {
        let n_samples = x.nrows();
        match self {
            KnnInit::Random(_) => {
                // Placeholder for random initialization logic
                let seeds = Array1::<f64>::random(n_clusters, random_state);
                x.select(
                    Axis(0),
                    &seeds
                        .to_vec()
                        .iter()
                        .map(|&v| (v * n_samples as f64) as usize)
                        .collect::<Vec<usize>>(),
                )
            }
            KnnInit::KMeansPlusPlus(_) => {
                // Placeholder for k-means++ initialization logic
                kmeans_plus_plus(
                    x,
                    n_clusters,
                    sample_weight,
                    x_squared_norms,
                    random_state,
                    None,
                )
                .0
            }
        }
    }
}

impl Default for KnnInit {
    fn default() -> Self {
        KnnInit::Random(DEFAULT_N_INIT)
    }
}

pub struct Knn<D>
where
    D: Distribution<f64> + Copy,
{
    pub k: usize,
    pub init: KnnInit,
    pub max_iterations: usize,
    pub tolerace: f64,
    pub distr: D,
    cluster_centers: Option<Array2<f64>>,
    n_features_out: Option<usize>,
    labels: Option<Array1<usize>>,
    inertia: Option<f64>,
    n_iter: Option<usize>,
}

impl Default for Knn<Uniform<f64>> {
    fn default() -> Self {
        Knn {
            k: DEFAILT_K,
            init: KnnInit::default(),
            max_iterations: DEFAUTL_MAX_ITER,
            tolerace: DEFAULT_TOLERACE,
            distr: Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution"),
            cluster_centers: None,
            n_features_out: None,
            labels: None,
            inertia: None,
            n_iter: None,
        }
    }
}

impl Knn<Uniform<f64>> {
    pub fn new(k: usize) -> Self {
        Knn {
            k,
            ..Default::default()
        }
    }
}

impl<D> Knn<D>
where
    D: Distribution<f64> + Copy,
{
    pub fn set_k(&mut self, k: usize) -> &mut Self {
        self.k = k;
        self
    }

    pub fn set_init(&mut self, init: KnnInit) -> &mut Self {
        self.init = init;
        self
    }

    pub fn set_n_init(&mut self, n_init: usize) -> &mut Self {
        self.init.set_n_init(n_init);
        self
    }

    pub fn set_max_iterations(&mut self, max_iterations: usize) -> &mut Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn set_tolerace(&mut self, tolerace: f64) -> &mut Self {
        self.tolerace = tolerace;
        self
    }

    pub fn set_distr(&mut self, distr: D) -> &mut Self {
        self.distr = distr;
        self
    }

    pub fn fit(&mut self, x: &Array2<f64>) -> AppResult<&mut Self> {
        let mut x = x.clone(); // x = (n_samples, n_features)
        let sample_weight = Array1::<f64>::ones(x.nrows()); // sample_weight = (n_samples,)
        let x_mean: Array1<f64> = x
            .mean_axis(Axis(0))
            .unwrap_or(Array1::<f64>::zeros(x.ncols())); // x_mean = (n_features,)
        x -= &x_mean;
        let x_squared_norms = row_norms(&x, true); // x_squared_norms = (n_samples,)

        let mut best_inertia = None;
        let mut best_labels = None;
        let mut best_centers = None;
        let mut best_n_iter = None;

        // Buffers reused across iterations
        let mut labels: Array1<usize>;
        let mut inertia: f64;
        let mut centers: Array2<f64>;
        let mut n_iter: usize;

        for _ in 0..self.init.n_init() {
            let centers_init = self.init.init_centroids(
                &x,               // (n_samples, n_features)
                &x_squared_norms, // (n_samples,)
                self.distr,
                &sample_weight, // (n_samples,)
                self.k,         // n_clusters
            ); // centers_init = (k, n_features)
            (labels, inertia, centers, n_iter) = lloyd::kmeans_single_lloyd(
                &x,             // (n_samples, n_features)
                &sample_weight, // (n_samples,)
                &centers_init,  // (k, n_features)
                self.max_iterations,
                self.tolerace,
            );
            if best_inertia.is_none_or(|bi| inertia < bi) {
                best_labels = Some(labels);
                best_centers = Some(centers);
                best_inertia = Some(inertia);
                best_n_iter = Some(n_iter);
            }
        }
        x += &x_mean;
        let distinct_clusters = best_labels
            .clone()
            .unwrap()
            .into_iter()
            .collect::<std::collections::HashSet<usize>>()
            .len();
        if distinct_clusters < self.k {
            warn!(
                "Number of distinct clusters ({}) found smaller than n_clusters ({}). Possibly due to duplicate points in X.",
                distinct_clusters, self.k
            );
        }

        self.n_features_out = best_centers.as_ref().map(|bc| bc.dim().0);
        self.cluster_centers = best_centers;
        self.labels = best_labels;
        self.inertia = best_inertia;
        self.n_iter = best_n_iter;
        Ok(self)
    }

    pub fn transform(&mut self, x: &Array2<f64>) -> AppResult<Array2<f64>> {
        if self.cluster_centers.is_none() {
            self.fit(x)?;
        }
        let centers: &Array2<f64> = unsafe { self.cluster_centers.as_ref().unwrap_unchecked() }; // centers = (k, n_features)
        let distances = utils::euclidean_distances(centers, x, None, None, false); // distances = (k, n_samples)
        Ok(distances)
    }

    /// Distances to k-nearest neighbors for each sample (excluding self).
    /// Returns (n_samples, k), where row i contains the sorted k smallest distances to other points.
    pub fn distances(&self, x: &Array2<f64>) -> AppResult<Array2<f64>> {
        let n_samples = x.nrows();
        assert!(self.k < n_samples, "k must be < number of samples");

        let mut full = utils::euclidean_distances(x, x, None, None, false); // full = (n_samples, n_samples)
        for i in 0..n_samples {
            full[(i, i)] = f64::INFINITY; // ignore self
        }

        let mut knn = Array2::<f64>::zeros((n_samples, self.k)); // knn = (n_samples, k)
        for (i, mut row) in knn.axis_iter_mut(Axis(0)).enumerate() {
            let mut dists = full.row(i).to_vec();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            for j in 0..self.k {
                row[j] = dists[j];
            }
        }

        Ok(knn)
    }

    /// Compute the within-cluster sum of squares for each cluster.
    /// Returns a vector of length k where entry i is sum_{j in cluster i} ||x_j - c_i||^2.
    /// x = (n_samples, n_features)
    pub fn wcss(&self, x: &Array2<f64>) -> Option<Array1<f64>> {
        let centers = self.cluster_centers.as_ref()?;
        let labels = self.labels.as_ref()?;

        if labels.len() != x.nrows() {
            return None;
        }
        let n_clusters = centers.nrows();
        let n_features = centers.ncols();

        let mut per_cluster = Array1::<f64>::zeros(n_clusters);
        for (idx, &label) in labels.iter().enumerate() {
            if label >= n_clusters {
                return None;
            }
            // diff = (n_features,)
            let mut sq = 0.0;
            for k in 0..n_features {
                let d = x[(idx, k)] - centers[(label, k)];
                sq += d * d;
            }
            per_cluster[label] += sq;
        }
        Some(per_cluster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wcss_computes_expected_per_cluster() {
        let mut knn = Knn::default();
        let centers = array![[1.5, 1.5], [8.5, 8.5]]; // centers = (2, 2)
        let labels = arr1(&[0_usize, 0, 1, 1]); // labels = (4,)
        knn.cluster_centers = Some(centers);
        knn.labels = Some(labels);

        let x = array![[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]]; // x = (4, 2)
        let wcss = knn.wcss(&x).unwrap();
        // Cluster 0: two points at distance sqrt(0.5^2+0.5^2)=~0.707 each => 0.5 per feature -> total 1.0
        // Cluster 1: same -> total 1.0
        assert_eq!(wcss, array![1.0, 1.0]);
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn wcss_returns_none_on_mismatched_shapes() {
        let mut knn = Knn::default();
        knn.cluster_centers = Some(array![[0.0, 0.0]]);
        knn.labels = Some(arr1(&[0, 0]));

        let x = array![[1.0, 1.0]]; // only one sample, labels expect two
        assert!(knn.wcss(&x).is_none());
    }

    #[test]
    fn distances_returns_k_neighbors_per_sample() {
        let mut knn = Knn::default();
        knn.set_k(2);
        let x = array![[0.0], [1.0], [3.0]]; // x = (3, 1)

        let dists = knn.distances(&x).unwrap(); // (3, 2)

        let expected = array![
            [1.0, 3.0], // from 0 -> neighbors at 1 and 3
            [1.0, 2.0], // from 1 -> neighbors at 0 and 3
            [2.0, 3.0], // from 3 -> neighbors at 1 and 0
        ];
        assert_eq!(dists, expected);
    }
}
