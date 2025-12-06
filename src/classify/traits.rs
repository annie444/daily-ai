use std::collections::HashMap;

use futures::future::BoxFuture;
use ndarray::Array2;

use crate::AppResult;

/// Trait for converting text into vector embeddings.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts.
    /// Returns a vector of embeddings, where each embedding is a vector of floats.
    fn embed<'a>(&'a self, texts: &'a [String]) -> BoxFuture<'a, AppResult<Vec<Vec<f32>>>>;
}

/// Trait for clustering vector embeddings.
pub trait Clusterer: Send + Sync {
    /// Cluster the given embeddings.
    /// Returns a map where key is cluster ID and value is list of indices into the input array.
    fn cluster(&self, embeddings: &Array2<f64>) -> AppResult<HashMap<usize, Vec<usize>>>;
}
