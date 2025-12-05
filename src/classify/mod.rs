pub(super) mod bert;
pub(super) mod convert;
pub(super) mod knn;
pub(super) mod linalg;
pub(super) mod pca;

use std::collections::HashMap;

use async_openai::{Client, config::Config};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, info_span, trace};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::style::ProgressStyle;

use crate::AppResult;
use crate::ai::label_urls::label_url_cluster;
use crate::safari::SafariHistoryItem;

/// Cluster of Safari URLs with a human-friendly label.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UrlCluster {
    pub label: String,
    pub urls: Vec<SafariHistoryItem>,
}

#[tracing::instrument(
    name = "Labeling browser history groups",
    level = "info",
    skip(client, grouped)
)]
async fn build_cluster_output<C: Config>(
    client: &Client<C>,
    grouped: HashMap<usize, Vec<SafariHistoryItem>>,
) -> AppResult<Vec<UrlCluster>> {
    let mut clusters = Vec::new();
    let mut misc = Vec::new();

    let header_span = info_span!("Labeling URL groups...");
    header_span.pb_set_message("Labeling...");
    header_span.pb_set_finish_message("Labeling complete");
    header_span.pb_set_length(grouped.len() as u64);
    header_span.pb_set_style(
        &ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );
    let header_span_enter = header_span.enter();

    for (_cid, urls) in grouped.into_iter() {
        if urls.is_empty() {
            continue;
        } else if urls.len() < 3 {
            misc.extend(urls);
            continue;
        }
        let label = label_url_cluster(client, &urls).await?;
        clusters.push(UrlCluster {
            label: label.label,
            urls,
        });
        header_span.pb_inc(1);
    }

    if !misc.is_empty() {
        info!("Labeling miscellaneous URLs...");
        let label = label_url_cluster(client, &misc).await?;
        clusters.push(UrlCluster {
            label: label.label,
            urls: misc,
        });
    }

    std::mem::drop(header_span_enter);
    std::mem::drop(header_span);

    Ok(clusters)
}

/// Entry point: embed Safari URLs, cluster them, and produce labeled clusters via the model.
#[tracing::instrument(name = "Grouping browser history", level = "info", skip(client, urls))]
pub async fn embed_urls<C: Config>(
    client: &Client<C>,
    urls: Vec<SafariHistoryItem>,
) -> AppResult<Vec<UrlCluster>> {
    let starting_count = urls.len();

    let embedder = bert::BertEmbedder::new_from_pretrained("intfloat/e5-small-v2").await?;
    let embeddings = embedder.embed_batch(&urls).await?;

    // Normalize
    let embs_only: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|(_, v)| v.clone())
        .collect::<Vec<Vec<f32>>>();
    let flattened: Vec<f32> = embs_only.iter().flatten().copied().collect();
    debug!(
        "Embedding value range: min={} max={}",
        flattened
            .iter()
            .copied()
            .reduce(|a, b| a.min(b))
            .unwrap_or(0.0),
        flattened
            .iter()
            .copied()
            .reduce(|a, b| a.max(b))
            .unwrap_or(0.0)
    );
    let raw_arr: Array2<f64> = convert::embeddings_to_ndarray(&embs_only);
    let arr: Array2<f64> = linalg::normalize_embedding(raw_arr);
    debug!(
        "Normalized embeddings range: min={} max={}",
        arr.iter().copied().reduce(|a, b| a.min(b)).unwrap_or(0.0),
        arr.iter().copied().reduce(|a, b| a.max(b)).unwrap_or(0.0)
    );
    debug!("Generated embeddings of shape: {:?}", arr.dim());
    trace!(
        "First 5 embeddings: {:?}",
        &arr.slice(s![..2.min(arr.dim().0), ..2.min(arr.dim().1)])
    );

    // PCA reduce
    let reduced: Array2<f64> = pca::pca_reduce(&arr, 25)?;
    debug!("Reduced embeddings to shape: {:?}", reduced.dim());
    trace!(
        "Reduced embeddings sample: {:?}",
        reduced.slice(s![..2.min(reduced.dim().0), ..2.min(reduced.dim().1)])
    );

    // compute k‐distance
    let mut knn = knn::Knn::default();
    knn.set_k(25).fit(&reduced)?;
    debug!("Computed k‐distance graph for k={}", knn.k);
    let kdists = knn.distances(&reduced)?;
    let dist_cols = kdists.ncols();
    let kdists_slice: ArrayView1<f64> = kdists.slice(s![.., dist_cols - 1]);
    trace!(
        "K‐distance sample: {:?}",
        kdists_slice.slice(s![..10.min(kdists_slice.len())])
    );
    let eps = linalg::elbow_kneedle(kdists_slice);
    debug!("Chosen eps for DBSCAN: {}", eps);

    // cluster with DBSCAN
    let labels = linalg::cluster_embeddings(&reduced, eps, 5)?;
    debug!(
        "Clustered embeddings into {} clusters",
        labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    trace!(
        "Cluster labels: {:?}",
        labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
    );
    let clustered = linalg::group_by_cluster(&embeddings, labels);
    let clustered_count: usize = clustered.values().map(|v| v.len()).sum();
    debug!("Grouped URLs into {} clusters", clustered.len());
    debug!(
        "Clustered URL count: {}, original URL count: {}",
        clustered_count, starting_count
    );

    info!(
        "Generating preliminary labels for {} url groups",
        clustered.len()
    );

    let ret = build_cluster_output(client, clustered).await?;

    Ok(ret)
}
