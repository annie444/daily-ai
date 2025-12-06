#[cfg(feature = "local-ml")]
pub(super) mod bert;
pub(super) mod convert;
pub(super) mod knn;
pub(super) mod linalg;
pub(super) mod openai;
pub(super) mod pca;
pub mod traits;

use std::collections::HashMap;

use async_openai::{Client, config::Config};
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, info_span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::style::ProgressStyle;

use crate::AppResult;
use crate::ai::label_urls::label_url_cluster;
use crate::classify::traits::{Clusterer, Embedder};
use crate::safari::SafariHistoryItem;

/// Cluster of Safari URLs with a human-friendly label.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UrlCluster {
    pub label: String,
    pub urls: Vec<SafariHistoryItem>,
}

pub struct HdbscanClusterer;

impl Clusterer for HdbscanClusterer {
    fn cluster(&self, embeddings: &Array2<f64>) -> AppResult<HashMap<usize, Vec<usize>>> {
        // compute k-distance
        let mut knn = knn::Knn::default();
        knn.set_k(25).fit(embeddings)?;
        debug!("Computed k-distance graph for k={}", knn.k);
        let kdists = knn.distances(embeddings)?;
        let dist_cols = kdists.ncols();
        let kdists_slice: ArrayView1<f64> = kdists.slice(s![.., dist_cols - 1]);
        let eps = linalg::elbow_kneedle(kdists_slice);
        debug!("Chosen eps for DBSCAN: {}", eps);

        // cluster with DBSCAN
        let labels = linalg::cluster_embeddings(embeddings, eps, 5)?;

        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, label) in labels.into_iter().enumerate() {
            if label >= 0 {
                map.entry(label as usize).or_default().push(i);
            }
        }
        Ok(map)
    }
}

pub struct Classifier<E, C> {
    embedder: E,
    clusterer: C,
}

impl<E: Embedder, C: Clusterer> Classifier<E, C> {
    pub fn new(embedder: E, clusterer: C) -> Self {
        Self {
            embedder,
            clusterer,
        }
    }

    pub async fn classify<ConfigType: Config>(
        &self,
        client: &Client<ConfigType>,
        items: Vec<SafariHistoryItem>,
    ) -> AppResult<Vec<UrlCluster>> {
        let texts: Vec<String> = items
            .iter()
            .map(|item| {
                format!(
                    "query: {} {}",
                    item.title.as_deref().unwrap_or_default(),
                    item.url
                )
            })
            .collect();

        let embeddings = self.embedder.embed(&texts).await?;

        // Normalize
        let embs_only = embeddings.clone();
        let raw_arr: Array2<f64> = convert::embeddings_to_ndarray(&embs_only);
        let arr: Array2<f64> = linalg::normalize_embedding(raw_arr);

        // PCA reduce
        let reduced: Array2<f64> = pca::pca_reduce(&arr, 25)?;

        let clusters = self.clusterer.cluster(&reduced)?;

        // Group items
        let mut grouped: HashMap<usize, Vec<SafariHistoryItem>> = HashMap::new();
        for (cid, indices) in clusters {
            let mut cluster_items = Vec::new();
            for idx in indices {
                cluster_items.push(items[idx].clone());
            }
            grouped.insert(cid, cluster_items);
        }

        build_cluster_output(client, grouped).await
    }
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
    #[cfg(feature = "local-ml")]
    let embedder = {
        let e = bert::BertEmbedder::new_from_pretrained("intfloat/e5-small-v2").await;
        // If local load fails (e.g. download error), we might fallback, but for now we just propagate
        // However, if local-ml is enabled but fails, or if we want to support both...
        // For this step, we will prioritize local if feature is on.
        e?
    };

    #[cfg(not(feature = "local-ml"))]
    let embedder =
        openai::OAIEmbedder::new(client, "text-embedding-nomic-embed-text-v1.5".to_string());

    let clusterer = HdbscanClusterer;
    let classifier = Classifier::new(embedder, clusterer);
    classifier.classify(client, urls).await
}
