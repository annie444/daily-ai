use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_openai::{Client, config::Config};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use futures::StreamExt;
use linfa::DatasetBase;
use linfa::prelude::*;
use linfa_clustering::Dbscan;
use linfa_preprocessing::norm_scaling::NormScaler;
use linfa_reduction::Pca;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;
use tokio::io::AsyncWriteExt;
use tracing::{Span, debug, info, info_span, trace, warn};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::style::ProgressStyle;

use crate::AppResult;
use crate::ai::label_urls::label_url_cluster;
use crate::dirs::DirType;
use crate::error::AppError;
use crate::safari::SafariHistoryItem;

/// Cluster of Safari URLs with a human-friendly label.
#[derive(Serialize, Deserialize, Debug)]
pub struct UrlCluster {
    pub label: String,
    pub urls: Vec<SafariHistoryItem>,
}

/// Wrapper around a BERT encoder for URL/title embeddings.
#[derive(Clone)]
pub struct BertEmbedder {
    device: Device,
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
}

impl BertEmbedder {
    /// Create a device; prefer Metal on macOS, fall back to CPU.
    fn create_device() -> AppResult<Device> {
        // If you only ever want Metal on this machine you can just do Device::new_metal(0)?
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // ordinal 0 is usually the integrated GPU
            Ok(Device::new_metal(0)?)
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(Device::Cpu)
        }
    }

    #[tracing::instrument(
        name = "Downloading embedding model from Hugging Face",
        level = "trace"
    )]
    pub async fn new_from_pretrained<S: AsRef<str> + std::fmt::Debug>(
        model_name: S,
    ) -> AppResult<Self> {
        let hf_cache_dir = DirType::Cache
            .ensure_dir_async()
            .await?
            .join("huggingface")
            .join("transformers");

        let model_dir = hf_cache_dir.join(model_name.as_ref().replace('/', "_"));

        if !model_dir.exists() {
            tokio::fs::create_dir_all(&model_dir).await?;
        }

        let base_url = format!(
            "https://huggingface.co/{}/resolve/main/",
            model_name.as_ref()
        );

        // Minimal fetcher for the few files we need; retries and progress for better UX.
        let client = reqwest::ClientBuilder::new()
            .user_agent(format!("daily-ai/{}", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::limited(10))
            .referer(true)
            .retry(
                reqwest::retry::for_host("huggingface.co")
                    .max_retries_per_request(3)
                    .max_extra_load(5.0),
            )
            .build()
            .unwrap();
        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                // Stream download into cache file.
                let mut open_file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file_path)
                    .await?;
                let url = format!("{}{}", base_url, file);
                let resp =
                    client.get(&url).send().await.map_err(|e| {
                        AppError::Other(format!("Failed to download {}: {}", file, e))
                    })?;
                let header_span = info_span!("Downloading model file", file = %file);
                header_span.pb_set_message("Downloading...");
                header_span.pb_set_finish_message("Download complete");
                let progress = if let Some(content_length) =
                    resp.headers().get(reqwest::header::CONTENT_LENGTH)
                {
                    let file_size: u64 = content_length.to_str()?.parse()?;
                    debug!("Expected file size: {} bytes", file_size);
                    header_span.pb_set_style(
                        &ProgressStyle::default_bar()
                            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                            .unwrap(),
                    );
                    header_span.pb_set_length(file_size);
                    header_span.enter()
                } else {
                    warn!(
                        "Content-Length header not found. Cannot determine file size beforehand."
                    );
                    header_span.pb_set_style(
                        &ProgressStyle::default_spinner()
                            .template("{msg} {spinner}")
                            .unwrap(),
                    );
                    header_span.enter()
                };

                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| {
                        AppError::Other(format!("Failed to download {}: {}", file, e))
                    })?;
                    open_file.write_all(&chunk).await?;
                    open_file.flush().await?;
                    Span::current().pb_inc(chunk.len() as u64);
                }
                open_file.sync_all().await?;
                open_file.shutdown().await?;
                std::mem::drop(progress);
                std::mem::drop(header_span);
            }
        }

        Self::new_from_dir(model_dir)
    }

    /// Load BERT from local files / HF cache.
    ///
    /// `model_dir` should contain:
    ///   - config.json
    ///   - model.safetensors (or multiple shard .safetensors files)
    ///   - tokenizer.json
    #[tracing::instrument(
        name = "Loading embedding model from directory",
        level = "trace",
        skip(model_dir)
    )]
    pub fn new_from_dir<P: AsRef<Path>>(model_dir: P) -> AppResult<Self> {
        let model_dir = model_dir.as_ref();

        // --- Load tokenizer ---------------------------------------------------
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AppError::Other(format!("failed to load tokenizer: {e}")))?;

        // --- Load config.json into BertConfig --------------------------------
        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)?;
        let config: BertConfig = serde_json::from_slice(&config_bytes)?;

        // --- Prepare device ---------------------------------------------------
        let device = Self::create_device()?;

        // --- Load safetensors weights ----------------------------------------
        //
        // Simplest case: single `model.safetensors`.
        // If you have sharded weights, you’d just push multiple `SafeTensors` into the vec.
        let weights_path = model_dir.join("model.safetensors");
        let weights_data = std::fs::read(&weights_path)?;

        // Candle usually has a helper VarBuilder for safetensors; depending on version
        // this is either `VarBuilder::from_safetensors` or similar. If the name differs,
        // search in your docs for "from_safetensors".
        let vb = VarBuilder::from_slice_safetensors(&weights_data, DType::F32, &device)?;

        // --- Build the BERT model --------------------------------------------
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            device,
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Synchronous embedding of one text. You will call this from `spawn_blocking`.
    fn embed_text_blocking(&self, text: &str) -> AppResult<Vec<f32>> {
        // 1) Tokenize
        let encoding = self.tokenizer.encode(text, true)?;

        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let attn_mask = encoding.get_attention_mask();

        let seq_len = ids.len();
        let batch_size = 1usize;

        // 2) Build tensors on our device
        let input_ids = Tensor::new(ids, &self.device)?.reshape((batch_size, seq_len))?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.reshape((batch_size, seq_len))?;
        let attention_mask =
            Tensor::new(attn_mask, &self.device)?.reshape((batch_size, seq_len))?;

        // 3) Forward pass.
        // NOTE: BERT forward signature is:
        //   (&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: Option<&Tensor>)
        let outputs = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // outputs shape: [batch, seq_len, hidden_dim]
        // We’ll average over seq_len to get a single embedding.
        let hidden_dim = outputs.dim(2)?;
        let seq_len = outputs.dim(1)?;

        // Mean pool over the second dimension (seq_len).
        let sum = outputs.sum(1)?;
        let mean = (sum / (seq_len as f64))?;

        // mean shape: [batch, hidden_dim] → [hidden_dim]
        let embedding = mean.squeeze(0)?.to_vec1::<f32>()?;
        debug_assert_eq!(embedding.len(), hidden_dim);

        Ok(embedding)
    }

    /// Asynchronously embed many texts. Runs in a blocking worker so Candle stays off Tokio.
    #[tracing::instrument(
        name = "Embedding browser history",
        level = "trace",
        skip(self, history)
    )]
    pub async fn embed_batch(
        &self,
        history: &[SafariHistoryItem],
    ) -> AppResult<Vec<(SafariHistoryItem, Vec<f32>)>> {
        // Clone what we need into the blocking task.
        let this = self.clone();
        let texts: Vec<String> = history
            .iter()
            .map(|item| {
                format!(
                    "query: {} {}",
                    item.clone().title.unwrap_or_default(),
                    item.url
                )
            })
            .collect();
        let items = history.to_vec();

        let embeddings = tokio::task::spawn_blocking(move || {
            let mut embeddings = Vec::new();
            let header_span = info_span!("Running embeddings for URLs");
            header_span.pb_set_message("Embedding...");
            header_span.pb_set_finish_message("Embedding complete");
            header_span.pb_set_length(texts.len() as u64);
            header_span.pb_set_style(
                &ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap(),
            );
            let header_span_enter = header_span.enter();

            for (i, t) in texts.iter().enumerate() {
                let emb = this.embed_text_blocking(t)?;
                embeddings.push((items[i].clone(), emb));
                Span::current().pb_inc(1);
            }
            std::mem::drop(header_span_enter);
            std::mem::drop(header_span);
            Result::<_, AppError>::Ok(embeddings)
        })
        .await??;
        Ok(embeddings)
    }
}

#[tracing::instrument(
    name = "Labeling browser history groups",
    level = "trace",
    skip(client, grouped)
)]
async fn build_cluster_output<C: Config>(
    client: &Client<C>,
    grouped: HashMap<usize, Vec<SafariHistoryItem>>,
) -> AppResult<Vec<UrlCluster>> {
    let mut clusters = Vec::new();

    for (_cid, urls) in grouped.into_iter() {
        let label = label_url_cluster(client, &urls).await?;
        clusters.push(UrlCluster {
            label: label.label,
            urls,
        });
    }

    Ok(clusters)
}

#[tracing::instrument(name = "Normalizing links", level = "trace", skip(v))]
fn normalize_embedding(mut v: Vec<f32>) -> Vec<f64> {
    let sum_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let norm = sum_sq.sqrt().max(f64::MAX);
    v.iter_mut()
        .map(|x| (*x as f64) / norm)
        .collect::<Vec<f64>>()
}

#[tracing::instrument(name = "Converting links", level = "trace", skip(embs))]
fn embeddings_to_dataset(embs: &[Vec<f64>]) -> Dataset<f64> {
    let rows = embs.len();
    let cols = embs[0].len();
    let mut arr = Array2::<f64>::zeros((rows, cols));
    for (i, vec) in embs.iter().enumerate() {
        for (j, &val) in vec.iter().enumerate() {
            arr[(i, j)] = val;
        }
    }
    Dataset::from(arr)
}

#[tracing::instrument(name = "Converting links", level = "trace", skip(embs))]
fn embeddings_to_ndarray(embs: &[Vec<f64>]) -> Array2<f64> {
    let rows = embs.len();
    let cols = embs[0].len();
    let mut arr = Array2::<f64>::zeros((rows, cols));
    for (i, vec) in embs.iter().enumerate() {
        for (j, &val) in vec.iter().enumerate() {
            arr[(i, j)] = val;
        }
    }
    arr
}

/// Compute pairwise distances between all rows in the data.
#[tracing::instrument(name = "Reducing links", level = "trace", skip(data))]
fn find_k_distances(data: &Array2<f64>, k: usize) -> Vec<f64> {
    let n = data.nrows();
    let mut k_dists = Vec::with_capacity(n);
    for i in 0..n {
        let vi = data.row(i);
        let mut dists = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j == i {
                continue;
            }
            let vj = data.row(j);
            let d = vi
                .iter()
                .zip(vj.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dists.push(d);
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        if dists.len() >= k {
            k_dists.push(dists[k - 1]);
        }
    }
    k_dists
}

/// Find the value at the `percentile` (0.0-1.0) of the sorted distances.
#[tracing::instrument(name = "Filtering browsing history", level = "trace", skip(kd))]
fn select_eps_from_k_distances(mut kd: Vec<f64>, percentile: f64) -> f64 {
    kd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let idx = ((kd.len() as f64) * percentile).floor() as usize;
    kd[idx].max(1e-6)
}

#[tracing::instrument(name = "Filtering browsing history", level = "trace", skip(kd))]
fn select_eps_from_k_dists_sc(mut kd: Vec<f64>) -> f64 {
    kd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    debug!("Sorted k-distances for SC method: {:?}", kd);
    let mut best: Vec<(usize, f64, usize)> = vec![(0, f64::MIN, 0)];
    kd.windows(2).enumerate().for_each(|(i, w)| {
        let reld = (w[0] - w[1]) / w[0];
        trace!("k-distance jump at index {} for point {:?}: {}", i, w, reld);
        if reld > best[0].1 {
            best.insert(0, (i, reld, (i as f64 / kd.len() as f64).floor() as usize));
        }
    });
    debug!("Best k-distance jumps: {:?}", &best);
    best.sort_by(|(_, _, a), (_, _, b)| a.cmp(b));
    if let Some((idx, _, _)) = best.first() {
        kd[*idx]
    } else {
        kd[kd.len() / 10].max(1e-6)
    }
}

#[tracing::instrument(name = "Reducing links", level = "trace", skip(data))]
fn reduce_dimensionality(data: &Array2<f64>, k: usize) -> Array2<f64> {
    let dataset = DatasetBase::from(data.to_owned());
    let pca = Pca::params(k)
        .whiten(false)
        .fit(&dataset)
        .map_err(|e| {
            AppError::Other(format!(
                "Failed to fit PCA model for dimensionality reduction: {}",
                e
            ))
        })
        .unwrap();
    let reduced = pca.predict(dataset);
    reduced.records.clone()
}

/// Cluster embeddings with DBSCAN and return a vector of Option<usize> labels.
#[tracing::instrument(name = "Transforming links", level = "trace", skip(data))]
fn cluster_embeddings(
    data: &Array2<f64>,
    eps: f64,
    min_points: usize,
) -> AppResult<Vec<Option<usize>>> {
    let params = Dbscan::params(min_points).tolerance(eps);
    let model = params
        .transform(data)
        .map_err(|e| AppError::Other(format!("Failed to cluster embeddings with DBSCAN: {}", e)))?;
    Ok(model.to_vec())
}

#[tracing::instrument(name = "Grouping links", level = "trace", skip(urls, labels))]
fn group_by_cluster(
    urls: &[(SafariHistoryItem, Vec<f32>)],
    labels: Vec<Option<usize>>,
) -> HashMap<usize, Vec<SafariHistoryItem>> {
    let mut map: HashMap<usize, Vec<SafariHistoryItem>> = HashMap::new();

    for (i, label) in labels.into_iter().enumerate() {
        if let Some(cid) = label {
            map.entry(cid).or_default().push(urls[i].0.clone());
        }
    }

    map
}

/// Entry point: embed Safari URLs, cluster them, and produce labeled clusters via the model.
#[tracing::instrument(name = "Grouping browser history", level = "debug", skip(client, urls))]
pub async fn embed_urls<C: Config>(
    client: &Client<C>,
    urls: Vec<SafariHistoryItem>,
) -> AppResult<Vec<UrlCluster>> {
    let starting_count = urls.len();

    let embedder = BertEmbedder::new_from_pretrained("intfloat/e5-small-v2").await?;
    let embeddings = embedder.embed_batch(&urls).await?;

    // Normalize
    let embs_only = embeddings
        .iter()
        .map(|(_, v)| v.clone())
        .collect::<Vec<_>>();
    let emb_abs = embs_only
        .iter()
        .flatten()
        .map(|x| x.abs())
        .collect::<Vec<_>>();
    debug!(
        "Embedding value range: min={} max={}",
        emb_abs
            .iter()
            .copied()
            .reduce(|a, b| a.min(b))
            .unwrap_or(0.0),
        emb_abs
            .iter()
            .copied()
            .reduce(|a, b| a.max(b))
            .unwrap_or(0.0)
    );
    let raw_arr = embeddings_to_ndarray(&embs_only);
    let norm_embs: Array2<f64> = debug!(
        "Normalized embeddings range: min={} max={}",
        norm_embs
            .iter()
            .flatten()
            .copied()
            .reduce(|a, b| a.min(b))
            .unwrap_or(0.0),
        norm_embs
            .iter()
            .flatten()
            .copied()
            .reduce(|a, b| a.max(b))
            .unwrap_or(0.0)
    );
    let arr = embeddings_to_ndarray(&norm_embs);
    debug!("Generated embeddings of shape: {:?}", arr.dim());
    trace!(
        "First 5 embeddings: {:?}",
        &arr.slice(s![..2.min(arr.dim().0), ..2.min(arr.dim().1)])
    );

    // PCA reduce
    let reduced = reduce_dimensionality(&arr, 30);
    debug!("Reduced embeddings to shape: {:?}", reduced.dim());
    trace!(
        "Reduced embeddings sample: {:?}",
        reduced.slice(s![..2.min(reduced.dim().0), ..2.min(reduced.dim().1)])
    );

    // compute k‐distance
    let dists = find_k_distances(&reduced, 15);
    debug!("Computed {} k-distances", dists.len());
    trace!("First 2 distances: {:?}", &dists[..2.min(dists.len())]);
    let eps = select_eps_from_k_distances(dists.clone(), 0.04);
    let new_eps = select_eps_from_k_dists_sc(dists);
    debug!("Chosen eps for DBSCAN: {} {}", eps, new_eps);

    // cluster with DBSCAN
    let labels = cluster_embeddings(&reduced, eps, 5)?;
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
    let clustered = group_by_cluster(&embeddings, labels);
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
