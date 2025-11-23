use crate::AppResult;
use crate::ai::label_urls::label_url_cluster;
use crate::dirs::DirType;
use crate::error::AppError;
use crate::safari::SafariHistoryItem;
use async_openai::{Client, config::Config};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use futures::StreamExt;
use linfa::DatasetBase;
use linfa::prelude::*;
use linfa_clustering::Dbscan;
use linfa_reduction::Pca;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;
use tokio::io::AsyncWriteExt;
use tracing::{Span, debug, info_span, warn};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::style::ProgressStyle;

#[derive(Serialize, Deserialize, Debug)]
pub struct ClusterOutput {
    pub label: String,
    pub urls: Vec<SafariHistoryItem>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BrowserClusters {
    pub clusters: Vec<ClusterOutput>,
}

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
        #[cfg(target_os = "macos")]
        {
            // ordinal 0 is usually the integrated GPU
            Ok(Device::new_metal(0)?)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(Device::Cpu)
        }
    }

    pub async fn new_from_pretrained<S: AsRef<str>>(model_name: S) -> AppResult<Self> {
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

    /// Asynchronously embed many texts.
    ///
    /// This spawns a single blocking worker thread and does the whole loop there,
    /// so Candle + Metal stay on one thread and Tokio stays happy.
    pub async fn embed_batch(
        &self,
        history: &[SafariHistoryItem],
    ) -> AppResult<Vec<(SafariHistoryItem, Vec<f32>)>> {
        // Clone what we need into the blocking task.
        let this = self.clone();
        let texts: Vec<String> = history
            .iter()
            .map(|item| format!("{} {}", item.clone().title.unwrap_or_default(), item.url))
            .collect();
        let items = history.to_vec();

        tokio::task::spawn_blocking(move || {
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

            let mut embeddings = Vec::with_capacity(texts.len());
            for (i, t) in texts.iter().enumerate() {
                let emb = this.embed_text_blocking(t)?;
                embeddings.push((items[i].clone(), emb));
                Span::current().pb_inc(1);
            }
            std::mem::drop(header_span_enter);
            std::mem::drop(header_span);
            Result::<_, _>::Ok(embeddings)
        })
        .await?
    }
}

async fn build_cluster_output<C: Config>(
    client: &Client<C>,
    grouped: HashMap<usize, Vec<SafariHistoryItem>>,
) -> AppResult<BrowserClusters> {
    let mut clusters = Vec::new();

    for (_cid, urls) in grouped.into_iter() {
        // TODO: implement label generation with `async_openai`
        let label = label_url_cluster(client, &urls).await?;
        clusters.push(ClusterOutput {
            label: label.label,
            urls,
        });
    }

    Ok(BrowserClusters { clusters })
}

fn embeddings_to_ndarray(embs: &[Vec<f32>]) -> Array2<f64> {
    let rows = embs.len();
    let cols = embs[0].len();
    let mut array = Array2::<f64>::zeros((rows, cols));

    for (i, vec) in embs.iter().enumerate() {
        for (j, val) in vec.iter().enumerate() {
            array[(i, j)] = *val as f64;
        }
    }
    array
}

fn reduce_dimensionality(data: Array2<f64>, k: usize) -> Array2<f64> {
    let dataset = DatasetBase::from(data);
    let pca = Pca::params(k).fit(&dataset).unwrap();
    let transformed = pca.transform(dataset);
    transformed.records
}

fn cluster_embeddings(data: &Array2<f64>) -> Vec<Option<usize>> {
    Dbscan::params(3)
        .tolerance(1e-2)
        .check()
        .unwrap()
        .transform(data)
        .to_vec()
}

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

pub async fn embed_urls<C: Config>(
    client: &Client<C>,
    urls: Vec<SafariHistoryItem>,
) -> AppResult<BrowserClusters> {
    let embedder = BertEmbedder::new_from_pretrained("intfloat/e5-small-v2").await?;
    let embeddings = embedder.embed_batch(&urls).await?;

    let embs_only: Vec<Vec<f32>> = embeddings.iter().map(|(_, emb)| emb.clone()).collect();
    let data = embeddings_to_ndarray(&embs_only);
    let reduced = reduce_dimensionality(data, 10);
    debug!("Reduced embeddings to shape: {:?}", reduced.dim());
    let labels = cluster_embeddings(&reduced);
    debug!(
        "Clustered embeddings into {} clusters",
        labels
            .iter()
            .filter_map(|l| *l)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    let clustered = group_by_cluster(&embeddings, labels);

    let ret = build_cluster_output(client, clustered).await?;

    Ok(ret)
}
