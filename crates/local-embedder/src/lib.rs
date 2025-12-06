use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use futures::StreamExt;
use murmur3::murmur3_x86_128;
use tokenizers::tokenizer::Tokenizer;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info_span, warn};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::style::ProgressStyle;

#[derive(thiserror::Error, Debug)]
pub enum EmbedderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::error::DecodeError),
    #[error("Bincode encode error: {0}")]
    BincodeEncode(#[from] bincode::error::EncodeError),
    #[error("Int parse error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("Header value error: {0}")]
    HeaderValue(#[from] reqwest::header::ToStrError),
    #[error("Other error: {0}")]
    Other(String),
    #[error("{0}")]
    Dir(#[from] daily_ai_dirs::DirError),
}

// Tokenizer error mapping
impl From<Box<dyn std::error::Error + Send + Sync>> for EmbedderError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Tokenizer(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EmbedderError>;

/// Wrapper around a BERT encoder for URL/title embeddings.
#[derive(Clone)]
pub struct BertEmbedder {
    device: Device,
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    cache_dir: PathBuf,
}

impl BertEmbedder {
    fn create_device() -> Result<Device> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            Ok(Device::new_metal(0)?)
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            Ok(Device::Cpu)
        }
    }

    #[tracing::instrument(name = "Downloading embedding model from Hugging Face", level = "info")]
    pub async fn new_from_pretrained<S: AsRef<str> + std::fmt::Debug>(
        model_name: S,
    ) -> Result<Self> {
        let cache_dir = daily_ai_dirs::DirType::Cache.ensure_dir_async().await?;
        let hf_cache_dir = cache_dir.join("huggingface").join("transformers");
        let model_dir = hf_cache_dir.join(model_name.as_ref().replace('/', "_"));

        if !model_dir.exists() {
            tokio::fs::create_dir_all(&model_dir).await?;
        }

        let base_url = format!(
            "https://huggingface.co/{}/resolve/main/",
            model_name.as_ref()
        );

        let client = reqwest::ClientBuilder::new()
            .user_agent("daily-ai-embedder/0.1.0")
            .redirect(reqwest::redirect::Policy::limited(10))
            .referer(true)
            .retry(reqwest::retry::for_host("huggingface.co").max_retries_per_request(3))
            .build()
            .unwrap(); // Unwrap safe for default builder

        for file in ["config.json", "model.safetensors", "tokenizer.json"] {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                let mut open_file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file_path)
                    .await?;
                let url = format!("{}{}", base_url, file);
                let resp = client.get(&url).send().await?;

                let header_span = info_span!("Downloading model file", file = %file);
                header_span.pb_set_message("Downloading...");
                header_span.pb_set_finish_message("Download complete");

                if let Some(content_length) = resp.headers().get(reqwest::header::CONTENT_LENGTH) {
                    let file_size: u64 = content_length.to_str()?.parse()?;
                    debug!("Expected file size: {} bytes", file_size);
                    header_span.pb_set_style(
                        &ProgressStyle::default_bar()
                            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                            .unwrap(),
                    );
                    header_span.pb_set_length(file_size);
                } else {
                    warn!("Content-Length header not found.");
                    header_span.pb_set_style(
                        &ProgressStyle::default_spinner()
                            .template("{msg} {spinner}")
                            .unwrap(),
                    );
                }
                let _enter = header_span.enter();

                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk?;
                    open_file.write_all(&chunk).await?;
                    open_file.flush().await?;
                    header_span.pb_inc(chunk.len() as u64);
                }
                open_file.sync_all().await?;
                open_file.shutdown().await?;
            }
        }

        Self::new_from_dir(model_dir)
    }

    #[tracing::instrument(
        name = "Loading embedding model from directory",
        level = "info",
        skip(model_dir)
    )]
    pub fn new_from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let cache_dir = daily_ai_dirs::DirType::Cache.ensure_dir()?;
        let model_dir = model_dir.as_ref();

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::Tokenizer(e.to_string()))?;

        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)?;
        let config: BertConfig = serde_json::from_slice(&config_bytes)?;

        let device = Self::create_device()?;

        let weights_path = model_dir.join("model.safetensors");
        let weights_data = std::fs::read(&weights_path)?;
        let vb = VarBuilder::from_slice_safetensors(&weights_data, DType::F32, &device)?;

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            device,
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            cache_dir,
        })
    }

    pub fn embed_text_blocking(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.trim();
        let hash_result = murmur3_x86_128(&mut Cursor::new(text), 0)?;
        let cache_path = self.cache_dir.join(format!("{hash_result}.bin"));
        if cache_path.exists() {
            let f = std::fs::File::open(&cache_path)?;
            let reader = std::io::BufReader::new(f);
            let vec: Vec<f32> = bincode::decode_from_reader(reader, bincode::config::standard())?;
            return Ok(vec);
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedderError::Tokenizer(e.to_string()))?;

        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let attn_mask = encoding.get_attention_mask();

        let seq_len = ids.len();
        let batch_size = 1usize;

        let input_ids = Tensor::new(ids, &self.device)?.reshape((batch_size, seq_len))?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.reshape((batch_size, seq_len))?;
        let attention_mask =
            Tensor::new(attn_mask, &self.device)?.reshape((batch_size, seq_len))?;

        let outputs = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let seq_len = outputs.dim(1)?;
        let sum = outputs.sum(1)?;
        let mean = (sum / (seq_len as f64))?;
        let embedding = mean.squeeze(0)?.to_vec1::<f32>()?;

        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&cache_path)?;
        bincode::encode_into_std_write(&embedding, &mut f, bincode::config::standard())?;

        Ok(embedding)
    }

    pub async fn embed_texts<'a>(&'a self, texts: &'a [String]) -> Result<Vec<Vec<f32>>> {
        let embedder = self.clone();
        let texts: Vec<String> = texts.to_vec();

        let mut embeddings = tokio::task::spawn_blocking(move || {
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
                let emb = embedder.embed_text_blocking(t)?;
                embeddings.push((i, emb));
                header_span.pb_inc(1);
            }
            std::mem::drop(header_span_enter);
            std::mem::drop(header_span);
            Result::<_>::Ok(embeddings)
        })
        .await
        .map_err(|e| EmbedderError::Other(e.to_string()))??;

        embeddings.sort_by(|(a, _), (b, _)| a.cmp(b));
        let embeddings: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, emb)| emb).collect();

        Ok(embeddings)
    }
}
