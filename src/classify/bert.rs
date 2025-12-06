use daily_ai_local_embedder::BertEmbedder as LocalBertEmbedder;
use futures::FutureExt;

use crate::AppResult;
use crate::classify::traits::Embedder;

/// Wrapper around a BERT encoder for URL/title embeddings.
#[derive(Clone)]
pub struct BertEmbedder {
    inner: LocalBertEmbedder,
}

impl Embedder for BertEmbedder {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
    ) -> futures::future::BoxFuture<'a, AppResult<Vec<Vec<f32>>>> {
        use futures::FutureExt;
        async move {
            match self.inner.embed_texts(texts).await {
                Ok(v) => Ok(v),
                Err(e) => Err(crate::error::AppError::Other(e.to_string())),
            }
        }
        .boxed()
    }
}

impl BertEmbedder {
    #[tracing::instrument(name = "Downloading embedding model from Hugging Face", level = "info")]
    pub async fn new_from_pretrained<S: AsRef<str> + std::fmt::Debug>(
        model_name: S,
    ) -> AppResult<Self> {
        let inner = LocalBertEmbedder::new_from_pretrained(model_name)
            .await
            .map_err(|e| crate::error::AppError::Other(e.to_string()))?;
        Ok(Self { inner })
    }
}
