use async_openai::types::embeddings::CreateEmbeddingRequestArgs;
use async_openai::{Client, config::Config};
use futures::FutureExt;

use crate::AppResult;
use crate::classify::traits::Embedder;

/// Embedding implementation that uses OpenAI API.
#[derive(Clone)]
#[allow(dead_code)]
pub struct OAIEmbedder<'a, C: Config> {
    client: &'a Client<C>,
    model: String,
}

#[allow(dead_code)]
impl<'a, C: Config> OAIEmbedder<'a, C> {
    pub fn new(client: &'a Client<C>, model: String) -> Self {
        Self { client, model }
    }
}

impl<'a, C: Config> Embedder for OAIEmbedder<'a, C> {
    fn embed<'e>(
        &'e self,
        texts: &'e [String],
    ) -> futures::future::BoxFuture<'e, AppResult<Vec<Vec<f32>>>> {
        async move {
            let request = CreateEmbeddingRequestArgs::default()
                .model(&self.model)
                .input(texts.to_vec())
                .build()
                .map_err(|e| crate::error::AppError::Other(e.to_string()))?;

            let response = self
                .client
                .embeddings()
                .create(request)
                .await
                .map_err(|e| crate::error::AppError::Other(e.to_string()))?;

            let embeddings = response.data.into_iter().map(|d| d.embedding).collect();
            Ok(embeddings)
        }
        .boxed()
    }
}
