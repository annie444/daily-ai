use ndarray::prelude::*;
use tracing::{trace, warn};

#[tracing::instrument(name = "Converting links", level = "info", skip(embs))]
pub fn embeddings_to_ndarray(embs: &[Vec<f32>]) -> Array2<f64> {
    let rows = embs.len();
    let cols = embs[0].len();
    let mut arr: Array2<f64> = Array2::<f64>::zeros((rows, cols));
    trace!("Initialized ndarray with shape: {:?}", arr.dim());
    for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            *val = embs[i][j] as f64;
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn converts_embeddings_to_f64_ndarray() {
        let embs = vec![vec![1.0_f32, 2.5_f32], vec![3.75_f32, -4.0_f32]];

        let arr = embeddings_to_ndarray(&embs);

        assert_eq!(arr.dim(), (2, 2));
        let expected = array![[1.0_f64, 2.5_f64], [3.75_f64, -4.0_f64]];
        assert_eq!(arr, expected);
    }

    #[test]
    #[should_panic]
    fn panics_on_empty_input() {
        let embs: Vec<Vec<f32>> = Vec::new();
        // function indexes embs[0]; ensure we catch the panic to document behavior
        let _ = embeddings_to_ndarray(&embs);
    }
}
