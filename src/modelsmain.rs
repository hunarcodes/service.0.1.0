use ndarray::Array1;
use ndarray::Axis;
use ndarray::{ArrayViewD, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::error::Error;
use tokenizers::Tokenizer;
// Import your tokenize function
mod tokenize;
use crate::tokenize::tokenize_text;

// Cosine similarity function remains the same
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dot / (norm_a * norm_b)
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- 1. Setup ---
    let model_path = "/home/garv/gemma-embedder-rust/model/model_quantized.onnx";
    let mut tokenizer =
        Tokenizer::from_file("/home/garv/gemma-embedder-rust/model/tokenizer.json")?;

    let mut session = Session::builder()?
        .with_intra_threads(num_cpus::get())?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    let s1 = "Rust is memory safe.";
    let s2 = "The compiler prevents data races.";
    let max_length = 32;

    // --- 2. Tokenization (now with padding) ---
    // Both enc1 and enc2 will now have a length of 32
    let enc1 = tokenize_text(s1, &mut tokenizer, max_length, true)?;
    let enc2 = tokenize_text(s2, &mut tokenizer, max_length, true)?;

    let ids1: Vec<i64> = enc1.get_ids().iter().map(|&id| id as i64).collect();
    let mask1: Vec<i64> = enc1
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();

    let ids2: Vec<i64> = enc2.get_ids().iter().map(|&id| id as i64).collect();
    let mask2: Vec<i64> = enc2
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();

    // --- 3. Batch Creation (This will now work) ---
    // The total number of elements will be 2 * 32 = 64, which matches the shape.
    let input_ids = ndarray::Array2::from_shape_vec((2, max_length), [ids1, ids2].concat())?;
    let attention_mask = ndarray::Array2::from_shape_vec((2, max_length), [mask1, mask2].concat())?;

    // --- 4. Run Inference (Using named inputs) ---
    // It's safer to name your inputs to match the model's expectations.
    // Correct and fully explicit
    let outputs = session.run(ort::inputs! {
        "input_ids" => Value::from_array(input_ids.clone())?,
        "attention_mask" => Value::from_array(attention_mask.clone())?
    })?;

    let last_hidden_view: ArrayViewD<f32> =
        outputs["last_hidden_state"].try_extract_array::<f32>()?;
    let last_hidden_state = last_hidden_view.into_dimensionality::<Ix3>()?; // shape [2, max_length, hidden]

    // Perform mean pooling
    let attention_mask_expanded = attention_mask
        .mapv(|x| x as f32)
        .into_shape_with_order((2, max_length, 1))?;

    // Correct
    let masked_hidden_state = &last_hidden_state * &attention_mask_expanded;

    let sum_masked = masked_hidden_state.sum_axis(Axis(1));
    let sum_mask = attention_mask_expanded.sum_axis(Axis(1));
    let mean_pooled_embeddings = &sum_masked / &sum_mask;

    // --- 6. Calculate Similarity ---
    let e1 = mean_pooled_embeddings.row(0).to_owned();
    let e2 = mean_pooled_embeddings.row(1).to_owned();

    println!("Cosine similarity = {}", cosine_similarity(&e1, &e2));

    Ok(())
}
