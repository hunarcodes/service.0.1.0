use ndarray::{Array2, ArrayViewD, Axis, Ix3};
use ort::session::Session;
use ort::value::Value;
use std::error::Error;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use std::time::Instant;
// A job sent to the inference task.
// Contains the text to embed and a channel to send the result back.
pub struct InferenceJob {
    pub text: String,
    pub response_sender: Option<oneshot::Sender<Vec<f32>>>,
}

pub async fn inference_batch_processor(
    mut receiver: mpsc::Receiver<InferenceJob>,
    tokenizer: Arc<Tokenizer>,
    session: Arc<Mutex<Session>>,
    max_batch_size: usize,
    max_wait_ms: u64,
) {
    let wait_duration = std::time::Duration::from_millis(max_wait_ms);

    loop {
        let Some(first_job) = receiver.recv().await else { return; };
        let t_batch = Instant::now();

        let mut batch = Vec::with_capacity(max_batch_size);
        batch.push(first_job);

        while batch.len() < max_batch_size {
            match tokio::time::timeout(wait_duration, receiver.recv()).await {
                Ok(Some(job)) => batch.push(job),
                _ => break,
            }
        }

        eprintln!("Batch size: {}, collection took {:?}", batch.len(), t_batch.elapsed());

        if let Err(e) = embed_batch(&mut batch, &tokenizer, &session) {
            eprintln!("Error processing batch: {}", e);
            for job in batch.iter_mut() {
                if let Some(tx) = job.response_sender.take() {
                    let _ = tx.send(vec![]);
                }
            }
        }
    }
}


fn embed_batch(
    batch: &mut Vec<InferenceJob>,
    tokenizer: &Tokenizer,
    session: &Arc<Mutex<Session>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let t_total = Instant::now();

    // ---- Tokenization ----
    let t_tok = Instant::now();
    let texts: Vec<&str> = batch.iter().map(|job| job.text.as_str()).collect();
    let encodings = tokenizer.encode_batch(texts, true)?;
    let max_length = encodings[0].get_ids().len();
    eprintln!("Tokenization took: {:?}", t_tok.elapsed());

    // ---- Tensor Prep ----
    let t_tensor = Instant::now();
    let (input_ids_vec, attention_mask_vec) = encodings.into_iter().fold(
        (Vec::new(), Vec::new()),
        |(mut ids, mut mask), enc| {
            ids.extend(enc.get_ids().iter().map(|&i| i as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&m| m as i64));
            (ids, mask)
        },
    );
    let batch_size = batch.len();
    let input_ids = Array2::from_shape_vec((batch_size, max_length), input_ids_vec)?;
    let attention_mask = Array2::from_shape_vec((batch_size, max_length), attention_mask_vec)?;
    eprintln!("Tensor prep took: {:?}", t_tensor.elapsed());

    // ---- Inference ----
    let t_infer = Instant::now();
    let mut session = session.lock().unwrap();
    let outputs = session.run(ort::inputs! {
        "input_ids" => Value::from_array(input_ids.clone())?,
        "attention_mask" => Value::from_array(attention_mask.clone())?
    })?;
    eprintln!("Inference took: {:?}", t_infer.elapsed());

    // ---- Pooling ----
    let t_pool = Instant::now();
    let last_hidden_view: ArrayViewD<f32> =
        outputs["last_hidden_state"].try_extract_array::<f32>()?;
    let last_hidden_state = last_hidden_view.into_dimensionality::<Ix3>()?;

    let attention_mask_expanded = attention_mask
        .mapv(|x| x as f32)
        .into_shape_with_order((batch_size, max_length, 1))?;

    let masked_hidden_state = &last_hidden_state * &attention_mask_expanded;
    let sum_masked = masked_hidden_state.sum_axis(Axis(1));
    let sum_mask = attention_mask_expanded.sum_axis(Axis(1));
    let mean_pooled_embeddings = &sum_masked / &sum_mask;
    eprintln!("Pooling took: {:?}", t_pool.elapsed());

    // ---- Send back ----
    for (i, job) in batch.iter_mut().enumerate() {
        if let Some(tx) = job.response_sender.take() {
            let _ = tx.send(mean_pooled_embeddings.row(i).to_vec());
        }
    }

    eprintln!("Total embed_batch took: {:?}", t_total.elapsed());
    Ok(())
}