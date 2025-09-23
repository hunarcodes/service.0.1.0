use std::error::Error;
use tokenizers::{
    tokenizer::Tokenizer, PaddingParams, PaddingStrategy, TruncationParams, TruncationStrategy,
};

pub fn tokenize_text(
    input: &str,
    tokenizer: &mut Tokenizer,
    max_length: usize,
    add_special_tokens: bool,
) -> Result<tokenizers::Encoding, Box<dyn Error + Send + Sync>> {
    // Grab [PAD] token id (default to 0 if missing)
    let pad_token_id = tokenizer.token_to_id("[PAD]").unwrap_or(0) as i64;

    // Enable both padding and truncation to the same max_length
    let padding_params = PaddingParams {
        strategy: PaddingStrategy::Fixed(max_length),
        pad_id: pad_token_id as u32, // field is u32, so cast back down
        ..Default::default()
    };
    let truncation_params = TruncationParams {
        max_length,
        strategy: TruncationStrategy::LongestFirst,
        ..Default::default()
    };

    // Apply the configurations
    tokenizer
        .with_padding(Some(padding_params))
        .with_truncation(Some(truncation_params));

    // Encode the input text
    let encoding = tokenizer.encode(input, add_special_tokens)?;

    Ok(encoding)
}
