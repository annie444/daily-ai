use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::env;
use std::fs;
use std::path::PathBuf;
use syn::{Error, LitByteStr, LitStr, parse_macro_input};

macro_rules! bail {
    ($call:expr) => {
        match $call {
            Ok(val) => val,
            Err(err) => return Error::new(Span::call_site(), err).to_compile_error().into(),
        }
    };
}

#[proc_macro]
pub fn include_zstd(input: TokenStream) -> TokenStream {
    let input_lit = parse_macro_input!(input as LitStr);
    let file_path = input_lit.value();

    let manifest_dir = bail!(env::var("CARGO_MANIFEST_DIR"));
    let full_path = PathBuf::from(manifest_dir).join(&file_path);

    let content = bail!(fs::read(&full_path));

    let compressed_data = bail!(zstd::stream::encode_all(&content[..], 19));

    let literal_bytes = LitByteStr::new(&compressed_data, Span::call_site());

    let output = quote!(#literal_bytes);

    output.into()
}
