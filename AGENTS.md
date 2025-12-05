# Agent Guidelines

## Build & Test
- **Build**: `cargo build`
- **Lint**: `cargo clippy` (fix suggestions) and `cargo fmt` (always run before committing)
- **Test**: `cargo test` (all tests)
- **Single Test**: `cargo test <test_name_substring>` (e.g., `cargo test my_feature`)

## Code Style & Conventions
- **Formatting**: Adhere strictly to `rustfmt`.
- **Error Handling**: Use `crate::error::AppResult` and `AppError`. Extend `AppError` using `thiserror` for new error types.
- **Logging**: Use `tracing` macros (`info!`, `debug!`, `error!`) instead of `println!`.
- **Async**: Use `tokio` for async runtime. Ensure functions are `async` where appropriate.
- **Database**: Use `sea-orm` entities (in `src/entity`).
- **Imports**: Group imports by crate. Prefer explicit imports over `*`.
- **Naming**: Snake_case for variables/functions; PascalCase for structs/enums/traits.
- **New Files**: Register new modules in `mod.rs`.
