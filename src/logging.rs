use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::filter::{EnvFilter, LevelFilter};
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub fn setup_logger() {
    let indicatif_layer = IndicatifLayer::new();

    let env_filter = EnvFilter::builder()
        .with_default_directive(if cfg!(debug_assertions) {
            LevelFilter::TRACE.into()
        } else {
            LevelFilter::INFO.into()
        })
        .with_env_var("DAILY_AI_LOG")
        .from_env_lossy();

    let fmt = fmt::layer()
        .with_ansi(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .with_thread_ids(false)
        .with_writer(indicatif_layer.get_stderr_writer())
        .pretty();

    tracing_subscriber::registry()
        .with(fmt) // Direct fmt logs to stderr writer
        .with(indicatif_layer)
        .with(env_filter)
        .init();
}
