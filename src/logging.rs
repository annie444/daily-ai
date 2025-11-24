use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub fn setup_logger(verbosity: &clap_verbosity_flag::Verbosity) {
    let indicatif_layer = IndicatifLayer::new();

    let env_filter = EnvFilter::builder()
        .with_default_directive(verbosity.tracing_level_filter().into())
        .with_env_var("DAILY_AI_LOG")
        .from_env()
        .unwrap_or_else(|_| EnvFilter::new(verbosity.to_string()));

    let fmt = if cfg!(debug_assertions) {
        fmt::layer()
            .with_ansi(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_writer(indicatif_layer.get_stderr_writer())
            .compact()
    } else {
        fmt::layer()
            .with_ansi(true)
            .with_writer(indicatif_layer.get_stderr_writer())
            .compact()
    };

    tracing_subscriber::registry()
        .with(fmt) // Direct fmt logs to stderr writer
        .with(indicatif_layer)
        .with(env_filter)
        .init();
}
