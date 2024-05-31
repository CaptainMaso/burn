use std::io::Write;

use tracing_subscriber::prelude::*;

/// Initialise and create a `env_logger::Builder` which follows the
/// GitHub Actions logging syntax when running on CI.
pub(crate) fn init_logger() {
    let mut filter = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive(get_log_level().into())
        .from_env_lossy();

    let fmt = tracing_subscriber::fmt::fmt()
        .with_writer(|| std::io::stdout().lock())
        .with_env_filter(filter);

    // Custom Formatter for Github Actions
    if std::env::var("CI").is_ok() {
        fmt.event_format(FmtCI).init();
    } else {
        fmt.init();
    }
}

/// Determine the LogLevel for the logger
fn get_log_level() -> tracing_subscriber::filter::LevelFilter {
    // DEBUG
    match std::env::var("DEBUG") {
        Ok(_value) => return tracing_subscriber::filter::LevelFilter::DEBUG,
        Err(_err) => (),
    }
    // ACTIONS_RUNNER_DEBUG
    match std::env::var("ACTIONS_RUNNER_DEBUG") {
        Ok(_value) => return tracing_subscriber::filter::LevelFilter::DEBUG,
        Err(_err) => (),
    };

    tracing_subscriber::filter::LevelFilter::INFO
}

/// Group Macro
#[macro_export]
macro_rules! group {
    // group!()
    ($($arg:tt)*) => {
        let title = format!($($arg)*);
        if std::env::var("CI").is_ok() {
            ::tracing::info!( "::group::{}", title)
        } else {
            ::tracing::info!("{}", title)
        }
    };
}

/// End Group Macro
#[macro_export]
macro_rules! endgroup {
    // endgroup!()
    () => {
        if std::env::var("CI").is_ok() {
            ::tracing::info!("::endgroup::")
        }
    };
}

struct FmtCI;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for FmtCI
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        match *event.metadata().level() {
            tracing::Level::DEBUG => write!(writer, "::debug:: ")?,
            tracing::Level::WARN => write!(writer, "::warning:: ")?,
            tracing::Level::ERROR => write!(writer, "::error:: ")?,
            _ => (),
        }

        // Format all the spans in the event's span context.
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                write!(writer, "{}", span.name())?;

                // `FormattedFields` is a formatted representation of the span's
                // fields, which is stored in its extensions by the `fmt` layer's
                // `new_span` method. The fields will have been formatted
                // by the same field formatter that's provided to the event
                // formatter in the `FmtContext`.
                let ext = span.extensions();
                let fields = &ext
                    .get::<tracing_subscriber::fmt::FormattedFields<N>>()
                    .expect("will never be `None`");

                // Skip formatting the fields if the span had no fields.
                if !fields.is_empty() {
                    write!(writer, "{{{}}}", fields)?;
                }
                write!(writer, ": ")?;
            }
        }

        Ok(())
    }
}
