use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TraceState};

pub fn trace_from_message_attributes(trace_id: &str, span_id: &str) -> eyre::Result<SpanContext> {
    tracing::info!(
        "Creating span context from message attributes. trace id: {}, span id: {}",
        trace_id,
        span_id
    );
    // Create and set the span parent context
    let parent_ctx = SpanContext::new(
        TraceId::from(trace_id.parse::<u128>()?),
        SpanId::from(span_id.parse::<u64>()?),
        TraceFlags::default(),
        true,
        TraceState::default(),
    );

    Ok(parent_ctx)
}
