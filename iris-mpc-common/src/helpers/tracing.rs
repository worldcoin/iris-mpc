use opentelemetry::{
    trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState},
    Context,
};
use tracing::info_span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub fn trace_from_message_attributes(trace_id: &str, span_id: &str) -> eyre::Result<SpanContext> {
    tracing::info!(
        "Creating span context from message attributes. trace id: {}, span id: {}",
        trace_id,
        span_id
    );

    // Create and set the span parent context
    let parent_span_ctx = SpanContext::new(
        TraceId::from(trace_id.parse::<u128>()?),
        SpanId::from(span_id.parse::<u64>()?),
        TraceFlags::default(),
        true,
        TraceState::default(),
    );
    // let parent_ctx = Context::new().with_remote_span_context(parent_span_ctx);
    // let item_span = info_span!("item", trace_id = trace_id, span_id = span_id);
    // item_span.set_parent(parent_ctx.clone());
    Ok(parent_span_ctx)
}

pub fn link_batch_spans(span_contexts: Vec<SpanContext>, batch_span: &tracing::Span) {
    tracing::info!("Linking batch spans to item spans");
    for span_ctx in span_contexts {
        let parent_ctx = Context::new().with_remote_span_context(span_ctx);
        let item_span = info_span!("item");
        item_span.set_parent(parent_ctx.clone());
        batch_span.follows_from(&item_span);
        item_span.follows_from(batch_span);
    }
}
