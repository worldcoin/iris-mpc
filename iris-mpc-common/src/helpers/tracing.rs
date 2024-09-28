use opentelemetry::{
    global,
    trace::{Span, SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState, Tracer},
    Context,
};

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
    let parent_ctx = Context::new().with_remote_span_context(parent_span_ctx);
    let tracer = global::tracer("mpcv2-batch-tracer");
    let mut span = tracer
        .span_builder("mpcv2-batch-item")
        .start_with_context(&tracer, &parent_ctx);
    span.add_event("Created batch span item", vec![]);
    tracing::info!("Created batch span item");
    Ok(span.span_context().clone())
}
