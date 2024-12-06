use aws_sdk_sns::types::MessageAttributeValue;
use std::collections::HashMap;
use telemetry_batteries::reexports::opentelemetry::trace::{
    SpanContext, SpanId, TraceFlags, TraceId, TraceState,
};

pub const TRACE_ID_MESSAGE_ATTRIBUTE_NAME: &str = "TraceID";
pub const SPAN_ID_MESSAGE_ATTRIBUTE_NAME: &str = "SpanID";
pub const NODE_ID_MESSAGE_ATTRIBUTE_NAME: &str = "NodeID";

pub fn construct_message_attributes(
    trace_id: &String,
    span_id: &String,
) -> eyre::Result<HashMap<String, MessageAttributeValue>> {
    let mut message_attributes = HashMap::new();

    let trace_id_message_attribute = MessageAttributeValue::builder()
        .data_type("String")
        .string_value(trace_id)
        .build()?;

    message_attributes.insert(
        TRACE_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
        trace_id_message_attribute,
    );

    let span_id_message_attribute = MessageAttributeValue::builder()
        .data_type("String")
        .string_value(span_id)
        .build()?;

    message_attributes.insert(
        SPAN_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
        span_id_message_attribute,
    );

    Ok(message_attributes)
}

// This would only ever be leveraged if the code had isolated flows for every
// message, leaving for now, maybe it will happen in the future
pub fn trace_from_message_attributes(
    message_attributes: &HashMap<String, MessageAttributeValue>,
    receipt_handle: &str,
) -> eyre::Result<()> {
    if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
        if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
            let trace_id = trace_id
                .string_value()
                .expect("Could not parse TraceID")
                .parse::<u128>()?;

            let span_id = span_id
                .string_value()
                .expect("Could not parse SpanID")
                .parse::<u64>()?;

            // Create and set the span parent context
            let parent_ctx = SpanContext::new(
                TraceId::from(trace_id),
                SpanId::from(span_id),
                TraceFlags::default(),
                true,
                TraceState::default(),
            );

            telemetry_batteries::tracing::trace_from_ctx(parent_ctx);
        } else {
            tracing::warn!(?receipt_handle, "SQS message missing SpanID");
        }
    } else {
        tracing::warn!(?receipt_handle, "SQS message missing TraceID");
    }

    Ok(())
}
