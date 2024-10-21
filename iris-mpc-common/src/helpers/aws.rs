use aws_sdk_sns::types::MessageAttributeValue;
use std::collections::HashMap;

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
