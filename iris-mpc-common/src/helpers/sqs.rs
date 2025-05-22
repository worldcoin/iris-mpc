use crate::config::Config;
use crate::helpers::smpc_request::SQSMessage;
use aws_sdk_sqs::Client;
use eyre::Context;
use eyre::Result;

/// SQS messages contain a sequence number when they originate from SNS and raw message delivery is disabled.
/// This function reads the top of the requests SQS queue and returns its sequence number.
pub async fn get_next_sns_seq_num(config: &Config, sqs_client: &Client) -> Result<Option<u128>> {
    let sqs_snoop_response = sqs_client
        .receive_message()
        .wait_time_seconds(config.sqs_sync_long_poll_seconds)
        .max_number_of_messages(1)
        .set_visibility_timeout(Some(0))
        .queue_url(&config.requests_queue_url)
        .send()
        .await
        .context("while reading from SQS to snoop on the next message")?;
    if let Some(msgs) = sqs_snoop_response.messages {
        if msgs.len() != 1 {
            return Err(eyre::eyre!(
                "Expected exactly one message in the queue, but found {}",
                msgs.len()
            ));
        }
        let sqs_message = msgs.first().expect("first sqs message is empty");
        match serde_json::from_str::<SQSMessage>(sqs_message.body().unwrap_or("")) {
            Ok(message) => {
                // found a valid message originating from SNS --> get its sequence number
                let sequence_number: u128 =
                    str::parse(&message.sequence_number).expect("sequence number is not a number");
                Ok(Some(sequence_number))
            }
            Err(err) => {
                tracing::error!(
                        "Found corrupt message in queue while parsing SNS message from body. The error is: '{}'. The SQS message body is '{}'", err, sqs_message.body().unwrap_or("None")
                    );
                Err(err)
                    .context("Found corrupt message in queue while parsing SNS message from body")
            }
        }
    } else {
        tracing::info!("Timeout while waiting for a message in the queue. Queue is clean.");
        Ok(None)
    }
}

/// Deletes all messages in the requests SQS queue until the sequence number is reached.
/// Leaves the message with given sequence number on top of the queue.
/// <https://docs.aws.amazon.com/sns/latest/dg/fifo-topic-message-ordering.html>
pub async fn delete_messages_until_sequence_num(
    config: &Config,
    sqs_client: &Client,
    my_sequence_num: Option<u128>,
    target_sequence_num: Option<u128>,
) -> Result<()> {
    tracing::info!(
        "Syncing queues. my_sequence_num: {:?}, target_sequence_num: {:?}.",
        my_sequence_num,
        target_sequence_num
    );

    // Exit early if there is no need to delete messages
    if target_sequence_num.is_none() {
        return if my_sequence_num.is_some() {
            Err(eyre::eyre!("SQS target sequence number is None, but my sequence number is Some. This should not happen."))
        } else {
            tracing::info!("Target sequence number is None. Queues are in clean state.");
            Ok(())
        };
    }

    let target_sequence_num = target_sequence_num.expect("could not unwrap target sequence number");
    if my_sequence_num.is_none() {
        tracing::info!(
            "My sequence number is None. Deleting all messages until target sequence number."
        );
    } else {
        let my_sequence_num = my_sequence_num.expect("could not unwrap my sequence number");
        if my_sequence_num == target_sequence_num {
            tracing::info!("My sequence number is already equal to target sequence number. No need to delete messages.");
            return Ok(());
        }
    }

    // Delete messages until the target sequence number is reached
    loop {
        let sqs_snoop_response = sqs_client
            .receive_message()
            .wait_time_seconds(config.sqs_sync_long_poll_seconds)
            .max_number_of_messages(1)
            .queue_url(&config.requests_queue_url)
            .send()
            .await
            .context("while reading from SQS to delete messages")?;
        if let Some(msgs) = sqs_snoop_response.messages {
            if msgs.len() != 1 {
                return Err(eyre::eyre!(
                    "Expected exactly one message in the queue, but found {}",
                    msgs.len()
                ));
            }
            let msg = msgs.first().expect("first sqs message is empty");
            let sequence_num: u128 = str::parse(
                &serde_json::from_str::<SQSMessage>(msg.body().unwrap_or(""))
                    .expect("message is not a valid SQS message")
                    .sequence_number,
            )
            .expect("sequence number is not a number");
            if sequence_num < target_sequence_num {
                tracing::warn!(
                    "Deleting message with sequence number: {}, body: {:?}",
                    sequence_num,
                    msg.body
                );
                sqs_client
                    .delete_message()
                    .queue_url(&config.requests_queue_url)
                    .receipt_handle(msg.receipt_handle.clone().unwrap_or_default())
                    .send()
                    .await
                    .context("while deleting message from SQS")?;
            } else {
                tracing::info!(
                    "Reached target sequence number. Top of queue has sequence num: {}",
                    sequence_num
                );
                // Leave the message with target sequence number on top of the queue
                sqs_client
                    .change_message_visibility()
                    .queue_url(&config.requests_queue_url)
                    .receipt_handle(msg.receipt_handle.clone().unwrap_or_default())
                    .visibility_timeout(0)
                    .send()
                    .await
                    .context("while changing message visibility in SQS")?;
                break;
            }
        } else {
            tracing::info!("Could not find more messages in the queue. Retrying.");
        }
    }

    Ok(())
}
