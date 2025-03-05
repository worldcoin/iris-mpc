use aws_sdk_sqs::{operation::delete_message::DeleteMessageOutput, Client};
use iris_mpc_common::helpers::smpc_request::ReceiveRequestError;
use std::time::Duration;

const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);

pub struct MessageReceiver<'a> {
    client: &'a Client,
    queue_url: String,
    max_messages: i32,
    pub wait_time: Duration,
}

impl<'a> MessageReceiver<'a> {
    pub fn new(client: &'a Client, queue_url: String) -> Self {
        Self {
            client,
            queue_url,
            max_messages: 1,
            wait_time: SQS_POLLING_INTERVAL,
        }
    }

    pub async fn receive_messages(
        &self,
    ) -> Result<Vec<aws_sdk_sqs::types::Message>, ReceiveRequestError> {
        let rcv_message_output = self
            .client
            .receive_message()
            .max_number_of_messages(self.max_messages)
            .queue_url(&self.queue_url)
            .send()
            .await
            .map_err(ReceiveRequestError::FailedToReadFromSQS)?;

        if let Some(messages) = rcv_message_output.messages {
            Ok(messages)
        } else {
            Ok(vec![])
        }
    }

    pub async fn delete_message(
        &self,
        receipt_handle: String,
    ) -> Result<DeleteMessageOutput, ReceiveRequestError> {
        self.client
            .delete_message()
            .queue_url(&self.queue_url)
            .receipt_handle(receipt_handle)
            .send()
            .await
            .map_err(ReceiveRequestError::FailedToDeleteFromSQS)
    }
}
