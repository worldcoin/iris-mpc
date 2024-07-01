use crate::QUERIES;
use aws_sdk_sqs::Client;
use gpu_iris_mpc::{
    helpers::sqs::{SMPCRequest, SQSMessage},
    setup::galois_engine::degree4::GaloisRingIrisCodeShare,
};
use tokio::task::spawn_blocking;

#[derive(Default)]
pub struct BatchQueryEntries {
    pub code: Vec<GaloisRingIrisCodeShare>,
    pub mask: Vec<GaloisRingIrisCodeShare>,
}

#[derive(Default)]
pub struct BatchQuery {
    pub request_ids: Vec<String>,
    pub query:       BatchQueryEntries,
    pub db:          BatchQueryEntries,
}

/// Receive batch of queries from SQS
pub async fn receive_batch(
    party_id: usize,
    client: &Client,
    queue_url: &String,
) -> eyre::Result<BatchQuery> {
    let mut batch_query = BatchQuery::default();

    while batch_query.db.code.len() < QUERIES {
        let rcv_message_output = client
            .receive_message()
            .max_number_of_messages(1i32)
            .queue_url(queue_url)
            .send()
            .await?;

        for sns_message in rcv_message_output.messages.unwrap_or_default() {
            let message: SQSMessage = serde_json::from_str(sns_message.body().unwrap())?;
            let message: SMPCRequest = serde_json::from_str(&message.message)?;
            batch_query.request_ids.push(message.clone().request_id);

            let (db_iris_shares, db_mask_shares, iris_shares, mask_shares) =
                spawn_blocking(move || {
                    let mut iris_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_iris_shares());
                    let mut mask_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_mask_shares());

                    let db_iris_shares = iris_share.all_rotations();
                    let db_mask_shares = mask_share.all_rotations();

                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut iris_share);
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut mask_share);

                    (
                        db_iris_shares,
                        db_mask_shares,
                        iris_share.all_rotations(),
                        mask_share.all_rotations(),
                    )
                })
                .await?;

            batch_query.db.code.extend(db_iris_shares);
            batch_query.db.mask.extend(db_mask_shares);
            batch_query.query.code.extend(iris_shares);
            batch_query.query.mask.extend(mask_shares);

            // TODO: we should only delete after processing
            client
                .delete_message()
                .queue_url(queue_url)
                .receipt_handle(sns_message.receipt_handle.unwrap())
                .send()
                .await?;
        }
    }

    Ok(batch_query)
}
