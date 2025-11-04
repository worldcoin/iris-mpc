use crate::{execution::player::Identity, proto_generated::party_node::SendRequest};
use eyre::Result;
use std::{collections::HashMap, time::Duration};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tonic::Status;

mod handle;
mod networking;
mod session;

#[allow(unused_imports)]
pub use handle::*;
pub use networking::*;

type TonicResult<T> = Result<T, Status>;

fn err_to_status(e: eyre::Error) -> Status {
    Status::internal(e.to_string())
}

type OutStream = UnboundedSender<SendRequest>;
type OutStreams = HashMap<Identity, OutStream>;
type InStream = UnboundedReceiver<SendRequest>;
type InStreams = HashMap<Identity, InStream>;

#[derive(Default, Clone, Debug)]
pub struct GrpcConfig {
    pub timeout_duration: Duration,
    // number of gRPC connections to create
    pub connection_parallelism: usize,
    // number of application level sessions per gRPC stream.
    pub stream_parallelism: usize,
}

#[cfg(test)]
mod tests {
    use super::{session::GrpcSession, *};
    use crate::{
        execution::{local::generate_local_identities, player::Role, session::SessionId},
        hawkers::aby3::{aby3_store::Aby3Query, test_utils::shared_random_setup},
        hnsw::HnswSearcher,
        network::{value::NetworkValue, NetworkType, Networking},
    };
    use aes_prng::AesRng;
    use futures::future::join_all;
    use iris_mpc_common::vector_id::VectorId;
    use rand::Rng;
    use rand::SeedableRng;
    use tokio::{task::JoinSet, time::sleep};
    use tracing_test::traced_test;

    // can only send NetworkValue over the network. PrfKey is easy to make so this is used here.
    fn get_prf() -> NetworkValue {
        let mut rng = rand::thread_rng();
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        NetworkValue::PrfKey(key)
    }

    async fn create_session_helper(
        session_id: SessionId,
        players: &[GrpcHandle],
    ) -> Result<Vec<GrpcSession>> {
        let mut jobs = vec![];
        for player in players.iter() {
            let player = player.clone();
            let task = tokio::spawn(async move {
                tracing::trace!(
                    "Player {:?} is creating session {:?}",
                    player.party_id(),
                    session_id
                );
                player.create_session(session_id).await.unwrap()
            });
            jobs.push(task);
        }
        join_all(jobs)
            .await
            .into_iter()
            .map(|r| r.map_err(eyre::Report::new))
            .collect::<Result<Vec<_>>>()
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_correct() -> Result<()> {
        let identities = generate_local_identities();
        let players = setup_local_grpc_networking(
            identities.clone(),
            NetworkType::default_connection_parallelism(),
            NetworkType::default_stream_parallelism(),
        )
        .await?;

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        {
            let players = players.clone();

            let session_id = SessionId::from(0);

            jobs.spawn(async move {
                let mut players = create_session_helper(session_id, &players).await.unwrap();

                // we don't need the last player here
                players.pop();

                let mut bob = players.pop().unwrap();
                let mut alice = players.pop().unwrap();

                // Send a message from the first party to the second party
                let message = get_prf();
                let message_copy = message.clone();

                let task1 = tokio::spawn(async move {
                    alice.send(message.clone(), &"bob".into()).await.unwrap();
                });
                let task2 = tokio::spawn(async move {
                    let received_message = bob.receive(&"alice".into()).await.unwrap();
                    assert_eq!(message_copy, received_message);
                });
                let _ = tokio::try_join!(task1, task2).unwrap();
            });
        }

        // Multiple parties sending messages to each other
        let all_parties_talk = |identities: Vec<Identity>, sessions: Vec<GrpcSession>| async move {
            let mut tasks = JoinSet::new();
            let message_to_next = get_prf();
            let message_to_prev = get_prf();
            for (player_id, session) in sessions.into_iter().enumerate() {
                let role = Role::new(player_id);
                let next = role.next(3).index();
                let prev = role.prev(3).index();

                let next_id = identities[next].clone();
                let prev_id = identities[prev].clone();

                let mut session = session;
                let msg_next = message_to_next.clone();
                let msg_prev = message_to_prev.clone();
                tasks.spawn(async move {
                    // Sending
                    session.send(msg_next.clone(), &next_id).await.unwrap();
                    session.send(msg_prev.clone(), &prev_id).await.unwrap();

                    // Receiving
                    let received_message_from_prev = session.receive(&prev_id).await.unwrap();
                    assert_eq!(received_message_from_prev, msg_next);
                    let received_message_from_next = session.receive(&next_id).await.unwrap();
                    assert_eq!(received_message_from_next, msg_prev);
                });
            }
            tasks.join_all().await;
        };

        // Each party sending and receiving messages to each other
        {
            let players = players.clone();
            let identities = identities.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);

                let players = create_session_helper(session_id, &players).await.unwrap();

                // Test that parties can send and receive messages
                all_parties_talk(identities, players).await;
            });
        }

        // Parties create a session asynchronously
        {
            let players = players.clone();
            let session_id = SessionId::from(2);
            // Session is consecutively created
            let sessions = {
                let mut jobs = vec![];
                for (i, player) in players.iter().enumerate() {
                    let player = player.clone();
                    let task = tokio::spawn(async move {
                        tracing::trace!(
                            "Player {:?} is creating session {:?}",
                            player.party_id(),
                            session_id
                        );
                        sleep(Duration::from_millis(200 * i as u64)).await;
                        player.create_session(session_id).await.unwrap()
                    });
                    jobs.push(task);
                }
                join_all(jobs)
                    .await
                    .into_iter()
                    .map(|r| r.map_err(eyre::Report::new))
                    .collect::<Result<Vec<_>>>()?
            };

            // Test that parties can send and receive messages
            all_parties_talk(identities, sessions).await;
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_fail() -> Result<()> {
        let parties = generate_local_identities();

        let players = setup_local_grpc_networking(
            parties.clone(),
            NetworkType::default_connection_parallelism(),
            NetworkType::default_stream_parallelism(),
        )
        .await?;

        let mut jobs = JoinSet::new();

        {
            // Send to a non-existing party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(0);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let message = get_prf();
                let res = sessions[0]
                    .send(message.clone(), &Identity::from("eve"))
                    .await;
                assert_eq!(
                    "Outgoing stream for Identity(\"eve\") in SessionId(0) not found",
                    res.unwrap_err().to_string()
                );
            });
        }

        {
            // Receive from a wrong party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let res = sessions[0].receive(&Identity::from("eve")).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Incoming stream for Identity(\"eve\") in SessionId(1) not found"
                );
            });
        }

        {
            // Send to itself
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let message = get_prf();
                let res = sessions[0]
                    .send(message.clone(), &Identity::from("alice"))
                    .await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Outgoing stream for Identity(\"alice\") in SessionId(2) not found",
                );
            });
        }

        {
            // Add the same session
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                let _ = create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.create_session(session_id).await;

                assert_eq!(
                    res.unwrap_err().to_string(),
                    "SessionId(3) has already been created by Identity(\"alice\")"
                );
            });
        }

        {
            // Receive from a party that didn't send a message (timeout error)
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let res = sessions[0].receive(&Identity::from("bob")).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Identity(\"alice\"): Timeout while waiting for message from \
                     Identity(\"bob\") in SessionId(4)"
                );
            });
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_hnsw_local() {
        use std::time::Instant;

        let t0 = Instant::now(); //measurte time it takes for test to run

        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 20;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut vectors_and_graphs = shared_random_setup(
            &mut rng,
            database_size,
            crate::network::NetworkType::default_grpc(),
        )
        .await
        .unwrap();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let mut jobs = JoinSet::new();

            for (store, graph) in vectors_and_graphs.iter_mut() {
                let searcher = searcher.clone();
                let q = store
                    .lock()
                    .await
                    .storage
                    .get_vector_or_empty(&vector_id)
                    .await;
                let q = Aby3Query::new(&q);
                let store = store.clone();
                let graph = graph.clone();
                jobs.spawn(async move {
                    let mut store_lock = store.lock().await;
                    let secret_neighbors = searcher
                        .search(&mut *store_lock, &graph, &q, 1)
                        .await
                        .unwrap();
                    searcher
                        .is_match(&mut *store_lock, &[secret_neighbors])
                        .await
                        .unwrap()
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
        let dt = t0.elapsed();
        eprintln!("TIME: {:?}", dt);
    }
}
