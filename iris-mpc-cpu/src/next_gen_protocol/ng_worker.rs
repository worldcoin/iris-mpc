use crate::{
    execution::{
        player::{Identity, Role, RoleAssignment},
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    next_gen_network::{
        local::{LocalNetworking, LocalNetworkingStore},
        value::NetworkValue::{self, PrfKey, RingElement16},
    },
    next_gen_protocol::binary::lift,
    prelude::IrisWorker,
    protocol::prf::{Prf, PrfSeed},
    shares::{int_ring::IntRing2k, ring_impl::RingElement, share::Share, vecshare::VecShare},
};
use eyre::eyre;
use iris_mpc_common::iris_db::iris::IrisCodeArray;
use rand::RngCore;
use std::{collections::HashMap, ops::SubAssign, sync::Arc};
use tokio::task::JoinSet;

pub struct LocalRuntime {
    pub identities:       Vec<Identity>,
    pub role_assignments: RoleAssignment,
    pub prf_setups:       Option<HashMap<Role, Prf>>,
    pub net_store:        LocalNetworkingStore,
    pub seeds:            Vec<PrfSeed>,
}

pub async fn setup_replicated_prf(session: &BootSession, my_seed: PrfSeed) -> eyre::Result<Prf> {
    let next_role = session.own_role()?.next(3);
    let prev_role = session.own_role()?.prev(3);
    let network = session.network();
    // send my_seed to the next party
    let _ = network
        .send(
            NetworkValue::PrfKey(my_seed).to_network(),
            session.identity(&next_role)?,
            &session.session_id,
        )
        .await;
    // received other seed from the previous party
    let serialized_other_seed = network
        .receive(session.identity(&prev_role)?, &session.session_id)
        .await;
    // deserializing received seed.
    let other_seed = match NetworkValue::from_network(serialized_other_seed) {
        Ok(NetworkValue::PrfKey(seed)) => seed,
        _ => return Err(eyre!("Could not deserialize PrfKey")),
    };
    // creating the two PRFs
    Ok(Prf::new(my_seed, other_seed))
}

pub(crate) async fn setup_session<Rng: RngCore>(session: BootSession, rng: &mut Rng) -> Session {
    let mut seed = [0_u8; 16];
    rng.fill_bytes(&mut seed);
    let prf = setup_replicated_prf(&session, seed).await.unwrap();
    Session {
        boot_session: session,
        setup:        prf,
    }
}

pub(crate) async fn open_single(
    session: &Session,
    x: Share<u16>,
) -> eyre::Result<RingElement<u16>> {
    let network = session.network();
    let next_role = session.identity(&session.own_role()?.next(3))?;
    let prev_role = session.identity(&session.own_role()?.prev(3))?;
    let _ = network
        .send(
            RingElement16(x.b).to_network(),
            next_role,
            &session.session_id(),
        )
        .await;
    let serialized_reply = network.receive(prev_role, &session.session_id()).await;
    let missing_share = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::RingElement16(element)) => element,
        _ => return Err(eyre!("Could not deserialize RingElement16")),
    };
    let (a, b) = x.get_ab();
    Ok(a + b + missing_share)
}

pub(crate) async fn replicated_dot(
    session: &mut Session,
    a: &Vec<Share<u16>>,
    b: &Vec<Share<u16>>,
) -> eyre::Result<Share<u16>> {
    if a.len() != b.len() {
        return Err(eyre!("Error::InvalidSize"));
    }
    let mut prf = session.prf_as_mut();

    let res_a = {
        let mut rand = prf.gen_zero_share();
        for (a__, b__) in a.iter().zip(b.iter()) {
            rand += a__ * b__;
        }
        rand
    };
    let network = session.network();
    let next_role = session.identity(&session.own_role()?.next(3))?;
    let prev_role = session.identity(&session.own_role()?.prev(3))?;

    let _ = network
        .send(
            RingElement16(res_a).to_network(),
            next_role,
            &session.session_id(),
        )
        .await;

    let serialized_reply = network.receive(prev_role, &session.session_id()).await;
    let res_b = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::RingElement16(element)) => element,
        _ => return Err(eyre!("Could not deserialize RingElement16")),
    };

    let res = Share::new(res_a, res_b);
    Ok(res)
}

pub(crate) async fn replicated_pairwise_distance(
    session: &mut Session,
    lhs_shares: &Vec<Share<u16>>,
    rhs_shares: &Vec<Share<u16>>,
    lhs_mask: &IrisCodeArray,
    rhs_mask: &IrisCodeArray,
) -> eyre::Result<(Share<u16>, usize)> {
    let combined_mask = *lhs_mask & *rhs_mask;
    let mask_dots = combined_mask.count_ones();
    let code_dots = replicated_dot(session, lhs_shares, rhs_shares).await?;
    Ok((code_dots, mask_dots))
}

pub(crate) async fn replicated_cross_mul(
    session: &mut Session,
    d1: Share<u16>,
    t1: u32,
    d2: Share<u16>,
    t2: u32,
) -> eyre::Result<Share<u32>> {
    let mut vd1 = VecShare::<u16>::with_capacity(1);
    // Do preprocessing to lift d1
    vd1.push(d1);
    // Compute (d1 + 2^{15}) % 2^{16}
    for x in vd1.iter_mut() {
        x.add_assign_const_role(1_u16 << 15, session.own_role()?);
    }
    let mut lifted_d1 = lift::<16>(session, vd1).await?;
    // Now we got shares of d1' over 2^32 such that d1' = (d1'_1 + d1'_2 + d1'_3) %
    // 2^{16} = d1 Next we subtract the 2^15 term we've added previously to
    // get signed shares over 2^{32}
    for x in lifted_d1.iter_mut() {
        x.add_assign_const_role(((1_u64 << 32) - (1_u64 << 15)) as u32, session.own_role()?);
    }

    // Compute d1 * t2
    for x in lifted_d1.iter_mut() {
        *x *= t2;
    }

    // Do preprocessing to lift d2
    let mut vd2 = VecShare::<u16>::with_capacity(1);
    vd2.push(d2);
    // Same process for d2, compute (d2 + 2^{15}) % 2^{16}
    for x in vd2.iter_mut() {
        x.add_assign_const_role(1_u16 << 15, session.own_role()?);
    }

    let mut lifted_d2 = lift::<16>(session, vd2).await?;
    // Now get rid of the 2^{15} term to get signed shares over 2^{32}
    for x in lifted_d2.iter_mut() {
        x.add_assign_const_role(((1_u64 << 32) - (1_u64 << 15)) as u32, session.own_role()?);
    }
    // Compute d2 * t1
    for x in lifted_d2.iter_mut() {
        *x *= t1;
    }
    // Compute d2*t1 - d1*t2
    lifted_d2.sub_assign(lifted_d1);
    Ok(lifted_d2.get_at(0))
}

impl LocalRuntime {
    fn new(identities: Vec<Identity>, seeds: Vec<PrfSeed>) -> Self {
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        let net_store = LocalNetworkingStore::from_host_ids(&identities);
        LocalRuntime {
            identities,
            role_assignments,
            net_store,
            prf_setups: None,
            seeds,
        }
    }

    async fn create_player_sessions(
        &self,
        sess_id: SessionId,
    ) -> eyre::Result<HashMap<Identity, Session>> {
        let boot_sessions: Vec<BootSession> = (0..self.seeds.len())
            .map(|i| {
                let identity = self.identities[i].clone();
                BootSession {
                    session_id:       sess_id,
                    role_assignments: Arc::new(self.role_assignments.clone()),
                    networking:       Arc::new(self.net_store.get_local_network(identity.clone())),
                    own_identity:     identity,
                }
            })
            .collect();

        let mut jobs = JoinSet::new();
        for (player_id, boot_session) in boot_sessions.iter().enumerate() {
            let player_seed = self.seeds[player_id].clone();
            let sess = boot_session.clone();
            jobs.spawn(async move {
                let prf = setup_replicated_prf(&sess, player_seed).await.unwrap();
                (sess, prf)
            });
        }
        let mut complete_sessions = HashMap::new();
        while let Some(t) = jobs.join_next().await {
            let (boot_session, prf) = t.unwrap();
            complete_sessions.insert(boot_session.own_identity(), Session {
                boot_session,
                setup: prf,
            });
        }
        Ok(complete_sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::session::{self, BootSession, SessionId},
        shares::ring_impl::RingElement,
    };
    use aes_prng::AesRng;
    use core::num;
    use num_traits::identities;
    use rand::{Rng, RngCore, SeedableRng};
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn test_async_prf_setup() {
        let num_parties = 3;
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let local = LocalRuntime::new(identities.clone(), seeds.clone());
        let mut ready_sessions = local.create_player_sessions(SessionId(0)).await.unwrap();

        // check whether parties have sent/received the correct seeds.
        // P0: [seed_0, seed_2]
        // P1: [seed_1, seed_0]
        // P2: [seed_2, seed_1]
        // This is done by calling next() on the PRFs and see whether they match with
        // the ones created from scratch.
        let mut prf0 = ready_sessions
            .get_mut(&"alice".into())
            .unwrap()
            .prf_as_mut();
        assert_eq!(
            prf0.get_my_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf0.get_prev_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_prev_prf().next_u64()
        );

        let mut prf1 = ready_sessions.get_mut(&"bob".into()).unwrap().prf_as_mut();
        assert_eq!(
            prf1.get_my_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf1.get_prev_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_prev_prf().next_u64()
        );

        let mut prf2 = ready_sessions
            .get_mut(&"charlie".into())
            .unwrap()
            .prf_as_mut();
        assert_eq!(
            prf2.get_my_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf2.get_prev_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_prev_prf().next_u64()
        );
    }

    struct LocalShares1D {
        p0: Vec<Share<u16>>,
        p1: Vec<Share<u16>>,
        p2: Vec<Share<u16>>,
    }

    fn create_single_sharing<R: RngCore>(
        rng: &mut R,
        input: u16,
    ) -> (Share<u16>, Share<u16>, Share<u16>) {
        let a = RingElement(rng.gen::<u16>());
        let b = RingElement(rng.gen::<u16>());
        let c = RingElement(input) - a - b;

        let share1 = Share::new(a, c);
        let share2 = Share::new(b, a);
        let share3 = Share::new(c, b);
        (share1, share2, share3)
    }

    fn create_array_sharing<R: RngCore>(rng: &mut R, input: &Vec<u16>) -> LocalShares1D {
        let mut player0 = Vec::new();
        let mut player1 = Vec::new();
        let mut player2 = Vec::new();

        for entry in input {
            let (a, b, c) = create_single_sharing(rng, *entry);
            player0.push(a);
            player1.push(b);
            player2.push(c);
        }
        LocalShares1D {
            p0: player0,
            p1: player1,
            p2: player2,
        }
    }

    #[tokio::test]
    async fn test_replicated_dot() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let v1 = vec![1, 2, 3, 4, 5_u16];
        let v2 = vec![1, 2, 3, 4, 5_u16];

        let v1_shares = create_array_sharing(&mut rng, &v1);
        let v2_shares = create_array_sharing(&mut rng, &v2);

        let num_parties = 3;
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];

        let v1_share_map = HashMap::from([
            (identities[0].clone(), v1_shares.p0),
            (identities[1].clone(), v1_shares.p1),
            (identities[2].clone(), v1_shares.p2),
        ]);
        let v2_share_map = HashMap::from([
            (identities[0].clone(), v2_shares.p0),
            (identities[1].clone(), v2_shares.p1),
            (identities[2].clone(), v2_shares.p2),
        ]);

        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let local = LocalRuntime::new(identities.clone(), seeds.clone());
        let mut ready_sessions = local.create_player_sessions(SessionId(0)).await.unwrap();

        let mut jobs = JoinSet::new();
        for player in identities.iter().cloned() {
            let mut player_session = ready_sessions.get(&player).unwrap().clone();
            let a = v1_share_map.get(&player).unwrap().clone();
            let b = v2_share_map.get(&player).unwrap().clone();
            jobs.spawn(async move {
                let out_shared = replicated_dot(&mut player_session, &a, &b).await.unwrap();
                let out = open_single(&player_session, out_shared).await.unwrap();
                out
            });
        }
        // check first party output is equal to the expected result.
        let t = jobs.join_next().await.unwrap();
        assert_eq!(t.unwrap(), RingElement(55));
    }
}
