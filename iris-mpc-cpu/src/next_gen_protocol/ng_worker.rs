use super::binary::{extract_msb_u16, single_extract_msb_u32};
use crate::{
    execution::{
        player::{Identity, Role, RoleAssignment},
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    next_gen_network::{
        local::LocalNetworkingStore,
        value::NetworkValue::{self, RingElement16},
    },
    next_gen_protocol::binary::{lift, open_bin},
    protocol::prf::{Prf, PrfSeed},
    shares::{
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
};
use eyre::eyre;
use iris_mpc_common::iris_db::iris::{IrisCodeArray, MATCH_THRESHOLD_RATIO};
use std::{collections::HashMap, ops::SubAssign, sync::Arc};
use tokio::task::JoinSet;

#[derive(Debug, Clone)]
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

// pub(crate) async fn setup_session<Rng: RngCore>(session: BootSession, rng:
// &mut Rng) -> Session {     let mut seed = [0_u8; 16];
//     rng.fill_bytes(&mut seed);
//     let prf = setup_replicated_prf(&session, seed).await.unwrap();
//     Session {
//         boot_session: session,
//         setup:        prf,
//     }
// }

pub(crate) async fn replicated_dot(
    session: &mut Session,
    a: &[Share<u16>],
    b: &[Share<u16>],
) -> eyre::Result<Share<u16>> {
    if a.len() != b.len() {
        return Err(eyre!("Error::InvalidSize"));
    }
    let res_a = {
        let mut rand = session.prf_as_mut().gen_zero_share();
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

pub async fn replicated_dot_many(
    session: &mut Session,
    a: SliceShare<'_, u16>,
    b: &[VecShare<u16>],
) -> eyre::Result<VecShare<u16>> {
    let len = b.len();
    if a.len() != IrisCodeArray::IRIS_CODE_SIZE {
        return Err(eyre!("Error::InvalidSize"));
    }

    let mut shares_a = Vec::with_capacity(len);

    for b_ in b.iter() {
        let mut rand = session.prf_as_mut().gen_zero_share();
        if a.len() != b_.len() {
            return Err(eyre!("Error::InvalidSize"));
        }
        for (a__, b__) in a.iter().zip(b_.iter()) {
            rand += a__ * b__;
        }
        shares_a.push(rand);
    }

    let network = session.network();
    let next_role = session.identity(&session.own_role()?.next(3))?;
    let prev_role = session.identity(&session.own_role()?.prev(3))?;

    let _ = network
        .send(
            NetworkValue::VecRing16(shares_a.clone()).to_network(),
            next_role,
            &session.session_id(),
        )
        .await;

    let serialized_reply = network.receive(prev_role, &session.session_id()).await;
    let shares_b = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::VecRing16(element)) => element,
        _ => return Err(eyre!("Could not deserialize Vec<RingElement16>")),
    };

    // Network: reshare
    let res = VecShare::from_ab(shares_a, shares_b);
    Ok(res)
}

pub(crate) async fn replicated_pairwise_distance(
    session: &mut Session,
    lhs_shares: &[Share<u16>],
    rhs_shares: &[Share<u16>],
    lhs_mask: &IrisCodeArray,
    rhs_mask: &IrisCodeArray,
) -> eyre::Result<(Share<u16>, usize)> {
    let combined_mask = *lhs_mask & *rhs_mask;
    let mask_dots = combined_mask.count_ones();
    let code_dots = replicated_dot(session, lhs_shares, rhs_shares).await?;
    Ok((code_dots, mask_dots))
}

// TODO(Dragos) revisit this as we can probably do 2 lifts at a time.
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

pub async fn replicated_lift_and_cross_mul(
    session: &mut Session,
    d1: Share<u16>,
    t1: u32,
    d2: Share<u16>,
    t2: u32,
) -> eyre::Result<bool> {
    let diff = replicated_cross_mul(session, d1, t1, d2, t2).await?;
    // Compute bit <- MSB(D2 * T1 - D1 * T2)
    let bit = single_extract_msb_u32::<32>(session, diff).await?;
    // Open bit
    let opened_b = open_bin(session, bit).await?;
    Ok(opened_b.convert())
}

pub async fn open_t_many(session: &Session, shares: VecShare<u64>) -> eyre::Result<Vec<u64>> {
    let next_party = session.next_identity()?;
    let network = session.network().clone();
    let sid = session.session_id();

    let shares_b: Vec<_> = shares.iter().map(|s| s.b).collect();
    let message = shares_b;
    let _ = tokio::spawn(async move {
        let _ = network
            .send(
                NetworkValue::VecRing64(message).to_network(),
                &next_party,
                &sid,
            )
            .await;
    })
    .await;

    // receiving from previous party
    let network = session.network().clone();
    let sid = session.session_id();
    let prev_party = session.prev_identity()?;
    let shares_c = tokio::spawn(async move {
        let serialized_other_share = network.receive(&prev_party, &sid).await;
        match NetworkValue::from_network(serialized_other_share) {
            Ok(NetworkValue::VecRing64(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in and_many operation")),
        }
    })
    .await??;

    let res = shares
        .into_iter()
        .zip(shares_c)
        .map(|(s, c)| {
            let (a, b) = s.get_ab();
            (a + b + c).convert()
        })
        .collect();
    Ok(res)
}

pub async fn rep3_single_iris_match_public_output(
    session: &mut Session,
    iris_to_match: SliceShare<'_, u16>,
    ground_truth: VecShare<u16>,
    mask_iris: &IrisCodeArray,
    mask_ground_truth: IrisCodeArray,
) -> eyre::Result<bool> {
    let res = replicated_compare_iris_public_mask_many(
        session,
        iris_to_match,
        &[ground_truth],
        mask_iris,
        &[mask_ground_truth],
    )
    .await?;
    let bit = open_t_many(session, res).await?;
    Ok(bit[0] != 0)
}

pub(crate) fn rep3_get_cmp_diff(
    session: &Session,
    dot: &mut Share<u16>,
    mask_ones: usize,
) -> eyre::Result<()> {
    let threshold: u16 = ((mask_ones as f64 * (1. - 2. * MATCH_THRESHOLD_RATIO)) as usize)
        .try_into()
        .expect("Sizes are checked in constructor");
    *dot = dot.sub_from_const_role(threshold, session.own_role()?);
    Ok(())
}

pub async fn replicated_compare_iris_public_mask_many(
    session: &mut Session,
    a: SliceShare<'_, u16>,
    b: &[VecShare<u16>],
    mask_a: &IrisCodeArray,
    mask_b: &[IrisCodeArray],
) -> eyre::Result<VecShare<u64>> {
    let amount = b.len();
    if (amount != mask_b.len()) || (amount == 0) {
        return Err(eyre!("InvalidSize"));
    }

    let masks = mask_b.iter().map(|b| *mask_a & *b).collect::<Vec<_>>();
    let mask_lens: Vec<_> = masks.iter().map(|m| m.count_ones()).collect();
    let mut dots = replicated_dot_many(session, a, b).await?;

    // a < b <=> msb(a - b)
    // Given no overflow, which is enforced in constructor
    for (dot, mask_len) in dots.iter_mut().zip(mask_lens) {
        rep3_get_cmp_diff(session, dot, mask_len)?;
    }

    extract_msb_u16::<{ u16::BITS as usize }>(session, dots).await
}

impl LocalRuntime {
    pub fn replicated_test_config() -> Self {
        let num_parties = 3;
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        LocalRuntime::new(identities, seeds)
    }
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

    pub async fn create_player_sessions(
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
            let player_seed = self.seeds[player_id];
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
    use crate::{execution::session::SessionId, shares::ring_impl::RingElement};
    use aes_prng::AesRng;
    use rand::{Rng, RngCore, SeedableRng};
    use tokio::task::JoinSet;

    async fn open_single(session: &Session, x: Share<u16>) -> eyre::Result<RingElement<u16>> {
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
        let prf0 = ready_sessions
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

        let prf1 = ready_sessions.get_mut(&"bob".into()).unwrap().prf_as_mut();
        assert_eq!(
            prf1.get_my_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf1.get_prev_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_prev_prf().next_u64()
        );

        let prf2 = ready_sessions
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
        let ready_sessions = local.create_player_sessions(SessionId(0)).await.unwrap();

        let mut jobs = JoinSet::new();
        for player in identities.iter() {
            let mut player_session = ready_sessions.get(player).unwrap().clone();
            let a = v1_share_map.get(player).unwrap().clone();
            let b = v2_share_map.get(player).unwrap().clone();
            jobs.spawn(async move {
                let out_shared = replicated_dot(&mut player_session, &a, &b).await.unwrap();
                open_single(&player_session, out_shared).await.unwrap()
            });
        }
        // check first party output is equal to the expected result.
        let t = jobs.join_next().await.unwrap();
        assert_eq!(t.unwrap(), RingElement(55));
    }
}
