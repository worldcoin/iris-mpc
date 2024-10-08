use super::binary::single_extract_msb_u32;
use crate::{
    database_generators::{GaloisRingSharedIris, NgSharedIris},
    execution::{
        player::{Identity, Role, RoleAssignment},
        session::{BootSession, Session, SessionHandles, SessionId},
    },
    next_gen_network::{
        local::LocalNetworkingStore,
        value::NetworkValue::{self},
    },
    next_gen_protocol::binary::{lift, mul_lift_2k, open_bin},
    protocol::{
        iris_worker::{A, A_BITS, B_BITS},
        prf::{Prf, PrfSeed},
    },
    shares::{bit::Bit, ring_impl::RingElement, share::Share, vecshare::VecShare},
};
use eyre::eyre;
use std::{collections::HashMap, sync::Arc};
use tokio::task::JoinSet;

#[derive(Debug, Clone)]
pub struct LocalRuntime {
    pub identities:       Vec<Identity>,
    pub role_assignments: RoleAssignment,
    pub prf_setups:       Option<HashMap<Role, Prf>>,
    pub seeds:            Vec<PrfSeed>,
}

pub async fn setup_replicated_prf(session: &BootSession, my_seed: PrfSeed) -> eyre::Result<Prf> {
    let next_role = session.own_role()?.next(3);
    let prev_role = session.own_role()?.prev(3);
    let network = session.network();
    // send my_seed to the next party
    network
        .send(
            NetworkValue::PrfKey(my_seed).to_network(),
            session.identity(&next_role)?,
            &session.session_id,
        )
        .await?;
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

/// Computes the dot product between the iris pairs; for both the code and the
/// mask of the irises. We batch the dot products into a single communication
/// round;
pub(crate) async fn ng_replicated_pairwise_distance(
    session: &mut Session,
    pairs: &[(NgSharedIris, NgSharedIris)],
) -> eyre::Result<Vec<Share<u16>>> {
    let mut exchanged_shares_a = Vec::with_capacity(2 * pairs.len());
    for pair in pairs.iter() {
        let (x, y) = pair;
        let (res_code, res_mask) = {
            let mut rand_code = session.prf_as_mut().gen_zero_share();
            for (a__, b__) in x.code.iter().zip(y.code.iter()) {
                rand_code += a__ * b__;
            }
            let mut rand_mask = session.prf_as_mut().gen_zero_share();
            for (a__, b__) in x.mask.iter().zip(y.mask.iter()) {
                rand_mask += a__ * b__;
            }
            (rand_code, rand_mask)
        };
        exchanged_shares_a.push(res_code);
        exchanged_shares_a.push(res_mask);
    }
    let network = session.network();
    let next_role = session.identity(&session.own_role()?.next(3))?;
    let prev_role = session.identity(&session.own_role()?.prev(3))?;

    network
        .send(
            NetworkValue::VecRing16(exchanged_shares_a.clone()).to_network(),
            next_role,
            &session.session_id(),
        )
        .await?;

    let serialized_reply = network.receive(prev_role, &session.session_id()).await;
    let res_b = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::VecRing16(element)) => element,
        _ => return Err(eyre!("Could not deserialize VecRing16")),
    };
    if exchanged_shares_a.len() != res_b.len() {
        Err(eyre!(
            "Expected a VecRing16 with length {:?} but received with length: {:?}",
            exchanged_shares_a.len(),
            res_b.len()
        ))
    } else {
        let mut res = Vec::with_capacity(2 * pairs.len());
        for (a_share, b_share) in exchanged_shares_a.into_iter().zip(res_b) {
            res.push(Share::new(a_share, b_share));
        }
        Ok(res)
    }
}

/// Computes the dot product between the iris pairs; for both the code and the
/// mask of the irises. We batch the dot products into a single communication
/// round;
pub(crate) async fn gr_replicated_pairwise_distance(
    _session: &mut Session,
    pairs: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
) -> eyre::Result<Vec<RingElement<u16>>> {
    let mut additive_shares = Vec::with_capacity(2 * pairs.len());
    for pair in pairs.iter() {
        let (x, y) = pair;
        let code_dot = x.code.trick_dot(&y.code);
        let mask_dot = x.mask.trick_dot(&y.mask);
        additive_shares.push(RingElement(code_dot));
        additive_shares.push(RingElement(mask_dot));
    }
    Ok(additive_shares)
}

pub async fn ng_compare_threshold_masked(
    session: &mut Session,
    code_dot: Share<u16>,
    mask_dot: Share<u16>,
) -> eyre::Result<Share<Bit>> {
    debug_assert!(A_BITS as u64 <= B_BITS);

    let y = mul_lift_2k::<B_BITS>(&code_dot);
    let mut x = lift::<{ B_BITS as usize }>(session, VecShare::new_vec(vec![mask_dot])).await?;
    debug_assert_eq!(x.len(), 1);
    let mut x = x.pop().expect("Enough elements present");
    x *= A as u32;
    x -= y;

    single_extract_msb_u32::<32>(session, x).await
}

pub(crate) async fn batch_signed_lift(
    session: &mut Session,
    mut pre_lift: VecShare<u16>,
) -> eyre::Result<VecShare<u32>> {
    // Compute (v + 2^{15}) % 2^{16}, to make values positive.
    for v in pre_lift.iter_mut() {
        v.add_assign_const_role(1_u16 << 15, session.own_role()?);
    }
    let mut lifted_values = lift::<16>(session, pre_lift).await?;
    // Now we got shares of d1' over 2^32 such that d1' = (d1'_1 + d1'_2 + d1'_3) %
    // 2^{16} = d1 Next we subtract the 2^15 term we've added previously to
    // get signed shares over 2^{32}
    for v in lifted_values.iter_mut() {
        v.add_assign_const_role(((1_u64 << 32) - (1_u64 << 15)) as u32, session.own_role()?);
    }
    Ok(lifted_values)
}

/// Computes [D1 * T2; D2 * T1] via lifting
pub(crate) async fn ng_cross_mul_via_lift(
    session: &mut Session,
    d1: Share<u16>,
    t1: Share<u16>,
    d2: Share<u16>,
    t2: Share<u16>,
) -> eyre::Result<(Share<u32>, Share<u32>)> {
    let mut pre_lift = VecShare::<u16>::with_capacity(4);
    // Do preprocessing to lift all values
    pre_lift.push(d1);
    pre_lift.push(t2);
    pre_lift.push(d2);
    pre_lift.push(t1);

    let lifted_values = batch_signed_lift(session, pre_lift).await?;

    // Compute d1 * t2; t2 * d1
    let mut exchanged_shares_a = Vec::with_capacity(2);
    let pairs = [
        (
            lifted_values.shares[0].clone(),
            lifted_values.shares[1].clone(),
        ),
        (
            lifted_values.shares[2].clone(),
            lifted_values.shares[3].clone(),
        ),
    ];
    for pair in pairs.iter() {
        let (x, y) = pair;
        let res = session.prf_as_mut().gen_zero_share() + x * y;
        exchanged_shares_a.push(res);
    }

    let network = session.network();
    let next_role = session.identity(&session.own_role()?.next(3))?;
    let prev_role = session.identity(&session.own_role()?.prev(3))?;

    network
        .send(
            NetworkValue::VecRing32(exchanged_shares_a.clone()).to_network(),
            next_role,
            &session.session_id(),
        )
        .await?;

    let serialized_reply = network.receive(prev_role, &session.session_id()).await;
    let res_b = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::VecRing32(element)) => element,
        _ => return Err(eyre!("Could not deserialize VecRing16")),
    };
    if exchanged_shares_a.len() != res_b.len() {
        return Err(eyre!(
            "Expected a VecRing32 with length {:?} but received with length: {:?}",
            exchanged_shares_a.len(),
            res_b.len()
        ));
    }

    // vec![D1 * T2; T2 * D1]
    let mut res = Vec::with_capacity(2);
    for (a_share, b_share) in exchanged_shares_a.into_iter().zip(res_b) {
        res.push(Share::new(a_share, b_share));
    }

    Ok((res[0].clone(), res[1].clone()))
}

/// Computes (d2*t1 - d1*t2) > 0 by first lifting the values in a batch
/// from Z_{2^16} to a bigger ring Z_{2^32}
/// Does the multiplication in Z_{2^32} and computes the MSB, to check the
/// comparison result.
/// d1, t1 are replicated shares that come from an iris code/mask dot product,
/// ie: d1 = dot(c_x, c_y); t1 = dot(m_x, m_y). d2, t2 are replicated shares
/// that come from an iris code and mask dot product, ie:
/// d2 = dot(c_u, c_w), t2 = dot(m_u, m_w)
pub async fn ng_cross_compare(
    session: &mut Session,
    d1: Share<u16>,
    t1: Share<u16>,
    d2: Share<u16>,
    t2: Share<u16>,
) -> eyre::Result<bool> {
    let (d1t2, d2t1) = ng_cross_mul_via_lift(session, d1, t1, d2, t2).await?;
    let diff = d2t1 - d1t2;
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
    network
        .send(
            NetworkValue::VecRing64(message).to_network(),
            &next_party,
            &sid,
        )
        .await?;

    // receiving from previous party
    let network = session.network().clone();
    let sid = session.session_id();
    let prev_party = session.prev_identity()?;
    let shares_c = {
        let serialized_other_share = network.receive(&prev_party, &sid).await;
        match NetworkValue::from_network(serialized_other_share) {
            Ok(NetworkValue::VecRing64(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in open_t_many operation")),
        }
    }?;

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

pub async fn ng_replicated_is_match(
    session: &mut Session,
    pairs: &[(NgSharedIris, NgSharedIris)],
) -> eyre::Result<bool> {
    let dots = ng_replicated_pairwise_distance(session, pairs).await?;
    // compute dots[0] - dots[1]
    let bit = ng_compare_threshold_masked(session, dots[0].clone(), dots[1].clone()).await?;
    let opened = open_bin(session, bit).await?;
    Ok(opened.convert())
}

pub async fn gr_to_rep3(
    session: &Session,
    items: Vec<RingElement<u16>>,
) -> eyre::Result<Vec<Share<u16>>> {
    let network = session.network().clone();
    let sid = session.session_id();
    let next_party = session.prev_identity()?;

    // sending to the next party
    let shares_a = items.clone();
    network
        .send(
            NetworkValue::VecRing16(items).to_network(),
            &next_party,
            &sid,
        )
        .await?;

    // receiving from previous party
    let network = session.network().clone();
    let sid = session.session_id();
    let prev_party = session.prev_identity()?;
    let shares_b = {
        let serialized_other_share = network.receive(&prev_party, &sid).await;
        match NetworkValue::from_network(serialized_other_share) {
            Ok(NetworkValue::VecRing16(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in gr_to_rep3 operation")),
        }
    }?;
    let res: Vec<Share<u16>> = shares_a
        .into_iter()
        .zip(shares_b)
        .map(|(a, b)| Share::new(a, b))
        .collect();
    Ok(res)
}

pub async fn gr_replicated_is_match(
    session: &mut Session,
    pairs: &[(GaloisRingSharedIris, GaloisRingSharedIris)],
) -> eyre::Result<bool> {
    let additive_dots = gr_replicated_pairwise_distance(session, pairs).await?;
    let rep_dots = gr_to_rep3(session, additive_dots).await?;
    // compute dots[0] - dots[1]
    let bit =
        ng_compare_threshold_masked(session, rep_dots[0].clone(), rep_dots[1].clone()).await?;
    let opened = open_bin(session, bit).await?;
    Ok(opened.convert())
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
        LocalRuntime {
            identities,
            role_assignments,
            prf_setups: None,
            seeds,
        }
    }

    pub async fn create_player_sessions(&self) -> eyre::Result<HashMap<Identity, Session>> {
        let network = LocalNetworkingStore::from_host_ids(&self.identities);
        let sess_id = SessionId::from(0_u128);
        let boot_sessions: Vec<BootSession> = (0..self.seeds.len())
            .map(|i| {
                let identity = self.identities[i].clone();
                BootSession {
                    session_id:       sess_id,
                    role_assignments: Arc::new(self.role_assignments.clone()),
                    networking:       Arc::new(network.get_local_network(identity.clone())),
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
    use crate::{
        next_gen_protocol::ng_worker::NetworkValue::RingElement32, shares::ring_impl::RingElement,
    };
    use aes_prng::AesRng;
    use rand::{Rng, RngCore, SeedableRng};

    async fn open_single(session: &Session, x: Share<u32>) -> eyre::Result<RingElement<u32>> {
        let network = session.network();
        let next_role = session.identity(&session.own_role()?.next(3))?;
        let prev_role = session.identity(&session.own_role()?.prev(3))?;
        network
            .send(
                RingElement32(x.b).to_network(),
                next_role,
                &session.session_id(),
            )
            .await?;
        let serialized_reply = network.receive(prev_role, &session.session_id()).await;
        let missing_share = match NetworkValue::from_network(serialized_reply) {
            Ok(NetworkValue::RingElement32(element)) => element,
            _ => return Err(eyre!("Could not deserialize RingElement32")),
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
        let mut ready_sessions = local.create_player_sessions().await.unwrap();

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
    struct LocalShares1D {
        p0: Vec<Share<u16>>,
        p1: Vec<Share<u16>>,
        p2: Vec<Share<u16>>,
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
    async fn test_replicated_cross_mul_lift() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let four_items = vec![1, 2, 3, 4];

        let four_shares = create_array_sharing(&mut rng, &four_items);

        let num_parties = 3;
        let identities: Vec<Identity> = vec!["alice".into(), "bob".into(), "charlie".into()];

        let four_share_map = HashMap::from([
            (identities[0].clone(), four_shares.p0),
            (identities[1].clone(), four_shares.p1),
            (identities[2].clone(), four_shares.p2),
        ]);

        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let local = LocalRuntime::new(identities.clone(), seeds.clone());
        let ready_sessions = local.create_player_sessions().await.unwrap();

        let mut jobs = JoinSet::new();
        for player in identities.iter() {
            let mut player_session = ready_sessions.get(player).unwrap().clone();
            let four_shares = four_share_map.get(player).unwrap().clone();
            jobs.spawn(async move {
                let out_shared = ng_cross_mul_via_lift(
                    &mut player_session,
                    four_shares[0].clone(),
                    four_shares[1].clone(),
                    four_shares[2].clone(),
                    four_shares[3].clone(),
                )
                .await
                .unwrap();
                (
                    open_single(&player_session, out_shared.0).await.unwrap(),
                    open_single(&player_session, out_shared.1).await.unwrap(),
                )
            });
        }
        // check first party output is equal to the expected result.
        let t = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(t.0, RingElement(4));
        assert_eq!(t.1, RingElement(6));
    }
}
