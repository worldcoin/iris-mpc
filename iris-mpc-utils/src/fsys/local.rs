use std::{
    io::Error,
    path::{Path, PathBuf},
};

use rand::{CryptoRng, Rng};

use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

use super::reader;
use crate::{
    constants::{N_PARTIES, PARTY_INDICES},
    irises,
    types::{NetConfig, NodeExecutionHost, PartyIdx},
};

/// Returns path to an asset within the crate assets sub-directory.
fn get_path_to_assets() -> PathBuf {
    get_path_to_subdir("assets")
}

/// Returns path to a node config file.
pub fn get_path_to_node_config(
    config_kind: &str,
    config_idx: usize,
    party_idx: &PartyIdx,
) -> PathBuf {
    get_path_to_assets().join(
        format!(
            "node-config/{}/{config_kind}-{config_idx}-node-{party_idx}.toml",
            NodeExecutionHost::assets_subdirectory(),
        )
        .as_str(),
    )
}

/// Returns path to an NDJSON file.
pub fn get_path_to_ndjson() -> PathBuf {
    get_path_to_assets().join("iris-codes-plaintext/20250710-1k.ndjson")
}

/// Returns path to root directory.
pub fn get_path_to_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).into()
}

/// Returns path to a service client simple options file.
pub fn get_path_to_service_client_simple_opts(opts_idx: usize) -> PathBuf {
    get_path_to_assets().join(format!("service-client/simple-{opts_idx}.toml",).as_str())
}

/// Returns path to sub-directory.
pub fn get_path_to_subdir(name: &str) -> PathBuf {
    get_path_to_root().join(name)
}

/// Returns an iterator over a local NDJSON file.
pub fn read_ndjson_file<'a, R: Rng + CryptoRng + 'a>(
    rng: &'a mut R,
) -> Result<impl Iterator<Item = [GaloisRingSharedIris; N_PARTIES]> + 'a, Error> {
    irises::reader::read_iris_shares(get_path_to_ndjson().as_path(), rng)
}

/// Returns a loaded node config file.
pub fn read_node_config(
    party_idx: &PartyIdx,
    config_kind: &str,
    config_idx: usize,
) -> Result<NodeConfig, Error> {
    let path_to_config = get_path_to_node_config(config_kind, config_idx, party_idx);

    reader::read_toml(&path_to_config)
}

/// Returns network wide configuration deserialized from a set of toml files.
pub fn read_node_config_set(config_kind: &str, config_idx: usize) -> Result<NetConfig, Error> {
    Ok(PARTY_INDICES
        .iter()
        .map(|party_idx| read_node_config(party_idx, config_kind, config_idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap())
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{
        get_path_to_assets, get_path_to_ndjson, get_path_to_node_config, get_path_to_root,
        get_path_to_service_client_simple_opts, get_path_to_subdir, read_ndjson_file,
        read_node_config, read_node_config_set,
    };
    use crate::constants::{NODE_CONFIG_KIND, PARTY_INDICES};

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_get_path_to_assets() {
        assert!(get_path_to_assets().exists());
    }

    #[test]
    fn test_path_to_ndjson() {
        assert!(get_path_to_ndjson().exists());
    }

    #[test]
    fn test_get_path_to_node_config() {
        PARTY_INDICES.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                assert!(get_path_to_node_config(kind, 0, party_idx).exists());
            });
        });
    }

    #[test]
    fn test_get_path_to_root() {
        assert!(get_path_to_root().exists());
    }

    #[test]
    fn test_get_path_to_service_client_simple_opts() {
        (1..=5).for_each(move |opts_idx| {
            assert!(get_path_to_service_client_simple_opts(opts_idx).exists());
        });
    }

    #[test]
    fn test_get_path_to_subdir() {
        assert!(get_path_to_subdir("assets").exists());
    }

    #[test]
    fn test_read_ndjson_file() {
        let mut rng = get_rng();
        let iterable = read_ndjson_file(&mut rng).unwrap();
        for _ in iterable {}
    }

    #[test]
    fn test_read_node_config() {
        PARTY_INDICES.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                let _ = read_node_config(party_idx, kind, 0).unwrap();
            });
        });
    }

    #[test]
    fn test_read_node_config_set() {
        NODE_CONFIG_KIND.iter().for_each(|kind| {
            let _ = read_node_config_set(kind, 0).unwrap();
        });
    }
}
