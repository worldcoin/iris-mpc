use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use eyre::{bail, eyre, OptionExt, Result};
use iris_mpc_common::iris_db::iris::IrisCode;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::Deserialize;

use crate::{
    hawkers::plaintext_deep_id_store::Int4Vector,
    hnsw::{
        searcher::{LayerDistribution, LayerMode, N_PARAM_LAYERS},
        GraphMem, HnswParams, HnswSearcher,
    },
    utils::serialization::{
        graph::GraphFormat,
        int4_ndjson::{int4_vectors_from_ndjson_iter, write_int4_vectors_ndjson},
        iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
        types::iris_base64::write_to_iris_ndjson,
    },
};

/********************* Load Irises *******************************/

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "option")]
pub enum IrisesConfig {
    /// Generate random iris codes.
    Random {
        /// Number of iris codes to generate
        number: usize,

        /// Optional deterministic seed to use for generation
        seed: Option<u64>,

        /// Optional path to write the generated iris codes as ndjson
        output_path: Option<PathBuf>,
    },

    /// Load iris codes from an NDJSON file.
    NdjsonFile {
        /// File path
        path: PathBuf,

        /// Optional limit on the number of iris codes to load
        limit: Option<usize>,

        /// Optional choice to read only even or odd codes from file
        selection: Option<IrisSelection>,
    },
}

/// Loads iris codes into a `PlaintextStore` based on `IrisesInit` config.
/// Returns the store and an RNG for use in later steps.
pub async fn load_irises(config: IrisesConfig) -> Result<Vec<IrisCode>> {
    let irises = match config {
        IrisesConfig::Random {
            number,
            seed,
            output_path,
        } => {
            tracing::info!("Generating {} random iris codes...", number);
            let mut rng: Box<dyn RngCore> = if let Some(seed) = seed {
                Box::new(StdRng::seed_from_u64(seed))
            } else {
                Box::new(rand::thread_rng())
            };

            let codes: Vec<_> = (0..number)
                .map(|_| IrisCode::random_rng(&mut rng))
                .collect();

            if let Some(path) = output_path {
                tracing::info!("Writing generated iris codes to {}", path.display());
                let file = File::create(&path)?;
                let mut writer = BufWriter::new(file);
                write_to_iris_ndjson(&mut writer, codes.iter().map(|c| c.into()))?;
                writer.flush()?;
            }

            codes
        }
        IrisesConfig::NdjsonFile {
            path,
            limit,
            selection,
        } => {
            tracing::info!("Loading irises from NDJSON file: {}", path.display());
            irises_from_ndjson_iter(&path, limit, selection.unwrap_or(IrisSelection::All))?
                .collect::<Vec<_>>()
        }
    };

    Ok(irises)
}

/********************* Load Graph ********************************/

#[derive(Clone, Debug, Deserialize)]
pub struct LoadGraphConfig {
    path: PathBuf,
    format: Option<GraphFormat>,
}

impl LoadGraphConfig {
    pub fn read_graph_from_file(&self) -> Result<GraphMem> {
        let format = self.format.unwrap_or(GraphFormat::Current);
        super::serialization::graph::read_graph_from_file(&self.path, format)
    }
}

/********************* Hnsw Searcher Config **********************/

/// Parameters for specifying an HNSW searcher
#[derive(Clone, Debug, Deserialize)]
#[allow(non_snake_case)]
pub struct SearcherConfig {
    pub params: SearcherParams,
    pub layer_mode: LayerMode,
    pub layer_distribution: Option<LayerDistribution>,
    #[serde(default)]
    pub fixed_layer_search_batch_size: Option<usize>,
}

impl TryFrom<&SearcherConfig> for HnswSearcher {
    type Error = eyre::Report;

    fn try_from(value: &SearcherConfig) -> Result<Self> {
        let params: HnswParams = (&value.params).try_into()?;
        let layer_mode = value.layer_mode.clone();
        let layer_distribution = value
            .layer_distribution
            .clone()
            .unwrap_or_else(|| LayerDistribution::new_geometric_from_M(params.get_M(0)));

        Ok(HnswSearcher {
            params,
            layer_mode,
            layer_distribution,
            fixed_layer_search_batch_size: value.fixed_layer_search_batch_size,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "option")]
#[allow(non_snake_case)]
pub enum SearcherParams {
    Standard {
        ef_constr: usize,
        ef_search: usize,
        M: usize,
    },
    Uniform {
        ef: usize,
        M: usize,
    },
    Custom {
        ef_constr_search: LayerValue<usize>,
        ef_constr_insert: LayerValue<usize>,
        ef_search: LayerValue<usize>,
        M: LayerValue<usize>,
        M_max: LayerValue<usize>,
        M_limit: LayerValue<usize>,
    },
}

impl TryFrom<&SearcherParams> for HnswParams {
    type Error = eyre::Report;

    fn try_from(value: &SearcherParams) -> Result<Self> {
        match value {
            SearcherParams::Standard {
                ef_constr,
                ef_search,
                M,
            } => Ok(HnswParams::new(*ef_constr, *ef_search, *M)),
            SearcherParams::Uniform { ef, M } => Ok(HnswParams::new_uniform(*ef, *M)),
            SearcherParams::Custom {
                ef_constr_search,
                ef_constr_insert,
                ef_search,
                M,
                M_max,
                M_limit,
            } => Ok(HnswParams {
                M: M.try_into()?,
                M_max: M_max.try_into()?,
                M_limit: M_limit.try_into()?,
                ef_constr_search: ef_constr_search.try_into()?,
                ef_constr_insert: ef_constr_insert.try_into()?,
                ef_search: ef_search.try_into()?,
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum LayerValue<T> {
    Single(T),
    PerLayer(Vec<T>),
}

impl<T: Copy> TryFrom<&LayerValue<T>> for [T; N_PARAM_LAYERS] {
    type Error = eyre::Report;

    fn try_from(value: &LayerValue<T>) -> Result<Self> {
        match value {
            LayerValue::Single(val) => Ok([*val; N_PARAM_LAYERS]),
            LayerValue::PerLayer(vals) => {
                if vals.len() > N_PARAM_LAYERS {
                    bail!("Too many parameters specified for layer values");
                }
                let last_elt = vals
                    .last()
                    .cloned()
                    .ok_or_eyre("Empty layer values list specified")?;
                let mut vals = vals.clone();
                vals.resize(N_PARAM_LAYERS, last_elt);
                vals.try_into()
                    .map_err(|_| eyre!("Unable to map values into parameters array"))
            }
        }
    }
}

/********************* Load Int4 Deep-ID Vectors *****************/

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "option")]
pub enum Int4VectorsConfig {
    /// Generate random Int4Vectors.
    Random {
        number: usize,
        seed: Option<u64>,
        output_path: Option<PathBuf>,
    },

    /// Load Int4Vectors from an NDJSON file.
    NdjsonFile { path: PathBuf, limit: Option<usize> },
}

/// Loads Int4 deep-ID vectors based on `Int4VectorsConfig`.
pub async fn load_int4_vectors(config: Int4VectorsConfig) -> Result<Vec<Int4Vector>> {
    let vectors = match config {
        Int4VectorsConfig::Random {
            number,
            seed,
            output_path,
        } => {
            tracing::info!("Generating {} random Int4Vectors...", number);
            let mut rng: Box<dyn RngCore> = if let Some(seed) = seed {
                Box::new(StdRng::seed_from_u64(seed))
            } else {
                Box::new(rand::thread_rng())
            };

            let vectors: Vec<Int4Vector> =
                (0..number).map(|_| Int4Vector::random(&mut rng)).collect();

            if let Some(path) = output_path {
                tracing::info!("Writing generated Int4Vectors to {}", path.display());
                write_int4_vectors_ndjson(&path, &vectors)?;
            }

            vectors
        }
        Int4VectorsConfig::NdjsonFile { path, limit } => {
            tracing::info!("Loading Int4Vectors from NDJSON file: {}", path.display());
            int4_vectors_from_ndjson_iter(&path, limit)?.collect::<Result<Vec<_>>>()?
        }
    };

    Ok(vectors)
}

#[cfg(test)]
mod int4_tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn load_random_int4_vectors_with_seed_is_deterministic() {
        let cfg = Int4VectorsConfig::Random {
            number: 4,
            seed: Some(7),
            output_path: None,
        };
        let a = load_int4_vectors(cfg.clone()).await.unwrap();
        let b = load_int4_vectors(cfg).await.unwrap();
        assert_eq!(a.len(), 4);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.packed, y.packed);
        }
    }

    #[tokio::test]
    async fn load_int4_vectors_random_with_output_path_writes_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.ndjson");
        let cfg = Int4VectorsConfig::Random {
            number: 3,
            seed: Some(1),
            output_path: Some(path.clone()),
        };
        let generated = load_int4_vectors(cfg).await.unwrap();
        assert_eq!(generated.len(), 3);

        let loaded = load_int4_vectors(Int4VectorsConfig::NdjsonFile {
            path,
            limit: Some(3),
        })
        .await
        .unwrap();
        for (g, l) in generated.iter().zip(loaded.iter()) {
            assert_eq!(g.packed, l.packed);
        }
    }
}
