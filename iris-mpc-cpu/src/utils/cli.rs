use std::path::PathBuf;

use eyre::{bail, eyre, OptionExt, Result};
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::Deserialize;

use crate::{
    hnsw::{
        searcher::{LayerDistribution, LayerMode, N_PARAM_LAYERS},
        GraphMem, HnswParams, HnswSearcher,
    },
    utils::serialization::{
        graph::GraphFormat,
        iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
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
        IrisesConfig::Random { number, seed } => {
            println!("Generating {} random iris codes...", number);
            let mut rng: Box<dyn RngCore> = if let Some(seed) = seed {
                Box::new(StdRng::seed_from_u64(seed))
            } else {
                Box::new(rand::thread_rng())
            };

            (0..number)
                .map(|_| IrisCode::random_rng(&mut rng))
                .collect::<Vec<_>>()
        }
        IrisesConfig::NdjsonFile {
            path,
            limit,
            selection,
        } => {
            println!("Loading irises from NDJSON file: {}", path.display());
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
    pub fn read_graph_from_file(&self) -> Result<GraphMem<IrisVectorId>> {
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
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
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
