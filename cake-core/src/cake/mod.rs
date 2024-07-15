use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use anyhow::Result;
use async_trait::async_trait;
use candle_core as candle;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};

use crate::{
    model::Cache,
    utils, Args,
};

mod client;
mod master;
mod proto;
mod topology;
mod worker;

pub use client::*;
pub use master::*;
pub use proto::*;
pub use topology::*;
pub use worker::*;

pub type Config = candle_transformers::models::llama::LlamaConfig;

/// Determines if we run in master or worker mode.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Mode {
    #[default]
    Master,
    Worker,
}

/// Main contect object used as a shared state.
#[derive(Clone)]
pub struct Context {
    pub args: Args,
    pub dtype: DType,
    pub topology: Topology,
    pub data_path: PathBuf,
    pub device: Device,
    pub config: Config,
    pub cache: Cache,
    pub var_builder: VarBuilder<'static>,
}

impl Context {
    /// Create the context from the parsed command line arguments.
    pub fn from_args(args: Args) -> Result<Self> {
        let dtype: DType = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        log::info!("loading topology from {}", &args.topology);

        let data_path = PathBuf::from(&args.model);

        let config_filename = data_path.join("config.json");
        // let config = LlamaConfig::from_path(&config_filename)?.into_config();

        let topology = Topology::from_path(&args.topology)?;

        // let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
        // let var_builder =
        //     utils::load_var_builder_from_index(model_tensors_index, dtype, device.clone())?;

        let api = Api::new()?;
        let model_id = "meta-llama/Meta-Llama-3-8B".to_string();
        println!("loading the model weights from {model_id}");
        let revision = "main".to_string();
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let config_filename = api.get("config.json")?;
        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let filenames = hub_load_safetensors(&api, "model.safetensors.index.json")?;

        let cache = Cache::new(true, dtype, &config, &device)?;
        println!("building the model");
        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

        Ok(Context {
            args,
            dtype,
            topology,
            data_path,
            device,
            config,
            cache,
            var_builder,
        })
    }
}

/// This is the trait that a shardable object must implement.
#[async_trait]
pub trait Forwarder: Debug + Send + Sync + Display {
    /// Create an instance of this object loading the specified layer(s) from a VarBuilder.
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>>
    where
        Self: Sized;

    /// Applies a forward operation to the input tensor, does not require mutability.
    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// Applies a forward operation to the input tensor, requires mutability.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// Applies a batch of forward operations to the input tensor.
    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        _cache: &mut Cache,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    /// Return the layer name.
    fn layer_name(&self) -> &str;

    /// Return the unique identity or local.
    fn ident(&self) -> &str {
        "local"
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(safetensors_files)
}
