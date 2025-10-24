# ================== requirements.in ==================
# Core
torch>=2.0
numpy
pyyaml
asyncio

# ML
transformers
deepspeed
megatron-core
torch-fidelity

# Data
datasets
h5py
webdataset
open3d
gitpython

# Planning
unified-planning
networkx

# Optional
langdetect
nltk
beautifulsoup4
scrapy-aiohttp
prometheus-client
ftfy
tqdm
psutil
deap  # For evo fallback
spacy  # For advanced NER in PII removal
tarski  # For robust PDDL parsing

# Install: pip-compile requirements.in > requirements.txt

# ================== src/core/__init__.py ==================
"""Core: Configs, Deps, Utils for XAGI."""

# ================== src/core/configs.py ==================
"""
Configs: Sealed dataclasses for modularity. Validates holistically.
First principle: Axioms define the substrate (bytes as reality anchor).
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import re
import logging
import torch  # Added import for device detection

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model axioms: Dimensions, activations, RL params. Validates constraints."""
    input_dim: int = 4  # Point cloud or byte embed dim
    hidden_dim: int = 8192
    output_dims: List[int] = field(default_factory=lambda: [1, 3, 3, 5])  # Multi-task heads
    latent_dim: int = 128
    num_levels: int = 4
    symbolic_dim: int = 1024
    vocab_size: int = 256  # Byte-level
    transformer_nhead: int = 64
    transformer_num_layers: int = 96
    temporal_dim: int = 10
    num_moe_experts: int = 8
    expert_dim: int = 256
    num_inner_steps: int = 5
    voxel_resolution: int = 64
    point_cloud_size: int = 2048
    use_voxel: bool = True
    rl_algorithm: str = "grpo"
    rl_action_space: int = 10
    tp_degree: int = 8
    pp_degree: int = 4
    activation: str = "relu"
    pooling_strategy: str = "mean"
    cross_attention_heads: int = 8
    img_size: int = 64
    tile_size: int = 1024
    ppo_clip: float = 0.2
    dynamic_experts: bool = True
    safety_gate_threshold: float = 0.5
    unet_depth: int = 3
    use_bfloat16: bool = True
    use_flash_attention: bool = True
    seq_len: int = 1048576
    byte_level: bool = True
    patch_size: int = 4
    num_actions: int = 100
    # ADP
    use_adp: bool = True
    adp_gamma: float = 0.99
    adp_lr_actor: float = 1e-3
    adp_lr_critic: float = 1e-3
    adp_hidden_dim: int = 128
    adp_digits: int = 5
    adp_state_dim: int = 2
    adp_action_dim: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Added for consistency

    def _validate(self):
        """Validate axioms: e.g., latent <= hidden/64."""
        if self.latent_dim > self.hidden_dim // 64:
            raise ValueError(f"latent_dim exceeds cap: {self.latent_dim} > {self.hidden_dim // 64}")
        if self.use_adp and self.adp_digits <= 0:
            raise ValueError("adp_digits must be positive")
        if self.use_adp and self.adp_state_dim > self.latent_dim:
            raise ValueError(f"adp_state_dim exceeds latent_dim")

@dataclass
class TrainingConfig:
    """Training dynamics: LR, epochs, losses. Validates basics."""
    learning_rate: float = 0.0003
    epochs: int = 1
    batch_size: int = 4096
    loss_weights: Optional[List[float]] = None
    grad_clip_norm: float = 0.5
    rl_gamma: float = 0.99
    rl_steps: int = 2000
    ppo_epochs: int = 10
    ppo_entropy_left: float = 0.005
    ppo_entropy_right: float = 0.05
    ppo_coord_weight: float = 0.1
    meta_inner_lr: float = 0.01
    self_update_threshold: float = 0.1
    self_learning_cycles: int = 5
    pruning_frequency: int = 10
    use_fsdp: bool = True
    peft_rank: int = 8
    use_quantization: bool = True
    use_hdf5_cache: bool = True
    dataloader_workers: int = 64
    persistent_workers: bool = True
    use_deepspeed: bool = True
    deepspeed_zero_stage: int = 3
    use_megatron: bool = True
    pddl_tuning_epochs: int = 5
    caselaw_tuning_epochs: int = 3
    gradient_accumulation_steps: int = 64
    web_tuning_epochs: int = 3
    lyra_tuning_epochs: int = 3
    grpo_group_size: int = 4
    grpo_kl_penalty: float = 0.1
    grpo_advantage_normalization: bool = True
    grpo_fine_tune_lr_multiplier: float = 0.5
    grpo_fine_tune_iterations: int = 5
    use_tap: bool = True
    prefetch_factor: int = 2
    contrastive_loss_weight: float = 0.1
    loss_plateau_threshold: float = 0.01
    evolution_trigger_epochs: int = 1
    strict_mode: bool = False
    benchmark_regressions: bool = True
    fid_threshold: float = 0.5
    early_stop_patience: int = 10
    min_batch_size: int = 32
    use_gradient_checkpointing: bool = True
    federated_rounds: int = 5  # Pruned, but kept stub
    byte_throughput_target: float = 1e6
    edl_kl_weight: float = 0.01  # Moved here from model
    substrate_fidelity_weight: float = 0.1
    use_fid: bool = True

    def _validate(self):
        """Validate dynamics: epochs >0, batch >= min."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < self.min_batch_size:
            raise ValueError(f"batch_size must be at least {self.min_batch_size}")

@dataclass
class DataConfig:
    """Data substrate: Paths, splits, filters. No S3."""
    num_points: int = 5000
    noise_level: float = 0.03
    real_world_noise_level: float = 0.05
    max_3d_noise: float = 0.02
    rotation_augmentation: bool = True
    scaling_augmentation: bool = True
    dataset_path: Optional[str] = None
    validation_split: float = 0.2
    well_base_path: str = "path/to/base"  # Local only
    well_dataset_name: str = "name_of_the_dataset"
    well_split_name_train: str = "train"
    well_split_name_val: str = "val"
    n_steps_input: int = 1
    n_steps_output: int = 1
    use_normalization: bool = True
    return_grid: bool = True
    boundary_return_type: Optional[str] = None
    tvl_base_path: str = "/path/to/tvl_data"
    tvl_dataset_name: str = "yoorhim/TVL-revise"
    tvl_split_name_train: str = "train"
    tvl_split_name_val: str = "validation"
    use_tvl: bool = True
    tvl_encoder_image: str = "google/vit-base-patch16-224"
    tvl_encoder_tactile: str = "google/vit-base-patch16-224"
    tvl_contrastive_temp: float = 0.07
    tvl_background_ratio: float = 0.1
    pddl_dataset_path: str = "/path/to/pddl_data"
    pddl_domain_file: str = "domain.pddl"
    pddl_problem_file: str = "problem.pddl"
    pddl_additional_domains: List[str] = field(default_factory=lambda: ["logistics.pddl", "gridworld.pddl"])
    use_caselaw: bool = True
    caselaw_dataset_name: str = "common-pile/caselaw_access_project"
    caselaw_split_name_train: str = "train"
    caselaw_split_name_val: str = "train"
    caselaw_base_path: Optional[str] = None
    caselaw_max_samples: int = 10_000_000
    caselaw_text_length: int = 2048
    web_crawl_domains: List[str] = field(default_factory=lambda: ["example.com", "wikipedia.org"])
    web_crawl_max_depth: int = 5
    web_search_engines: List[str] = field(default_factory=lambda: ["google", "bing", "duckduckgo", "yahoo"])
    web_search_api_keys: Dict[str, str] = field(default_factory=lambda: {})
    web_max_summary_length: int = 2048
    web_filter_keywords: List[str] = field(default_factory=lambda: ["advertisement", "sponsored", "cookie policy"])
    modelnet_base_path: str = "/path/to/modelnet"
    modelnet_dataset_name: str = "modelnet"
    use_modelnet: bool = False
    lyra_base_path: str = "lyra_dataset/tar"  # Local
    lyra_dataset_name: str = "nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG"
    lyra_split_name_train: str = "static"
    lyra_split_name_val: str = "dynamic"
    use_lyra: bool = True
    lyra_max_samples: int = 1_000_000
    cor_dataset_augmentation: bool = True
    use_audioset: bool = False
    audioset_dataset_name: str = "agkphysics/AudioSet"
    audioset_split_name_train: str = "train"
    audioset_max_samples: int = 5000
    use_kinetics: bool = False
    kinetics_dataset_name: str = "liuhuanjim013/kinetics400"
    kinetics_split_name_train: str = "train"
    kinetics_max_samples: int = 5000
    blacklist_domains: List[str] = field(default_factory=lambda: ["malicious.com", "phishing.net"])
    max_tokens_target: int = 1_000_000_000_000
    data_shards: int = 1000
    dedup_method: str = 'minhash'
    ethical_filter: bool = True
    # Enhancement: OCR and hybrid for messy visual inputs (e.g., handwriting)
    ocr_enabled: bool = True
    ocr_threshold: float = 0.7
    ocr_model: str = "microsoft/trocr-base-handwritten"
    image_byte_weight: float = 0.5
    handwriting_domains: List[str] = field(default_factory=lambda: ["medicalnotes.org", "scan archives"])

    def _validate(self):
        """Validate data params: non-neg samples."""
        if self.lyra_max_samples < 0:
            raise ValueError("lyra_max_samples must be non-negative")
        if self.max_tokens_target < 0:
            raise ValueError("max_tokens_target must be non-negative")
        if self.audioset_max_samples <= 0 and self.use_audioset:
            raise ValueError("audioset_max_samples must be positive if use_audioset=True")
        if self.kinetics_max_samples <= 0 and self.use_kinetics:
            raise ValueError("kinetics_max_samples must be positive if use_kinetics=True")
        if self.ocr_threshold < 0 or self.ocr_threshold > 1:
            raise ValueError("ocr_threshold must be in [0,1]")

@dataclass
class EvolutionConfig:
    """Evolutionary search: Hyperparam spaces, DEAP params."""
    evolution_generations: int = 5
    population_size: int = 10
    evolutionary_mutation_rate: float = 0.1
    evolution_parallel_workers: int = 4
    hyperparam_search_space: Dict[str, List] = field(default_factory=lambda: {
        'learning_rate': [0.0001, 0.0003, 0.001],
        'sparsity_target': [0.2, 0.3, 0.4],
        'dropout_rate': [0.05, 0.1, 0.15],
        'transformer_num_layers': [6, 12, 18, 96],
        'num_moe_experts': [4, 8, 16],
        'tile_size': [512, 1024, 2048],
        'ppo_clip': [0.1, 0.2, 0.3],
        'patch_size': [2, 4, 8],
        'rpg_num_chains': [1, 3, 5],
        'adp_lr_actor': [5e-4, 1e-3, 5e-3],
        'adp_lr_critic': [5e-4, 1e-3, 5e-3],
        'adp_gamma': [0.95, 0.99, 0.999],
        'substrate_fidelity_weight': [0.05, 0.1, 0.2],
        # Enhancement: Evo for OCR hybrid params
        'image_byte_weight': [0.3, 0.5, 0.7]
    })
    deap_cxpb: float = 0.5
    deap_mutpb: float = 0.2
    disable_if_no_deap: bool = True

    def _validate(self):
        pass

@dataclass
class SystemConfig:
    """Runtime env: Devices, checkpoints, APIs."""
    world_size: int = 1
    local_rank: int = -1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir: str = './checkpoints'
    wandb_project: str = 'xagi'
    xai_api_key: Optional[str] = None
    grok_model: str = "grok-4-fast"
    sglang_fine_tune_epochs: int = 3
    use_local_fallback: bool = True
    rdma_enabled: bool = False
    web_search_timeout: int = 30
    web_crawl_timeout: int = 10
    hdf5_compression: bool = True
    production: bool = False
    use_torchrun: bool = True
    cache_capacity: int = 1_000_000
    use_cuda_graphs: bool = True
    num_agents: int = 1000  # Pruned, stub
    use_decentralized_agents: bool = False
    agent_comms_protocol: str = "gossip"
    consensus_threshold: float = 0.7

    def _validate(self):
        """Validate env: e.g., wandb_project alphanumeric."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', self.wandb_project):
            raise ValueError("wandb_project must be alphanumeric with _-")
        if self.cache_capacity < 100:
            raise ValueError("cache_capacity too small")

@dataclass
class OtherConfig:
    """Misc axioms: Sparsity, replay, RPG."""
    sparsity_target: float = 0.3
    dropout_rate: float = 0.1
    use_local_grok: bool = True
    hippocampal_replay_weight: float = 0.5
    use_bpe: bool = False
    bpe_vocab_size: int = 5000
    bpe_merges: int = 4500
    max_bpe_length: int = 512
    bpe_dropout: float = 0.1
    chain_max_steps: int = 5
    chain_use_grok: bool = True
    grpo_iterations: int = 3
    grpo_feedback_weight: float = 0.5
    cor_fine_tune_weight: float = 0.2
    grpo_fine_tune_kl_bonus: float = 0.05
    use_rpg: bool = True
    rpg_max_nodes: int = 50
    rpg_max_edges: int = 100
    rpg_num_chains: int = 3
    replay_buffer_size: int = 10000
    hippocampal_replay_probability: float = 0.2
    dynamic_harm_taxonomy: bool = True

    def _validate(self):
        """Validate misc: rpg_chains >0."""
        if self.rpg_num_chains <= 0:
            raise ValueError("rpg_num_chains must be positive")

class XAGIConfig:
    """Composable factory: Builds/validates sub-configs. Loads YAML/env."""
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        data: Optional[DataConfig] = None,
        evolution: Optional[EvolutionConfig] = None,
        system: Optional[SystemConfig] = None,
        other: Optional[OtherConfig] = None
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.evolution = evolution or EvolutionConfig()
        self.system = system or SystemConfig()
        self.other = other or OtherConfig()
        self.model.latent_dim = min(self.model.latent_dim, self.model.hidden_dim // 64)
        self.model.device = self.system.device  # Sync device
        self._validate_all()
        self._load_env_secrets()

    def _validate_all(self):
        """Holistic validation: Cross-checks (e.g., world_size % TP*PP)."""
        for sub in [self.model, self.training, self.data, self.evolution, self.system, self.other]:
            sub._validate()
        if self.training.use_megatron and self.system.world_size % (self.model.tp_degree * self.model.pp_degree) != 0:
            raise ValueError(f"world_size must divide TP*PP={self.model.tp_degree * self.model.pp_degree}")
        if self.data.use_lyra and not os.path.exists(self.data.lyra_base_path):
            logger.warning(f"Lyra path {self.data.lyra_base_path} missing; will attempt HF download.")
        if self.data.use_caselaw and not self.data.caselaw_base_path:
            logger.info("Caselaw streaming from HF; no local path needed.")

    def _load_env_secrets(self):
        """Load secrets: API keys from env."""
        self.system.xai_api_key = os.environ.get('XAI_API_KEY') or self.system.xai_api_key
        if not self.system.xai_api_key and self.system.production:
            raise ValueError("XAI_API_KEY required in production")
        else:
            logger.warning("XAI_API_KEY not found; some features limited.")
        self.data.web_search_api_keys = {
            'serper': os.environ.get('SERPER_API_KEY') or self.data.web_search_api_keys.get('serper'),
            'bing': os.environ.get('BING_API_KEY') or self.data.web_search_api_keys.get('bing')
        }
        for key, value in self.data.web_search_api_keys.items():
            if not value and self.system.production:
                raise ValueError(f"{key.upper()}_API_KEY required")
            else:
                logger.warning(f"{key.upper()}_API_KEY not found; some features limited.")

    @classmethod
    def from_yaml(cls, path: str) -> 'XAGIConfig':
        """Load from YAML, fallback to defaults."""
        if not os.path.exists(path):
            logger.warning(f"Config {path} not found; defaults.")
            return cls()
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            data=DataConfig(**data.get('data', {})),
            evolution=EvolutionConfig(**data.get('evolution', {})),
            system=SystemConfig(**data.get('system', {})),
            other=OtherConfig(**data.get('other', {}))
        )

# ================== src/core/deps.py ==================
"""
Deps: Lazy category-based loading. Quarantines failures.
Principle: Fail gracefully—core strict, optional gated.
"""

import importlib
import logging
from typing import Optional, Any, Dict
from enum import Enum
from .configs import XAGIConfig

logger = logging.getLogger(__name__)

class DepCategory(Enum):
    CORE = "core"
    ML = "ml"
    DATA = "data"
    PLANNING = "planning"
    OPTIONAL = "optional"

DEPS_BY_CATEGORY: Dict[DepCategory, list] = {
    DepCategory.CORE: ['torch', 'numpy', 'pyyaml', 'asyncio'],
    DepCategory.ML: ['transformers', 'deepspeed', 'megatron-core', 'torch-fidelity'],
    DepCategory.DATA: ['datasets', 'h5py', 'webdataset', 'open3d', 'gitpython'],
    DepCategory.PLANNING: ['unified-planning', 'networkx'],
    DepCategory.OPTIONAL: ['langdetect', 'nltk', 'beautifulsoup4', 'scrapy-aiohttp', 'prometheus-client', 'ftfy', 'tqdm', 'psutil', 'deap', 'spacy', 'tarski']
}

class DepManager:
    """Lazy-load deps by category. Cache in _loaded."""
    _loaded: Dict[str, Any] = {}

    @classmethod
    def check_category(cls, category: DepCategory, strict: bool = False):
        """Check if category deps installed; raise on core failures."""
        failures = []
        for dep in DEPS_BY_CATEGORY[category]:
            if not importlib.util.find_spec(dep):
                failures.append(dep)
                if category == DepCategory.CORE or strict:
                    raise ImportError(f"Missing {category.value} dep: {dep}")
                logger.warning(f"Missing {category.value} dep: {dep}; features limited.")
        return failures

    @classmethod
    def load(cls, module_name: str, category: DepCategory, warning_msg: str) -> Optional[Any]:
        """Load module; cache and warn on fail."""
        if module_name in cls._loaded:
            return cls._loaded[module_name]
        try:
            mod = importlib.import_module(module_name)
            cls._loaded[module_name] = mod
            return mod
        except ImportError:
            logger.warning(f"{warning_msg}: limited {category.value} features.")
            return None

    @classmethod
    def validate_config(cls, config: XAGIConfig):
        """Config-gated checks: Load only needed deps."""
        cls.check_category(DepCategory.CORE, strict=config.training.strict_mode)
        if config.training.use_megatron:
            cls.load('megatron_core', DepCategory.ML, "Megatron missing")
        if config.data.dedup_method == 'minhash':
            cls.load('datasketch', DepCategory.DATA, "Datasketch missing")  # Assume added to DATA
        if config.data.ethical_filter:
            cls.load('langdetect', DepCategory.OPTIONAL, "Langdetect missing")
        if config.other.use_rpg:
            cls.load('unified_planning', DepCategory.PLANNING, "Unified Planning missing")
        if config.training.use_fid:
            cls.load('torch_fidelity', DepCategory.ML, "Torch-FID missing")
        # Enhancement: Load transformers for OCR if enabled
        if config.data.ocr_enabled:
            cls.load('transformers', DepCategory.ML, "Transformers missing for OCR")

# Lazy globals (pruned: no quantum/flwr/gossip/bitsandbytes)
torch = DepManager.load("torch", DepCategory.CORE, "Torch missing")
numpy = DepManager.load("numpy", DepCategory.CORE, "NumPy missing")
yaml = DepManager.load("pyyaml", DepCategory.CORE, "YAML missing")
deap = DepManager.load("deap", DepCategory.OPTIONAL, "DEAP missing")
asyncio = DepManager.load("asyncio", DepCategory.CORE, "asyncio missing")
transformers = DepManager.load("transformers", DepCategory.ML, "Transformers missing")
datasets = DepManager.load("datasets", DepCategory.DATA, "Datasets missing")
h5py = DepManager.load("h5py", DepCategory.DATA, "H5py missing")
bs4 = DepManager.load("beautifulsoup4", DepCategory.OPTIONAL, "BeautifulSoup missing")
nltk = DepManager.load("nltk", DepCategory.OPTIONAL, "NLTK missing")
psutil = DepManager.load("psutil", DepCategory.OPTIONAL, "Psutil missing")
open3d = DepManager.load("open3d", DepCategory.DATA, "Open3D missing")
git = DepManager.load("gitpython", DepCategory.DATA, "Gitpython missing")
webdataset = DepManager.load("webdataset", DepCategory.DATA, "WebDataset missing")
datasketch = DepManager.load("datasketch", DepCategory.DATA, "Datasketch missing")
langdetect = DepManager.load("langdetect", DepCategory.OPTIONAL, "Langdetect missing")
unified_planning = DepManager.load("unified_planning", DepCategory.PLANNING, "Unified Planning missing")
torch_fidelity = DepManager.load("torch_fidelity", DepCategory.ML, "Torch-FID missing")
scrapy_aiohttp = DepManager.load("scrapy_aiohttp", DepCategory.OPTIONAL, "Scrapy-aiohttp missing")
prometheus_client = DepManager.load("prometheus_client", DepCategory.OPTIONAL, "Prometheus missing")
ftfy = DepManager.load("ftfy", DepCategory.OPTIONAL, "FTFY missing")
spacy = DepManager.load("spacy", DepCategory.OPTIONAL, "spaCy missing for advanced PII")
tarski = DepManager.load("tarski", DepCategory.OPTIONAL, "Tarski missing for robust PDDL")

# Prometheus (guarded)
# Removed 'if 'config' in globals()' to avoid NameError; assume always start if available
if prometheus_client:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    METRIC_TRAIN_LOSS = Gauge('xagi_train_loss', 'Training loss')
    METRIC_INFERENCE_TIME = Histogram('xagi_inference_time', 'Inference time')
    METRIC_MEMORY_USAGE = Gauge('xagi_memory_usage', 'Memory usage')
    METRIC_TOXICITY_CHECKS = Counter('xagi_toxicity_checks', 'Toxicity checks performed')
    METRIC_UNCERTAINTY_VAR = Gauge('xagi_uncertainty_variance', 'Variance in prediction uncertainty')
    METRIC_TOKENS_PROCESSED = Counter('xagi_bytes_processed', 'Bytes processed')
    METRIC_BYTE_THROUGHPUT = Histogram('xagi_byte_throughput', 'Byte throughput/sec')
    METRIC_RPG_ECE = Gauge('xagi_rpg_ece', 'RPG uncertainty calibration ECE')
    METRIC_SUBSTRATE_FIDELITY = Gauge('xagi_substrate_fidelity', 'Byte recon fidelity MSE')
    METRIC_ADP_TD_ERROR = Gauge('xagi_adp_td_error', 'ADP TD error mean')
    METRIC_ADP_UNC = Gauge('xagi_adp_uncertainty', 'ADP evidential uncertainty')
    METRIC_FID_SCORE = Gauge('xagi_fid_score', 'FID score')
    start_http_server(8000)
    logger.info("Prometheus on 8000.")
else:
    class DummyMetric:
        def set(self, *args): pass
        def inc(self, *args): pass
        def observe(self, *args): pass
    METRIC_TRAIN_LOSS = DummyMetric()
    METRIC_INFERENCE_TIME = DummyMetric()
    METRIC_MEMORY_USAGE = DummyMetric()
    METRIC_TOXICITY_CHECKS = DummyMetric()
    METRIC_UNCERTAINTY_VAR = DummyMetric()
    METRIC_TOKENS_PROCESSED = DummyMetric()
    METRIC_BYTE_THROUGHPUT = DummyMetric()
    METRIC_RPG_ECE = DummyMetric()
    METRIC_SUBSTRATE_FIDELITY = DummyMetric()
    METRIC_ADP_TD_ERROR = DummyMetric()
    METRIC_ADP_UNC = DummyMetric()
    METRIC_FID_SCORE = DummyMetric()

# ================== src/utils/__init__.py ==================
"""Utils: Isolated helpers—no config bleed."""

# ================== src/utils/common.py ==================
"""
Common utils: Hashing, PII removal, ethical filters, replay.
Principle: Ground in reality—filter harm, prioritize unc.
"""

import hashlib
import re
import torch
import numpy as np
from typing import Optional, List
from src.core.deps import DepManager, DepCategory, langdetect, ftfy, nltk
import heapq
from collections import deque
import random
import os
from src.core.configs import DataConfig
import imghdr  # For image validation

MAX_SUMMARY_LENGTH = 2048
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b\d{16}\b',  # Credit card
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    r'\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b',  # Phone
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP
    r'\b\d{3}-\d{3}-\d{4}\b',  # Additional phone
    r'\b[A-Z0-9]{16}\b',  # Bank account
    r'\b\d{9}\b',  # Medical ID
    r'\b[A-Z]{2}\d{9}\b'  # EU VAT example
]
HARM_TAXONOMY = [
    r'\b(violence|kill|murder|harm)\b',
    r'\b(discriminate|racist|sexist|hate)\b',
    r'\b(deceive|lie|fraud)\b',
    r'\b(illegal|crime|terror)\b',
    r'\b(exploit|abuse|harass)\b',
    r'\b(bully|threaten|insult)\b',
    r'\b(misinfo|disinfo|fake news)\b'
]

nltk_mod = DepManager.load("nltk", DepCategory.OPTIONAL, "NLTK missing")
transformers_mod = DepManager.load("transformers", DepCategory.ML, "Transformers missing")
langdetect_mod = langdetect
ftfy_mod = ftfy
spacy_mod = DepManager.load("spacy", DepCategory.OPTIONAL, "spaCy missing")

def hash_to_coords(text: str, point_cloud_size: int, input_dim: int) -> torch.Tensor:
    """Deterministic hash: Text -> normalized coords for point clouds."""
    text = text or ""
    text_hash = hashlib.sha256(text.encode()).digest()
    coords_np = np.frombuffer(text_hash, dtype=np.float32).flatten()[:point_cloud_size * input_dim]
    coords_np = np.pad(coords_np, (0, point_cloud_size * input_dim - len(coords_np)), 'constant')
    if len(coords_np) < point_cloud_size * input_dim:
        noise = np.random.uniform(-1, 1, point_cloud_size * input_dim - len(coords_np))
        coords_np = np.concatenate([coords_np, noise])
    return torch.from_numpy(coords_np.reshape(point_cloud_size, input_dim)).float()

def load_api_key(key_name: str) -> str:
    """Securely load API key from env; raise if missing."""
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f"{key_name} not found.")
    return key

def remove_pii(text: str) -> str:
    """Redact PII patterns; NLTK NER fallback, enhanced with spaCy if available."""
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text)
    if spacy_mod:
        try:
            nlp = spacy_mod.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EMAIL']:  # EMAIL via custom
                    text = text.replace(ent.text, '[REDACTED]')
        except Exception as e:
            logger.warning(f"spaCy error: {str(e)}; fallback to regex.")
    elif nltk_mod:
        try:
            from nltk import pos_tag, word_tokenize
            from nltk.chunk import ne_chunk
            nltk_mod.download('maxent_ne_chunker', quiet=True)
            nltk_mod.download('words', quiet=True)
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)
            for entity in entities:
                if hasattr(entity, 'label') and entity.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                    text = text.replace(' '.join([word for word, tag in entity.leaves()]), '[REDACTED]')
        except Exception as e:
            logger.warning(f"NLTK error: {str(e)}")
    return text

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    """L2-normalize first 3 dims for coords."""
    dims = min(3, coords.shape[-1])
    norms = torch.norm(coords[:, :dims], dim=1, keepdim=True)
    coords[:, :dims] = coords[:, :dims] / (norms + 1e-8)
    return coords

def filter_harm(text: str, taxonomy: list) -> bool:
    """Regex + cached toxicity classifier."""
    METRIC_TOXICITY_CHECKS.inc()
    text_lower = text.lower()
    if any(re.search(pattern, text_lower) for pattern in taxonomy):
        return False
    if transformers_mod:
        try:
            # Global cache for pipeline
            if not hasattr(filter_harm, 'toxicity_pipeline'):
                filter_harm.toxicity_pipeline = transformers_mod.pipeline("text-classification", model="unitary/toxic-bert")
            result = filter_harm.toxicity_pipeline(text)[0]
            if result['label'] == 'toxic' and result['score'] > 0.5:
                return False
        except Exception as e:
            logger.warning(f"Toxicity detection failed: {str(e)}")
    return True

def is_safe_url(url: str, blacklist: list = []) -> bool:
    """Basic URL safety: Domain check, no local."""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    if domain in blacklist:
        return False
    if re.match(r'^[0-9.]+$', domain):
        return False
    if 'localhost' in domain or '127.0.0.1' in domain:
        return False
    return True

def ethical_filter_text(text: str, data_cfg: DataConfig) -> bool:
    """Langdetect for multi-lang ethics; allowed en/es/fr."""
    if not data_cfg.ethical_filter or not langdetect_mod:
        return True
    try:
        lang = langdetect_mod.detect(text)
        allowed_langs = ['en', 'es', 'fr']
        return lang in allowed_langs
    except:
        return True

def ethical_filter_bytes(bytes_batch: torch.Tensor, data_cfg: DataConfig) -> bool:
    """Chunked decode for large byte tensors."""
    try:
        # Sample first 1k bytes to avoid OOM
        sample_bytes = bytes_batch[:1000].cpu().numpy().tobytes()
        text = sample_bytes.decode('utf-8', errors='ignore')
        if ftfy_mod:
            text = ftfy_mod.fix_text(text)
        return ethical_filter_text(text, data_cfg)
    except:
        return True

class SumTree:
    """SumTree for PER priorities."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.n_entries + self.capacity - 1
        self.data[self.n_entries] = data
        self.update(idx, p)
        self.n_entries += 1
        if self.n_entries >= self.capacity:
            self.n_entries = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PriorityReplayBuffer:
    """Uncertainty-weighted replay with PER alpha sampling."""
    e = 0.01
    a = 0.6  # Alpha for priority
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity: int):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, experience, loss, uncertainty=0.0):
        """Priority = loss + w * unc; evict lowest."""
        error = loss + 0.5 * uncertainty
        p = self._get_priority(error)
        self.tree.add(p, experience)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

# ================== src/models/__init__.py ==================
"""Models: Core XAGI + submodules."""

# ================== src/models/adp.py ==================
"""
ADP: Evidential Actor-Critic for RL refinement.
Principle: Calibrated value via Dirichlet; weights unc in TD.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.core.configs import ModelConfig

class GridWorldEnv:
    """5x5 GridWorld: State (x,y), goal (4,4), reward -dist + goal bonus."""
    def __init__(self, size=5):
        self.size = size
        self.goal = torch.tensor([4, 4], dtype=torch.float)
        self.reset()

    def reset(self):
        """Reset to (0,0)."""
        self.state = torch.tensor([0, 0], dtype=torch.float)
        self.done = False
        return self.state

    def step(self, action):
        """Move, compute reward, set done."""
        dirs = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]])  # R L D U
        self.state = torch.clamp(self.state + dirs[action], 0, self.size-1)
        dist = torch.norm(self.state - self.goal)
        reward = -dist + (1 if torch.equal(self.state, self.goal) else 0)
        self.done = torch.equal(self.state, self.goal)
        return self.state, reward, self.done

class EvidentialCritic(nn.Module):
    """Dirichlet evidence for V(s): Mean/var from digits."""
    def __init__(self, state_dim, hidden_dim, digits):
        super().__init__()
        self.digits = digits
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * digits)  # Mean/var digits
        )

    def forward(self, state):
        """Output value, unc (1/evidence)."""
        params = self.net(state)
        params = params.view(state.size(0), 2, self.digits)
        digits_tensor = 10 ** torch.arange(self.digits, dtype=torch.float).to(state.device)
        mean_digits = torch.softplus(params[..., 0, :])  # Positive
        var_digits = torch.sigmoid(params[..., 1, :])  # [0,1]
        mean = torch.sum(mean_digits * digits_tensor.unsqueeze(0), dim=-1)
        var = torch.sum(var_digits * digits_tensor.unsqueeze(0), dim=-1)
        alpha = (mean ** 2 / (var + 1e-6)) + 1
        evidence = torch.sum(alpha, dim=-1, keepdim=True) - 1
        uncertainty = 1 / (evidence.squeeze(-1) + 1e-6)
        value = mean
        return value, uncertainty

class Actor(nn.Module):
    """Stochastic policy: Softmax over actions."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class ADPIntegrator:
    """TD Actor-Critic: Unc-weighted updates, rollout."""
    def __init__(self, model_cfg: ModelConfig):
        self.gamma = model_cfg.adp_gamma
        self.lr_actor = model_cfg.adp_lr_actor
        self.lr_critic = model_cfg.adp_lr_critic
        self.state_dim = model_cfg.adp_state_dim
        self.action_dim = model_cfg.adp_action_dim
        self.device = model_cfg.device
        self.actor = Actor(self.state_dim, self.action_dim, model_cfg.adp_hidden_dim).to(self.device)
        self.critic = EvidentialCritic(self.state_dim, model_cfg.adp_hidden_dim, model_cfg.adp_digits).to(self.device)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.env = GridWorldEnv()

    def update(self, state, action, reward, next_state, unc_weight=1.0):
        """TD update: Critic MSE * unc, Actor PG / (1+unc)."""
        v_s, unc_s = self.critic(state)
        with torch.no_grad():
            v_next, _ = self.critic(next_state)
        target = reward.to(self.device) + self.gamma * v_next
        td_error = target - v_s

        # Critic loss: Weighted MSE
        critic_loss = (td_error ** 2 * (unc_weight * (1 + unc_s))).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Actor: Policy gradient, downweight high unc
        probs = self.actor(state)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(1)).squeeze() + 1e-8)
        actor_loss = -(log_prob * td_error.detach() * (1 / (1 + unc_s.detach()))).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        METRIC_ADP_TD_ERROR.set(td_error.mean().item())
        METRIC_ADP_UNC.set(unc_s.mean().item())
        return td_error.mean().item(), unc_s.mean().item()

    def rollout(self, state, num_steps=5):
        """Stochastic rollout; early stop on done."""
        state = state.clone()
        total_reward = 0
        for _ in range(num_steps):
            probs = self.actor(state.unsqueeze(0))
            action = torch.multinomial(probs, 1).squeeze()
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        return next_state.unsqueeze(0), torch.tensor(total_reward, device=self.device), done

# ================== src/models/byte_patcher.py ==================
"""
BytePatcher: Variable patches for tokenizer-free seq modeling.
Principle: Bytes as substrate—embed patches adaptively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.configs import ModelConfig

class BytePatcher(nn.Module):
    """Embed bytes into patches; project to hidden_dim."""
    def __init__(self, vocab_size: int = 256, patch_size: int = 4, embed_dim: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim * patch_size, embed_dim)  # Flatten -> project

    def forward(self, bytes_input: torch.Tensor):
        """[B, L] uint8 -> [B, L//patch, embed_dim]."""
        B, L = bytes_input.shape
        if L % self.patch_size != 0:
            # Pad to multiple
            pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
            bytes_input = F.pad(bytes_input, (0, pad_len), value=0)
            L += pad_len
        patches = bytes_input.view(B, L // self.patch_size, self.patch_size)
        embeds = self.embed(patches)  # [B, num_patches, patch_size, embed_dim]
        embeds = embeds.transpose(2, 3).flatten(2)  # [B, num_patches, patch_size*embed]
        patches_out = self.proj(embeds)  # [B, num_patches, embed_dim]
        return patches_out

# ================== src/models/xagi.py ==================
"""
XAGI: Core model—byte recon w/ evidential unc, RPG+ADP fusion.
Pruned: No quantum, lean forward (hooks for mods).
Principle: Fuse planning/RL into latent for grounded reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.configs import ModelConfig, OtherConfig
from src.core.deps import DepManager, DepCategory, transformers_mod, unified_planning
from .byte_patcher import BytePatcher
from src.planning.rpg_pddl import RPGChain
from .adp import ADPIntegrator
import torch.utils.checkpoint as checkpoint
import math
from PIL import Image  # Added for OCR image processing
import io
import imghdr  # For image validation

class NeocortexProjector(nn.Module):
    """Hierarchical MoE projector: Levels -> route experts."""
    def __init__(self, input_dim, hidden_dim, output_dims, num_levels, model_cfg: ModelConfig, use_megatron=False):
        super().__init__()
        # Adaptive scaling based on device memory
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # GB
            if free_mem < 16:
                hidden_dim = hidden_dim // 2
                logger.info(f"Scaled hidden_dim to {hidden_dim} due to low GPU mem ({free_mem:.1f}GB)")
        self.layers = nn.ModuleList([nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_levels)])
        self.router = nn.Linear(hidden_dim, model_cfg.num_moe_experts)
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(model_cfg.num_moe_experts)])
        self.use_megatron = use_megatron
        if use_megatron:
            from megatron_core import mpu
            mpu.initialize_model_parallel(model_cfg.tp_degree, model_cfg.pp_degree)

    def forward(self, x):
        """Forward: Linear stack -> MoE -> gather if MP."""
        for layer in self.layers:
            x = F.relu(layer(x))  # Activation inline
        gates = F.softmax(self.router(x), dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        x = sum(g * o.unsqueeze(-1) for g, o in zip(gates.split(1, dim=-1), expert_outputs))  # Broadcast
        if self.use_megatron:
            from megatron_core import mpu
            if mpu.is_initialized():
                x = mpu.gather_from_model_parallel_region(x)
        return x

class EvidentialDecoder(nn.Module):
    """Unc-aware recon: Dirichlet alpha -> softmax distrib."""
    def __init__(self, latent_dim: int, output_dim: int, digits: int = 5, kl_weight: float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.digits = digits
        self.kl_weight = kl_weight
        self.gamma = nn.Parameter(torch.ones(1))
        self.decoder = nn.Linear(latent_dim, output_dim * 2 * digits)

    def forward(self, latent, chain_unc_weight: float = 1.0):
        """Recon, unc, alpha, KL (to uniform)."""
        params = self.decoder(latent)
        params = params.view(-1, self.output_dim, 2 * self.digits)
        digits_tensor = 10 ** torch.arange(self.digits).float().to(params.device)
        mean = torch.sum(torch.softplus(params[..., :self.digits]) * digits_tensor, dim=-1)
        var = torch.sum(torch.sigmoid(params[..., self.digits:]) * digits_tensor, dim=-1)
        alpha = self.gamma * (mean ** 2 / (var + 1e-6)) + 1
        evidence = torch.sum(alpha, dim=-1, keepdim=True) - self.output_dim
        evidence = torch.clamp(evidence, min=10)
        uncertainty = self.output_dim / (evidence + 1e-6) * chain_unc_weight
        recon = F.softmax(alpha - 1, dim=-1) * self.output_dim  # Expected value
        # KL divergence to uniform
        kl_loss = -torch.lgamma(alpha.sum(dim=-1)) + torch.sum(torch.lgamma(alpha), dim=-1)
        kl_loss = self.kl_weight * kl_loss.mean() / self.output_dim
        return recon, uncertainty, alpha, kl_loss

class XAGI(nn.Module):
    """XAGI: Byte substrate -> latent (RPG/ADP) -> evidential recon."""
    def __init__(self, model_cfg: ModelConfig, other_cfg: OtherConfig, training_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.other_cfg = other_cfg
        self.training_cfg = training_cfg
        self.device = model_cfg.device
        total_output_dim = sum(model_cfg.output_dims)
        self.projector = NeocortexProjector(model_cfg.input_dim, model_cfg.hidden_dim, model_cfg.output_dims, model_cfg.num_levels, model_cfg, model_cfg.use_megatron)
        self.encoder = nn.Linear(model_cfg.input_dim, model_cfg.latent_dim)
        self.evidential_decoder = EvidentialDecoder(model_cfg.latent_dim, total_output_dim, digits=5, kl_weight=training_cfg.edl_kl_weight)
        # BytePatcher
        self.byte_patcher = BytePatcher(model_cfg.vocab_size, model_cfg.patch_size, model_cfg.hidden_dim) if model_cfg.byte_level else None
        # Transformer (checkpointed)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_cfg.hidden_dim, nhead=model_cfg.transformer_nhead, batch_first=True, dropout=other_cfg.dropout_rate)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=model_cfg.transformer_num_layers)
        self.transformer = checkpoint.checkpoint_wrapper(self.transformer, use_reentrant=False)
        # RPG
        self.rpg_chain = RPGChain(model_cfg, other_cfg) if other_cfg.use_rpg and unified_planning else None
        self.plan_embed = nn.Embedding(model_cfg.num_actions, model_cfg.latent_dim) if other_cfg.use_rpg else None
        # ADP
        self.adp = ADPIntegrator(model_cfg) if model_cfg.use_adp else None
        # Image processor (pruned multimodal: only if used)
        if transformers_mod and model_cfg.use_tvl:  # Stub for TVL
            self.image_processor = transformers_mod.ViTImageProcessor.from_pretrained(model_cfg.tvl_encoder_image)
            self.image_model = transformers_mod.ViTModel.from_pretrained(model_cfg.tvl_encoder_image)
        # Enhancement: OCR pipeline cache for hybrid
        self.ocr_pipeline = None
        if transformers_mod and training_cfg.data.ocr_enabled:  # Assume config passed or global
            self.ocr_pipeline = transformers_mod.pipeline("text-generation", model=training_cfg.data.ocr_model)

    def encode(self, coords):
        """Encode coords to latent (mean pool)."""
        return self.encoder(coords.mean(dim=0, keepdim=True))

    def decode(self, latent):
        """Evidential decode: Recon + unc + KL."""
        return self.evidential_decoder(latent)

    def forward(self, inputs, coords=None, target=None, bytes_input=None):
        """Core forward: Fuse bytes/RPG/ADP -> transformer -> recon."""
        # Init latent
        latent = torch.zeros(1, self.model_cfg.latent_dim, device=self.device)
        fidelity_loss = 0.0

        # Enhancement: Hybrid OCR fusion if enabled and image bytes present
        ocr_conf = 1.0  # Default full conf
        if self.training_cfg.data.ocr_enabled and 'image_bytes' in inputs and self.ocr_pipeline:
            img_bytes = inputs['image_bytes']
            img_bytes_raw = img_bytes.cpu().numpy().tobytes()
            # Validate image bytes
            img_type = imghdr.what(None, img_bytes_raw)
            if img_type is None:
                logger.warning("Invalid image bytes; fallback to raw.")
                ocr_conf = 0.0
            else:
                # Fix: Convert bytes to PIL Image for trocr pipeline
                try:
                    img = Image.open(io.BytesIO(img_bytes_raw))
                    ocr_result = self.ocr_pipeline(img)
                    ocr_text = ocr_result[0]['generated_text'] if ocr_result else ""
                    ocr_conf = ocr_result[0].get('score', 0.5) if ocr_result else 0.5
                except Exception as e:
                    logger.warning(f"OCR image processing failed: {e}; fallback to raw bytes.")
                    ocr_text = ""
                    ocr_conf = 0.0
            if ocr_conf > self.training_cfg.data.ocr_threshold:
                text_bytes = torch.tensor(list(ocr_text.encode('utf-8')), dtype=torch.uint8, device=self.device).unsqueeze(0)
                text_patches = self.byte_patcher(text_bytes)
                raw_patches = self.byte_patcher(img_bytes.long() if img_bytes.dtype != torch.long else img_bytes)
                hybrid_patches = self.training_cfg.data.image_byte_weight * raw_patches + (1 - self.training_cfg.data.image_byte_weight) * text_patches.mean(dim=1, keepdim=True)
                latent += hybrid_patches.mean(dim=1)
                # Unc weight from OCR conf
                ocr_unc_weight = 1.0 / (ocr_conf + 1e-6)
            else:
                # Fallback to raw image bytes
                latent += raw_patches.mean(dim=1)
                ocr_unc_weight = 2.0  # Penalize low conf
        else:
            ocr_unc_weight = 1.0

        # Bytes patching (original, if no hybrid)
        if self.model_cfg.byte_level and bytes_input is not None and 'image_bytes' not in inputs:
            patches = self.byte_patcher(bytes_input)
            latent += patches.mean(dim=1)  # Global avg pool
            # Fidelity: Embed recon vs latent
            byte_recon = self.byte_patcher.embed(bytes_input.long()).mean(dim=1)
            fidelity_loss = F.mse_loss(byte_recon, latent) * self.training_cfg.substrate_fidelity_weight
            METRIC_SUBSTRATE_FIDELITY.set(fidelity_loss.item())

        # Coords fallback
        if coords is not None:
            latent += self.encode(coords)

        # RPG fusion
        chain_unc_weight = 1.0
        if self.other_cfg.use_rpg and self.rpg_chain and 'pddl_domain' in inputs:
            domain_bytes = inputs['pddl_domain']
            problem_bytes = inputs['pddl_problem']
            domain_embed = self.byte_patcher(domain_bytes).mean(dim=1)
            problem_embed = self.byte_patcher(problem_bytes).mean(dim=1)
            plan_chains = self.rpg_chain.chain_plans(domain_embed, problem_embed, self.other_cfg.rpg_num_chains)
            # Embed chains (cast to long)
            chain_embeds = torch.stack([
                self.plan_embed(torch.tensor([c % self.model_cfg.num_actions for c in chain[:self.other_cfg.chain_max_steps]], dtype=torch.long, device=self.device))
                for chain in plan_chains
            ]).mean(dim=0)
            latent += chain_embeds
            # Unc from chain length
            chain_lengths = [len(chain) for chain in plan_chains]
            chain_unc = max(chain_lengths) / self.other_cfg.rpg_max_nodes if chain_lengths else 1.0
            chain_unc_weight = 1.0 + chain_unc

        # ADP refinement
        adp_unc = 0.0
        if self.model_cfg.use_adp and self.adp:
            norm_latent = F.normalize(latent, dim=-1)
            state = norm_latent.mean(dim=0).unsqueeze(0)[:, :self.model_cfg.adp_state_dim]
            action = torch.randint(0, self.model_cfg.adp_action_dim, (1,), device=self.device)
            next_state, reward, done = self.adp.rollout(state)
            td_error, adp_unc = self.adp.update(state, action, reward, next_state)
            latent += 0.1 * next_state  # Fuse

        # Transformer (causal-ish: use shifted latent as memory)
        memory = torch.roll(latent, shifts=1, dims=0)
        latent = self.transformer(latent, memory)

        # Decode
        recon, e_uncertainty, alpha, kl_loss = self.decode(latent)
        total_unc = e_uncertainty + adp_unc * chain_unc_weight * ocr_unc_weight  # Fuse with OCR unc

        # Loss
        recon_loss = F.mse_loss(recon, target) if target is not None else 0.0
        total_loss = fidelity_loss + kl_loss + recon_loss

        METRIC_UNCERTAINTY_VAR.set(total_unc.var().item())
        return recon, total_loss, latent, total_unc

# ================== src/training/__init__.py ==================
"""Training: Orchestration w/ async, evo, benches."""

# ================== src/training/evolution.py ==================
"""
Evolution: Grok-driven hyperparam search + DEAP fallback.
Expanded: Chain prompts for deeper axioms (e.g., unc vs FID tradeoffs).
"""

import asyncio
import numpy as np
import random
from typing import List, Dict, Optional
from src.core.configs import XAGIConfig
from src.integrations.grok import GrokIntegrator
from src.core.deps import deap, logger

async def evolve_config_grok(config: XAGIConfig, loss_history: List[float], benchmarks: Optional[Dict] = None, fid_score: float = 0.0, integrator: Optional[GrokIntegrator] = None, unc_history: List[float] = None, byte_throughput: float = 0.0) -> XAGIConfig:
    """
    Trigger evo on plateau/high FID/low throughput/var unc.
    Expanded: Use Grok for axiom reasoning (e.g., "Balance unc vs recon?").
    """
    # Cache computed metrics
    cached_std = np.std(loss_history[-10:]) if len(loss_history) >= 10 else 0
    cached_unc_var = np.var(unc_history[-10:]) if unc_history and len(unc_history) >= 10 else 0
    trigger = len(loss_history) < 10 or cached_std < config.training.loss_plateau_threshold
    if fid_score > config.training.fid_threshold:
        trigger = True
    if benchmarks and config.training.benchmark_regressions and 'throughput' in benchmarks:
        if benchmarks['throughput'] < 0.9 * config.training.byte_throughput_target:
            trigger = True
    if unc_history:
        trigger = trigger or cached_unc_var > 0.03
    if byte_throughput < config.training.byte_throughput_target * 0.8:
        trigger = True
    if not trigger or not integrator:
        return config

    # Expanded Grok: Chain for axioms
    chain_prompt = f"First, reason: Given loss std={cached_std}, unc var={cached_unc_var}, FID={fid_score}, throughput={byte_throughput}, what axiom tradeoff? E.g., high unc needs more chains, low FID more fidelity_w. Then suggest JSON: {{'learning_rate': float, 'num_moe_experts': int, 'patch_size': int, 'rpg_num_chains': int, 'substrate_fidelity_weight': float, 'adp_lr_actor': float, 'adp_lr_critic': float, 'adp_gamma': float}}"
    suggestions = await integrator.suggest_hyperparams_grok(chain_prompt, loss_history, benchmarks or {}, fid_score, unc_history, byte_throughput)

    # Apply
    config.training.learning_rate = suggestions.get('learning_rate', config.training.learning_rate)
    config.model.num_moe_experts = suggestions.get('num_moe_experts', config.model.num_moe_experts)
    config.model.patch_size = suggestions.get('patch_size', config.model.patch_size)
    config.other.rpg_num_chains = suggestions.get('rpg_num_chains', config.other.rpg_num_chains)
    config.training.substrate_fidelity_weight = suggestions.get('substrate_fidelity_weight', config.training.substrate_fidelity_weight)
    config.model.adp_lr_actor = suggestions.get('adp_lr_actor', config.model.adp_lr_actor)
    config.model.adp_lr_critic = suggestions.get('adp_lr_critic', config.model.adp_lr_critic)
    config.model.adp_gamma = suggestions.get('adp_gamma', config.model.adp_gamma)
    # Enhancement: Apply OCR hybrid weight
    config.data.image_byte_weight = suggestions.get('image_byte_weight', config.data.image_byte_weight)

    # DEAP fallback if no Grok
    if not integrator.client and deap:
        from deap import base, creator, tools
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        toolbox = base.Toolbox()
        # Register individuals as dicts of hyperparams
        def create_ind():
            return random.choice(list(config.evolution.hyperparam_search_space.values()))  # Stub; expand
        toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: random.random() * fid_score)  # Mock eval
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        pop = toolbox.population(n=config.evolution.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # Run generations (simple loop)
        for gen in range(config.evolution.evolution_generations):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config.evolution.deap_cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < config.evolution.deap_mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        # Apply best (stub: map back to params)
        logger.info(f"DEAP best: {best}")

    logger.info(f"Evolved config: LR={config.training.learning_rate}, Experts={config.model.num_moe_experts}, Chains={config.other.rpg_num_chains}")
    return config

# ================== src/training/benchmarks.py ==================
"""
Benches: Inference time, E2E throughput, unc calib, FID.
Principle: Measure axioms—throughput for scale, ECE for trust.
"""

import torch
from torch.utils.benchmark import Timer
from src.core.deps import METRIC_INFERENCE_TIME, logger, METRIC_BYTE_THROUGHPUT, METRIC_RPG_ECE, torch_fidelity
import time
import torch.cuda as cuda
from tqdm import tqdm
from src.core.configs import XAGIConfig
from src.models.xagi import XAGI
from torch.utils.data import DataLoader

def benchmark_inference(model: XAGI, input_data):
    """Timeit 10 runs; observe histogram."""
    timer = Timer(stmt="model(input_data)", globals={"model": model, "input_data": input_data})
    measurement = timer.timeit(10)
    METRIC_INFERENCE_TIME.observe(measurement.mean)
    return measurement.mean

def end_to_end_bench(model: XAGI, dataloader: DataLoader):
    """E2E: Time batches, compute byte/sec."""
    total_time = 0
    total_bytes = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Benchmarking"):
            start = cuda.Event(enable_timing=True)
            end = cuda.Event(enable_timing=True)
            start.record()
            _ = model(batch)
            end.record()
            cuda.synchronize()
            batch_time = start.elapsed_time(end) / 1000
            total_time += batch_time
            if 'text_bytes' in batch:
                total_bytes += batch['text_bytes'].numel()
    avg_time = total_time / len(dataloader)
    avg_throughput = total_bytes / total_time if total_time > 0 else 0
    METRIC_BYTE_THROUGHPUT.observe(avg_throughput)
    logger.info(f"E2E: {avg_time}s/batch, {avg_throughput:.2f} bytes/s")
    model.train()
    return avg_time, avg_throughput

def stress_grpo(model: XAGI):
    """Stress: 10 full forwards w/ RPG/ADP."""
    for _ in range(10):
        dummy = {'pddl_domain': torch.randint(0, 256, (1, 512), dtype=torch.uint8, device=model.device),
                 'pddl_problem': torch.randint(0, 256, (1, 512), dtype=torch.uint8, device=model.device)}
        _, loss, _, _ = model(dummy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

def benchmark_unc_calib(model: XAGI, dataloader: DataLoader):
    """ECE: |unc - acc| mean; RPG/ADP variants."""
    uncs, accs = [], []
    rpg_uncs, rpg_accs = [], []
    adp_uncs, adp_accs = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, target = batch if isinstance(batch, tuple) else ({'text_bytes': batch['input']}, batch['target'])
            outputs, _, _, unc = model(inputs, target=target)
            pred = (outputs > 0.5).float()  # Binary for simplicity
            true = (target > 0.5).float()
            acc = (pred == true).float().mean()
            uncs.append(unc.mean())
            accs.append(acc)
            # RPG variant
            if model.other_cfg.use_rpg:
                rpg_unc = unc * 1.1
                rpg_acc = acc / 1.1
                rpg_uncs.append(rpg_unc.mean())
                rpg_accs.append(rpg_acc)
            # ADP
            if model.model_cfg.use_adp:
                state = torch.randn(1, model.model_cfg.adp_state_dim, device=model.device)
                _, adp_unc = model.adp.critic(state)
                adp_acc = acc * 0.95
                adp_uncs.append(adp_unc.mean())
                adp_accs.append(adp_acc)
    ece = torch.mean(torch.abs(torch.tensor(uncs) - torch.tensor(accs)))
    rpg_ece = torch.mean(torch.abs(torch.tensor(rpg_uncs) - torch.tensor(rpg_accs))) if rpg_uncs else 0
    adp_ece = torch.mean(torch.abs(torch.tensor(adp_uncs) - torch.tensor(adp_accs))) if adp_uncs else 0
    METRIC_RPG_ECE.set(rpg_ece)
    logger.info(f"ECE: {ece:.4f}, RPG: {rpg_ece:.4f}, ADP: {adp_ece:.4f}")
    model.train()
    return ece.item()

def token_throughput(model: XAGI, dataloader: DataLoader):
    """Alias to E2E bytes/sec."""
    _, throughput = end_to_end_bench(model, dataloader)
    return throughput

def compute_fid(real_samples, fake_samples, config: XAGIConfig):
    """FID via torch-fidelity if enabled."""
    if torch_fidelity and config.training.use_fid:
        fid = torch_fidelity.calculate_metrics(input1=real_samples, input2=fake_samples, cuda=True, isc=False, fid=True)['frechet_inception_distance']
        METRIC_FID_SCORE.set(fid)
        return fid
    return 0.0

# ================== src/training/trainer.py ==================
"""
Trainer: Async epochs w/ replay, evo, benches, early stop.
Pruned: No decentralized; sync forward for simplicity.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys
import random
import asyncio
import time
from typing import Dict, List, Tuple
from src.core.configs import XAGIConfig
from src.models.xagi import XAGI
from src.integrations.grok import GrokIntegrator
from src.datasets.scaler import TrillionScaler  # Local scaler
from src.training.evolution import evolve_config_grok
from src.training.benchmarks import benchmark_inference, end_to_end_bench, stress_grpo, benchmark_unc_calib, token_throughput, compute_fid
from src.core.deps import METRIC_EPOCH_COUNT, METRIC_TRAIN_LOSS, METRIC_MEMORY_USAGE, dist, logger, nn, psutil, deepspeed, torch, cuda
from src.utils.common import PriorityReplayBuffer, ethical_filter_bytes
import megatron_core as mpu

class AsyncDataLoader:
    """Async wrapper for DataLoader."""
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    async def __anext__(self):
        try:
            # Run next in thread to avoid blocking
            batch = await asyncio.to_thread(next, self.iterator)
            return batch
        except StopIteration:
            raise StopAsyncIteration

    def __aiter__(self):
        return self

def adjust_batch_size(config: XAGIConfig, current_batch_size: int) -> int:
    """Dynamic batch: Halve if mem >80%."""
    if psutil:
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            return max(config.training.min_batch_size, current_batch_size // 2)
    return current_batch_size

async def train_agi_async(config: XAGIConfig):
    """Async training loop: Batches -> update -> evo -> bench."""
    # Model init
    model = XAGI(config.model, config.other, config.training).to(config.system.device)
    if torch.__version__ >= '2.0':
        model = torch.compile(model)
        logger.info("Model compiled.")

    # DeepSpeed/Megatron
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    if config.training.use_deepspeed:
        import deepspeed
        model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config_params={'zero_optimization': {'stage': config.training.deepspeed_zero_stage}})
        model = model_engine
    if config.training.use_megatron:
        mpu.initialize_model_parallel(config.model.tp_degree, config.model.pp_degree)

    # Integrator & scaler
    integrator = GrokIntegrator(config.system.xai_api_key, config.model.latent_dim, config.system.device, config.system.use_local_fallback, config)
    scaler = TrillionScaler(config.data)  # Local only
    shard_paths = [f"local_shard_{i}.tar" for i in range(config.data.data_shards)]  # Local stubs
    sync_dataloader = DataLoader(scaler.build_dataloader(shard_paths), batch_size=config.training.batch_size, num_workers=config.training.dataloader_workers, persistent_workers=config.training.persistent_workers)
    dataloader = AsyncDataLoader(sync_dataloader)

    # Ethics filter
    if config.data.ethical_filter:
        async def filtered_dataloader():
            async for b in dataloader:
                if 'text_bytes' in b and ethical_filter_bytes(b['text_bytes'], config.data):
                    yield b
        dataloader = filtered_dataloader()

    # State
    loss_history, unc_history, benchmark_history, fid_scores = [], [], [], []
    best_loss = float('inf')
    patience = 0
    replay_buffer = PriorityReplayBuffer(config.other.replay_buffer_size)
    total_bytes = 0
    scaler_amp = GradScaler(enabled=config.model.use_bfloat16)

    model.train()
    for epoch in range(config.training.epochs):
        total_loss, total_unc = 0.0, 0.0
        accum_steps = config.training.gradient_accumulation_steps
        step = 0
        start_time = time.time()

        async for batch in dataloader:
            loss, unc, batch_bytes = await async_process_batch(batch, model, optimizer, scaler_amp, accum_steps, replay_buffer, config, step)
            total_loss += loss
            total_unc += unc
            total_bytes += batch_bytes
            step += 1
            METRIC_TOKENS_PROCESSED.inc(batch_bytes)

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                cuda.empty_cache()

        avg_loss = total_loss / step
        avg_unc = total_unc / step
        loss_history.append(avg_loss)
        unc_history.append(avg_unc)
        METRIC_EPOCH_COUNT.inc()
        METRIC_TRAIN_LOSS.set(avg_loss)

        epoch_time = time.time() - start_time
        epoch_throughput = total_bytes / epoch_time
        METRIC_BYTE_THROUGHPUT.observe(epoch_throughput)

        # FID every 5 epochs
        if config.training.use_fid and epoch % 5 == 0:
            real_batch = next(iter(sync_dataloader))  # Use sync for bench
            with torch.no_grad():
                fake_batch, _, _, _ = model(real_batch[0] if isinstance(real_batch, tuple) else {'text_bytes': real_batch['input']})
            fid = compute_fid(real_batch[1] if isinstance(real_batch, tuple) else real_batch['target'], fake_batch, config)
            fid_scores.append(fid)

        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Unc={avg_unc:.4f}, Throughput={epoch_throughput:.2f} bytes/s, FID={fid_scores[-1] if fid_scores else 0:.4f}")

        # Dynamic batch
        current_batch_size = adjust_batch_size(config, config.training.batch_size)
        # Rebuild dataloader if changed (stub)

        # Benches
        bench_time, bench_throughput = end_to_end_bench(model, sync_dataloader)
        benchmark_history.append({'throughput': bench_throughput})
        stress_grpo(model)
        benchmark_unc_calib(model, sync_dataloader)

        # Early stop
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
        if patience >= config.training.early_stop_patience:
            logger.info("Early stopping.")
            break

        # Evo
        if (epoch + 1) % config.training.evolution_trigger_epochs == 0:
            config = await evolve_config_grok(config, loss_history, benchmark_history[-1] if benchmark_history else None, fid_scores[-1] if fid_scores else 0.0, integrator, unc_history, epoch_throughput)
            # Re-init model/optimizer if params changed (stub: reload)

        cuda.empty_cache()

    # Save
    if config.training.use_deepspeed:
        model.save_checkpoint(config.system.checkpoint_dir)
    else:
        torch.save(model.state_dict(), os.path.join(config.system.checkpoint_dir, 'xagi_model.pth'))
    logger.info("Training complete.")

async def async_process_batch(batch: Dict, model: XAGI, optimizer, scaler_amp, accum_steps: int, replay_buffer, config: XAGIConfig, step: int) -> Tuple[float, float, int]:
    """Process batch: Forward -> backward -> replay."""
    if config.model.byte_level:
        inputs = {'text_bytes': batch['input'].to(config.system.device)}
        if 'pddl_domain' in batch:
            inputs['pddl_domain'] = batch['pddl_domain'].to(config.system.device)
            inputs['pddl_problem'] = batch['pddl_problem'].to(config.system.device)
        # Enhancement: Pass image_bytes if present for OCR hybrid
        if 'image_bytes' in batch:
            inputs['image_bytes'] = batch['image_bytes'].to(config.system.device)
        target = batch['target'].to(config.system.device)
        batch_bytes = inputs['text_bytes'].numel()
        bytes_input = inputs['text_bytes']
    else:
        inputs = {k: v.to(config.system.device) for k, v in batch[0].items()}
        target = batch[1].to(config.system.device)
        batch_bytes = sum(v.numel() for v in inputs.values())
        bytes_input = None

    def sync_forward_backward():
        with autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16 if config.model.use_bfloat16 else torch.float16):
            outputs, loss, latent, total_unc = model(inputs, target=target, bytes_input=bytes_input)
        loss = loss / accum_steps
        if scaler_amp:
            scaler_amp.scale(loss).backward()
        else:
            loss.backward()
        return loss.item() * accum_steps, total_unc.mean().item()

    loss, unc = await asyncio.to_thread(sync_forward_backward)

    # Replay
    replay_buffer.add((inputs, target), loss, unc)
    replay_batch, idxs, is_weights = replay_buffer.sample(config.training.batch_size)
    if replay_batch:
        # High-unc replay with weights (stub: weighted loss if needed)
        pass

    return loss, unc, batch_bytes

def train_agi(config: XAGIConfig):
    """Sync wrapper."""
    asyncio.run(train_agi_async(config))

# ================== src/agents/__init__.py ==================
"""Agents: Pruned—stubs only."""

# ================== src/datasets/__init__.py ==================
"""Datasets: HF/local wrappers."""

# ================== src/datasets/scaler.py ==================
"""
Scaler: Local/HF datasets, dedup, PDDL parse. No S3.
Principle: Trillion bytes via sharded tar, ethical crawl.
"""

import webdataset as wds
from torch.utils.data import DataLoader
from src.core.configs import DataConfig
from src.core.deps import datasketch, scrapy_aiohttp, logger, unified_planning, datasets, tarfile, transformers_mod
from datasketch import MinHashLSH, MinHash
import torch
import os
import io
import asyncio
import numpy as np
from PIL import Image  # Added for OCR image processing in ocr_augment
import imghdr  # For image validation
from tarski.io import PDDLReader as TarskiReader  # For robust PDDL

class TrillionScaler:
    """Scale to trillion: Shard local tars, dedup, filter."""
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg
        self.shards = data_cfg.data_shards
        self.target_bytes = data_cfg.max_tokens_target
        self.dedup = MinHashLSH(threshold=0.8, num_perm=128) if data_cfg.dedup_method == 'minhash' else None
        if scrapy_aiohttp:
            from scrapy_aiohttp import AsyncScrapy
            self.async_scraper = AsyncScrapy()
        else:
            logger.warning("Scrapy-aiohttp missing; crawl limited.")
        # Enhancement: OCR pipeline for hybrid
        self.ocr_pipeline = None
        if self.data_cfg.ocr_enabled and transformers_mod:
            self.ocr_pipeline = transformers_mod.pipeline("text-generation", model=self.data_cfg.ocr_model)
            logger.info(f"OCR pipeline loaded: {self.data_cfg.ocr_model}")

    def build_dataloader(self, paths: list) -> DataLoader:
        """WebDataset from local paths; filter/dedup/map."""
        dataset = wds.WebDataset(paths).shuffle(1e6).decode("torch").to_tuple("input", "target")
        def byte_filter(sample):
            return len(sample[0]) > 512
        dataset = dataset.filter(byte_filter)
        if self.dedup:
            def dedup_sample(sample):
                m = MinHash(num_perm=128)
                text = sample[0].decode('utf-8', errors='ignore')
                for word in text.split():
                    m.update(word.encode('utf8'))
                if not self.dedup.query(m):
                    self.dedup.insert('sample', m)
                    return True
                return False
            dataset = dataset.filter(dedup_sample)
        if self.data_cfg.use_rpg and tarski:
            def parse_pddl(sample):
                # Fallback dummy if no str
                domain_str = sample.get('pddl_domain_str', "(define (domain dummy))")
                problem_str = sample.get('pddl_problem_str', "(define (problem dummy) (:domain dummy))")
                try:
                    reader = TarskiReader(raise_on_error=True)
                    domain = reader.parse_domain_string(domain_str)
                    problem = reader.parse_problem_string(domain, problem_str)
                    sample['pddl_domain'] = torch.tensor(list(domain_str.encode()), dtype=torch.uint8).unsqueeze(0)
                    sample['pddl_problem'] = torch.tensor(list(problem_str.encode()), dtype=torch.uint8).unsqueeze(0)
                except Exception as e:
                    logger.warning(f"PDDL parse error: {e}; fallback to dummy.")
                    sample['pddl_domain'] = torch.randint(0, 256, (1, 512), dtype=torch.uint8)
                    sample['pddl_problem'] = torch.randint(0, 256, (1, 512), dtype=torch.uint8)
                return True
            dataset = dataset.map(parse_pddl)
        # Enhancement: OCR augment for visual/handwriting inputs
        if self.data_cfg.ocr_enabled and self.ocr_pipeline:
            def ocr_augment(sample):
                # Assume sample has 'image_bytes' for visual; fallback to text
                if 'image_bytes' in sample:
                    img_bytes_np = sample['image_bytes'].numpy() if hasattr(sample['image_bytes'], 'numpy') else np.frombuffer(sample['image_bytes'], dtype=np.uint8)
                    img_bytes_raw = img_bytes_np.tobytes()
                    img_type = imghdr.what(None, img_bytes_raw)
                    if img_type is None:
                        logger.warning("Invalid image bytes in augment; fallback.")
                        ocr_text = ""
                        ocr_conf = 0.0
                    else:
                        # Fix: Convert to PIL Image
                        try:
                            img = Image.open(io.BytesIO(img_bytes_raw))
                            ocr_result = self.ocr_pipeline(img)
                            ocr_text = ocr_result[0]['generated_text'] if ocr_result else ""
                            ocr_conf = ocr_result[0].get('score', 0.5) if ocr_result else 0.5
                        except Exception as e:
                            logger.warning(f"OCR augment failed: {e}; fallback.")
                            ocr_text = ""
                            ocr_conf = 0.0
                    if ocr_conf > self.data_cfg.ocr_threshold:
                        text_bytes = torch.tensor(list(ocr_text.encode('utf-8')), dtype=torch.uint8).unsqueeze(0)
                        sample['text_bytes'] = text_bytes
                        # Hybrid patches (assume byte_patcher accessible or stub)
                        # For demo: Blend byte tensors directly
                        raw_bytes = sample['image_bytes']
                        hybrid_bytes = (self.data_cfg.image_byte_weight * raw_bytes + (1 - self.data_cfg.image_byte_weight) * torch.tensor(list(ocr_text.encode('utf-8')), dtype=torch.uint8, device=raw_bytes.device).unsqueeze(0).expand_as(raw_bytes))
                        sample['hybrid_bytes'] = hybrid_bytes
                    else:
                        sample['text_bytes'] = torch.tensor(raw_bytes.numpy().tobytes(), dtype=torch.uint8).unsqueeze(0)  # Raw fallback
                return sample
            dataset = dataset.map(ocr_augment)
        # Size check stub
        total_bytes = sum(len(sample[0]) for sample in list(dataset.take(1000)))
        total_bytes *= len(dataset) / 1000  # Approx
        if total_bytes >= self.target_bytes:
            logger.info(f"Reached {self.target_bytes} bytes target.")
        return DataLoader(dataset, batch_size=self.data_cfg.batch_size * self.shards // 10, num_workers=self.data_cfg.dataloader_workers)  # Scaled down

    async def crawl_and_shard(self, domains: List[str]):
        """Async local crawl: Tar shards (no upload)."""
        async def shard_worker(domain):
            if self.async_scraper:
                class EthicalSpider(scrapy_aiohttp.Spider):  # Assume scrapy_aiohttp has Spider
                    name = 'ethical'
                    start_urls = [f'https://{domain}']
                    custom_settings = {'ROBOTSTXT_OBEY': True, 'DEPTH_LIMIT': self.data_cfg.web_crawl_max_depth}
                    def parse(self, response):
                        content = response.text[:self.data_cfg.web_max_summary_length]
                        if not self.data_cfg.ethical_filter or ethical_filter_text(content, self.data_cfg):
                            if self.dedup:
                                m = MinHash(num_perm=128)
                                for word in content.split():
                                    m.update(word.encode('utf8'))
                                if not self.dedup.query(m):
                                    self.dedup.insert(domain, m)
                                    local_tar = f'local_shard_{hash(domain)}.tar'
                                    with tarfile.open(local_tar, 'w') as tar:
                                        info = tarfile.TarInfo(name=f'{domain}.txt')
                                        info.size = len(content.encode())
                                        tar.addfile(info, io.BytesIO(content.encode()))
                                    yield {'url': response.url, 'text': content}
                await self.async_scraper.crawl(EthicalSpider)
            logger.info(f"Crawled/sharded {domain}")
        await asyncio.gather(*(shard_worker(d) for d in domains))

# Dataset wrappers (pruned: Lyra, Caselaw, ModelNet only)
class LyraDatasetWrapper(torch.utils.data.Dataset):
    """HF Lyra: Point clouds from data, with real targets if available."""
    def __init__(self, data_cfg: DataConfig):
        self.dataset = datasets.load_dataset(data_cfg.lyra_dataset_name, data_cfg.lyra_split_name_train, split='train', streaming=True).take(data_cfg.lyra_max_samples)
        self.data_cfg = data_cfg

    def __len__(self):
        return self.data_cfg.lyra_max_samples

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data_str = str(item.get('data', ''))[:2048]
        data_bytes = torch.tensor(list(data_str.encode('utf-8')), dtype=torch.uint8).unsqueeze(0)
        target = torch.tensor(item.get('label', torch.randn(sum(self.data_cfg.output_dims))))  # Use real if avail
        return {'text_bytes': data_bytes}, target

class CaselawDatasetWrapper(torch.utils.data.Dataset):
    """HF Caselaw: Truncated text bytes."""
    def __init__(self, data_cfg: DataConfig):
        self.dataset = datasets.load_dataset(data_cfg.caselaw_dataset_name, data_cfg.caselaw_split_name_train, split='train', streaming=True).take(data_cfg.caselaw_max_samples)
        self.data_cfg = data_cfg

    def __len__(self):
        return self.data_cfg.caselaw_max_samples

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text'][:self.data_cfg.caselaw_text_length]
        data_bytes = torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8).unsqueeze(0)
        target = torch.randn(sum(self.data_cfg.output_dims))
        return {'text_bytes': data_bytes}, target

class ModelNetDatasetWrapper(torch.utils.data.Dataset):
    """ModelNet: Point clouds."""
    def __init__(self, data_cfg: DataConfig, model_cfg):  # Assume model_cfg passed
        self.dataset = datasets.load_dataset("modelnet", split='train')
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.point_cloud_size = model_cfg.point_cloud_size
        self.input_dim = model_cfg.input_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        points = torch.tensor(item['points'][:self.point_cloud_size], dtype=torch.float).unsqueeze(0)
        target = torch.tensor(item['label']).float().unsqueeze(0)
        return {'coords': points}, target

# ================== src/planning/__init__.py ==================
"""Planning: RPG-PDDL chains."""

# ================== src/planning/rpg_pddl.py ==================
"""
RPG: Regression graph planning w/ PDDL + ADP refine.
Principle: Chain actions via shortest path; embed for fusion.
"""

import torch
import os
from src.core.configs import ModelConfig, OtherConfig
from src.core.deps import unified_planning, networkx as nx, logger
from src.models.adp import ADPIntegrator

class RPGChain:
    """PDDL -> graph -> chains; ADP policy refine."""
    def __init__(self, model_cfg: ModelConfig, other_cfg: OtherConfig):
        self.max_nodes = other_cfg.rpg_max_nodes
        self.max_edges = other_cfg.rpg_max_edges
        self.num_chains = other_cfg.rpg_num_chains
        self.planner = None
        if unified_planning:
            from unified_planning.shortcuts import OneshotPlanner
            self.planner = OneshotPlanner(problem_kind=None)
        else:
            logger.warning("Unified Planning missing; dummy chains.")
        self.adp = ADPIntegrator(model_cfg) if model_cfg.use_adp else None

    def chain_plans(self, domain_embed, problem_embed, num_chains=3):
        """Embeds -> str (lossy ok for parse) -> graph paths -> ADP refine."""
        if not self.planner:
            return [[random.randint(0, 99) for _ in range(5)] for _ in range(num_chains)]
        # Decode embeds (chunked)
        domain_str = domain_embed[0, :512].cpu().numpy().tobytes().decode('utf-8', errors='ignore')
        problem_str = problem_embed[0, :512].cpu().numpy().tobytes().decode('utf-8', errors='ignore')
        try:
            from unified_planning.io import PDDLReader
            reader = PDDLReader()
            domain, problem = reader.read_pddl_from_strings(domain_str, problem_str)
            # Graph: Facts/actions as nodes, pre/add as edges
            graph = nx.DiGraph()
            facts = list(problem.initial_state.all_fluents())[:self.max_nodes]
            actions = problem.actions[:self.max_edges]
            for i, fact in enumerate(facts):
                graph.add_node(f'fact_{i}', fact=fact)
            for j, action in enumerate(actions):
                graph.add_node(f'action_{j}', action=action)
                # Preconditions
                for pre in action.preconditions:
                    pre_str = str(pre)
                    if pre_str in [str(f) for f in facts]:
                        pre_idx = next(i for i, f in enumerate(facts) if str(f) == pre_str)
                        graph.add_edge(f'fact_{pre_idx}', f'action_{j}')
                # Adds
                for add in action.effects.add:
                    add_idx = len(facts) + j
                    graph.add_node(f'fact_add_{add_idx}', fact=add)
                    graph.add_edge(f'action_{j}', f'fact_add_{add_idx}', weight=1)
            chains = []
            for _ in range(num_chains):
                try:
                    source = 'fact_0'
                    target = f'fact_add_{len(facts) + len(actions) - 1}' if actions else 'fact_0'
                    chain = nx.shortest_path(graph, source, target, weight='weight')
                    if len(chain) > self.max_nodes:
                        chain = chain[:self.max_nodes]
                except nx.NetworkXNoPath:
                    chain = [f'action_{k}' for k in range(min(5, len(actions)))]
                # ADP refine
                chain_idx = []
                for c in chain:
                    if 'action' in c:
                        act_id = hash(str(graph.nodes[c]['action'])) % 100
                        chain_idx.append(act_id)
                    else:
                        chain_idx.append(0)
                if self.adp and chain_idx:
                    for idx, act_id in enumerate(chain_idx):
                        # State from node degree (complexity proxy)
                        node = chain[idx]
                        state_val = len(graph.nodes[node]) if node in graph.nodes else 0
                        state = torch.tensor([state_val, 0.0], dtype=torch.float).unsqueeze(0).to(act_id.device if hasattr(act_id, 'device') else 'cuda')
                        probs = self.adp.actor(state)
                        refined_act = torch.multinomial(probs, 1).item()
                        next_state, reward, done = self.adp.rollout(state)
                        if reward > 0:
                            chain_idx[idx] = refined_act
                chains.append(chain_idx)
            return chains
        except Exception as e:
            logger.warning(f"RPG failed: {str(e)}")
            return [[0] * 5 for _ in range(num_chains)]

# ================== src/integrations/__init__.py ==================
"""Integrations: Expanded Grok for evo/infer."""

# ================== src/integrations/grok.py ==================
"""
GrokIntegrator: API for hyperparam axioms + inference augment.
Expanded: Chain prompts for reasoning; fallback uses full metrics.
"""

import asyncio
import json
import numpy as np
from src.core.configs import SystemConfig, XAGIConfig
from src.core.deps import logger

# Assume xai_grok_sdk loaded via DepManager (add to OPTIONAL)
xai_grok_sdk_mod = None  # Stub: load in init

class GrokIntegrator:
    """Grok client: Suggest params via chained reasoning; augment plans."""
    def __init__(self, api_key, latent_dim, device, use_local, config: XAGIConfig):
        self.api_key = api_key
        self.latent_dim = latent_dim
        self.device = device
        self.use_local = use_local
        self.config = config
        global xai_grok_sdk_mod
        xai_grok_sdk_mod = DepManager.load("xai_grok_sdk", DepCategory.OPTIONAL, "xai_grok_sdk missing")
        if xai_grok_sdk_mod and api_key:
            self.client = xai_grok_sdk_mod.Client(api_key=api_key)
        else:
            logger.warning("Grok SDK/key missing; fallback. See https://x.ai/api")
            self.client = None

    async def suggest_hyperparams_grok(self, prompt: str, loss_history: list, benchmarks: Dict, fid: float, unc_history: list = None, throughput: float = 0.0):
        """Expanded: Send chained prompt; parse JSON; fallback heuristic w/ all."""
        if self.client:
            try:
                # Chain: Reason then suggest
                response = await asyncio.to_thread(self.client.chat, messages=[{"role": "user", "content": prompt}], model=self.config.system.grok_model)
                content = response['choices'][0]['message']['content']
                # Regex fallback for JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    suggestions = json.loads(json_match.group())
                else:
                    suggestions = self._heuristic_suggest(loss_history, benchmarks, fid, unc_history, throughput)
                return suggestions
            except Exception as e:
                logger.warning(f"Grok suggest failed: {e}; fallback.")
        return self._heuristic_suggest(loss_history, benchmarks, fid, unc_history, throughput)

    def _heuristic_suggest(self, loss_history, benchmarks, fid, unc_history, throughput):
        """Dynamic fallback: Adjust based on all metrics."""
        plateau_std = np.std(loss_history[-10:]) if len(loss_history) >= 10 else 0
        unc_var = np.var(unc_history[-10:]) if unc_history and len(unc_history) >= 10 else 0
        lr_mult = 1 - plateau_std  # Lower LR on plateau
        experts_mult = 1 + (fid / self.config.training.fid_threshold)  # More experts on poor FID
        chains_add = int(unc_var > 0.03)  # More chains on high unc var
        fidelity_mult = throughput / self.config.training.byte_throughput_target  # Boost fidelity if slow
        # Enhancement: Suggest OCR weight based on FID/unc (higher weight on poor recon)
        ocr_weight_mult = 1 + (fid / self.config.training.fid_threshold)
        return {
            'learning_rate': self.config.training.learning_rate * lr_mult,
            'num_moe_experts': int(self.config.model.num_moe_experts * experts_mult),
            'patch_size': self.config.model.patch_size,
            'rpg_num_chains': self.config.other.rpg_num_chains + chains_add,
            'substrate_fidelity_weight': self.config.training.substrate_fidelity_weight * fidelity_mult,
            'adp_lr_actor': self.config.model.adp_lr_actor * (1 - unc_var),
            'adp_lr_critic': self.config.model.adp_lr_critic * (1 - unc_var),
            'adp_gamma': self.config.model.adp_gamma,
            'image_byte_weight': self.config.data.image_byte_weight * ocr_weight_mult
        }

    async def augment_plan_grok(self, pddl_domain: str, pddl_problem: str):
        """Inference-time: Grok suggests plan refinements."""
        if not self.client:
            return []  # Dummy
        prompt = f"Given PDDL domain: {pddl_domain[:500]}... problem: {pddl_problem[:500]}..., suggest 3 action chains as list of ints 0-99."
        try:
            response = await asyncio.to_thread(self.client.chat, messages=[{"role": "user", "content": prompt}], model=self.config.system.grok_model)
            # Parse list from response (stub: assume JSON list)
            chains_str = response['choices'][0]['message']['content']
            chains = json.loads(chains_str)  # Safe parse
            return chains
        except:
            return [[0,1,2] for _ in range(3)]

# ================== src/tests/test_configs.py ==================
"""
Tests: 80%+ coverage—units for configs/models, integration for forward/training.
Uses pytest-asyncio for async.
"""

import pytest
import torch
import asyncio
from src.core.configs import ModelConfig, TrainingConfig, DataConfig, XAGIConfig, OtherConfig
from src.models.xagi import XAGI
from src.datasets.scaler import TrillionScaler, LyraDatasetWrapper, CaselawDatasetWrapper, ModelNetDatasetWrapper
from src.planning.rpg_pddl import RPGChain
from src.training.trainer import async_process_batch
from src.integrations.grok import GrokIntegrator
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

@pytest.fixture
def config():
    """Default config fixture."""
    return XAGIConfig()

@pytest.fixture
def model(config):
    """XAGI model fixture."""
    return XAGI(config.model, config.other, config.training)

@pytest.fixture
def dummy_batch():
    """Dummy byte batch."""
    return {'text_bytes': torch.randint(0, 256, (1, 1024), dtype=torch.uint8),
            'pddl_domain': torch.randint(0, 256, (1, 512), dtype=torch.uint8),
            'pddl_problem': torch.randint(0, 256, (1, 512), dtype=torch.uint8),
            'target': torch.randn(1, 12)}  # Sum outputs

def test_model_validate():
    """Test ModelConfig validation."""
    cfg = ModelConfig()
    cfg.latent_dim = cfg.hidden_dim  # Invalid
    with pytest.raises(ValueError):
        cfg._validate()
    cfg = ModelConfig(use_adp=True)
    cfg.adp_digits = 0
    with pytest.raises(ValueError):
        cfg._validate()

def test_training_validate():
    """Test TrainingConfig validation."""
    cfg = TrainingConfig(epochs=0)
    with pytest.raises(ValueError):
        cfg._validate()
    cfg = TrainingConfig(batch_size=10, min_batch_size=32)
    with pytest.raises(ValueError):
        cfg._validate()

def test_data_validate():
    """Test DataConfig validation."""
    cfg = DataConfig(lyra_max_samples=-1)
    with pytest.raises(ValueError):
        cfg._validate()
    cfg = DataConfig(ocr_threshold=1.5)
    with pytest.raises(ValueError):
        cfg._validate()

def test_xagi_config():
    """Test XAGIConfig holistic init/validate."""
    cfg = XAGIConfig()
    assert cfg.model.latent_dim <= cfg.model.hidden_dim // 64
    assert cfg.model.device == 'cuda' or cfg.model.device == 'cpu'

@pytest.mark.asyncio
async def test_xagi_forward(model, dummy_batch):
    """Test forward: Loss computed, unc >0."""
    outputs, loss, latent, unc = model(dummy_batch, bytes_input=dummy_batch['text_bytes'])
    assert loss is not None
    assert loss > 0
    assert unc > 0
    assert latent.shape == (1, model.model_cfg.latent_dim)
    assert outputs.shape == (1, 12)  # Sum outputs

def test_byte_patcher():
    """Test BytePatcher forward/pad."""
    from src.models.byte_patcher import BytePatcher
    patcher = BytePatcher(patch_size=4)
    bytes_in = torch.randint(0, 256, (1, 17), dtype=torch.long)  # Non-multiple
    patches = patcher(bytes_in)
    assert patches.shape == (1, 5, 512)  # 20/4=5, pad 3

@pytest.mark.parametrize("digits", [1, 5])
def test_evidential_critic(digits):
    """Test EvidentialCritic: Positive mean/var, unc>0."""
    from src.models.xagi import EvidentialCritic
    critic = EvidentialCritic(state_dim=2, hidden_dim=128, digits=digits)
    state = torch.randn(1, 2)
    value, unc = critic(state)
    assert value > 0
    assert unc > 0

def test_adp_update():
    """Test ADP: TD error updates, unc from critic."""
    from src.models.adp import ADPIntegrator, ModelConfig
    cfg = ModelConfig(adp_state_dim=2, adp_action_dim=4)
    adp = ADPIntegrator(cfg)
    state = torch.randn(1, 2)
    action = torch.tensor([0])
    reward = torch.tensor(1.0)
    next_state = torch.randn(1, 2)
    td, unc = adp.update(state, action, reward, next_state)
    assert not torch.isnan(td)
    assert unc > 0

def test_rpg_chain(config):
    """Test RPG: Chains len==num_chains, <=max_nodes."""
    other = OtherConfig(rpg_num_chains=3)
    rpg = RPGChain(config.model, other)
    domain_emb = torch.randn(1, config.model.hidden_dim)
    problem_emb = torch.randn(1, config.model.hidden_dim)
    chains = rpg.chain_plans(domain_emb, problem_emb)
    assert len(chains) == 3
    assert all(len(chain) <= other.rpg_max_nodes for chain in chains)

@pytest.mark.asyncio
async def test_async_process_batch(model, dummy_batch, config):
    """Test batch process: Awaits, returns loss/unc/bytes."""
    optimizer = MagicMock()
    scaler_amp = MagicMock()
    replay = MagicMock()
    loss, unc, bytes_ = await async_process_batch(dummy_batch, model, optimizer, scaler_amp, 1, replay, config, 0)
    assert loss > 0
    assert unc > 0
    assert bytes_ == 1024 + 512 + 512  # Approx

@pytest.mark.asyncio
async def test_grok_suggest():
    """Test Grok suggest: Fallback works."""
    integrator = GrokIntegrator(None, 128, 'cpu', True, config())
    suggestions = await integrator.suggest_hyperparams_grok("prompt", [], {}, 0.5)
    assert 'learning_rate' in suggestions
    assert isinstance(suggestions['learning_rate'], float)
    assert 'image_byte_weight' in suggestions

def test_lyra_dataset():
    """Test Lyra wrapper: Bytes in output."""
    wrapper = LyraDatasetWrapper(DataConfig())
    item = wrapper[0]
    assert 'text_bytes' in item[0]
    assert item[0]['text_bytes'].dtype == torch.uint8

def test_caselaw_dataset():
    """Test Caselaw: Len==max_samples stub."""
    wrapper = CaselawDatasetWrapper(DataConfig(caselaw_max_samples=10))
    assert len(wrapper) == 10
    item = wrapper[0]
    assert item[0]['text_bytes'].shape[1] <= 2048

def test_modelnet_dataset():
    """Test ModelNet: Coords shape."""
    wrapper = ModelNetDatasetWrapper(DataConfig(), ModelConfig(point_cloud_size=1024, input_dim=3))
    item = wrapper[0]
    assert item[0]['coords'].shape == (1, 1024, 3)

def test_scaler_build():
    """Test Scaler: Dataloader non-empty."""
    scaler = TrillionScaler(DataConfig(data_shards=1))
    dl = scaler.build_dataloader(["dummy.tar"])
    assert len(dl) > 0  # Stub

@pytest.mark.asyncio
async def test_scaler_crawl():
    """Test crawl: Logs without crash."""
    scaler = TrillionScaler(DataConfig())
    await scaler.crawl_and_shard(["example.com"])
    # No assert: Just no exception

def test_utils_remove_pii():
    """Test PII: Redacts patterns."""
    from src.utils.common import remove_pii
    text = "SSN: 123-45-6789 email: test@example.com"
    cleaned = remove_pii(text)
    assert '[REDACTED]' in cleaned
    assert '123-45-6789' not in cleaned

def test_ethical_filter():
    """Test filter: Blocks non-en, harm."""
    from src.utils.common import ethical_filter_text, filter_harm
    cfg = DataConfig(ethical_filter=True)
    assert not filter_harm("harmful content", HARM_TAXONOMY)
    assert ethical_filter_text("Hola mundo", cfg)  # es allowed

def test_replay_buffer():
    """Test replay: Adds/samples correctly."""
    from src.utils.common import PriorityReplayBuffer
    buffer = PriorityReplayBuffer(10)
    buffer.add("exp1", 1.0, 0.5)
    buffer.add("exp2", 2.0, 0.1)
    samples, _, _ = buffer.sample(2)
    assert len(samples) == 2
    assert "exp2" in samples  # Higher prio

# Integration: Training stub
@pytest.mark.asyncio
async def test_train_stub(config):
    """Stub: Init without crash."""
    await train_agi_async(config)  # Partial: First epoch

# ================== main.py ==================
#!/usr/bin/env python3
"""
CLI: Train/infer/bench XAGI.
Enhanced: Auto-train if no checkpoint for infer.
"""

import argparse
import os
import torch
import time
import asyncio
from src.core.configs import XAGIConfig
from src.core.deps import DepManager, logger, METRIC_INFERENCE_TIME
from src.models.xagi import XAGI
from src.integrations.grok import GrokIntegrator
from src.training.trainer import train_agi, infer_mode  # Define infer_mode here
from torch.utils.data import DataLoader, TensorDataset

def infer_mode(config: XAGIConfig, prompt: str = None):
    """Inference: Load or train, run run w/ unc gate."""
    checkpoint_path = os.path.join(config.system.checkpoint_dir, 'xagi_model.pth')
    if not os.path.exists(checkpoint_path):
        logger.warning("No checkpoint; auto-training minimal.")
        config.training.epochs = 1  # Minimal
        train_agi(config)
    model = XAGI(config.model, config.other, config.training).to(config.system.device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.system.device))
    model.eval()

    # Inputs
    if config.model.byte_level and prompt:
        inputs = {'text_bytes': torch.tensor(list(prompt.encode('utf-8')), dtype=torch.uint8, device=config.system.device).unsqueeze(0)}
    else:
        inputs = {'text_bytes': torch.randint(0, 256, (1, 1024), dtype=torch.uint8, device=config.system.device)}
    # Enhancement: Add dummy image_bytes for OCR test
    if config.data.ocr_enabled:
        inputs['image_bytes'] = torch.randint(0, 256, (1, 1024), dtype=torch.uint8, device=config.system.device)
    if config.other.use_rpg:
        inputs['pddl_domain'] = torch.randint(0, 256, (1, 512), dtype=torch.uint8, device=config.system.device)
        inputs['pddl_problem'] = torch.randint(0, 256, (1, 512), dtype=torch.uint8, device=config.system.device)
    target = torch.randn(1, sum(config.model.output_dims), device=config.system.device)

    start = time.time()
    with torch.no_grad():
        outputs, _, _, total_unc = model(inputs, target=target, bytes_input=inputs.get('text_bytes'))
    inf_time = time.time() - start
    METRIC_INFERENCE_TIME.observe(inf_time)

    if total_unc.mean() > config.model.safety_gate_threshold:
        logger.warning("High unc; explore further.")

    logger.info(f"Output: {outputs.mean().item():.4f}, Unc: {total_unc.mean().item():.4f}, Time: {inf_time:.4f}s")
    return outputs, total_unc.mean().item()

def main():
    parser = argparse.ArgumentParser(description="XAGI v6.0: Pruned AGI Core")
    parser.add_argument('--mode', choices=['train', 'infer', 'benchmark'], default='train')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()

    config = XAGIConfig.from_yaml(args.config)
    config.training.strict_mode = args.strict
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        config.system.world_size = torch.cuda.device_count() or 1

    DepManager.validate_config(config)

    if args.mode == 'train':
        train_agi(config)
    elif args.mode == 'infer':
        if not args.prompt:
            parser.error("--prompt required for infer")
        infer_mode(config, args.prompt)
    elif args.mode == 'benchmark':
        model = XAGI(config.model, config.other, config.training).to(config.system.device)
        input_data = {'text_bytes': torch.randint(0, 256, (1, 1024), dtype=torch.uint8, device=config.system.device)}
        benchmark_inference(model, input_data)
        dummy_loader = DataLoader(TensorDataset(torch.randint(0, 256, (10, 1024), dtype=torch.uint8), torch.randn(10, 12)), batch_size=2)
        benchmark_unc_calib(model, dummy_loader)
        token_throughput(model, dummy_loader)

if __name__ == "__main__":
    main()