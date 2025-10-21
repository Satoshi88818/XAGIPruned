# XAGI: Bytes-First AGI Framework

**XAGI** (eXtensible AGI ) is a modular, bytes-grounded framework for building scalable AGI systems. From first principles: *All computation reduces to bytes as the unyielding substrate of reality*. XAGI fuses raw byte reconstruction with evidential uncertainty, symbolic planning (RPG-PDDL chains), adaptive RL (ADP), and hybrid OCR for grounded reasoning. No tokenizers—direct byte patching enables trillion-scale throughput while anchoring outputs in verifiable fidelity.

Designed for evidential calibration: Every prediction carries uncertainty, weighted by planning complexity and substrate fidelity. Evolve hyperparameters via Grok-augmented chains or DEAP fallbacks. Train on local shards or HF datasets; infer with safety gates.

> **Axiom**: Intelligence emerges from fusing substrate (bytes/coords) with hierarchy (MoE projectors) and deliberation (RPG/ADP). XAGI enforces this via composable configs and lazy deps.

## 🚀 Features

- **Bytes as Substrate**: Tokenizer-free patching; evidential decoders for calibrated recon (Dirichlet KL to uniform prior).
- **Planning Fusion**: RPG graphs from PDDL bytes → multi-chain embeds, refined by evidential ADP (GridWorld-tested).
- **Hybrid Multimodal**: OCR (TrOCR) blends handwriting/scans with raw bytes; weighted fusion tunable via evo.
- **Evolutionary Adaptation**: Grok-driven hyperparam search on loss plateaus/FID/unc var; DEAP fallback.
- **Scalable Training**: Async epochs with DeepSpeed/Megatron; priority replay; trillion-byte sharding (WebDataset).
- **Benchmarks Built-In**: E2E throughput, ECE calibration, FID; Prometheus metrics.
- **Ethical Grounding**: PII redaction (NLTK/regex), harm filters (Toxic-BERT), langdetect for en/es/fr.
- **Modular Extensibility**: Dataclass configs; lazy deps (fail gracefully); stubs for agents/quantum.

| Component | Principle | Key Param |
|-----------|-----------|-----------|
| **BytePatcher** | Direct byte embeds | `patch_size=4` |
| **NeocortexProjector** | Hierarchical MoE | `num_moe_experts=8` |
| **EvidentialDecoder** | Unc-aware recon | `edl_kl_weight=0.01` |
| **RPGChain** | Symbolic paths | `rpg_num_chains=3` |
| **ADPIntegrator** | RL refinement | `adp_gamma=0.99` |
| **TrillionScaler** | Ethical scaling | `data_shards=1000` |

## 🛠 Quick Start

1. **Clone & Install**:
   ```bash
   git clone https://github.com/Satoshi88818/XAGIPruned.git   cd xagi
   pip install pip-tools
   pip-compile requirements.in > requirements.txt
   pip install -r requirements.txt
   ```

2. **Config**: Edit `config.yaml` (or use defaults):
   ```yaml
   model:
     hidden_dim: 8192
     byte_level: true
   training:
     epochs: 1  # Minimal for test
   data:
     ocr_enabled: true
   system:
     xai_api_key: "your_key_here"  # For Grok evo; see https://x.ai/api
   ```

3. **Train Minimal**:
   ```bash
   python main.py --mode train --config config.yaml
   ```
   - Outputs to `./checkpoints/xagi_final.pth`.
   - Monitors: Loss, unc var, byte throughput (Prometheus @8000).

4. **Infer**:
   ```bash
   python main.py --mode infer --prompt "Ground bytes in reality" --config config.yaml
   ```
   - Returns recon mean + unc; gates if >0.5.

5. **Benchmark**:
   ```bash
   python main.py --mode benchmark --config config.yaml
   ```
   - Logs inference time, ECE, FID.

For distributed: Add `--distributed`; scales with `world_size`.

## 📐 Architecture

XAGI's forward pass: Substrate → Latent Fusion → Transformer → Evidential Recon.

```
Bytes/Coords/Image → BytePatcher/Encoder → Latent [128]
                  ↓
RPG (PDDL bytes → Chains) + ADP (Rollout) + OCR (Hybrid Blend)
                  ↓
NeocortexProjector (MoE Hierarchy) → TransformerDecoder (96 layers)
                  ↓
EvidentialDecoder → Recon [sum(output_dims)] + Unc (Dirichlet)
```

- **Substrate Fidelity**: MSE(recon embeds, latent) weighted 0.1.
- **Unc Fusion**: e_unc + adp_unc * rpg_unc * ocr_unc.
- **Evo Loop**: Triggers every epoch on plateau/FID>0.5; Grok suggests via chained axioms.

See `src/models/xagi.py` for core; `src/training/trainer.py` for async loop.

## 🔧 Datasets & Scaling

- **Built-In Wrappers**: Lyra (spatial), Caselaw (legal text), ModelNet (points).
- **TrillionScaler**: Async crawl → tar shards; minhash dedup; OCR augment for visuals.
- **Ethical Crawl**: Scrapy-aiohttp with harm/PII filters; shards to local (no S3).

Load: `scaler = TrillionScaler(config.data); dl = scaler.build_dataloader(shards)`.

## 🤝 Contributing

From axioms: Contributions must validate (configs, tests) and enhance substrate fidelity.

1. Fork & PR: Fix issues (e.g., "Add quantum stub") or features (e.g., "Federated rounds").
2. Tests: Run `pytest src/tests/` (>90% coverage); add for new params.
3. Docs: Update README/examples; use Sphinx for API.
4. Good First: Labelled in issues; e.g., "Tune ocr_threshold evo space".

Code of Conduct: [Contributor Covenant](https://www.contributor-covenant.org/). Questions? Open an issue.

## 📄 License

MIT License—use freely, attribute, build upon. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

- **xAI**: Grok integration for evo reasoning (API: https://x.ai/api).
- **Deps**: PyTorch, Transformers, Unified-Planning, DeepSpeed.
- **Inspiration**: First-principles from bytes-as-reality; evidential DL papers.

**Star/Fork if bytes-grounded AGI resonates!** Questions? [@yourhandle on X](https://x.com/yourhandle).

---

*Built with ❤️ by [James Squire], Oct 2025. Evolving axioms toward AGI.*
