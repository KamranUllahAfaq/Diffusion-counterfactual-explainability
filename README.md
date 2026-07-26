# Diffusion Counterfactual Explainability

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research code for generating and evaluating image counterfactuals with diffusion-based
structural causal models. The project implements the abduction–action–prediction
workflow and supports spatial, semantic, and dynamically guided image mechanisms.

> **Research status:** this is experimental research software, not a production
> decision system. Model checkpoints and datasets are not distributed in this
> repository.

<p align="center">
  <img src="assets/demo.gif" alt="Counterfactual image generation demonstration" width="760">
</p>

## What is included

- Diffusion autoencoder components for conditional image reconstruction.
- Structural causal model abstractions for numerical and image mechanisms.
- Spatial, semantic, VAE, hierarchical VAE, VCI, and CTA mechanism variants.
- Experiment drivers for Morpho-MNIST, CelebA-HQ, and EMBED-style evaluations.
- Metrics for effectiveness, composition, reversibility, causal effects, LPIPS,
  and L1 distance.
- Slurm launch scripts for reproducible cluster experiments.

## Method at a glance

Counterfactual generation follows three steps:

1. **Abduction** — infer exogenous state from an observed image and its attributes.
2. **Action** — intervene on one or more causal variables.
3. **Prediction** — decode the retained exogenous state under the intervention.

Semantic abduction retains identity-relevant information in a learned latent
representation. Dynamic abduction can additionally align the denoising trajectory
to balance intervention effectiveness and reconstruction fidelity.

## Repository layout

```text
.
├── model/          # diffusion autoencoder, encoder, UNet, training and sampling
├── scm/            # structural causal models and image/numerical mechanisms
├── benchmarking/   # evaluation programs and metrics
├── scripts/        # training, evaluation, and classifier entry points
├── assets/         # documentation media and small illustrative samples
└── model/cfg/      # experiment configuration files
```

The installed Python namespace is `counterfactuals`. The source folders remain
top-level to keep the original experiment paths stable.

## Installation

The experiments were developed around Python 3.9 and CUDA-capable PyTorch.
Create an isolated environment before installing:

```bash
git clone https://github.com/KamranUllahAfaq/Diffusion-counterfactual-explainability.git
cd Diffusion-counterfactual-explainability
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the matching PyTorch build from the
[official selector](https://pytorch.org/get-started/locally/) when GPU support is
required. EMBED experiments also require the private or separately distributed
`mammo_artifacts` data module; it is intentionally not bundled here.

## Data and checkpoints

Configure local paths in the relevant file under `model/cfg/` before training.
You are responsible for obtaining each dataset under its own terms:

- [Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST)
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- EMBED (subject to its data-access agreement)

Do not commit datasets, model checkpoints, experiment logs, or generated batches.
The included `.gitignore` excludes their common locations and extensions.

## Running experiments

Inspect and edit a matching configuration first, for example
`model/cfg/64_model.yml`.

```bash
# Train a 64 × 64 model
counterfactual-train --size 64 --expn celeba64

# Evaluate a saved run
counterfactual-test --output outputs/celeba64 --model-ckpt last_ckpt.pth

# Show a benchmark's available options
python -m counterfactuals.benchmarking.celeba --help
```

The shell and Slurm files in the repository are experiment templates. Their
filesystem paths, partitions, and checkpoint identifiers are environment-specific
and should be reviewed before use.

## Evaluation

The benchmark suite separates intervention success from preservation quality:

| Dimension | Representative implementation |
| --- | --- |
| Intervention effectiveness | `benchmarking/effectiveness.py` |
| Composition consistency | `benchmarking/composition.py` |
| Reversibility | `benchmarking/reversibility.py` |
| Perceptual similarity | `benchmarking/lpips.py` |
| Pixel-space distance | `benchmarking/l1_distance.py` |

For meaningful comparisons, report dataset split, random seed, image resolution,
checkpoint, guidance scale, number of denoising steps, and mechanism variant.

## Reproducibility checklist

- Pin the environment and record the CUDA/PyTorch versions.
- Keep preprocessing and dataset splits identical across baselines.
- Set and report seeds for every benchmark.
- Evaluate the same observations and interventions for each mechanism.
- Store configuration files alongside checkpoints.
- Report both causal effectiveness and identity-preservation metrics.


## License

Distributed under the MIT License. See [LICENSE](LICENSE). Existing copyright and
license notices must remain intact in redistributions and substantial derivatives.

<div align="center">
Built by <a href="https://github.com/KamranUllahAfaq">Kamran Ullah Afaq</a>.
</div>
