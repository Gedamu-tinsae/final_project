# Group 1 Notebook -> `src` Full Migration Trace

Source notebook: `legacy/CS159FP_GROUP1/scripts/LLaVA_Public.ipynb`

## Notebook Inventory

- Total cells: 65
- code cells: 47
- markdown cells: 10
- raw cells: 8
- Code cells with saved outputs in file: 0

Interpretation: the notebook currently stores no pre-rendered outputs, so all inline outputs are generated only when you run it.

## Cell-By-Cell Trace

| Cell | Type | Summary | Migration Status | Destination | Notebook Story Value |
|---|---|---|---|---|---|
| 1 | markdown | `### The purpose of this notebook is to recreate the baseline visual instruction tuning pip` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 2 | raw | `import importlib.util` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 3 | markdown | `## This is how our baseline workload looks like` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 4 | raw | `Frozen CLIP outside trainer` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 5 | raw | `We will prepare 2 datasets for the two stages of training:` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 6 | raw | `TPU-Compatible Vision Encoder Loader` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 7 | raw | `Offline Stage:` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 8 | raw | `Image` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 9 | markdown | `## Project Structure` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 10 | raw | `CS159FinalProject/` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 11 | code | `import os` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 12 | code | `#UTILITY FUNCTIONS` | migrated logic | `src/training/memory.py` | high |
| 13 | markdown | `## Our Datasets` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 14 | code | `# Download the COCO 2017 dataset and annotations` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 15 | code | `# Match the COCO captions with the corresponding images and save in a new JSON file for st` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 16 | code | `# We are adding a simple instruction to the captions to make them more suitable for traini` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 17 | code | `# Loggin to HF` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 18 | code | `# ====== Sharding ======` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 19 | markdown | `# Get HF access to llama 3.2 1B Instruct model files` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 20 | code | `# We need to bypass the standard transformer loader as it would have a stuck progress bar ` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 21 | code | `# Export HF API Key to get access to the model files. Make sure to set HF_TOKEN in your en` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 22 | code | `! HF_HUB_ENABLE_HF_TRANSFER` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 23 | code | `# Choose the correct config for the model you downloaded. This should match the config.jso` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 24 | code | `llama_dir = "CS159FinalProject/temp/hf_models/Llama-3.2-1B-Instruct"` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 25 | code | `# WE can display the model architecture using nnx.display. This will show the layers and p` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 26 | code | `from transformers import AutoTokenizer` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 27 | markdown | `download CLIP locally` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 28 | code | `# Now the same deal for CLIP. We need the vision encoder to extract image features for the` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 29 | code | `from clip_helpers import build_clip_vision_tower` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 30 | code | `#Sanity Check` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 31 | code | `model = clip_bundle.model` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 32 | markdown | `### Sanity Check` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 33 | code | `# Check available JAX devices, essential for setting up mesh geometry` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 34 | code | `# Stage 1: Feature Alignment` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 35 | raw | `image` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 36 | code | `from flax import nnx` | migrated logic | `src/training/projector.py; src/training/tokenization.py` | medium |
| 37 | code | `def masked_cross_entropy_loss(logits, labels):` | migrated logic | `src/training/losses.py` | medium |
| 38 | code | `# During training, confirm these dimensions:` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 39 | markdown | `## Stage 1 Orchestration` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 40 | code | `#Compute stage 1 tokenized dataset` | migrated logic | `src/training/tokenization.py` | medium |
| 41 | code | `#Build tokenized dataset for stage 1 training. This will create a new JSON file where each` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 42 | code | `def make_clip_feature_fn(clip_bundle):` | migrated logic | `src/training/clip_features.py` | medium |
| 43 | code | `import json` | migrated logic | `src/training/clip_features.py` | high |
| 44 | code | `#Double check that we are using FlaxCLIPVISIONModel` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 45 | code | `# 3. Build trainable examples` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 46 | code | `# Check manifest is valid` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 47 | code | `# 4 Collator` | migrated logic | `src/training/batching.py` | medium |
| 48 | code | `#5 Build multimodal batch inside training loop, since we need to feed the raw images to th` | migrated logic | `src/training/multimodal.py` | medium |
| 49 | code | `# 6 Optimizer and training step` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 50 | code | `@nnx.jit # Use nnx.jit for compatibility with nnx modules` | migrated logic | `src/training/stage1.py` | medium |
| 51 | code | `import random` | migrated logic | `src/training/batching.py; src/training/stage1.py` | high |
| 52 | code | `run_stage1_training(` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 53 | markdown | `## Step 2: Multimodal Finetuning` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 54 | code | `#2. Serilize multimodal instruction data` | migrated logic | `src/training/tokenization.py` | medium |
| 55 | code | `#3: Multimodal input builder` | migrated logic | `src/training/multimodal.py` | medium |
| 56 | code | `tx = optax.adamw(` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 57 | code | `@jax.jit` | migrated logic | `src/training/stage2.py` | medium |
| 58 | code | `opt_state = tx.init({` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 59 | markdown | `## Stage 2 Orchestration` | narrative content | `Keep/port as narrative in workflow notebook` | high |
| 60 | code | `def build_tokenized_stage2_dataset(tokenizer, input_json, output_json, max_len=256):` | migrated logic | `src/training/tokenization.py` | medium |
| 61 | code | `# Reuse the pre-computed CLIP features and the manifest from stage 1, since the dataset is` | orchestration | `Keep as orchestration/runtime notebook cell` | high |
| 62 | code | `#build manifest for stage 2 training. This will be similar to the stage 1 manifest, but it` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 63 | code | `# Reuse the same collate function since the data format is the same, just with longer text` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |
| 64 | code | `def run_stage2_training(` | migrated logic | `src/training/stage2.py` | high |
| 65 | code | `with open(STAGE2_PROJECTOR_PATH, "wb") as f:` | orchestration | `Keep as orchestration/runtime notebook cell` | medium |

## Function/Class Coverage Check

All function/class definitions found in the original notebook are now present in `code/group1_baseline/src`.

Migrated definitions:
- `show_hbm_usage` -> `src/training/memory.py`
- `VisionProjector` -> `src/training/projector.py`
- `serialize_stage1_sample` -> `src/training/tokenization.py`
- `masked_cross_entropy_loss` -> `src/training/losses.py`
- `build_tokenized_stage1_dataset` -> `src/training/tokenization.py`
- `make_clip_feature_fn` -> `src/training/clip_features.py`
- `precompute_clip_features_jitted` -> `src/training/clip_features.py`
- `pad_list` -> `src/training/batching.py`
- `stage1_collate_fn` -> `src/training/batching.py`
- `make_multimodal_inputs` -> `src/training/multimodal.py`
- `train_step` -> `src/training/stage1.py`
- `iterate_minibatches` -> `src/training/batching.py`
- `run_stage1_training` -> `src/training/stage1.py`
- `serialize_stage2_sample` -> `src/training/tokenization.py`
- `train_step_stage2` -> `src/training/stage2.py`
- `build_tokenized_stage2_dataset` -> `src/training/tokenization.py`
- `run_stage2_training` -> `src/training/stage2.py`

## Data Download Coverage (Original Cell 14)

Original notebook cell 14 used inline shell commands (`wget`/`unzip`) to fetch COCO.
That logic is now accounted for in:

- `src/data_prep/acquire_coco.py`
  - `acquire_coco_2017(...)`
  - `coco_files_status(...)`

The workflow notebook now calls this `src` module directly during Stage 1 orchestration.

## Notebook-First Cells (Recommended to Run in Notebook for Storytelling)

These are better demonstrated interactively, even when backed by scripts:
- Environment/device sanity checks (imports, JAX devices, mesh, memory reporting).
- CLIP and model sanity checks (shape prints, architecture display).
- Stage orchestration calls with progress logs.
- Manifest/data spot checks and quick inspection cells.
