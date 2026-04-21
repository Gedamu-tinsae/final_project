# Group 2 Migration Trace

Notebook source: `legacy/LLaVA_Public_Group2.ipynb`
Total cells traced: 94

## Unique Group2 Logic (Compared To Group1)

- Multi-variant stage2 datasets (`gemma`, `qwen`, `llama`)
- Shared quality pool + deterministic train/val image split
- Variant-specific stage2 tokenization/features/manifests
- Quality diagnostics, qualitative sample packs, pairwise judge request generation
- Quantity ablation bookkeeping and experiment result aggregation

Detailed cell-to-module mapping:
- `code/group2_baseline/notes/CELL_TO_MODULE_MAPPING.md`

## Cell-by-Cell Trace

- `000 | markdown | ### The purpose of this notebook is to recreate the baseline visual instruction tuning pipeline in the LLaVA paper`
- `001 | code     | import importlib.util  if importlib.util.find_spec('tensorflow') is None:   print("Installing required packages...")   %pip install -q dotenv   %pip install -q kagglehub   %pip ins...`
- `002 | markdown | ## This is how our baseline workload looks like`
- `003 | raw      | Frozen CLIP outside trainer Precompute visual embeddings offline Store embeddings in dataset Train Tunix only on:    visual embeddings + text`
- `004 | raw      | We will prepare 2 datasets for the two stages of training:  Stage 1 Dataset: Image + Caption (for projector alignment)  Stage 2 Dataset: Image + Instruction + Response (for multimo...`
- `005 | raw      | TPU-Compatible Vision Encoder Loader         ? Convert image ? visual embeddings         ? Project embeddings ? LLM token dim         ? Inject into Tunix/JAX LLM pipeline         ?...`
- `006 | raw      | Offline Stage: COCO/Instruction Dataset    ? Run CLIP encoder once    ? Save visual embeddings  Training Stage: Load embeddings    ? Project    ? Inject into Llama    ? Train Tunix`
- `007 | raw      | Image  ? FlaxCLIP Vision Tower  ? Vision Hidden States  ? Projection Layer  ? Projected Visual Tokens  ? Concat w/ Token Embeddings  ? Tunix Llama Decoder  ? Loss`
- `008 | markdown | ## Project Structure  ### Please create such directories`
- `009 | raw      | CS159FinalProject/ ? ??? data/ ?   ??? raw/ ?   ?   ??? images/ ?   ?   ??? annotations/ ?   ? ?   ??? processed/ ?   ?   ??? clip_embeddings/ ?   ?   ??? stage1_alignment/ ?   ?  ...`
- `010 | code     | import os from pathlib import Path  # Get the directory where the script is located #TODO: SCRIPT_DIR = "/home/cengjiehui/CS159FinalProject/scripts"  # The project root is one leve...`
- `011 | code     | #UTILITY FUNCTIONS def show_hbm_usage():   """Displays memory usage per device."""   fmt_size = functools.partial(humanize.naturalsize, binary=True)    for d in jax.local_devices()...`
- `012 | markdown | ## Our Datasets   ### For baseline, we are using COCO for both stage 1 and stage 2, instead of CC3M and LLaVA-instruct-158k, for simplicity and debugging`
- `013 | code     | %cd /home/cengjiehui/CS159FinalProject !pwd`
- `014 | code     | !python3 scripts/prepare_stage1_dataset.py !python3 scripts/convert_alignment_format.py`
- `015 | code     | # Download the COCO 2017 dataset and annotations  !mkdir -p /home/cengjiehui/CS159FinalProject/data/raw %cd /home/cengjiehui/CS159FinalProject/data/raw !pwd  !wget -c http://images...`
- `016 | code     | # Match the COCO captions with the corresponding images and save in a new JSON file for stage 1 training. !python /home/cengjiehui/CS159FinalProject/scripts/prepare_stage1_dataset....`
- `017 | code     | # We are adding a simple instruction to the captions to make them more suitable for training the projector in stage 1. # Make sure to execute this at project root directory (CS159F...`
- `018 | code     | # Loggin to HF  import os  import json import jax import jax.numpy as jnp import numpy as np  from PIL import Image import os import kagglehub  try:   from google.colab import user...`
- `019 | code     | # ====== Sharding ====== # Adjust mesh based on your TPU memory and model size. NUM_TPUS = len(jax.devices()) if NUM_TPUS == 8:   MESH_COUNTS = (2, 4) elif NUM_TPUS == 4:   MESH_CO...`
- `020 | markdown | # Get HF access to llama 3.2 1B Instruct model files  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`
- `021 | code     | !find /home/cengjiehui/CS159FinalProject -name model.py -o -name params.py`
- `022 | code     | %pip uninstall -y flax %pip install -q git+https://github.com/google/flax  # We need to bypass the standard transformer loader as it would have a stuck progress bar on TPU. Setting...`
- `023 | code     | # Export HF API Key to get access to the model files. Make sure to set HF_TOKEN in your environment variables before running this cell.  #TODO: Again, set HF API KEY os.environ["HF...`
- `024 | code     | ! HF_HUB_ENABLE_HF_TRANSFER  !hf download meta-llama/Llama-3.2-1B-Instruct \   --include "config.json" "generation_config.json" \             "tokenizer.json" "tokenizer_config.jso...`
- `025 | code     | # Choose the correct config for the model you downloaded. This should match the config.json file in the model directory. # Config is used to define the architecture of the model an...`
- `026 | code     | llama_dir = "CS159FinalProject/temp/hf_models/Llama-3.2-1B-Instruct" mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])) with mesh:     llama3_1b =...`
- `027 | code     | # WE can display the model architecture using nnx.display. This will show the layers and parameters of the model, which can be useful for debugging and understanding the model stru...`
- `028 | code     | from transformers import AutoTokenizer  tokenizer = AutoTokenizer.from_pretrained(     llama_dir,     local_files_only=True, )`
- `029 | markdown | download CLIP locally ? load FlaxCLIPModel from local files ? keep only vision tower for feature extraction ? return processor + model + vision hidden size`
- `030 | code     | # Now the same deal for CLIP. We need the vision encoder to extract image features for the projector training in stage 1, and also for the multimodal finetuning in stage 2. from cl...`
- `031 | code     | from clip_helpers import build_clip_vision_tower  clip_bundle = build_clip_vision_tower() print("CLIP hidden size:", clip_bundle.hidden_size) print("Image size:", clip_bundle.image...`
- `032 | code     | #Sanity Check from clip_helpers import load_clip_flax_local clip_bundle = load_clip_flax_local() print(type(clip_bundle.model))`
- `033 | code     | model = clip_bundle.model`
- `034 | markdown | ### Sanity Check from PIL import Image import requests from transformers import AutoProcessor, FlaxCLIPModel  model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32") ...`
- `035 | code     | # Check available JAX devices, essential for setting up mesh geometry jax.devices()`
- `036 | code     | # Stage 1: Feature Alignment`
- `037 | raw      | image ? CLIP vision tower (frozen) ? visual hidden states ? trainable projector W ? projected visual tokens ? prepend to text prompt ? Llama (frozen) ? predict caption tokens ? cro...`
- `038 | code     | from flax import nnx import jax import jax.numpy as jnp   class VisionProjector(nnx.Module):     def __init__(self, in_dim: int=768, out_dim: int = 2048, *, rngs: nnx.Rngs):       ...`
- `039 | code     | def masked_cross_entropy_loss(logits, labels):     # logits: [B, L, V]     # labels: [B, L]     #B: Batch size, L: Sequence length, V: Vocabulary size     shift_logits = logits[:, ...`
- `040 | code     | # During training, confirm these dimensions:   #vision_feats: [B, N_vis, D_clip] #vis_embeds: [B, N_vis, 2048] #text_embeds: [B, T, 2048] #input_embeds: [B, N_vis + T, 2048] #logit...`
- `041 | markdown | ## Stage 1 Orchestration  Now let?s wire the whole thing.  The best first orchestration is:  1. load processed Stage 1 JSON 2. tokenize text examples 3. precompute CLIP vision feat...`
- `042 | code     | #Compute stage 1 tokenized dataset  import json import os  def build_tokenized_stage1_dataset(tokenizer, input_json, output_json, max_len=128):     with open(input_json, "r") as f:...`
- `043 | code     | #Build tokenized dataset for stage 1 training. This will create a new JSON file where each sample contains the tokenized input IDs and labels, ready for training the vision project...`
- `044 | code     | def make_clip_feature_fn(clip_bundle):          '''Returns a function that takes in raw pixel values and returns the CLIP vision features.      This function is JIT-compiled for ef...`
- `045 | code     | import json import os import numpy as np import jax.numpy as jnp from PIL import Image   def precompute_clip_features_jitted(     clip_bundle,     tokenized_json,     image_root,  ...`
- `046 | code     | #Double check that we are using FlaxCLIPVISIONModel print(type(clip_bundle.model)) print(clip_bundle.model.__class__.__name__) print(type(clip_bundle.model.config)) print(clip_bund...`
- `047 | code     | # 3. Build trainable examples #Build the manifest for stage 1 training. This manifest will be used by the training loop to load the precomputed CLIP features and the tokenized text...`
- `048 | code     | # Check manifest is valid with open(MANIFEST_JSON, "r") as f:     sample_manifest = json.load(f)[0]      print("Manifest Sample Check:") print(f" - Vision Path: {sample_manifest['v...`
- `049 | code     | # 4 Collator  import numpy as np import ml_dtypes   def pad_list(x, length, pad_value):     return x + [pad_value] * (length - len(x))   def stage1_collate_fn(batch):     max_text_...`
- `050 | code     | #5 Build multimodal batch inside training loop, since we need to feed the raw images to the CLIP vision tower and get the projected embeddings on the fly for each batch.   import j...`
- `051 | code     | # 6 Optimizer and training step import optax from flax import nnx import jax #Freeze the CLIP vision tower and LLaMA model during stage 1 training, only train the projector.  tx = ...`
- `052 | code     | import jax import jax.numpy as jnp  def llama_forward_from_embeddings(llama_model, input_embeds, positions, attention_mask):     x = input_embeds     cache = None      for layer in...`
- `053 | code     | @nnx.jit # Use nnx.jit for compatibility with nnx modules def train_step(projector_state, opt_state, batch, projector_graphdef, llama_state, llama_graphdef):     def loss_fn(proj_s...`
- `054 | code     | import random import json import jax.numpy as jnp  # split the LLM once so stage 1 can use the frozen state correctly llama_graphdef, llama_state = nnx.split(llama3_1b)  def iterat...`
- `055 | code     | run_stage1_training(     manifest_json=MANIFEST_JSON,     num_epochs=1,     batch_size=4, )`
- `056 | code     | import os import pickle  PROJECTOR_STATE_PATH = "/home/cengjiehui/CS159FinalProject/train/checkpoints/projector_stage1.pkl" os.makedirs(os.path.dirname(PROJECTOR_STATE_PATH), exist...`
- `057 | markdown | ## Step 2: Multimodal Finetuning   1. initialize projector from Stage 1 2. keep CLIP frozen 3. keep projector trainable 4. unfreeze Llama 5. train on image + instruction + response...`
- `058 | code     | from pathlib import Path import json from collections import Counter  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  BASELINE_VARIANT =...`
- `059 | code     | POOL_REFERENCE_VARIANT = "gemma"   # temporary reference for now QUALITY_IMAGE_COUNT = 5000 VAL_IMAGE_COUNT = 1000 SPLIT_SEED = 42`
- `060 | code     | # 2. Serialize multimodal instruction data  import json import pickle from pathlib import Path from collections import Counter  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProje...`
- `061 | code     | from pathlib import Path import json import random  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  variant_image_sets = {}  for variant...`
- `062 | code     | from pathlib import Path import json  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  with open(STAGE2_ROOT / "shared_quality_pool.json"...`
- `063 | code     | import json from pathlib import Path  import numpy as np import jax.numpy as jnp from PIL import Image  STAGE2_FEATURE_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processe...`
- `064 | code     | import json from pathlib import Path   def build_stage2_manifest(variant, data_split="train"):     variant_dir, _, tokenized_json = resolve_stage2_paths(STAGE2_ROOT, variant, data_...`
- `065 | code     | prep_results = []  for variant in ALL_VARIANTS:     for split in ["train", "val"]:         print(f"\n===== PREPARING {variant} | {split} =====")         tok_result = tokenize_stage...`
- `066 | code     | # 3. Multimodal input builder  import jax.numpy as jnp  def make_multimodal_inputs(llama_model, projector, batch):     vision_feats = jnp.asarray(batch["vision_feats"], dtype=jnp.f...`
- `067 | code     | import optax import pickle  STAGE2_LR = 2e-5 STAGE2_WEIGHT_DECAY = 0.0   def init_stage2_run():     """     Reset Stage 2 to a clean, comparable starting point.      Returns:      ...`
- `068 | code     | def llama_forward_from_embeddings_stage2(     llama_model,     input_embeds,     positions,     attention_mask,     cache=None, ):     x = input_embeds     running_cache = cache   ...`
- `069 | code     | @nnx.jit def train_step_stage2(     projector_state,     llama_state,     opt_state,     batch, ):     def loss_fn(projector_state, llama_state):         projector = nnx.merge(proj...`
- `070 | code     | # Duplicate Stage-2 optimizer cell: # do NOT reinitialize opt_state here. # The real Stage-2 optimizer was already created in the earlier optimizer cell # using the correct dict-sh...`
- `071 | code     | import random import numpy as np import ml_dtypes  def stage2_collate_fn(batch):     if len(batch) == 0:         raise ValueError("stage2_collate_fn received an empty batch.")     ...`
- `072 | code     | @nnx.jit def eval_step_stage2(     projector_state,     llama_state,     batch, ):     projector = nnx.merge(projector_graphdef, projector_state)     llama_model = nnx.merge(llama_...`
- `073 | code     | import json from pathlib import Path  def run_stage2_training(     manifest_json,     val_manifest_json=None,     num_epochs=1,     batch_size=8,     log_every_steps=20, ):     glo...`
- `074 | code     | import pickle from pathlib import Path  RESET_ROOT = STAGE2_ROOT / "_fresh_state_snapshot" RESET_ROOT.mkdir(parents=True, exist_ok=True)  PROJECTOR_RESET_PATH = RESET_ROOT / "proje...`
- `075 | code     | save_fresh_stage2_snapshot()`
- `076 | code     | from pathlib import Path  def run_stage2_experiment(     variant,     num_epochs=1,     batch_size=8,     log_every_steps=20, ):     global projector_state, llama_state, tx, opt_st...`
- `077 | code     | from pathlib import Path import json import statistics from collections import Counter, defaultdict  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_in...`
- `078 | code     | import json from pathlib import Path  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction") RESULTS_PATH = STAGE2_ROOT / "all_results_manual.js...`
- `079 | code     | import json from pathlib import Path  if CURRENT_VARIANT is None:     print("No variant selected to run.") else:     print(f"Running variant: {CURRENT_VARIANT}")      result = run_...`
- `080 | code     | from pathlib import Path import json from collections import Counter  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  # Compare prompts ...`
- `081 | code     | from pathlib import Path import json  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction") RESULTS_PATH = STAGE2_ROOT / "all_results_manual.js...`
- `082 | code     | from pathlib import Path import json  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  ENGINE_SUMMARY_PATH = STAGE2_ROOT / "engine_compar...`
- `083 | code     | from pathlib import Path import json import random  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  with open(STAGE2_ROOT / "shared_qual...`
- `084 | code     | from pathlib import Path import shutil import json  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  QUANTITY_REGISTRATION_PATH = STAGE2_...`
- `085 | code     | import json from pathlib import Path  QUANTITY_PREP_STATUS_PATH = STAGE2_ROOT / "quantity_prep_status.json"  try:     with open(QUANTITY_PREP_STATUS_PATH, "r", encoding="utf-8") as...`
- `086 | code     | import json from pathlib import Path  QUANTITY_RESULTS_PATH = STAGE2_ROOT / "quantity_results.json"  # Safety flag: keep False unless you intentionally want to rerun saved quantity...`
- `087 | code     | import json from pathlib import Path  QUANTITY_RESULTS_PATH = STAGE2_ROOT / "quantity_results.json"  if "quantity_results" not in globals() or not quantity_results:     if not QUAN...`
- `088 | code     | from pathlib import Path import json import random from collections import defaultdict  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  ...`
- `089 | code     | import sys print(sys.executable) import sys !{sys.executable} -m pip install matplotlib`
- `090 | code     | from pathlib import Path import json import math import matplotlib.pyplot as plt  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  engine...`
- `091 | code     | from pathlib import Path import json  if "all_results" not in globals() or not all_results:     if not RESULTS_PATH.exists():         raise FileNotFoundError(             f"Missing...`
- `092 | code     | from pathlib import Path import json import random from collections import defaultdict  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  ...`
- `093 | code     | from pathlib import Path import json import random  STAGE2_ROOT = Path("/home/cengjiehui/CS159FinalProject/data/processed/stage2_instruction")  PAIRWISE_JUDGE_SEED = 2026  eval_pac...`
