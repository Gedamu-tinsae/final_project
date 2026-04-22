# File Tree: final_project

**Generated:** 4/22/2026, 12:50:03 AM
**Root Path:** `/root/final_project`

```
├── 📁 group1_baseline
│   ├── 📁 artifacts
│   │   ├── 📄 llama_stage2.pkl
│   │   ├── 📄 projector_stage1.pkl
│   │   └── 📄 projector_stage2.pkl
│   ├── 📁 configs
│   │   └── ⚙️ workflow_paths.json
│   ├── 📁 data
│   │   ├── 📁 models
│   │   │   └── 📁 Llama-3.2-1B-Instruct
│   │   │       ├── ⚙️ config.json
│   │   │       ├── ⚙️ generation_config.json
│   │   │       ├── 📄 model.safetensors
│   │   │       ├── ⚙️ special_tokens_map.json
│   │   │       ├── ⚙️ tokenizer.json
│   │   │       └── ⚙️ tokenizer_config.json
│   │   ├── 📁 processed
│   │   │   ├── 📁 clip_embeddings
│   │   │   │   ├── 📄 000000000775.npy
│   │   │   │   ├── 📄 000000002881.npy...
│   │   │   │   └── 📄 000000581177.npy
│   │   │   ├── 📁 stage1_alignment
│   │   │   │   ├── ⚙️ alignment.json
│   │   │   │   ├── ⚙️ alignment_chat.json
│   │   │   │   ├── ⚙️ alignment_tokenized.json
│   │   │   │   ├── ⚙️ stage1_manifest.json
│   │   │   │   └── ⚙️ stage1_manifest_smoke.json
│   │   │   ├── 📁 stage2_finetuning
│   │   │   │   ├── ⚙️ alignment_tokenized_stage2.json
│   │   │   │   ├── ⚙️ stage2_manifest.json
│   │   │   │   └── ⚙️ stage2_manifest_smoke.json
│   │   │   └── 📁 stage2_instruction
│   │   │       ├── 📁 gemma
│   │   │       │   └── 📄 stage2_dataset.jsonl
│   │   │       ├── 📁 llama
│   │   │       │   └── 📄 stage2_dataset.jsonl
│   │   │       └── 📁 qwen
│   │   │           └── 📄 stage2_dataset.jsonl
│   │   └── 📁 raw
│   │       ├── 📁 annotations
│   │       │   ├── ⚙️ captions_train2017.json
│   │       │   ├── ⚙️ captions_val2017.json
│   │       │   ├── ⚙️ instances_train2017.json
│   │       │   ├── ⚙️ instances_val2017.json
│   │       │   ├── ⚙️ person_keypoints_train2017.json
│   │       │   └── ⚙️ person_keypoints_val2017.json
│   │       ├── 📁 train2017
│   │       │   ├── 🖼️ 000000000009.
│   │       │   └── 🖼️ 000000581929.jpg
│   │       ├── 📦 annotations_trainval2017.zip
│   │       └── 📦 train2017.zip
│   ├── 📁 notebooks
│   │   ├── 📄 LLaVA_Baseline_Workflow.ipynb
│   │   └── 📄 LLaVA_Public_legacy.ipynb
│   ├── 📁 notes
│   │   ├── 📝 DATA_DIRECTORY_GUIDE.md
│   │   ├── 📝 GPU_ENABLE_CLOUDEXE.md
│   │   ├── 📝 HF_GATED_MODEL_ACCESS.md
│   │   ├── 📝 HF_SETUP_STEPS.md
│   │   ├── 📝 MIGRATION_TRACE.md
│   │   ├── 📝 TPU_PROVISIONING_GUIDE.md
│   │   └── 📝 TPU_RUN_SETUP.md
│   ├── 📁 scripts
│   │   ├── 🐍 check_accelerator.py
│   │   ├── 🐍 check_env.py
│   │   ├── 🐍 run_baseline_workflow.py
│   │   └── 🐍 run_tpu_smoke.py
│   ├── 📁 src
│   │   ├── 📁 data_prep
│   │   │   ├── 🐍 acquire_coco.py
│   │   │   ├── 🐍 convert_alignment_format.py
│   │   │   ├── 🐍 prepare_stage1_dataset.py
│   │   │   └── 🐍 stage1_pipeline.py
│   │   ├── 📁 model_internals
│   │   │   ├── 🐍 loader_pipeline.py
│   │   │   ├── 🐍 model.py
│   │   │   └── 🐍 params.py
│   │   ├── 📁 training
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 batching.py
│   │   │   ├── 🐍 clip_features.py
│   │   │   ├── 🐍 losses.py
│   │   │   ├── 🐍 memory.py
│   │   │   ├── 🐍 multimodal.py
│   │   │   ├── 🐍 projector.py
│   │   │   ├── 🐍 stage1.py
│   │   │   ├── 🐍 stage2.py
│   │   │   ├── 🐍 tokenization.py
│   │   │   ├── 🐍 tokenization_pipeline.py
│   │   │   └── 🐍 train_pipeline.py
│   │   ├── 📁 training_manifests
│   │   │   ├── 🐍 build_stage1_manifest.py
│   │   │   ├── 🐍 build_stage2_manifest.py
│   │   │   └── 🐍 manifest_pipeline.py
│   │   ├── 📁 vision_features
│   │   │   ├── 🐍 clip_helpers.py
│   │   │   ├── 🐍 feature_pipeline.py
│   │   │   └── 🐍 precompute_clip_features.py
│   │   └── 🐍 config_loader.py
│   ├── ⚙️ .gitignore
│   ├── 📝 SETUP.md
│   ├── 📄 requirements-core.txt
│   ├── 📄 requirements-notebook.txt
│   ├── 📄 requirements-tpu.txt
│   └── 📄 requirements.txt
├── 📁 group2_baseline
│   ├── 📁 configs
│   │   └── ⚙️ workflow_paths.json
│   ├── 📁 data
│   │   ├── 📁 processed
│   │   │   ├── 📁 stage2_features
│   │   │   │   ├── 📄 000000000109.npy...
│   │   │   │   └── 📄 000000581899.npy
│   │   │   └── 📁 stage2_instruction
│   │   │       ├── 📁 gemma
│   │   │       │   ├── 📄 stage2_dataset.jsonl
│   │   │       │   ├── ⚙️ stage2_manifest_train.json
│   │   │       │   ├── ⚙️ stage2_manifest_val.json
│   │   │       │   ├── ⚙️ stage2_tokenized_train.json
│   │   │       │   ├── ⚙️ stage2_tokenized_val.json
│   │   │       │   ├── 📄 stage2_train.jsonl
│   │   │       │   └── 📄 stage2_val.jsonl
│   │   │       ├── 📁 llama
│   │   │       │   ├── 📄 stage2_dataset.jsonl
│   │   │       │   ├── ⚙️ stage2_manifest_train.json
│   │   │       │   ├── ⚙️ stage2_manifest_val.json
│   │   │       │   ├── ⚙️ stage2_tokenized_train.json
│   │   │       │   ├── ⚙️ stage2_tokenized_val.json
│   │   │       │   ├── 📄 stage2_train.jsonl
│   │   │       │   └── 📄 stage2_val.jsonl
│   │   │       ├── 📁 qwen
│   │   │       │   ├── 📄 stage2_dataset.jsonl
│   │   │       │   ├── ⚙️ stage2_manifest_train.json
│   │   │       │   ├── ⚙️ stage2_manifest_val.json
│   │   │       │   ├── ⚙️ stage2_tokenized_train.json
│   │   │       │   ├── ⚙️ stage2_tokenized_val.json
│   │   │       │   ├── 📄 stage2_train.jsonl
│   │   │       │   └── 📄 stage2_val.jsonl
│   │   │       ├── ⚙️ all_results_manual.json
│   │   │       ├── ⚙️ dataset_quality_diagnostics.json
│   │   │       ├── ⚙️ heldout_eval_pack.json
│   │   │       ├── ⚙️ pairwise_judge_requests.json
│   │   │       ├── ⚙️ prompt_alignment_audit.json
│   │   │       ├── ⚙️ qualitative_comparison_samples.json
│   │   │       ├── ⚙️ shared_quality_pool.json
│   │   │       └── ⚙️ shared_split.json
│   │   └── 📁 raw
│   │       └── 📄 train2017
│   ├── 📁 notebooks
│   │   └── 📄 LLaVA_Group2_Workflow.ipynb
│   ├── 📁 notes
│   │   ├── 📝 CELL_TO_MODULE_MAPPING.md
│   │   ├── 📝 DATA_DIRECTORY_GUIDE.md
│   │   ├── 📝 GROUP2_UNIQUE_COMPONENTS.md
│   │   └── 📝 MIGRATION_TRACE.md
│   ├── 📁 scripts
│   │   ├── 🐍 run_group2_nonmodel.py
│   │   └── 🐍 run_group2_workflow.py
│   ├── 📁 src
│   │   ├── 📁 group2_stage2
│   │   │   ├── 📁 data
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 audit.py
│   │   │   │   ├── 🐍 features.py
│   │   │   │   ├── 🐍 manifests.py
│   │   │   │   ├── 🐍 pipeline.py
│   │   │   │   ├── 🐍 splits.py
│   │   │   │   └── 🐍 tokenization.py
│   │   │   ├── 📁 eval
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 evaluation_pack.py
│   │   │   │   ├── 🐍 quality_eval.py
│   │   │   │   └── 🐍 reporting.py
│   │   │   ├── 📁 experiments
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 experiment_tracking.py
│   │   │   │   ├── 🐍 quantity_ablation.py
│   │   │   │   └── 🐍 training_orchestration.py
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 bootstrap_runtime.py
│   │   │   └── 🐍 common.py
│   │   └── 🐍 __init__.py
│   ├── ⚙️ .gitignore
│   ├── 📝 SETUP.md
│   └── 📄 requirements.txt
└── 📝 File_Tree.md
```

---
*Generated by FileTree Pro Extension*