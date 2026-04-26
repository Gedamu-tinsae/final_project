# Final Cross-Group Presentation Pack (Group1 + Group2 + Group4)

This report is based on generated artifacts on TPU at:
`/home/tinsae_kiya_gedamu/final_project`

No cleanup or deletion was performed.

## 1) Cross-Group Key Metrics (for slides)

| Group | Main objective | Key completed run | Key quality metric(s) | Key efficiency/compute metric(s) | Primary output path |
|---|---|---|---|---|---|
| Group1 | Build baseline multimodal pipeline and train projector + LLaMA stages | `outputs/group1/20260423_043833_group1_workflow` | Stage5 `val_loss_last=2.8514`, Stage6 `val_loss_last=2.5322` | Total wall time `15000.74s` (Stage5 `6254.54s`, Stage6 `8528.25s`) | `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow` |
| Group2 | Stage2 engine comparison + tracking/reporting | `engine_comparison_summary.json` | Best variant `llama`, `val_mean_loss=0.9701` | Tracking run wall time `58874.50s` (`stage4_tracking_summary`) | `/home/tinsae_kiya_gedamu/final_project/group2_baseline/data/processed/subsets/subset_10000_seed42/stage2_instruction/engine_comparison_summary.json` |
| Group4 | PEFT experiments (LoRA + selective FT), compare methods and summarize | `logs/runs/20260424_144442_group4_full.log` (12/12 succeeded) | Best experiment `g4_sft_012`, `val_loss=2.6895`, `win_rate_vs_baseline=0.7930` | `steps_per_sec=5.4315`, `wall_time_sec=1472.88`, `trainable_params_millions=1.5749` | `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/group4_results_summary.json` |

## 2) Group4 PEFT Method Comparison (headline)

| Experiment | Method | Target | Trainable Params (M) | Val Loss | Steps/sec | Wall Time (sec) | Win rate vs baseline |
|---|---|---|---:|---:|---:|---:|---:|
| `g4_lora_001` | LoRA (`qv`) | `qv` | 2.4248 | 2.6893 | 4.3839 | 1824.84 | 0.7915 |
| `g4_lora_002` | LoRA (`all_weights`) | `all` | 7.2090 | 2.6922 | 3.0397 | 2631.82 | 0.7860 |
| `g4_sft_012` | selective_ft | `all` (`budget=1.0%`) | 1.5749 | 2.6895 | 5.4315 | 1472.88 | 0.7930 |

Interpretation (slide-ready):
- `g4_sft_012` won overall by score because it balanced quality with lower trainable params and higher throughput.
- LoRA-`all_weights` increased trainable params and runtime substantially vs LoRA-`qv` without improving val loss.

## 3) Final Figures to Use in Slides

### Group1
- Stage timing:  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/fig_stage_timing.png`
- Stage memory:  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/fig_stage_memory.png`
- Stage5 train/val curves:  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/stage5/fig_train_loss.png`  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/stage5/fig_val_loss.png`
- Stage6 train/val curves:  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/stage6/fig_train_loss.png`  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group1/20260423_043833_group1_workflow/stage6/fig_val_loss.png`

### Group2
- Workflow timing/memory dashboard:  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group2/20260423_142651_group2_workflow/fig_stage_timing.png`  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group2/20260423_142651_group2_workflow/fig_stage_memory.png`  
  `/home/tinsae_kiya_gedamu/final_project/outputs/group2/20260423_142651_group2_workflow/fig_overview_dashboard.png`
- Engine comparison table markdown:  
  `/home/tinsae_kiya_gedamu/final_project/group2_baseline/data/processed/subsets/subset_10000_seed42/stage2_instruction/report_figures/engine_results_table.md`

### Group4
- Method comparison charts (final):  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_trainable_params.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_runtime_sec.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_val_loss.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_steps_per_sec.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_samples_per_sec.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_tpu_mem_max_mb.png`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/fig_method_comparison_rss_kb_max.png`
- Group4 final summary files:  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/group4_results_summary.json`  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/data/processed/subsets/subset_10000_seed42/group4_results_summary.md`

## 4) Final Narrative (Methodology, Setup, Results, Tradeoffs, Limitations)

### Methodology
- Group1: baseline multimodal pipeline from data prep -> tokenization -> CLIP precompute -> manifests -> Stage5/Stage6 training.
- Group2: Stage2 variant comparison and tracking (gemma/qwen/llama), then reporting artifacts.
- Group4: PEFT experiment grid across LoRA and selective FT, with 12 planned experiments, each run tracked and summarized.

### Experimental setup
- Compute: TPU VM (JAX backend TPU for Group4 runs).
- Dataset profile: subset-10k workflow for practical full-pipeline runs.
- Group4 training budget: `max_rows=10000`, `batch_size=1`, `epochs=1`, validation every 200 steps.

### Results
- Group1 produced required Stage5/Stage6 artifacts and baseline train/val curves.
- Group2 produced engine summary with `llama` best on final validation mean loss.
- Group4 completed all 12 experiments successfully and produced comparison charts + ranked summary; best run was `g4_sft_012`.

### Tradeoffs
- LoRA `all_weights` increases trainable parameters and runtime relative to LoRA `qv`.
- Selective FT (best run) achieved better throughput and lower trainable parameter count with competitive val loss.
- One-epoch budget gives a reliable directional comparison but is still a bounded-budget regime.

### Limitations
- Budget-constrained runs (`epochs=1`) are not full convergence studies.
- Judge-style evaluation (human/API) is available but not part of this default run (`EVAL_MODE=none` in full runner).

### Why Group4 method X won
- `g4_sft_012` won because it provided the strongest efficiency-quality balance in the current scoring setup:
  - lower trainable parameter footprint than LoRA alternatives,
  - higher steps/sec and lower wall time,
  - competitive val loss and highest win rate among final top candidates.

## 5) Evaluation Status (your question)

Evaluation is integrated in Group4, but optional:
- Implemented modes: `template`, `human_pack`, `human_aggregate`, `api_judge`
- Entry point:  
  `/home/tinsae_kiya_gedamu/final_project/group4_baseline/scripts/run_group4_eval.py`
- Full runner default: `EVAL_MODE=none`, so no judge pass is executed unless explicitly requested.

## 6) Primary Logs

- Group1:  
  `/home/tinsae_kiya_gedamu/final_project/logs/runs/20260423_043328_all_groups_subset10k_tpu.log`
- Group2:  
  `/home/tinsae_kiya_gedamu/final_project/logs/runs/20260423_142214_group2_tpu.log`
- Group4 (final 12-plan successful run):  
  `/home/tinsae_kiya_gedamu/final_project/logs/runs/20260424_144442_group4_full.log`
