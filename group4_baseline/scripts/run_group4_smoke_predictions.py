"""Run quick prediction comparisons from Group4 smoke checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
GROUP1_ROOT = REPO_ROOT / "group1_baseline"
GROUP4_BACKBONES_ROOT = PROJECT_ROOT / "src" / "group4_backbones"

for p in (GROUP1_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.config_loader import load_dotenv_file  # type: ignore
from src.model_internals.loader_pipeline import ensure_llama_artifacts, load_llama_model_and_tokenizer  # type: ignore
from src.training.batching import stage1_collate_fn  # type: ignore
from src.training.losses import masked_cross_entropy_loss  # type: ignore
from src.training.multimodal import make_multimodal_inputs  # type: ignore
from src.training.projector import VisionProjector  # type: ignore
from src.training.train_pipeline import _infer_clip_dim_from_manifest  # type: ignore


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prediction comparisons for Group4 smoke checkpoints.")
    p.add_argument(
        "--metrics-json",
        nargs="+",
        required=True,
        help="One or more metrics JSON files in artifacts/peft_smoke.",
    )
    p.add_argument("--manifest-json", default=str(PROJECT_ROOT / "artifacts" / "peft_smoke" / "stage1_manifest_smoke_group4.json"))
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument("--max-decode-tokens", type=int, default=40)
    p.add_argument("--output-json", default=str(PROJECT_ROOT / "data" / "processed" / "group4_prediction_samples.json"))
    p.add_argument("--output-md", default=str(PROJECT_ROOT / "data" / "processed" / "group4_prediction_samples.md"))
    return p.parse_args()


def _decode_ids(tokenizer, ids, max_tokens: int) -> str:
    ids = [int(x) for x in ids[:max_tokens]]
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def _load_llama_for_run(run: dict[str, Any], llama_dir: Path):
    method = str(run.get("method"))
    dtype = str(run.get("dtype", "bfloat16"))
    if method == "lora":
        lora_variant = str(run.get("lora_variant", "qv"))
        if lora_variant == "all_weights":
            model_file = GROUP4_BACKBONES_ROOT / "model_all_weights.py"
            params_file = GROUP4_BACKBONES_ROOT / "params_all_weights.py"
        else:
            model_file = GROUP4_BACKBONES_ROOT / "model_qv.py"
            params_file = GROUP4_BACKBONES_ROOT / "params_qv.py"
        g4_model = _load_module_from_file("group4_pred_model", model_file)
        g4_params = _load_module_from_file("group4_pred_params", params_file)
        mcfg = g4_model.ModelConfig.llama3p2_1b_instruct()
        model_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
        llama_model = g4_params.create_model_from_safe_tensors(
            file_dir=str(llama_dir),
            config=mcfg,
            mesh=None,
            dtype=model_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(llama_dir), local_files_only=True)
        return llama_model, tokenizer
    loaded = load_llama_model_and_tokenizer(local_dir=llama_dir, dtype=dtype, use_mesh=False)
    return loaded["llama_model"], loaded["tokenizer"]


def main() -> int:
    args = parse_args()
    load_dotenv_file(GROUP1_ROOT / ".env")

    manifest_path = Path(args.manifest_json).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = rows[: max(1, int(args.max_samples))]

    llama_dir = GROUP1_ROOT / "data" / "models" / "Llama-3.2-1B-Instruct"
    ensure_llama_artifacts(repo_id="meta-llama/Llama-3.2-1B-Instruct", local_dir=llama_dir)

    in_dim = _infer_clip_dim_from_manifest(manifest_path)
    all_results: list[dict[str, Any]] = []

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", [str(d) for d in jax.devices()])
    print("manifest rows used:", len(rows))

    for metrics_path_raw in args.metrics_json:
        metrics_path = Path(metrics_path_raw).resolve()
        run = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = str(run.get("run_id", metrics_path.stem))

        llama_model, tokenizer = _load_llama_for_run(run, llama_dir)
        out_dim = int(llama_model.config.embed_dim)
        projector = VisionProjector(in_dim=in_dim, out_dim=out_dim, rngs=nnx.Rngs(0))
        projector_graphdef, _ = nnx.split(projector)
        llama_graphdef, _ = nnx.split(llama_model)

        with Path(run["projector_state_path"]).open("rb") as f:
            projector_state = pickle.load(f)
        with Path(run["llama_state_path"]).open("rb") as f:
            llama_state = pickle.load(f)

        proj = nnx.merge(projector_graphdef, projector_state)
        llm = nnx.merge(llama_graphdef, llama_state)

        sample_rows: list[dict[str, Any]] = []
        losses: list[float] = []
        accuracies: list[float] = []

        for i, row in enumerate(rows):
            batch = stage1_collate_fn([row])
            mm = make_multimodal_inputs(llm, proj, batch)
            logits, _ = llm.forward_from_embeddings(
                input_embeds=mm["input_embeds"],
                positions=mm["positions"],
                cache=None,
                attention_mask=mm["attention_mask"],
            )
            loss = float(masked_cross_entropy_loss(logits, mm["labels"]))
            losses.append(loss)

            shift_logits = logits[:, :-1, :]
            shift_labels = mm["labels"][:, 1:]
            pred_ids = jnp.argmax(shift_logits, axis=-1)[0]
            true_ids = shift_labels[0]
            valid = true_ids != -100
            pred_valid = pred_ids[valid]
            true_valid = true_ids[valid]

            acc = 0.0
            if true_valid.size > 0:
                acc = float(jnp.mean((pred_valid == true_valid).astype(jnp.float32)))
            accuracies.append(acc)

            sample_rows.append(
                {
                    "sample_index": i,
                    "vision_path": row.get("vision_path", ""),
                    "loss": loss,
                    "token_accuracy": acc,
                    "pred_text": _decode_ids(tokenizer, pred_valid.tolist(), args.max_decode_tokens),
                    "label_text": _decode_ids(tokenizer, true_valid.tolist(), args.max_decode_tokens),
                }
            )

        run_summary = {
            "run_id": run_id,
            "method": run.get("method"),
            "lora_variant": run.get("lora_variant"),
            "target_modules": run.get("target_modules"),
            "avg_loss": float(sum(losses) / max(1, len(losses))),
            "avg_token_accuracy": float(sum(accuracies) / max(1, len(accuracies))),
            "num_samples": len(sample_rows),
            "samples": sample_rows,
            "metrics_json": str(metrics_path),
        }
        all_results.append(run_summary)
        print(
            f"[{run_id}] avg_loss={run_summary['avg_loss']:.4f} "
            f"avg_token_accuracy={run_summary['avg_token_accuracy']:.4f}"
        )

    out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_json": str(manifest_path),
        "max_samples": len(rows),
        "results": all_results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.output_md).resolve()
    lines = [
        "# Group4 Smoke Prediction Comparison",
        "",
        f"- manifest: `{manifest_path}`",
        f"- samples: {len(rows)}",
        "",
    ]
    for r in all_results:
        lines += [
            f"## {r['run_id']}",
            "",
            f"- method: {r.get('method')}",
            f"- lora_variant: {r.get('lora_variant')}",
            f"- target_modules: {r.get('target_modules')}",
            f"- avg_loss: {r['avg_loss']:.4f}",
            f"- avg_token_accuracy: {r['avg_token_accuracy']:.4f}",
            "",
        ]
        for s in r["samples"]:
            lines += [
                f"### sample {s['sample_index']}",
                f"- loss: {s['loss']:.4f}",
                f"- token_accuracy: {s['token_accuracy']:.4f}",
                f"- vision_path: `{s['vision_path']}`",
                f"- label_text: `{s['label_text']}`",
                f"- pred_text: `{s['pred_text']}`",
                "",
            ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", out_json)
    print("saved:", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

