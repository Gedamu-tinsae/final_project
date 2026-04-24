"""Modular evaluation pipeline for Group4 (human and API-based judging)."""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _extract_prompt(row: dict[str, Any]) -> str:
    for k in ("prompt", "text", "question", "instruction", "input_text", "user_text"):
        v = row.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_sample_id(row: dict[str, Any], idx: int) -> str:
    for k in ("sample_id", "image_id", "id"):
        v = row.get(k, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return f"sample_{idx:06d}"


def build_generations_template_from_manifest(
    manifest_json: Path,
    *,
    methods: list[str],
    out_jsonl: Path,
    max_samples: int = 0,
) -> dict[str, Any]:
    rows_any = _read_json(manifest_json)
    if not isinstance(rows_any, list):
        raise ValueError(f"Manifest is not a JSON list: {manifest_json}")
    rows: list[dict[str, Any]] = [r for r in rows_any if isinstance(r, dict)]
    if max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]

    out_rows: list[dict[str, Any]] = []
    for idx, r in enumerate(rows):
        sample_id = _extract_sample_id(r, idx)
        prompt = _extract_prompt(r)
        for m in methods:
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "method": m,
                    "output": "",
                }
            )
    _write_jsonl(out_jsonl, out_rows)
    return {
        "manifest_json": str(manifest_json),
        "output_jsonl": str(out_jsonl),
        "num_samples": len(rows),
        "num_methods": len(methods),
        "num_rows": len(out_rows),
        "methods": methods,
    }


def build_pairwise_requests(
    generations_jsonl: Path,
    *,
    baseline_method: str,
    candidate_methods: list[str] | None,
    seed: int = 42,
    max_requests: int = 0,
) -> dict[str, Any]:
    rows = _read_jsonl(generations_jsonl)
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        sample_id = str(r.get("sample_id", "")).strip()
        method = str(r.get("method", "")).strip()
        if not sample_id or not method:
            continue
        by_sample.setdefault(sample_id, {})[method] = r

    all_methods = sorted({str(r.get("method", "")).strip() for r in rows if str(r.get("method", "")).strip()})
    if baseline_method not in all_methods:
        raise ValueError(f"baseline_method '{baseline_method}' not found in generations.")
    candidates = [m for m in all_methods if m != baseline_method]
    if candidate_methods:
        requested = [m.strip() for m in candidate_methods if m.strip()]
        candidates = [m for m in candidates if m in requested]
    if not candidates:
        raise ValueError("No candidate methods available after filtering.")

    reqs: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for sample_id in sorted(by_sample.keys()):
        sample_map = by_sample[sample_id]
        if baseline_method not in sample_map:
            continue
        base = sample_map[baseline_method]
        prompt = str(base.get("prompt", ""))
        base_output = str(base.get("output", ""))
        for method in candidates:
            cand = sample_map.get(method)
            if cand is None:
                continue
            cand_output = str(cand.get("output", ""))
            if not base_output.strip() or not cand_output.strip():
                continue

            method_on_a = bool(rng.getrandbits(1))
            answer_a = cand_output if method_on_a else base_output
            answer_b = base_output if method_on_a else cand_output
            reqs.append(
                {
                    "request_id": f"{sample_id}__{method}",
                    "sample_id": sample_id,
                    "method": method,
                    "baseline_method": baseline_method,
                    "prompt": prompt,
                    "answer_a": answer_a,
                    "answer_b": answer_b,
                    "method_side": "A" if method_on_a else "B",
                    "baseline_side": "B" if method_on_a else "A",
                }
            )

    if max_requests > 0 and len(reqs) > max_requests:
        reqs = reqs[:max_requests]

    return {
        "baseline_method": baseline_method,
        "candidate_methods": candidates,
        "num_requests": len(reqs),
        "requests": reqs,
    }


def write_human_eval_pack(out_dir: Path, pairwise: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = list(pairwise["requests"])
    req_path = out_dir / "pairwise_requests.jsonl"
    _write_jsonl(req_path, requests)

    template_rows: list[dict[str, Any]] = []
    for r in requests:
        template_rows.append(
            {
                "request_id": r["request_id"],
                "winner": None,  # "A" | "B" | "TIE"
                "reason": "",
            }
        )
    template_path = out_dir / "human_results_template.jsonl"
    _write_jsonl(template_path, template_rows)

    instructions = (
        "# Group4 Pairwise Human Evaluation\n\n"
        "For each request, choose winner in {A, B, TIE}.\n"
        "- Use factual correctness, relevance, and clarity.\n"
        "- Do not use method names as signal (blind judging).\n"
        "- Write a short reason.\n\n"
        "Files:\n"
        "- pairwise_requests.jsonl\n"
        "- human_results_template.jsonl\n"
    )
    instructions_path = out_dir / "instructions.md"
    instructions_path.write_text(instructions, encoding="utf-8")

    return {
        "pairwise_requests_jsonl": str(req_path),
        "human_results_template_jsonl": str(template_path),
        "instructions_md": str(instructions_path),
        "num_requests": len(requests),
    }


def aggregate_pairwise_results(
    requests_jsonl: Path,
    judged_results_jsonl: Path,
) -> dict[str, Any]:
    req_rows = _read_jsonl(requests_jsonl)
    judged_rows = _read_jsonl(judged_results_jsonl)
    req_by_id = {str(r["request_id"]): r for r in req_rows}

    per_method: dict[str, dict[str, int]] = {}
    used = 0
    skipped = 0
    for row in judged_rows:
        rid = str(row.get("request_id", ""))
        winner = str(row.get("winner", "")).strip().upper()
        req = req_by_id.get(rid)
        if req is None or winner not in {"A", "B", "TIE"}:
            skipped += 1
            continue
        used += 1
        method = str(req["method"])
        method_side = str(req["method_side"])
        stats = per_method.setdefault(method, {"wins": 0, "ties": 0, "losses": 0, "total": 0})
        stats["total"] += 1
        if winner == "TIE":
            stats["ties"] += 1
        elif winner == method_side:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    summary_rows: list[dict[str, Any]] = []
    for method in sorted(per_method.keys()):
        s = per_method[method]
        total = max(1, int(s["total"]))
        win_rate = (float(s["wins"]) + 0.5 * float(s["ties"])) / float(total)
        summary_rows.append(
            {
                "method": method,
                "wins": int(s["wins"]),
                "ties": int(s["ties"]),
                "losses": int(s["losses"]),
                "total": int(s["total"]),
                "win_rate_vs_baseline": float(win_rate),
            }
        )

    return {
        "num_requests_total": len(req_rows),
        "num_judged_used": used,
        "num_judged_skipped": skipped,
        "method_summaries": summary_rows,
    }


def _openai_judge_one(
    req: dict[str, Any],
    *,
    model: str,
    api_key: str,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    system = (
        "You are a strict evaluator. Compare two candidate answers for the same prompt. "
        "Return JSON with keys: winner (A|B|TIE), reason (short)."
    )
    user = (
        f"Prompt:\n{req['prompt']}\n\n"
        f"Answer A:\n{req['answer_a']}\n\n"
        f"Answer B:\n{req['answer_b']}\n\n"
        "Judge based on correctness, relevance, and clarity."
    )
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "text": {"format": {"type": "json_object"}},
    }
    raw = json.dumps(payload).encode("utf-8")
    http_req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=raw,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"winner": "TIE", "reason": f"HTTPError: {e.code}", "raw_error": body}
    except Exception as e:
        return {"winner": "TIE", "reason": f"RequestError: {e}"}

    try:
        obj = json.loads(body)
        # responses API shape: output_text available for simple parsing.
        text = str(obj.get("output_text", "")).strip()
        if not text:
            # fallback: try nested extraction
            chunks = obj.get("output", [])
            for c in chunks:
                if isinstance(c, dict):
                    for item in c.get("content", []):
                        if isinstance(item, dict) and item.get("type") in {"output_text", "text"}:
                            text = str(item.get("text", "")).strip()
                            if text:
                                break
                    if text:
                        break
        parsed = json.loads(text) if text else {}
        winner = str(parsed.get("winner", "TIE")).upper()
        if winner not in {"A", "B", "TIE"}:
            winner = "TIE"
        reason = str(parsed.get("reason", ""))
        return {"winner": winner, "reason": reason, "raw": text}
    except Exception:
        return {"winner": "TIE", "reason": "parse_failed", "raw": body}


def run_openai_judging(
    requests_jsonl: Path,
    *,
    model: str,
    api_key: str | None = None,
    max_requests: int = 0,
) -> dict[str, Any]:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY missing. Set env var or pass --openai-api-key.")

    reqs = _read_jsonl(requests_jsonl)
    if max_requests > 0 and len(reqs) > max_requests:
        reqs = reqs[:max_requests]

    judged: list[dict[str, Any]] = []
    for r in reqs:
        out = _openai_judge_one(r, model=model, api_key=key)
        judged.append(
            {
                "request_id": r["request_id"],
                "winner": out.get("winner", "TIE"),
                "reason": out.get("reason", ""),
                "model": model,
            }
        )
    return {"num_requests": len(reqs), "judged_rows": judged}


def update_group4_results_with_eval(
    results_manual_json: Path,
    eval_summary: dict[str, Any],
) -> dict[str, Any]:
    data = _read_json(results_manual_json) if results_manual_json.exists() else {"results": []}
    rows = data.setdefault("results", [])
    by_method = {str(r.get("method", "")): r for r in eval_summary.get("method_summaries", [])}

    updated = 0
    for row in rows:
        method = str(row.get("method", ""))
        s = by_method.get(method)
        if s is None:
            continue
        row["win_rate_vs_baseline"] = float(s.get("win_rate_vs_baseline", row.get("win_rate_vs_baseline", 0.0)))
        row["eval_wins"] = int(s.get("wins", 0))
        row["eval_ties"] = int(s.get("ties", 0))
        row["eval_losses"] = int(s.get("losses", 0))
        row["eval_total"] = int(s.get("total", 0))
        updated += 1

    _write_json(results_manual_json, data)
    return {"updated_rows": updated, "results_manual_json": str(results_manual_json)}
