"""SelVA Textual Inversion Scheduler — sweeps TI training experiments from a JSON file.

Each experiment inherits from a shared `base` config and overrides specific keys.
The dataset is loaded once and reused across all experiments. Results are written
to `experiment_summary.json` (updated after each completed run) and a comparison
loss-curve image showing all runs on the same axes.

JSON format:
    {
      "name": "ti_sweep_1",
      "description": "optional human note",
      "data_dir": "dataset/bj_sounds",
      "output_root": "ti_output/sweep_1",
      "base": {
        "n_tokens": 4,
        "lr": 1e-3,
        "steps": 3000,
        "batch_size": 16,
        "warmup_steps": 100,
        "seed": 42,
        "save_every": 1000
      },
      "experiments": [
        {"id": "baseline",       "description": "default 4 tokens"},
        {"id": "n8_tokens",      "n_tokens": 8},
        {"id": "lr_5e4",         "lr": 5e-4},
        {"id": "warm_init",      "init_text": "industrial sound design"},
        {"id": "n4_more_steps",  "steps": 5000}
      ]
    }
"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device
from .selva_lora_trainer import (
    _prepare_dataset,
    _smooth_losses,
    _pil_to_tensor,
)
from .selva_textual_inversion_trainer import SelvaTextualInversionTrainer


# ---------------------------------------------------------------------------
# Helpers (shared with LoRA scheduler, inlined to keep modules independent)
# ---------------------------------------------------------------------------

def _get_system_info() -> dict:
    info: dict = {
        "torch_version": torch.__version__,
        "cuda_version":  torch.version.cuda or "N/A",
        "gpu_name":      None,
        "gpu_vram_gb":   None,
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"]    = torch.cuda.get_device_name(0)
            props               = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1e9, 1)
        except Exception:
            pass
    return info


_PARAM_DEFAULTS = {
    "n_tokens":     4,
    "lr":           2e-4,
    "steps":        3000,
    "batch_size":   4,
    "warmup_steps": 100,
    "seed":         42,
    "save_every":   1000,
    "init_text":    "",
    "inject_mode":  "suffix",
}

_PALETTE = [
    (66,  133, 244),
    (234,  67,  53),
    (52,  168,  83),
    (251, 188,   5),
    (155,  89, 182),
    (26,  188, 156),
    (230, 126,  34),
    (149, 165, 166),
]


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    unix_style_on_windows = (
        sys.platform == "win32" and p.is_absolute() and not p.drive
    )
    if not p.is_absolute() or unix_style_on_windows:
        p = Path(folder_paths.get_output_directory()) / p.relative_to(p.anchor)
    return p


def _merge_config(base: dict, experiment: dict) -> dict:
    cfg = dict(_PARAM_DEFAULTS)
    cfg.update(base)
    cfg.update({k: v for k, v in experiment.items() if k not in ("id", "description")})
    return cfg


def _loss_at_steps(loss_history: list, log_interval: int, save_every: int,
                   total_steps: int) -> dict:
    result = {}
    for target in range(save_every, total_steps + 1, save_every):
        idx = target // log_interval - 1
        if 0 <= idx < len(loss_history):
            result[str(target)] = round(loss_history[idx], 6)
    return result


def _draw_comparison_curves(experiments_data: list) -> "Image.Image":
    from PIL import Image, ImageDraw

    W, H = 900, 420
    pl, pr, pt, pb = 75, 160, 30, 50

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw   = W - pl - pr
    ph   = H - pt - pb

    series = []
    for i, ed in enumerate(experiments_data):
        lh = ed.get("loss_history") or []
        if len(lh) < 2:
            continue
        sm = _smooth_losses(lh)
        series.append({
            "id":       ed["id"],
            "smoothed": sm,
            "color":    _PALETTE[i % len(_PALETTE)],
        })

    if not series:
        draw.text((pl + 10, pt + 10), "No data to plot", fill=(80, 80, 80))
        return img

    all_vals = [v for s in series for v in s["smoothed"]]
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1e-6
    rng = hi - lo

    for i in range(5):
        y   = pt + int(i * ph / 4)
        val = hi - i * rng / 4
        draw.line([(pl, y), (W - pr, y)], fill=(220, 220, 220), width=1)
        draw.text((2, y - 7), f"{val:.4f}", fill=(100, 100, 100))

    for s in series:
        n   = len(s["smoothed"])
        pts = []
        for j, v in enumerate(s["smoothed"]):
            x = pl + int(j * pw / max(n - 1, 1))
            y = pt + int((1.0 - (v - lo) / rng) * ph)
            pts.append((x, y))
        draw.line(pts, fill=s["color"], width=2)

    draw.line([(pl, pt), (pl, H - pb)],         fill=(40, 40, 40), width=1)
    draw.line([(pl, H - pb), (W - pr, H - pb)], fill=(40, 40, 40), width=1)
    draw.text((pl + 4, 8), "TI loss comparison (smoothed)", fill=(40, 40, 40))

    lx, ly = W - pr + 10, pt
    for s in series:
        draw.rectangle([(lx, ly + 3), (lx + 14, ly + 13)], fill=s["color"])
        draw.text((lx + 18, ly), s["id"][:20], fill=(40, 40, 40))
        ly += 20

    return img


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SelvaTiScheduler:
    """Runs a sweep of Textual Inversion experiments defined in a JSON file.

    The dataset is loaded once and reused. Each experiment calls
    SelvaTextualInversionTrainer._train_inner() with its own config.
    Results are written to experiment_summary.json after every completed run.
    """

    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "run"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("summary_path", "comparison_curves")
    OUTPUT_TOOLTIPS = (
        "Path to experiment_summary.json — compare runs across sweeps.",
        "All smoothed loss curves overlaid on the same axes.",
    )
    DESCRIPTION = (
        "Runs a series of Textual Inversion experiments from a JSON sweep file. "
        "The dataset is encoded once and reused. Results (loss, config, embeddings "
        "paths) are collected in experiment_summary.json after each run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "experiments_file": ("STRING", {
                    "default": "ti_experiments.json",
                    "tooltip": (
                        "Path to JSON sweep file. Relative paths resolve to the ComfyUI "
                        "output directory. See node description for the file format."
                    ),
                }),
            }
        }

    def run(self, model, experiments_file):
        # ------------------------------------------------------------------
        # 1. Read + validate JSON
        # ------------------------------------------------------------------
        exp_path = Path(experiments_file.strip())
        if not exp_path.is_absolute():
            candidate = Path(folder_paths.models_dir) / exp_path
            if not candidate.exists():
                candidate = Path(folder_paths.get_output_directory()) / exp_path
            exp_path = candidate
        if not exp_path.exists():
            raise FileNotFoundError(
                f"[TI Scheduler] Experiment file not found: {exp_path}"
            )

        spec = json.loads(exp_path.read_text(encoding="utf-8"))

        if "experiments" not in spec or not spec["experiments"]:
            raise ValueError("[TI Scheduler] 'experiments' list is missing or empty.")
        for i, exp in enumerate(spec["experiments"]):
            if "id" not in exp:
                raise ValueError(
                    f"[TI Scheduler] Experiment at index {i} is missing required 'id' field."
                )

        sweep_name  = spec.get("name", exp_path.stem)
        description = spec.get("description", "")
        base_cfg    = spec.get("base", {})

        # ------------------------------------------------------------------
        # 2. Resolve data_dir and output_root
        # ------------------------------------------------------------------
        if "data_dir" not in spec:
            raise ValueError("[TI Scheduler] 'data_dir' is required in the sweep file.")
        data_dir    = _resolve_path(spec["data_dir"])
        output_root = _resolve_path(spec.get("output_root", f"ti_sweeps/{sweep_name}"))
        output_root.mkdir(parents=True, exist_ok=True)

        device = get_device()
        dtype  = model["dtype"]
        mode   = model["mode"]
        seq_cfg            = model["seq_cfg"]
        feature_utils_orig = model["feature_utils"]

        print(f"\n[TI Scheduler] Sweep '{sweep_name}': "
              f"{len(spec['experiments'])} experiment(s)", flush=True)
        if description:
            print(f"[TI Scheduler] {description}", flush=True)
        print(f"[TI Scheduler] data_dir    = {data_dir}", flush=True)
        print(f"[TI Scheduler] output_root = {output_root}\n", flush=True)

        # ------------------------------------------------------------------
        # 3. Load dataset once
        # ------------------------------------------------------------------
        n_clips = len(list(data_dir.glob("*.npz")))
        dataset = _prepare_dataset(model, data_dir, device)

        # ------------------------------------------------------------------
        # 4. Build or restore summary (resume-aware)
        # ------------------------------------------------------------------
        summary_path   = output_root / "experiment_summary.json"
        completed_ids  = set()
        all_curve_data = []

        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                for rec in existing.get("experiments", []):
                    if rec.get("results", {}).get("status") == "completed":
                        completed_ids.add(rec["id"])
                        all_curve_data.append({
                            "id":           rec["id"],
                            "loss_history": rec["results"].get("loss_history", []),
                        })
                summary = existing
                summary["completed_at"] = None
                if completed_ids:
                    print(f"[TI Scheduler] Resuming — skipping {len(completed_ids)} "
                          f"completed experiment(s): {sorted(completed_ids)}", flush=True)
            except Exception as e:
                print(f"[TI Scheduler] Could not read existing summary ({e}) — starting fresh",
                      flush=True)
                completed_ids  = set()
                all_curve_data = []
                summary        = None

        if not completed_ids:
            summary = {
                "sweep_name":   sweep_name,
                "description":  description,
                "sweep_file":   str(exp_path),
                "started_at":   datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "system":       _get_system_info(),
                "data_dir":     str(data_dir),
                "n_clips":      n_clips,
                "experiments":  [],
            }

        comparison_img_path = output_root / "loss_comparison.png"

        def _write_summary():
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        def _save_comparison():
            try:
                img = _draw_comparison_curves(all_curve_data)
                img.save(str(comparison_img_path))
            except Exception as e:
                print(f"[TI Scheduler] Comparison image failed: {e}", flush=True)

        _write_summary()

        # ------------------------------------------------------------------
        # 5. Run each experiment
        # ------------------------------------------------------------------
        trainer    = SelvaTextualInversionTrainer()
        pbar_outer = comfy.utils.ProgressBar(len(spec["experiments"]))
        log_interval = 50   # matches _train_inner

        for exp in spec["experiments"]:
            exp_id   = exp["id"]
            exp_desc = exp.get("description", "")

            if exp_id in completed_ids:
                print(f"[TI Scheduler] Skipping '{exp_id}' (already completed)", flush=True)
                pbar_outer.update(1)
                continue

            cfg = _merge_config(base_cfg, exp)

            n_tokens    = int(cfg["n_tokens"])
            lr          = float(cfg["lr"])
            steps       = int(cfg["steps"])
            batch_size  = int(cfg["batch_size"])
            warmup      = int(cfg["warmup_steps"])
            seed        = int(cfg["seed"])
            save_every  = int(cfg["save_every"])
            init_text   = str(cfg["init_text"])
            inject_mode = str(cfg["inject_mode"])

            output_dir = output_root / exp_id
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "embeddings.pt"

            print(f"\n[TI Scheduler] ── Experiment '{exp_id}' ──", flush=True)
            if exp_desc:
                print(f"[TI Scheduler] {exp_desc}", flush=True)
            print(f"[TI Scheduler] n_tokens={n_tokens}  lr={lr:.2e}  steps={steps}  "
                  f"batch_size={batch_size}  warmup={warmup}  seed={seed}  "
                  f"inject_mode={inject_mode}", flush=True)
            if init_text:
                print(f"[TI Scheduler] init_text='{init_text}'", flush=True)

            exp_record = {
                "id":          exp_id,
                "description": exp_desc,
                "config": {
                    "n_tokens":     n_tokens,
                    "lr":           lr,
                    "steps":        steps,
                    "batch_size":   batch_size,
                    "warmup_steps": warmup,
                    "seed":         seed,
                    "save_every":   save_every,
                    "init_text":    init_text,
                    "inject_mode":  inject_mode,
                },
                "results":        {"status": "running"},
                "embeddings_path": None,
                "output_dir":      str(output_dir),
            }
            summary["experiments"].append(exp_record)
            _write_summary()

            t_start = time.monotonic()
            try:
                with torch.inference_mode(False), torch.enable_grad():
                    r = trainer._train_inner(
                        model, dataset, feature_utils_orig, seq_cfg,
                        device, dtype, mode,
                        data_dir, out_path,
                        n_tokens, steps, lr, batch_size,
                        warmup, seed, save_every, init_text, inject_mode,
                    )

                duration     = time.monotonic() - t_start
                loss_history = r["loss_history"]
                smoothed     = _smooth_losses(loss_history) if loss_history else []

                final_loss    = round(smoothed[-1], 6) if smoothed else None
                min_loss      = round(min(smoothed), 6) if smoothed else None
                min_idx       = smoothed.index(min(smoothed)) if smoothed else None
                min_loss_step = (min_idx + 1) * log_interval if min_idx is not None else None

                loss_std_last_quarter = None
                if loss_history:
                    quarter = max(1, len(loss_history) // 4)
                    loss_std_last_quarter = round(float(np.std(loss_history[-quarter:])), 6)

                exp_record["results"] = {
                    "status":                "completed",
                    "final_loss":            final_loss,
                    "min_loss":              min_loss,
                    "min_loss_step":         min_loss_step,
                    "loss_std_last_quarter": loss_std_last_quarter,
                    "loss_at_steps":         _loss_at_steps(
                        loss_history, log_interval, save_every, steps
                    ),
                    "loss_history":          [round(v, 6) for v in loss_history],
                    "log_interval":          log_interval,
                    "duration_seconds":      round(duration, 1),
                }
                exp_record["embeddings_path"] = r["embeddings_path"]

                all_curve_data.append({
                    "id":           exp_id,
                    "loss_history": loss_history,
                })

            except Exception as e:
                duration = time.monotonic() - t_start
                print(f"[TI Scheduler] Experiment '{exp_id}' failed: {e}", flush=True)
                traceback.print_exc()
                exp_record["results"] = {
                    "status":           "failed",
                    "error":            str(e),
                    "duration_seconds": round(duration, 1),
                }
                _write_summary()
                pbar_outer.update(1)
                continue

            _write_summary()
            _save_comparison()
            pbar_outer.update(1)

        # ------------------------------------------------------------------
        # 6. Finalise
        # ------------------------------------------------------------------
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_summary()
        print(f"\n[TI Scheduler] Sweep complete. Summary: {summary_path}", flush=True)

        # ------------------------------------------------------------------
        # 7. Comparison image (final update, then return to ComfyUI)
        # ------------------------------------------------------------------
        _save_comparison()
        comparison_img = _draw_comparison_curves(all_curve_data)
        return (str(summary_path), _pil_to_tensor(comparison_img))
