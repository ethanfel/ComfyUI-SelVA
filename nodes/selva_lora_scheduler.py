"""SelVA LoRA Scheduler — runs a sweep of training experiments from a JSON file.

Each experiment inherits from a shared `base` config and overrides specific keys.
The dataset is loaded once and reused across all experiments. Results are written
to `experiment_summary.json` (updated after each completed run) and a comparison
loss-curve image showing all runs on the same axes.

JSON format:
    {
      "name": "tier1_sweep",
      "description": "optional human note",
      "data_dir": "dataset/dog_bark",
      "output_root": "lora_output/tier1_sweep",
      "base": { "rank": 16, "lr": 1e-4, "steps": 2000, ... },
      "experiments": [
        {"id": "baseline", "description": "..."},
        {"id": "lora_plus_16", "lora_plus_ratio": 16.0},
        ...
      ]
    }
"""

import copy
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device
from .selva_lora_trainer import (
    SelvaLoraTrainer,
    SkipExperiment,
    _prepare_dataset,
    _smooth_losses,
    _pil_to_tensor,
)


def _get_system_info() -> dict:
    """Collect GPU / torch version info for the summary header."""
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


# Defaults mirror SelvaLoraTrainer INPUT_TYPES defaults
_PARAM_DEFAULTS = {
    "alpha":               0.0,
    "target":              "attn.qkv",
    "batch_size":          4,
    "warmup_steps":        100,
    "grad_accum":          1,
    "save_every":          500,
    "resume_path":         "",
    "seed":                42,
    "timestep_mode":       "uniform",
    "logit_normal_sigma":  1.0,
    "curriculum_switch":   0.6,
    "lora_dropout":        0.0,
    "lora_plus_ratio":     1.0,
}

# Palette for comparison chart: one color per experiment (cycles if > 8)
_PALETTE = [
    (66,  133, 244),   # blue
    (234,  67,  53),   # red
    (52,  168,  83),   # green
    (251, 188,   5),   # yellow
    (155,  89, 182),   # purple
    (26,  188, 156),   # teal
    (230, 126,  34),   # orange
    (149, 165, 166),   # grey
]


def _resolve_path(raw: str) -> Path:
    """Resolve path the same way SelvaLoraTrainer does (relative → ComfyUI output dir)."""
    p = Path(raw.strip())
    unix_style_on_windows = (
        sys.platform == "win32" and p.is_absolute() and not p.drive
    )
    if not p.is_absolute() or unix_style_on_windows:
        p = Path(folder_paths.get_output_directory()) / p.relative_to(p.anchor)
    return p


def _merge_config(base: dict, experiment: dict) -> dict:
    """Merge base defaults + file base + experiment overrides."""
    cfg = dict(_PARAM_DEFAULTS)
    cfg.update(base)
    # Don't carry id/description into the training params
    cfg.update({k: v for k, v in experiment.items() if k not in ("id", "description")})
    return cfg


def _loss_at_steps(loss_history: list, log_interval: int, save_every: int,
                   start_step: int, total_steps: int) -> dict:
    """Build a dict of {step: loss} at each save_every boundary.

    loss_history[i] = average loss over steps [start + i*log_interval + 1 …
                                                 start + (i+1)*log_interval].
    """
    result = {}
    targets = range(save_every, total_steps + 1, save_every)
    for target in targets:
        # index of the loss entry nearest to this step
        idx = (target - start_step) // log_interval - 1
        if 0 <= idx < len(loss_history):
            result[str(target)] = round(loss_history[idx], 6)
    return result


def _draw_comparison_curves(
    experiments_data: list,   # list of dicts: {id, loss_history, log_interval, start_step}
) -> Image.Image:
    """Draw all smoothed loss curves on the same axes, one color per experiment."""
    W, H = 900, 420
    pl, pr, pt, pb = 75, 160, 30, 50  # wider right margin for legend

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = W - pl - pr
    ph = H - pt - pb

    # Collect all smoothed series
    series = []
    for i, ed in enumerate(experiments_data):
        lh = ed.get("loss_history") or []
        if len(lh) < 2:
            continue
        sm = _smooth_losses(lh)
        series.append({
            "id":         ed["id"],
            "smoothed":   sm,
            "log_interval": ed.get("log_interval", 50),
            "start_step": ed.get("start_step", 0),
            "color":      _PALETTE[i % len(_PALETTE)],
        })

    if not series:
        draw.text((pl + 10, pt + 10), "No data to plot", fill=(80, 80, 80))
        return img

    all_vals = [v for s in series for v in s["smoothed"]]
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1e-6
    rng = hi - lo

    # Horizontal grid + y-axis labels
    for i in range(5):
        y   = pt + int(i * ph / 4)
        val = hi - i * rng / 4
        draw.line([(pl, y), (W - pr, y)], fill=(220, 220, 220), width=1)
        draw.text((2, y - 7), f"{val:.4f}", fill=(100, 100, 100))

    # Draw each curve
    for s in series:
        n    = len(s["smoothed"])
        pts  = []
        for j, v in enumerate(s["smoothed"]):
            x = pl + int(j * pw / max(n - 1, 1))
            y = pt + int((1.0 - (v - lo) / rng) * ph)
            pts.append((x, y))
        draw.line(pts, fill=s["color"], width=2)

    # Axes
    draw.line([(pl, pt), (pl, H - pb)],         fill=(40, 40, 40), width=1)
    draw.line([(pl, H - pb), (W - pr, H - pb)], fill=(40, 40, 40), width=1)
    draw.text((pl + 4, 8), "Loss comparison (smoothed)", fill=(40, 40, 40))

    # Legend (right side)
    lx = W - pr + 10
    ly = pt
    for s in series:
        draw.rectangle([(lx, ly + 3), (lx + 14, ly + 13)], fill=s["color"])
        draw.text((lx + 18, ly), s["id"][:20], fill=(40, 40, 40))
        ly += 20

    return img


class SelvaLoraScheduler:
    """Runs a sweep of LoRA training experiments defined in a JSON file.

    The dataset (VAE encoding + .npz loading) is performed once and shared
    across all experiments. Each experiment deep-copies the generator and trains
    independently. Results are written to `experiment_summary.json` after every
    completed run so partial results are preserved if the sweep is interrupted.
    """

    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "run"
    RETURN_TYPES  = ("STRING", "IMAGE")
    RETURN_NAMES  = ("summary_path", "comparison_curves")
    OUTPUT_TOOLTIPS = (
        "Path to experiment_summary.json — share this file to compare runs.",
        "All smoothed loss curves overlaid on the same axes.",
    )
    DESCRIPTION = (
        "Runs a series of LoRA training experiments defined in a JSON sweep file. "
        "The dataset is encoded once and reused across all experiments. "
        "Results (loss, config, adapter paths) are collected in experiment_summary.json."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "experiments_file": ("STRING", {
                    "default": "experiments.json",
                    "tooltip": (
                        "Path to JSON sweep file. Relative paths resolve to the ComfyUI "
                        "models directory; absolute paths are used as-is. "
                        "See LORA_TRAINING.md for the file format."
                    ),
                }),
            }
        }

    def run(self, model, experiments_file):
        # ------------------------------------------------------------------
        # 1. Read + validate the JSON file
        # ------------------------------------------------------------------
        exp_path = Path(experiments_file.strip())
        if not exp_path.is_absolute():
            # Try relative to ComfyUI models dir first, then output dir
            candidate = Path(folder_paths.models_dir) / exp_path
            if not candidate.exists():
                candidate = Path(folder_paths.get_output_directory()) / exp_path
            exp_path = candidate
        if not exp_path.exists():
            raise FileNotFoundError(
                f"[LoRA Scheduler] Experiment file not found: {exp_path}"
            )

        spec = json.loads(exp_path.read_text(encoding="utf-8"))

        if "experiments" not in spec or not spec["experiments"]:
            raise ValueError("[LoRA Scheduler] 'experiments' list is missing or empty.")
        for i, exp in enumerate(spec["experiments"]):
            if "id" not in exp:
                raise ValueError(
                    f"[LoRA Scheduler] Experiment at index {i} is missing required 'id' field."
                )

        sweep_name  = spec.get("name", exp_path.stem)
        description = spec.get("description", "")
        base_cfg    = spec.get("base", {})

        # ------------------------------------------------------------------
        # 2. Resolve data_dir and output_root
        # ------------------------------------------------------------------
        if "data_dir" not in spec:
            raise ValueError("[LoRA Scheduler] 'data_dir' is required in the sweep file.")
        data_dir    = _resolve_path(spec["data_dir"])
        output_root = _resolve_path(spec.get("output_root", f"lora_sweeps/{sweep_name}"))
        output_root.mkdir(parents=True, exist_ok=True)

        device = get_device()
        dtype  = model["dtype"]

        print(f"\n[LoRA Scheduler] Sweep '{sweep_name}': "
              f"{len(spec['experiments'])} experiment(s)", flush=True)
        if description:
            print(f"[LoRA Scheduler] {description}", flush=True)
        print(f"[LoRA Scheduler] data_dir    = {data_dir}", flush=True)
        print(f"[LoRA Scheduler] output_root = {output_root}\n", flush=True)

        # ------------------------------------------------------------------
        # 3. Load + encode dataset once
        # ------------------------------------------------------------------
        n_clips = len(list(data_dir.glob("*.npz")))
        dataset = _prepare_dataset(model, data_dir, device)

        # ------------------------------------------------------------------
        # 4. Build or restore the summary (resume-aware)
        # ------------------------------------------------------------------
        summary_path   = output_root / "experiment_summary.json"
        completed_ids  = set()
        all_curve_data = []   # collected for comparison image

        if summary_path.exists():
            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                for rec in existing.get("experiments", []):
                    if rec.get("results", {}).get("status") == "completed":
                        completed_ids.add(rec["id"])
                        lh = rec["results"].get("loss_history", [])
                        all_curve_data.append({
                            "id":           rec["id"],
                            "loss_history": lh,
                            "log_interval": rec["results"].get("log_interval", 50),
                            "start_step":   0,
                        })
                # Restore the original summary, clear completed_at so it gets set again
                summary = existing
                summary["completed_at"] = None
                if completed_ids:
                    print(f"[LoRA Scheduler] Resuming — skipping {len(completed_ids)} "
                          f"completed experiment(s): {sorted(completed_ids)}", flush=True)
            except Exception as e:
                print(f"[LoRA Scheduler] Could not read existing summary ({e}) — starting fresh",
                      flush=True)
                completed_ids = set()
                all_curve_data = []
                summary = None

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

        def _write_summary():
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _write_summary()

        # ------------------------------------------------------------------
        # 5. Run each experiment
        # ------------------------------------------------------------------
        trainer      = SelvaLoraTrainer()
        pbar_outer   = comfy.utils.ProgressBar(len(spec["experiments"]))
        log_interval = 50   # matches _train_inner

        feature_utils_orig = model["feature_utils"]
        seq_cfg            = model["seq_cfg"]
        variant            = model["variant"]
        mode               = model["mode"]

        for exp in spec["experiments"]:
            exp_id  = exp["id"]
            exp_desc = exp.get("description", "")

            if exp_id in completed_ids:
                print(f"[LoRA Scheduler] Skipping '{exp_id}' (already completed)", flush=True)
                pbar_outer.update(1)
                continue
            cfg     = _merge_config(base_cfg, exp)

            # Required training params
            steps       = int(cfg.get("steps",       2000))
            rank        = int(cfg.get("rank",        16))
            lr          = float(cfg.get("lr",         1e-4))
            alpha       = float(cfg.get("alpha",      0.0))
            target      = str(cfg.get("target",       "attn.qkv"))
            batch_size  = int(cfg.get("batch_size",   4))
            warmup      = int(cfg.get("warmup_steps", 100))
            grad_accum  = int(cfg.get("grad_accum",   1))
            save_every  = int(cfg.get("save_every",   500))
            resume_path = str(cfg.get("resume_path",  ""))
            seed        = int(cfg.get("seed",         42))
            ts_mode     = str(cfg.get("timestep_mode",      "uniform"))
            ln_sigma    = float(cfg.get("logit_normal_sigma", 1.0))
            curr_switch = float(cfg.get("curriculum_switch",  0.6))
            dropout     = float(cfg.get("lora_dropout",       0.0))
            plus_ratio  = float(cfg.get("lora_plus_ratio",    1.0))
            alpha_val   = alpha if alpha > 0.0 else float(rank)
            target_suffixes = tuple(target.strip().split())

            output_dir = output_root / exp_id
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[LoRA Scheduler] ── Experiment '{exp_id}' ──", flush=True)
            if exp_desc:
                print(f"[LoRA Scheduler] {exp_desc}", flush=True)

            exp_record = {
                "id":          exp_id,
                "description": exp_desc,
                "config": {
                    "rank": rank, "alpha": alpha_val, "lr": lr, "steps": steps,
                    "batch_size": batch_size, "warmup_steps": warmup,
                    "grad_accum": grad_accum, "save_every": save_every,
                    "seed": seed, "target": list(target_suffixes),
                    "timestep_mode": ts_mode, "logit_normal_sigma": ln_sigma,
                    "curriculum_switch": curr_switch,
                    "lora_dropout": dropout, "lora_plus_ratio": plus_ratio,
                },
                "results":     {"status": "running"},
                "adapter_path": None,
                "output_dir":   str(output_dir),
            }
            summary["experiments"].append(exp_record)
            _write_summary()

            t_start = time.monotonic()
            try:
                with torch.inference_mode(False), torch.enable_grad():
                    r = trainer._train_inner(
                        model, dataset, feature_utils_orig, seq_cfg,
                        device, dtype, variant, mode,
                        data_dir, output_dir, steps, rank, lr,
                        alpha_val, target_suffixes, batch_size, warmup,
                        grad_accum, save_every, resume_path, seed,
                        ts_mode, ln_sigma, curr_switch, dropout, plus_ratio,
                    )

                duration          = time.monotonic() - t_start
                loss_history      = r["loss_history"]
                grad_norm_history = r.get("grad_norm_history", [])
                spectral_metrics  = r.get("spectral_metrics", {})
                run_start_step    = r.get("start_step", 0)
                smoothed          = _smooth_losses(loss_history) if loss_history else []

                # Scalar summary metrics
                final_loss    = round(smoothed[-1], 6) if smoothed else None
                min_loss      = round(min(smoothed), 6) if smoothed else None
                min_idx       = smoothed.index(min(smoothed)) if smoothed else None
                min_loss_step = (
                    run_start_step + (min_idx + 1) * log_interval
                    if min_idx is not None else None
                )

                # Stability: std-dev of raw loss over last 25% of steps
                if loss_history:
                    quarter    = max(1, len(loss_history) // 4)
                    last_q     = loss_history[-quarter:]
                    loss_std_last_quarter = round(float(np.std(last_q)), 6)
                else:
                    loss_std_last_quarter = None

                exp_record["results"] = {
                    "status":               "completed",
                    "final_loss":           final_loss,
                    "min_loss":             min_loss,
                    "min_loss_step":        min_loss_step,
                    "loss_std_last_quarter": loss_std_last_quarter,
                    "loss_at_steps":        _loss_at_steps(
                        loss_history, log_interval, save_every, run_start_step, steps
                    ),
                    "loss_history":         [round(v, 6) for v in loss_history],
                    "grad_norm_history":    grad_norm_history,
                    "spectral_metrics":     {str(k): v for k, v in spectral_metrics.items()},
                    "log_interval":         log_interval,
                    "duration_seconds":     round(duration, 1),
                }
                exp_record["adapter_path"] = r["adapter_path"]

                all_curve_data.append({
                    "id":           exp_id,
                    "loss_history": loss_history,
                    "log_interval": log_interval,
                    "start_step":   0,
                })

            except SkipExperiment as e:
                duration = time.monotonic() - t_start
                print(f"[LoRA Scheduler] Experiment '{exp_id}' skipped: {e}", flush=True)
                exp_record["results"] = {
                    "status":           "skipped",
                    "error":            str(e),
                    "duration_seconds": round(duration, 1),
                }
                _write_summary()
                pbar_outer.update(1)
                continue

            except Exception as e:
                duration = time.monotonic() - t_start
                print(f"[LoRA Scheduler] Experiment '{exp_id}' failed: {e}", flush=True)
                traceback.print_exc()
                exp_record["results"] = {
                    "status":           "failed",
                    "error":            str(e),
                    "duration_seconds": round(duration, 1),
                }
                _write_summary()
                pbar_outer.update(1)
                # Continue to next experiment rather than aborting the whole sweep
                continue

            _write_summary()
            pbar_outer.update(1)

        # ------------------------------------------------------------------
        # 6. Finalise summary
        # ------------------------------------------------------------------
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_summary()
        print(f"\n[LoRA Scheduler] Sweep complete. Summary: {summary_path}", flush=True)

        # ------------------------------------------------------------------
        # 7. Comparison image
        # ------------------------------------------------------------------
        comparison_img = _draw_comparison_curves(all_curve_data)
        comparison_img.save(str(output_root / "loss_comparison.png"))
        comparison_tensor = _pil_to_tensor(comparison_img)

        return (str(summary_path), comparison_tensor)
