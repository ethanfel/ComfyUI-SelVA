from pathlib import Path

import folder_paths

from .utils import SELVA_CATEGORY


class SelvaSkipExperiment:
    """Writes skip_current.flag into a sweep output_root.

    Queue this node while a SelVA LoRA Scheduler sweep is running to skip
    the current experiment and move to the next one. The trainer picks up
    the flag within 50 steps (~a few seconds).
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_root": ("STRING", {
                    "default": "",
                    "tooltip": "output_root of the running sweep — same value as in your experiments JSON.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flag_path",)
    OUTPUT_TOOLTIPS = ("Path where the flag was written.",)
    FUNCTION = "skip"
    CATEGORY = SELVA_CATEGORY
    DESCRIPTION = (
        "Signals the running SelVA LoRA Scheduler to skip the current experiment "
        "and move to the next one. Queue this node while the scheduler is running. "
        "Partial scalars collected so far are saved in the summary."
    )

    def skip(self, output_root: str):
        p = Path(output_root.strip())
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p
        if not p.exists():
            raise FileNotFoundError(f"[SelVA Skip] output_root not found: {p}")

        flag = p / "skip_current.flag"
        flag.touch()
        print(f"[SelVA Skip] Flag written: {flag}", flush=True)
        return (str(flag),)
