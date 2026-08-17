"""Kaggle runner for the GPT-2 Large fixed-code robustness sweep."""

# The embedded package archive is intentionally a single long base64 literal.
# flake8: noqa: E501

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
SOURCE = ROOT / "decoding-robustness"
RESULTS = ROOT / "results_v2"
PACKAGE_ZIP_B64 = "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"
MODELS = ["gpt2-large"]
PERTURBATIONS = ["shuffle", "typo", "synonym"]
N_SAMPLES = int(os.environ.get("ROBUSTNESS_N_SAMPLES", "300"))


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
) -> None:
    print("$", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {command}"
        )


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers<4.50",
            "datasets>=2.21",
            "pandas",
            "numpy",
            "scipy",
            "nltk",
            "python-Levenshtein",
        ]
    )


def prepare_source() -> None:
    import base64
    import io
    import shutil
    import zipfile

    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    SOURCE.mkdir(parents=True, exist_ok=True)
    package = base64.b64decode(PACKAGE_ZIP_B64)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        archive.extractall(SOURCE)


def main() -> None:
    started = time.time()
    log_path = ROOT / "five_model_run.log"
    complete_path = ROOT / "run_complete.json"
    complete_path.unlink(missing_ok=True)
    # A Kaggle retry can reuse /kaggle/working; clear stale failure logs.
    log_path.write_text("", encoding="utf-8")
    child_log_path = RESULTS / "run_cross_model.log"
    child_log_path.parent.mkdir(parents=True, exist_ok=True)
    child_log_path.write_text("", encoding="utf-8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    install_dependencies()
    prepare_source()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run on CPU")
    manifest = {
        "models": MODELS,
        "perturbations": PERTURBATIONS,
        "n_samples": N_SAMPLES,
        "cuda": torch.cuda.get_device_name(0),
        "cuda_count": torch.cuda.device_count(),
    }
    (ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "run_cross_model.py",
            "--models",
            ",".join(MODELS),
            "--ptb-types",
            ",".join(PERTURBATIONS),
            "--n-samples",
            str(N_SAMPLES),
            "--out-root",
            str(RESULTS),
        ],
        cwd=SOURCE,
        log_path=log_path,
    )
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (log_path, RESULTS / "run_cross_model.log")
        if path.exists()
    )
    if "!!! FAILED" in log_text:
        raise RuntimeError(
            "One or more model/perturbation jobs failed; " "see five_model_run.log"
        )
    complete_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
