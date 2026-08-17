"""Pre-download model weights (no GPU needed). Usage: python predownload.py [--models ...]"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_INFO, CROSS_MODEL_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(CROSS_MODEL_MODELS))
    args = parser.parse_args()
    for key in args.models.split(","):
        key = key.strip()
        if key not in MODEL_INFO:
            continue
        name = MODEL_INFO[key]["model_name"]
        print(f"Downloading {key} ({name})...", flush=True)
        AutoTokenizer.from_pretrained(name)
        AutoModelForCausalLM.from_pretrained(
            name,
            attn_implementation="eager",
            dtype="float16" if MODEL_INFO[key]["dtype"] == "fp16" else "float32",
        )
        print(f"  done {key}", flush=True)


if __name__ == "__main__":
    main()
