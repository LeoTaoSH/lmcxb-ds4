#!/usr/bin/env python3
"""Download a pinned Kimi Linear checkpoint for DS4 conversion.

The immutable revision is mandatory so a GGUF can be reproduced and compared
against the same upstream checkpoint during review.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MODELS = {
    "base": "moonshotai/Kimi-Linear-48B-A3B-Base",
    "instruct": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
}

ALLOW_PATTERNS = [
    "*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tiktoken.model",
    "chat_template.jinja",
    "modeling_kimi.py",
    "configuration_kimi.py",
    "tokenization_kimi.py",
    "tokenizer_kimi.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--revision", required=True,
                        help="immutable 40-character Hugging Face commit SHA")
    parser.add_argument("--out", required=True, type=Path,
                        help="local checkpoint directory")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit SHA")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required; install it in the conversion environment"
        ) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=MODELS[args.model],
        revision=args.revision,
        local_dir=str(args.out),
        allow_patterns=ALLOW_PATTERNS,
    )
    print(path)


if __name__ == "__main__":
    main()
