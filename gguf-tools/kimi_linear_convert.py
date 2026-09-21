#!/usr/bin/env python3
"""Convert a Kimi Linear (KimiLinearForCausalLM) HF checkpoint into a DS4 GGUF.

Wraps llama.cpp's conversion package, so the file keeps the upstream `kimi-linear`
schema (20 KDA linear-attention layers + 7 MLA full-attention layers +
DeepSeek-V3-style MoE with 256 routed + 1 shared expert), then overrides the
tensor types to the DS4-supported Q8_0 profile. Norms, scalar params and the
per-expert correction stay F32; embeddings and the output head stay F16.

The ds4 loader decides per layer whether it is a KDA layer (uses the `kda_*`
tensor names) or an MLA layer (uses the `attn_*` tensor names) based on the
`kimi.linear_attn.kda_layers` metadata array emitted here.

Needs a llama.cpp master checkout (--llama-cpp or $LLAMA_CPP) and a Python with
torch, safetensors and transformers. Output is written to <out>.incomplete and
renamed when complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="HF checkpoint directory")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--source-repo", required=True,
                    choices=["moonshotai/Kimi-Linear-48B-A3B-Base",
                             "moonshotai/Kimi-Linear-48B-A3B-Instruct"],
                    help="Hugging Face repository used for the checkpoint")
    ap.add_argument("--source-revision", required=True, help="pinned HF checkpoint commit SHA")
    ap.add_argument("--llama-revision", required=True,
                    help="pinned llama.cpp commit SHA providing the converter")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0"],
                    help="type of the dense projections (only the tested Q8_0 profile is supported)")
    ap.add_argument("--experts", default="q8_0", choices=["q8_0"],
                    help="routed/shared expert type (only Q8_0 is supported)")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", os.path.expanduser("~/llamacpp")),
                    help="llama.cpp checkout with conversion/kimi_linear.py (default: $LLAMA_CPP or ~/llamacpp)")
    ap.add_argument("--dry-run", action="store_true", help="plan the tensor types without writing")
    ap.add_argument("--verbose", action="store_true", help="pass --verbose to the llama.cpp converter")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.llama_cpp, "conversion", "kimi_linear.py")):
        sys.exit(f"{args.llama_cpp}/conversion/kimi_linear.py not found: point --llama-cpp at a llama.cpp master checkout")
    if not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        sys.exit("--source-revision must be an immutable 40-character commit SHA")
    if not re.fullmatch("[0-9a-f]{40}", args.llama_revision):
        sys.exit("--llama-revision must be an immutable 40-character commit SHA")
    try:
        actual_llama_revision = subprocess.check_output(
            ["git", "-C", args.llama_cpp, "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f"cannot determine llama.cpp revision: {exc}")
    if actual_llama_revision != args.llama_revision:
        sys.exit(
            "llama.cpp checkout revision mismatch: "
            f"expected {args.llama_revision}, found {actual_llama_revision}"
        )
    sys.path.insert(0, os.path.join(args.llama_cpp, "gguf-py"))
    sys.path.insert(0, args.llama_cpp)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    import gguf
    from conversion.kimi_linear import KimiLinearModel

    Q = gguf.GGMLQuantizationType
    T = gguf.MODEL_TENSOR
    experts_t = Q.Q8_0

    class DS4KimiLinearModel(KimiLinearModel):
        model_arch = gguf.MODEL_ARCH.KIMI_LINEAR

        def set_gguf_parameters(self):
            # Persist the KDA layer classification so the ds4 loader can map each
            # blk.N to KDA or MLA without re-deriving it from tensor presence.
            super().set_gguf_parameters()
            self.gguf_writer.add_string(
                "general.source.url", f"https://huggingface.co/{args.source_repo}"
            )
            self.gguf_writer.add_string(
                "general.source.revision", args.source_revision
            )
            self.gguf_writer.add_string(
                "ds4.converter.llama_cpp_revision", args.llama_revision
            )
            hashtag = self.hparams.get("linear_attn_config", {})
            kda_layers = hashtag.get("kda_layers", []) if isinstance(hashtag, dict) else []
            if kda_layers:
                self.gguf_writer.add_array("kimi.linear_attn.kda_layers", [int(x) for x in kda_layers])

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            # Norms, scalar params and scalar-shaped KDA parameters stay F32.
            if n_dims <= 1 or new_name.endswith(("conv1d.weight", "o_norm.weight",
                                                 "kv_a_layernorm.weight", "q_a_layernorm.weight")):
                return Q.F32
            # The vocabulary embeddings and the un-tied output head stay F16
            # (the ds4 layout expects a full-precision embedding table).
            for key in (T.TOKEN_EMBD, T.OUTPUT):
                if self.match_model_tensor_name(new_name, key, bid):
                    return Q.F16
            # Routed and shared expert projections follow --experts.
            for key in (T.FFN_GATE_EXP, T.FFN_UP_EXP, T.FFN_DOWN_EXP,
                        T.FFN_GATE_SHEXP, T.FFN_UP_SHEXP, T.FFN_DOWN_SHEXP):
                if self.match_model_tensor_name(new_name, key, bid):
                    return experts_t
            return super().tensor_force_quant(name, new_name, bid, n_dims)

    ftype = gguf.LlamaFileType.MOSTLY_Q8_0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".incomplete")
    if out.exists() or tmp.exists():
        sys.exit("Output already exists; inspect it or choose a new path")
    model = DS4KimiLinearModel(Path(args.src), ftype, tmp, use_temp_file=False, dry_run=args.dry_run)
    model.write()
    if args.dry_run:
        return
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
