#!/usr/bin/env python3
"""Small CPU/CUDA numerical regression suite; requires the full Qwen model.

This is a short-context regression guard, not a model-quality certification.
Each invocation computes the CPU reference and compares CUDA chunk endpoints.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="./ds4")
    parser.add_argument("--max-relative-rms", type=float, default=0.005)
    parser.add_argument("--max-absolute", type=float, default=0.05)
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="qwen-numerics-"))
    print(f"Logs: {directory}", flush=True)
    cases = [
        ("english", "The quick brown fox jumps over the lazy dog."),
        ("code", "def add(a, b): return a + b"),
    ]
    results = []
    for name, prompt in cases:
        for chunk in (1, 9):
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("DS4_QWEN4_")}
            env.update(DS4_QWEN4_GPU="1", DS4_QWEN4_GPU_CHUNK=str(chunk))
            path = directory / f"{name}-chunk{chunk}.log"
            print(f"Running {name}, chunk={chunk}: {path}", flush=True)
            with path.open("w") as log:
                proc = subprocess.run([
                    args.binary, "--cuda", "--first-token-test", "--raw",
                    "-m", args.model, "-c", "256", "-p", prompt,
                ], env=env, stdout=log, stderr=subprocess.STDOUT)
            data = path.read_text()
            rows = re.findall(
                r"pos\s+(\d+): max\|gpu-cpu\|=(\S+) top1 cpu=(\d+) gpu=(\d+)[^\n]*\n"
                r"\s+rms=(\S+) relative_rms=(\S+) nonfinite=(\d+)", data)
            summary = re.search(r"GPU-vs-CPU: (\d+) tokens .*top1 agree (\d+)/(\d+)", data)
            passed = proc.returncode == 0 and bool(rows) and summary is not None
            if summary:
                count = int(summary[1])
                passed &= len(rows) == (count + chunk - 1) // chunk
                passed &= int(rows[-1][0]) == count - 1 if rows else False
            for _, absolute, cpu, gpu, rms, relative, nonfinite in rows:
                passed &= all(math.isfinite(float(x)) for x in (absolute, rms, relative))
                passed &= float(absolute) <= args.max_absolute
                passed &= float(relative) <= args.max_relative_rms
                passed &= nonfinite == "0" and cpu == gpu
            result = dict(case=name, chunk=chunk, passed=bool(passed),
                          positions=len(rows),
                          max_absolute=max((float(r[1]) for r in rows), default=None),
                          max_relative_rms=max((float(r[5]) for r in rows), default=None),
                          log=str(path))
            results.append(result)
            print(json.dumps(result), flush=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
