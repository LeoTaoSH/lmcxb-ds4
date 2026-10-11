#!/usr/bin/env python3
"""Frozen, activation-aware two-layer expert staging on the pinned MiMo MOPD.

Prepare six capture commands (does not execute inference):
  python3 speed-bench/expert_activation_prefetch.py capture-commands --capture-root DIR
After the initial three captures and a CPU-fitting lease:
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 \
    speed-bench/expert_activation_prefetch.py train --capture-root DIR --output-dir MODEL
Only after freeze.json exists, capture/evaluate the three held-out prompts:
  python3 speed-bench/expert_activation_prefetch.py evaluate \
    --capture-root DIR --freeze MODEL/freeze.json --output MODEL/holdout-replay.json

The recorder writes unchanged v1 JSONL plus DS4ACT01 f32 sidecars. Predictors see
only the current normalized activation and current real router IDs. Future route
IDs are training labels. All staging accounting comes from expert_prefetch_replay.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import struct
import sys
import time

import numpy as np

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import expert_prefetch_replay as prefetch

replay = prefetch.replay
ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
DIM = 4096
EXPERTS = 256
SELECTED = 8
FEATURES = 256
TRAIN_NAMES = ("coding", "reasoning", "tool-result")
HOLDOUT_NAMES = ("rust-parser", "probability", "tool-debug")
MODEL_NAME = "MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf"
REVISION = "b12f5573350bb1704513fcf119fca48e498b0423"
HEADER = struct.Struct("<8s6IQ")
FOOTER = struct.Struct("<8sQ")
RECORD_BYTES = 24 + 4 * SELECTED + 4 * DIM
RECORD_DTYPE = np.dtype([("index", "<u8"), ("token", "<u8"),
                         ("layer", "<u4"), ("count", "<u4"),
                         ("ids", "<i4", (SELECTED,)), ("x", "<f4", (DIM,))])
SETTINGS = {"model": MODEL_NAME, "source_revision": REVISION,
            "lookahead_layer_events": prefetch.LOOKAHEAD,
            "widths": list(prefetch.WIDTHS), "total_cache_slots": prefetch.CACHE_SLOTS,
            "input_dim": DIM, "expert_domain": EXPERTS, "selected": SELECTED,
            "activation_features": FEATURES,
            "feature_selection": "highest training-prefix variance; coordinate-ID ties",
            "inputs": "standardized selected current ffn_norm coordinates plus current route multi-hot",
            "head": "per-layer regularized linear multi-label least squares; unpenalized intercept",
            "ridge_candidates": [0.001, 0.01, 0.1, 1.0],
            "selection": "maximize aggregate top-four precision on final 20% of initial prompts; stronger-ridge ties",
            "validation_boundary": "target wholly in prefix or source wholly in suffix; no cross-prompt pairs",
            "final_fit": "refit feature selection, normalization and head on all initial three prompts",
            "tie_break": "ascending expert ID", "staging": prefetch.SETTINGS}
LIMITS = ["Only the initial three prompts fit coordinates, normalization, ridge strength and weights",
          "Inference reads current normalized activation and current router IDs only",
          "Held-out prompts are evaluated only against the hash-frozen model and source",
          "Immediate-ready staging ignores actual SSD latency, contention and dispatch scheduling",
          "Every prefetch consumes the original 5134-slot budget and its bytes are charged, including mistakes and tail loads",
          "Logical requested bytes and CPU NumPy prediction costs are simulated traffic evidence, not serving throughput"]


@dataclass
class Capture:
    name: str
    trace: object
    path: Path
    records: object
    model_bytes: int

    @classmethod
    def load(cls, name, trace_path, sidecar_path):
        trace = replay.load_trace(trace_path)
        sidecar_path = Path(sidecar_path)
        size = sidecar_path.stat().st_size
        if size < HEADER.size + FOOTER.size:
            raise ValueError("activation sidecar header/footer is incomplete")
        with open(sidecar_path, "rb") as source:
            header = HEADER.unpack(source.read(HEADER.size))
            source.seek(-FOOTER.size, 2)
            magic, count = FOOTER.unpack(source.read(FOOTER.size))
        expected = (b"DS4ACT01", VERSION, 0x01020304, DIM, EXPERTS, SELECTED, RECORD_BYTES)
        if header[:-1] != expected or header[-1] < 1:
            raise ValueError("activation sidecar header/version/shape differs")
        if magic != b"DS4END01" or count != len(trace.routes):
            raise ValueError("activation sidecar footer/route count differs")
        if size != HEADER.size + count * RECORD_BYTES + FOOTER.size:
            raise ValueError("activation sidecar record size or trailing bytes differ")
        records = np.memmap(sidecar_path, mode="r", dtype=RECORD_DTYPE,
                            offset=HEADER.size, shape=(count,))
        for index, route in enumerate(trace.routes):
            record = records[index]
            if (int(record["index"]), int(record["token"]), int(record["layer"]),
                int(record["count"]), tuple(map(int, record["ids"]))) != (
                    index, route.token, route.layer, len(route.ids), route.ids):
                raise ValueError(f"activation sidecar metadata differs at route {index}")
        for start in range(0, count, 256):
            if not np.isfinite(records["x"][start:start + 256]).all():
                raise ValueError("activation sidecar contains nonfinite input")
        replay.replay(trace)
        return cls(name, trace, sidecar_path, records, header[-1])

    def manifest(self):
        return {"name": self.name, "trace": prefetch.input_record(self.trace.source, self.trace),
                "activations": {"source": str(self.path), "sha256": prefetch.file_hash(self.path),
                                "file_bytes": self.path.stat().st_size,
                                "records": len(self.records), "input_dim": DIM,
                                "model_file_bytes": self.model_bytes}}


def prompt_path(name):
    directory = ROOT / "speed-bench/mimo-cache"
    return directory / (("holdout-20261011/" if name in HOLDOUT_NAMES else "") + name + ".txt")


def capture_commands(root):
    root = Path(root).resolve()
    commands = []
    for split, names in (("train", TRAIN_NAMES), ("holdout", HOLDOUT_NAMES)):
        for name in names:
            prompt = prompt_path(name)
            directory = root / split
            env = {"DS4_METAL_STREAMING_EXPERT_CACHE_TRACE": str(directory / (name + ".jsonl")),
                   "DS4_MIMO2_ACTIVATION_TRACE": str(directory / (name + ".bin"))}
            argv = ["./ds4", "-m", "gguf/" + MODEL_NAME, "--ssd-streaming", "--ctx", "4096",
                    "--raw-prompt", "--prompt-file", str(prompt), "--temp", "0", "--nothink", "--tokens", "512",
                    "--dump-logprobs", str(directory / (name + ".logprobs.json"))]
            commands.append({"name": name, "split": split, "prompt": str(prompt),
                             "prompt_sha256": prefetch.file_hash(prompt), "env": env, "argv": argv,
                             "shell": " ".join([*(key + "=" + shlex.quote(value) for key, value in env.items()),
                                                  *(shlex.quote(value) for value in argv)])})
    return {"version": VERSION, "model": MODEL_NAME, "source_revision": REVISION,
            "capture_root": str(root), "tokens": 512, "ordinary_real_routing": True,
            "estimated_sidecar_bytes_per_case": HEADER.size + 512 * 47 * RECORD_BYTES + FOOTER.size,
            "commands": commands}


def load_captures(root, split):
    root = Path(root).resolve()
    with open(root / "capture-commands.json", encoding="utf-8") as source:
        manifest = json.load(source)
    if (manifest["version"], manifest["model"], manifest["source_revision"], manifest["tokens"],
        manifest["ordinary_real_routing"]) != (VERSION, MODEL_NAME, REVISION, 512, True):
        raise ValueError("capture manifest differs from pinned ordinary MOPD target")
    names = TRAIN_NAMES if split == "train" else HOLDOUT_NAMES
    records = {record["name"]: record for record in manifest["commands"] if record["split"] == split}
    if set(records) != set(names):
        raise ValueError("capture manifest prompt membership differs")
    initial_prompt_hashes = {}
    if split == "train":
        with open(ROOT / "speed-bench/mimo-cache/20261010/manifest.json", encoding="utf-8") as source:
            initial = json.load(source)
        if initial["model"] != MODEL_NAME or initial["source_revision"] != REVISION:
            raise ValueError("initial prompt manifest model differs")
        initial_prompt_hashes = {row["workload"]: row["prompt_sha256"] for row in initial["cases"]}
    captures = []
    for name in names:
        record = records[name]
        if Path(record["prompt"]).resolve() != prompt_path(name).resolve() or (
                prefetch.file_hash(prompt_path(name)) != record["prompt_sha256"]):
            raise ValueError("capture prompt identity differs")
        if split == "train" and record["prompt_sha256"] != initial_prompt_hashes[name]:
            raise ValueError("training must use the original three fixed prompts")
        capture = Capture.load(name, root / split / (name + ".jsonl"), root / split / (name + ".bin"))
        if capture.trace.snapshot.budget != prefetch.CACHE_SLOTS:
            raise ValueError("activation experiment requires 5134 original expert slots")
        layers = prefetch.layer_sequence(capture.trace)
        if layers != tuple(range(1, 48)) or len(capture.trace.routes) != 512 * 47:
            raise ValueError("capture must contain 512 ordinary complete MiMo routed-layer evaluations")
        captures.append(capture)
    if len({capture.model_bytes for capture in captures}) != 1:
        raise ValueError("capture model file sizes differ")
    return captures


def layer_samples(captures, layer, subset="all"):
    xs, current, targets = [], [], []
    for capture in captures:
        routes = capture.trace.routes
        # Split by token, with a lookahead gap that prevents boundary-label overlap.
        boundary_token = routes[0].token + int(0.8 * (routes[-1].token - routes[0].token + 1))
        for index, source in enumerate(routes[:-prefetch.LOOKAHEAD]):
            if source.layer != layer:
                continue
            target = routes[index + prefetch.LOOKAHEAD]
            if subset == "prefix" and target.token >= boundary_token:
                continue
            if subset == "suffix" and source.token < boundary_token:
                continue
            xs.append(capture.records["x"][index])
            current.append(source.ids)
            targets.append(target.ids)
    if not xs:
        raise ValueError("layer has no training pairs")
    labels = np.zeros((len(xs), EXPERTS), dtype=np.float64)
    for row, ids in enumerate(targets):
        labels[row, list(ids)] = 1.0
    return np.asarray(xs, dtype=np.float64), current, labels


def feature_stats(x, count=FEATURES):
    variance = x.var(axis=0)
    indices = np.argsort(-variance, kind="stable")[:count]
    mean = x[:, indices].mean(axis=0)
    scale = np.maximum(x[:, indices].std(axis=0), 1e-6)
    return indices.astype(np.int32), mean, scale


def design(x, ids, indices, mean, scale):
    result = np.zeros((len(x), len(indices) + EXPERTS), dtype=np.float64)
    result[:, :len(indices)] = (x[:, indices] - mean) / scale
    for row, route in enumerate(ids):
        result[row, len(indices) + np.asarray(route, dtype=np.int32)] = 1.0
    return result


def ridge(design_matrix, labels, strength):
    xm, ym = design_matrix.mean(axis=0), labels.mean(axis=0)
    centered, targets = design_matrix - xm, labels - ym
    gram = centered.T @ centered / len(centered)
    gram.flat[::len(gram) + 1] += strength
    weights = np.linalg.solve(gram, centered.T @ targets / len(centered))
    bias = ym - xm @ weights
    return weights, bias


def top_ids(scores, width):
    return np.argsort(-scores, axis=-1, kind="stable")[..., :width]


def fit(captures, feature_count=FEATURES, candidates=None):
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    candidates = tuple(SETTINGS["ridge_candidates"] if candidates is None else candidates)
    layers = prefetch.layer_sequence(captures[0].trace)
    totals = {strength: [0, 0] for strength in candidates}
    per_layer = []
    for layer in layers:
        x, ids, labels = layer_samples(captures, layer, "prefix")
        vx, vids, vlabels = layer_samples(captures, layer, "suffix")
        indices, mean, scale = feature_stats(x, feature_count)
        training, validation = design(x, ids, indices, mean, scale), design(vx, vids, indices, mean, scale)
        scores = {}
        for strength in candidates:
            weights, bias = ridge(training, labels, strength)
            selected = top_ids(validation @ weights + bias, 4)
            correct = int(np.take_along_axis(vlabels, selected, axis=1).sum())
            totals[strength][0] += correct
            totals[strength][1] += len(validation) * 4
            scores[str(strength)] = {"correct": correct, "predictions": len(validation) * 4}
        per_layer.append({"layer": layer, "prefix_pairs": len(x), "suffix_pairs": len(vx), "scores": scores})
    strength = max(candidates, key=lambda value: (totals[value][0] / totals[value][1], value))
    arrays = {"layers": np.asarray(layers, dtype=np.int32)}
    pair_count = 0
    for layer in layers:
        x, ids, labels = layer_samples(captures, layer)
        indices, mean, scale = feature_stats(x, feature_count)
        weights, bias = ridge(design(x, ids, indices, mean, scale), labels, strength)
        arrays[f"indices_{layer}"] = indices
        arrays[f"mean_{layer}"] = mean.astype(np.float32)
        arrays[f"scale_{layer}"] = scale.astype(np.float32)
        arrays[f"weights_{layer}"] = weights.astype(np.float32)
        arrays[f"bias_{layer}"] = bias.astype(np.float32)
        pair_count += len(x)
    costs = {"cpu_seconds": time.process_time() - cpu_start,
             "wall_seconds": time.perf_counter() - wall_start, "training_pairs": pair_count,
             "selected_ridge": strength, "validation_per_layer": per_layer,
             "validation_aggregate": [{"ridge": value, "correct": totals[value][0],
                                       "predictions": totals[value][1],
                                       "precision": totals[value][0] / totals[value][1]} for value in candidates],
             "parameter_bytes": sum(value.nbytes for value in arrays.values())}
    return arrays, costs


class Predictor:
    def __init__(self, arrays):
        self.layers = tuple(map(int, arrays["layers"]))
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("invalid predictor layers")
        self.targets = {layer: self.layers[(index + prefetch.LOOKAHEAD) % len(self.layers)]
                        for index, layer in enumerate(self.layers)}
        self.heads = {}
        expected = {"layers"}
        for layer in self.layers:
            keys = [f"{name}_{layer}" for name in ("indices", "mean", "scale", "weights", "bias")]
            expected.update(keys)
            indices, mean, scale, weights, bias = (arrays[key] for key in keys)
            count = len(indices)
            if (not 1 <= count <= FEATURES or indices.dtype.kind not in "iu" or len(set(map(int, indices))) != count or
                np.any(indices < 0) or np.any(indices >= DIM) or
                mean.shape != (count,) or scale.shape != (count,) or
                weights.shape != (count + EXPERTS, EXPERTS) or bias.shape != (EXPERTS,) or
                np.any(scale <= 0) or not all(np.isfinite(value).all() for value in (mean, scale, weights, bias))):
                raise ValueError("invalid predictor head shape or values")
            self.heads[layer] = indices, mean, scale, weights, bias
        if set(arrays) != expected:
            raise ValueError("unexpected predictor arrays")

    def predict_current(self, route, x, width):
        if width not in prefetch.WIDTHS or route.layer not in self.heads:
            raise ValueError("route or width outside frozen predictor domain")
        if np.shape(x) != (DIM,) or not np.isfinite(x).all() or any(not 0 <= expert < EXPERTS for expert in route.ids):
            raise ValueError("current predictor input outside domain")
        indices, mean, scale, weights, bias = self.heads[route.layer]
        # One current activation, one current routing set. No future record is consulted.
        scores = ((x[indices] - mean) / scale) @ weights[:len(indices)] + bias
        for expert in route.ids:
            scores += weights[len(indices) + expert]
        return tuple((self.targets[route.layer], int(expert)) for expert in top_ids(scores, width))

    def bind(self, capture):
        return BoundPredictor(self, capture)


class BoundPredictor:
    def __init__(self, predictor, capture):
        self.predictor, self.capture, self.layers = predictor, capture, predictor.layers
        self.lookup = {(route.token, route.layer): index for index, route in enumerate(capture.trace.routes)}
        if len(self.lookup) != len(capture.trace.routes):
            raise ValueError("ordinary capture repeats token/layer metadata")

    def predict(self, route, width):
        index = self.lookup[(route.token, route.layer)]
        if route.ids != self.capture.trace.routes[index].ids:
            raise ValueError("prediction routing does not match current activation")
        return self.predictor.predict_current(route, self.capture.records["x"][index], width)


def load_frozen(path):
    path = Path(path)
    with open(path, encoding="utf-8") as source:
        freeze = json.load(source)
    if freeze["version"] != VERSION or freeze["settings"] != SETTINGS:
        raise ValueError("frozen activation parameters differ")
    for source, key in ((Path(__file__), "predictor_script_sha256"),
                        (Path(prefetch.__file__), "staging_script_sha256"),
                        (Path(replay.__file__), "baseline_script_sha256")):
        if prefetch.file_hash(source) != freeze[key]:
            raise ValueError("frozen activation source hash differs")
    model = path.parent / "model.npz"
    if prefetch.file_hash(model) != freeze["model_sha256"]:
        raise ValueError("frozen activation model hash differs")
    with np.load(model, allow_pickle=False) as source:
        predictor = Predictor({key: source[key] for key in source.files})
    if predictor.layers != tuple(range(1, 48)) or any(len(head[0]) != FEATURES for head in predictor.heads.values()):
        raise ValueError("frozen activation model topology or feature count differs")
    return predictor, freeze


def report(captures, predictor, freeze, experiment):
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    evaluations = []
    for capture in captures:
        value = prefetch.compare(capture.trace, predictor.bind(capture))
        value["name"] = capture.name
        for result in value["staging"]:
            result["policy"] = "activation_ridge_staging"
        evaluations.append(value)
    totals = []
    for index, width in enumerate(prefetch.WIDTHS):
        summed = {key: sum(value["staging"][index]["aggregate"][key] for value in evaluations)
                  for key in evaluations[0]["staging"][index]["aggregate"]}
        misses = summed["baseline_demand_misses"]
        summed["exposed_miss_reduction_pct"] = 100 * summed["exposed_misses_saved"] / misses
        summed["logical_read_change_pct"] = -100 * summed["logical_reads_saved"] / misses
        totals.append({"width": width, **summed})
    return {"version": VERSION, "experiment": experiment, "settings": SETTINGS,
            "model_sha256": freeze["model_sha256"], "model_file_bytes": freeze["model_file_bytes"],
            "model_parameter_bytes": freeze["training_compute"]["parameter_bytes"],
            "training_inputs": freeze["training_inputs"], "inputs": [capture.manifest() for capture in captures],
            "evaluations": evaluations, "aggregate": totals, "limits": LIMITS,
            "compute": {"cpu_seconds": time.process_time() - start_cpu,
                        "wall_seconds": time.perf_counter() - start_wall},
            "numpy_version": np.__version__, "python_version": sys.version}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("capture-commands")
    prepare.add_argument("--capture-root", required=True)
    train = commands.add_parser("train")
    train.add_argument("--capture-root", required=True)
    train.add_argument("--output-dir", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--capture-root", required=True)
    evaluate.add_argument("--freeze", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture-commands":
            root = Path(args.capture_root).resolve()
            root.mkdir(parents=True, exist_ok=True)
            for split in ("train", "holdout"):
                (root / split).mkdir(exist_ok=True)
            manifest = capture_commands(root)
            prefetch.write_json(root / "capture-commands.json", manifest)
            print(json.dumps(manifest, indent=2))
        elif args.command == "train":
            root = Path(args.output_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            if any((root / name).exists() for name in ("model.npz", "freeze.json", "training-replay.json")):
                raise ValueError("refusing to overwrite frozen activation artifacts")
            captures = load_captures(args.capture_root, "train")
            arrays, costs = fit(captures)
            model = root / "model.npz"
            with open(model, "xb") as target:
                np.savez(target, **arrays)
            freeze = {"version": VERSION, "settings": SETTINGS, "training_compute": costs,
                      "training_inputs": [capture.manifest() for capture in captures],
                      "model_sha256": prefetch.file_hash(model), "model_file_bytes": model.stat().st_size,
                      "predictor_script_sha256": prefetch.file_hash(__file__),
                      "staging_script_sha256": prefetch.file_hash(prefetch.__file__),
                      "baseline_script_sha256": prefetch.file_hash(replay.__file__)}
            prefetch.write_json(root / "freeze.json", freeze)
            prefetch.write_json(root / "training-replay.json", report(captures, Predictor(arrays), freeze, "initial-three-in-sample"))
            print(json.dumps({"freeze": str(root / "freeze.json"), "training_compute": costs}, indent=2))
        else:
            predictor, freeze = load_frozen(args.freeze)  # Validate before opening any held-out feature data.
            captures = load_captures(args.capture_root, "holdout")
            training_hashes = {item["activations"]["sha256"] for item in freeze["training_inputs"]}
            if any(prefetch.file_hash(capture.path) in training_hashes for capture in captures):
                raise ValueError("held-out capture duplicates training activations")
            if any(capture.model_bytes != freeze["training_inputs"][0]["activations"]["model_file_bytes"] for capture in captures):
                raise ValueError("held-out model size differs from frozen training")
            value = report(captures, predictor, freeze, "heldout-three-frozen")
            prefetch.write_json(args.output, value)
            print(json.dumps({"output": args.output, "aggregate": value["aggregate"]}, indent=2))
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, f"activation staging error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
