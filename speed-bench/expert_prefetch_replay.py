#!/usr/bin/env python3
"""CPU-only, same-budget two-event predictive expert staging.

Freeze the initial three audited traces, then replay them:
  python3 speed-bench/expert_prefetch_replay.py train --output-dir DIR
Evaluate fresh traces with unchanged parameters:
  python3 speed-bench/expert_prefetch_replay.py evaluate TRACE... \
      --freeze DIR/freeze.json --output evaluation.json

Only fitting reads future labels. Prediction takes one current Route and frozen
training counts. Layer order wraps across tokens, never across trace boundaries.
Staging is optimistically ready immediately after the source event. Every staged
load consumes a slot within the original cache budget and is charged even if
wrong, displaced, or issued at the end of a trace. No serving-speed claim follows.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

_REPLAY_DIR = str(Path(__file__).resolve().parent)
if _REPLAY_DIR not in sys.path:
    sys.path.insert(0, _REPLAY_DIR)
import expert_cache_replay as replay

VERSION = 1
LOOKAHEAD = 2
WIDTHS = (1, 2, 4)
EXPERTS = 256
SHRINK = 32
CACHE_SLOTS = 5134
TRAINING_HASHES = {
    "coding": "2a304275f9c189dd85bb69d52691bce01b07ae38b520682d40ba81e63504465a",
    "reasoning": "0b787445292f5e407bc36d9b4990db74df329ac7e37533e96fb6b9cf88e354da",
    "tool-result": "86166889598bdce648d2d78b1c0538f861bbaf6e10324918316fd6e5e07ac8d5",
}
SETTINGS = {"lookahead_layer_events": LOOKAHEAD, "widths": list(WIDTHS),
            "expert_domain": EXPERTS, "prior_pseudoroutes": SHRINK,
            "total_cache_slots": CACHE_SLOTS,
            "staging_capacity": "min(total_cache_slots, lookahead_layer_events * width)",
            "layer_sequence": "training-frozen cyclic order; no cross-trace training pairs",
            "score": "mean over current IDs of (coactivation_count + 32 * target_prior) / (source_count + 32)",
            "tie_break": "ascending expert ID", "ready_model": "all staging loads ready immediately after source event",
            "prefetch_hotness": "none; only demand advances hotness and cache clock",
            "expiry": "after target event; early demand may promote; footer wastes pending loads",
            "reservation": "dynamic occupied staging slots inside the original total, not extra slots"}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_sequence(trace):
    first_token = trace.routes[0].token
    return tuple(route.layer for route in trace.routes if route.token == first_token)


class Predictor:
    """Training-only conditional expert marginals, shrunk toward target priors."""

    def __init__(self, artifact):
        if artifact["version"] != VERSION or artifact["settings"] != SETTINGS:
            raise ValueError("frozen predictor version or parameters differ")
        self.artifact = artifact
        self.layers = tuple(artifact["layers"])
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("invalid frozen layer sequence")
        self.targets = {layer: self.layers[(index + LOOKAHEAD) % len(self.layers)]
                        for index, layer in enumerate(self.layers)}
        self.tables = {}
        for layer, record in zip(self.layers, artifact["counts"], strict=True):
            if record["source_layer"] != layer or record["target_layer"] != self.targets[layer]:
                raise ValueError("frozen source/target layer mismatch")
            events = record["events"]
            if type(events) is not int or events < 1:
                raise ValueError("each source layer needs training pairs")
            prior = record["target_counts"]
            occurrences = record["source_counts"]
            rows = record["coactivation_counts"]
            if len(prior) != EXPERTS or len(occurrences) != EXPERTS or len(rows) != EXPERTS:
                raise ValueError("frozen expert domain differs")
            if any(type(count) is not int or not 0 <= count <= events for count in (*prior, *occurrences)):
                raise ValueError("invalid marginal counts")
            table = []
            for count, row in zip(occurrences, rows):
                if len(row) != EXPERTS or any(type(value) is not int or not 0 <= value <= count for value in row):
                    raise ValueError("invalid coactivation counts")
                table.append(tuple((value + SHRINK * target / events) / (count + SHRINK)
                                   for value, target in zip(row, prior)))
            self.tables[layer] = tuple(table)

    def predict(self, route, width):
        if width not in WIDTHS:
            raise ValueError("prefetch width must be 1, 2, or 4")
        if route.layer not in self.tables or any(expert >= EXPERTS for expert in route.ids):
            raise ValueError("route outside frozen predictor domain")
        rows = [self.tables[route.layer][expert] for expert in route.ids]
        # Division by len(ids) cannot change ranking. All inputs are current IDs.
        scores = tuple(map(sum, zip(*rows)))
        ranked = sorted(range(EXPERTS), key=lambda expert: (-scores[expert], expert))
        return tuple((self.targets[route.layer], expert) for expert in ranked[:width])


def fit(traces, training_inputs):
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    if not traces:
        raise ValueError("training requires traces")
    layers = layer_sequence(traces[0])
    counts = {}
    for index, layer in enumerate(layers):
        counts[layer] = {"source_layer": layer, "target_layer": layers[(index + LOOKAHEAD) % len(layers)],
                         "events": 0, "source_counts": [0] * EXPERTS,
                         "target_counts": [0] * EXPERTS,
                         "coactivation_counts": [[0] * EXPERTS for _ in range(EXPERTS)]}
    for trace in traces:
        if layer_sequence(trace) != layers:
            raise ValueError("training layer sequences differ")
        for index, source in enumerate(trace.routes):
            if source.layer != layers[index % len(layers)] or any(expert >= EXPERTS for expert in source.ids):
                raise ValueError("training route outside frozen sequence or domain")
            if index + LOOKAHEAD >= len(trace.routes):
                continue
            target = trace.routes[index + LOOKAHEAD]
            record = counts[source.layer]
            if target.layer != record["target_layer"] or any(expert >= EXPERTS for expert in target.ids):
                raise ValueError("training target outside frozen sequence or domain")
            record["events"] += 1
            for expert in target.ids:
                record["target_counts"][expert] += 1
            for expert in source.ids:
                record["source_counts"][expert] += 1
                row = record["coactivation_counts"][expert]
                for target_expert in target.ids:
                    row[target_expert] += 1
    artifact = {"version": VERSION, "settings": SETTINGS, "layers": list(layers),
                "training_inputs": training_inputs, "counts": [counts[layer] for layer in layers]}
    return artifact, {"fit_cpu_seconds": time.process_time() - cpu_start,
                      "fit_wall_seconds": time.perf_counter() - wall_start,
                      "training_pairs": sum(record["events"] for record in counts.values())}


def metrics():
    return {name: 0 for name in (
        "routes", "lookups", "demand_hits", "exposed_demand_misses", "demand_read_bytes",
        "predictions", "useful_predictions", "wasted_predictions", "end_predictions",
        "resident_predictions", "pending_predictions", "capacity_skips", "protection_skips",
        "prefetch_loads", "prefetch_read_bytes", "useful_prefetch_loads", "wasted_prefetch_loads",
        "expiry_wastes", "pressure_wastes", "end_wastes", "demand_evictions", "staging_evictions")}


@dataclass(frozen=True)
class Staged:
    due: int


class StagingCache:
    """Bounded physical staging plus the audited current-policy demand cache.

    A promotion inserts the already-loaded key without touching its clock. The
    unchanged Cache.resolve then performs its one normal demand touch. Prefetch
    and expiry never advance that clock or hotness. Slots are shared dynamically.
    """

    def __init__(self, snapshot, width):
        if width not in WIDTHS:
            raise ValueError("prefetch width must be 1, 2, or 4")
        self.cache = replay.Cache(snapshot, "current")
        self.total_budget = snapshot.budget
        self.width = width
        self.capacity = min(self.total_budget, LOOKAHEAD * width)
        self.expert_bytes = snapshot.expert_bytes
        self.staged = {}
        self.forecasts = deque()
        self.total = metrics()
        self.layers = {}
        self.peak_slots = len(self.cache.resident)
        self.peak_staging_slots = 0
        self.last_index = -1

    def add(self, layer, field, value=1):
        self.total[field] += value
        self.layers.setdefault(layer, metrics())[field] += value

    def check_budget(self):
        slots = len(self.cache.resident) + len(self.staged)
        if slots > self.total_budget or len(self.staged) > self.capacity:
            raise AssertionError("staging exceeded original slot budget")
        if self.cache.resident.keys() & self.staged.keys():
            raise AssertionError("expert occupies both demand and staging slots")
        self.peak_slots = max(self.peak_slots, slots)
        self.peak_staging_slots = max(self.peak_staging_slots, len(self.staged))

    def discard(self, key, reason):
        del self.staged[key]
        self.add(key[0], "wasted_prefetch_loads")
        self.add(key[0], reason + "_wastes")

    def demand(self, route, index):
        if index != self.last_index + 1:
            raise ValueError("staging events must be contiguous")
        self.last_index = index
        requested = set(route.keys)
        while self.forecasts and self.forecasts[0][0] <= index:
            due, keys = self.forecasts.popleft()
            if due != index:
                raise AssertionError("forecast target was skipped")
            for key in keys:
                self.add(key[0], "useful_predictions" if key in requested else "wasted_predictions")
        for key, staged in list(self.staged.items()):
            if staged.due < index or (staged.due == index and key not in requested):
                self.discard(key, "expiry")
        promoted = tuple(key for key in route.keys if key in self.staged)
        for key in promoted:
            del self.staged[key]
            # Cache.resolve touches every hit before victim selection.
            self.cache.resident[key] = self.cache.clock
            self.add(key[0], "useful_prefetch_loads")
        # Complete current requests take precedence over every pending forecast.
        while len(self.staged) + len(requested) > self.total_budget:
            victim = max(self.staged, key=lambda key: (self.staged[key].due, key))
            self.discard(victim, "pressure")
        self.cache.budget = self.total_budget - len(self.staged)
        mask, victims = self.cache.resolve(route)
        hits = mask.bit_count()
        self.add(route.layer, "routes")
        self.add(route.layer, "lookups", len(route.ids))
        self.add(route.layer, "demand_hits", hits)
        self.add(route.layer, "exposed_demand_misses", len(route.ids) - hits)
        self.add(route.layer, "demand_read_bytes", (len(route.ids) - hits) * self.expert_bytes)
        for layer, _ in victims:
            self.add(layer, "demand_evictions")
        self.check_budget()
        return mask, promoted

    def issue(self, keys, index, protected):
        if index != self.last_index:
            raise ValueError("predictions must follow the current demand event")
        if len(keys) > self.width or len(set(keys)) != len(keys):
            raise ValueError("invalid prediction width")
        self.forecasts.append((index + LOOKAHEAD, tuple(keys)))
        loaded = []
        for key in keys:
            self.add(key[0], "predictions")
            if key in self.cache.resident:
                self.add(key[0], "resident_predictions")
                continue
            if key in self.staged:
                self.add(key[0], "pending_predictions")
                continue
            if len(self.staged) >= self.capacity:
                self.add(key[0], "capacity_skips")
                continue
            if len(self.cache.resident) + len(self.staged) == self.total_budget:
                if len(self.cache.resident) == sum(key in self.cache.resident for key in protected):
                    self.add(key[0], "protection_skips")
                    continue
                for layer, _ in self.cache._evict(1, protected):
                    self.add(layer, "staging_evictions")
            self.staged[key] = Staged(index + LOOKAHEAD)
            loaded.append(key)
            self.add(key[0], "prefetch_loads")
            self.add(key[0], "prefetch_read_bytes", self.expert_bytes)
            self.check_budget()
        if len(self.cache.heap) > max(64, 3 * self.total_budget):
            self.cache._rebuild()
        return tuple(loaded)

    def finish(self):
        for key in tuple(self.staged):
            self.discard(key, "end")
        while self.forecasts:
            _, keys = self.forecasts.popleft()
            for key in keys:
                self.add(key[0], "wasted_predictions")
                self.add(key[0], "end_predictions")
        for row in [self.total, *self.layers.values()]:
            row["total_logical_reads"] = row["exposed_demand_misses"] + row["prefetch_loads"]
            row["total_logical_read_bytes"] = row["demand_read_bytes"] + row["prefetch_read_bytes"]
            if row["prefetch_loads"] != row["useful_prefetch_loads"] + row["wasted_prefetch_loads"]:
                raise AssertionError("a prefetched load was lost or counted twice")
            if row["predictions"] != row["useful_predictions"] + row["wasted_predictions"]:
                raise AssertionError("a forecast was lost or counted twice")
        self.check_budget()


def staging_replay(trace, predictor, width, observer=None):
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    staged = StagingCache(trace.snapshot, width)
    prediction_cpu = 0.0
    for index, route in enumerate(trace.routes):
        if route.layer != predictor.layers[index % len(predictor.layers)]:
            raise ValueError("evaluation changed the frozen cyclic layer sequence")
        mask, promoted = staged.demand(route, index)
        prediction_start = time.process_time()
        predictions = predictor.predict(route, width)
        prediction_cpu += time.process_time() - prediction_start
        loaded = staged.issue(predictions, index, set(route.keys))
        if observer is not None:
            observer({"index": index, "predictions": predictions, "loaded": loaded,
                      "promoted": promoted, "hit_mask": mask, "clock": staged.cache.clock,
                      "resident": dict(staged.cache.resident),
                      "staged": {key: value.due for key, value in staged.staged.items()},
                      "counters": dict(staged.total)})
    if staged.cache.clock != trace.snapshot.clock + sum(len(route.ids) for route in trace.routes):
        raise AssertionError("prefetch, promotion or expiry changed demand clock increments")
    staged.finish()
    return {"policy": "causal_coactivation_staging", "width": width,
            "total_slots": trace.snapshot.budget, "staging_capacity": staged.capacity,
            "peak_slots": staged.peak_slots, "peak_staging_slots": staged.peak_staging_slots,
            "aggregate": staged.total,
            "per_layer": [{"layer": layer, **row} for layer, row in sorted(staged.layers.items())],
            "compute": {"cpu_seconds": time.process_time() - cpu_start,
                        "wall_seconds": time.perf_counter() - wall_start,
                        "prediction_cpu_seconds": prediction_cpu,
                        "prediction_cpu_microseconds_per_event": 1e6 * prediction_cpu / len(trace.routes)},
            "final_clock": staged.cache.clock}


def compare(trace, predictor):
    cpu_start = time.process_time()
    baseline = replay.replay(trace)  # Audits every recorded hit, byte and clock.
    baseline["compute"] = {"cpu_seconds": time.process_time() - cpu_start}
    baseline_layers = {row["layer"]: row for row in baseline["per_layer"]}
    results = []
    for width in WIDTHS:
        result = staging_replay(trace, predictor, width)
        for row, reference in [(result["aggregate"], baseline["aggregate"]),
                               *((row, baseline_layers.get(row["layer"], replay._metrics()))
                                 for row in result["per_layer"])]:
            row["baseline_demand_misses"] = reference["misses"]
            row["exposed_misses_saved"] = reference["misses"] - row["exposed_demand_misses"]
            row["logical_reads_saved"] = reference["misses"] - row["total_logical_reads"]
        totals = result["aggregate"]
        result["result"] = ("negative" if totals["logical_reads_saved"] < 0 or totals["exposed_misses_saved"] < 0
                            else "optimistic_replay_only")
        results.append(result)
    return {"source": trace.source, "baseline_validated": True, "baseline": baseline, "staging": results}


def input_record(path, trace):
    return {"source": str(path), "sha256": file_hash(path), "file_bytes": Path(path).stat().st_size,
            "routes": len(trace.routes), "tokens": trace.routes[-1].token - trace.routes[0].token + 1,
            "total_slots": trace.snapshot.budget, "expert_bytes": trace.snapshot.expert_bytes}


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


def write_model(path, artifact):
    with open(path, "xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            compressed.write(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def load_frozen(freeze_path):
    freeze_path = Path(freeze_path)
    with open(freeze_path, encoding="utf-8") as source:
        freeze = json.load(source)
    if freeze["version"] != VERSION or freeze["settings"] != SETTINGS:
        raise ValueError("frozen experiment parameters differ")
    for path, expected in ((Path(__file__), freeze["predictor_script_sha256"]),
                           (Path(replay.__file__), freeze["baseline_script_sha256"])):
        if file_hash(path) != expected:
            raise ValueError("frozen replay source hash differs")
    model_path = freeze_path.parent / "model.json.gz"
    if file_hash(model_path) != freeze["model_sha256"]:
        raise ValueError("frozen model hash differs")
    with gzip.open(model_path, "rt", encoding="utf-8") as source:
        artifact = json.load(source)
    if artifact["training_inputs"] != freeze["training_inputs"]:
        raise ValueError("frozen training input manifest differs")
    return Predictor(artifact), freeze


LIMITS = ["Only routed IDs from the three initial traces train the predictor; no hidden activations are available",
          "Immediate ready-in-time staging ignores SSD latency, bandwidth contention and scheduling",
          "Demand and staging share the exact original expert-slot budget; metadata and Python heap memory are outside this logical model",
          "All wrong and footer-tail prefetch loads count; promotion and expiry add no load or cache clock",
          "Prediction quality is scored at the target event; early promotions are useful loads even if their forecast later misses",
          "Training replays are in-sample evidence; only separately supplied frozen evaluation is fresh-holdout evidence",
          "Logical reads and Python CPU costs do not establish physical I/O savings or serving throughput"]


def report(traces, predictor, freeze, experiment):
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    evaluations = [compare(trace, predictor) for trace in traces]
    totals = []
    for index, width in enumerate(WIDTHS):
        summed = {field: sum(item["staging"][index]["aggregate"][field] for item in evaluations)
                  for field in evaluations[0]["staging"][index]["aggregate"]}
        totals.append({"width": width, **summed,
                       "result": "negative" if summed["logical_reads_saved"] < 0 or summed["exposed_misses_saved"] < 0
                       else "optimistic_replay_only"})
    return {"version": VERSION, "experiment": experiment, "settings": SETTINGS,
            "model_sha256": freeze["model_sha256"], "training_inputs": freeze["training_inputs"],
            "inputs": [input_record(trace.source, trace) for trace in traces],
            "evaluations": evaluations, "aggregate": totals, "limits": LIMITS,
            "compute": {"evaluation_cpu_seconds": time.process_time() - cpu_start,
                        "evaluation_wall_seconds": time.perf_counter() - wall_start},
            "python_version": sys.version}


def load_inputs(paths):
    paths = [Path(path).resolve() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("each input trace must occur exactly once")
    traces = [replay.load_trace(path) for path in paths]
    if any(trace.snapshot.budget != CACHE_SLOTS for trace in traces):
        raise ValueError("MiMo experiment requires the recorded 5134-slot total budget")
    return traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--output-dir", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("traces", nargs="+")
    evaluate.add_argument("--freeze", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            directory = Path(args.output_dir).resolve()
            paths = [Path(__file__).parent / "mimo-cache/20261010" / (name + ".jsonl.gz")
                     for name in TRAINING_HASHES]
            for name, path in zip(TRAINING_HASHES, paths):
                if file_hash(path) != TRAINING_HASHES[name]:
                    raise ValueError("initial training trace hash differs: " + name)
            for name in ("model.json.gz", "freeze.json", "training-replay.json"):
                if (directory / name).exists():
                    raise ValueError("refusing to overwrite frozen experiment artifacts")
            traces = load_inputs(paths)
            for trace in traces:
                replay.replay(trace)
            inputs = [input_record(trace.source, trace) for trace in traces]
            artifact, costs = fit(traces, inputs)
            prepare_start = time.process_time()
            predictor = Predictor(artifact)
            costs["predictor_prepare_cpu_seconds"] = time.process_time() - prepare_start
            directory.mkdir(parents=True, exist_ok=True)
            write_model(directory / "model.json.gz", artifact)
            freeze = {"version": VERSION, "settings": SETTINGS, "training_inputs": inputs,
                      "model_sha256": file_hash(directory / "model.json.gz"),
                      "predictor_script_sha256": file_hash(__file__),
                      "baseline_script_sha256": file_hash(replay.__file__), "fit": costs,
                      "model": "MiMo-V2.6-Flash-MOPD-MXFP4-Q8Attn.gguf",
                      "hf_revision": "b12f5573350bb1704513fcf119fca48e498b0423",
                      "freeze_order": "model and input hashes written before evaluation; no fresh holdout read"}
            write_json(directory / "freeze.json", freeze)
            result = report(traces, predictor, freeze, "initial_training_replay")
            output = directory / "training-replay.json"
        else:
            output = Path(args.output).resolve()
            frozen = Path(args.freeze).resolve()
            forbidden = {Path(path).resolve() for path in args.traces} | {frozen, frozen.parent / "model.json.gz"}
            if output in forbidden or output.exists():
                raise ValueError("report must be a new path distinct from model, freeze and inputs")
            prepare_start = time.process_time()
            predictor, freeze = load_frozen(frozen)  # Freeze checked before holdout is opened.
            prepare_cpu = time.process_time() - prepare_start
            traces = load_inputs(args.traces)
            if any(input_record(trace.source, trace)["sha256"] in
                   {record["sha256"] for record in freeze["training_inputs"]} for trace in traces):
                raise ValueError("fresh holdout must not reuse an initial training trace")
            result = report(traces, predictor, freeze, "fresh_frozen_holdout")
            result["compute"]["frozen_load_and_prepare_cpu_seconds"] = prepare_cpu
        write_json(output, result)
        print(json.dumps({"report": str(output), "model_sha256": result["model_sha256"],
                          "aggregate": result["aggregate"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
