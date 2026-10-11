#!/usr/bin/env python3
"""CPU-only causal expert-cache ranking experiment using the audited replay.

Train and leave one input workload out at a time:
  python3 speed-bench/expert_cache_learned.py train TRACE... --model model.json \
      --output validation.json
Evaluate a frozen model without training or reading future labels:
  python3 speed-bench/expert_cache_learned.py evaluate TRACE... --model model.json \
      --output evaluation.json

Only training labels use future routes. Validation reports include the audited
current policy, the existing FIFO variant, and the finite-trace oracle. Logical
read reductions and Python CPU timings are not serving-throughput measurements.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

# The script directory is also added when imported by path from unittest.
_REPLAY_DIR = str(Path(__file__).resolve().parent)
if _REPLAY_DIR not in sys.path:
    sys.path.insert(0, _REPLAY_DIR)
import expert_cache_replay as replay


VERSION = 1
DOMAIN = replay.MAX_LAYER * replay.MAX_EXPERT
FEATURES = (
    {"name": "clock_recency", "definition": "log1p((clock-last_used)/256); snapshot last_used is exact"},
    {"name": "hotness", "definition": "log1p(integer current-policy hotness, half-life 16 tokens)"},
    {"name": "frequency_4", "definition": "log1p(sum of observed accesses weighted by exp2(-token_age/4))"},
    {"name": "frequency_64", "definition": "log1p(sum of observed accesses weighted by exp2(-token_age/64))"},
    {"name": "token_recency", "definition": "log1p(tokens since last observed access); zero when unobserved"},
    {"name": "last_gap", "definition": "log1p(last observed token reuse gap); zero before second access"},
    {"name": "mean_gap", "definition": "log1p(reuse-gap EMA, new gap weight 0.25); first gap initializes EMA"},
    {"name": "observed", "definition": "1 after at least one post-snapshot access, otherwise 0"},
    {"name": "gap_known", "definition": "1 after at least two post-snapshot accesses, otherwise 0"},
)


@dataclass(frozen=True)
class Settings:
    horizon_tokens: int = 16
    sample_every: int = 32
    candidates_per_sample: int = 32
    seed: int = 20261010
    l2: float = 0.001
    max_iterations: int = 30
    tolerance: float = 1e-8

    def validate(self):
        for name in ("horizon_tokens", "sample_every", "candidates_per_sample", "max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("l2", "tolerance"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Ranker:
    weights: tuple
    intercept: float

    def __post_init__(self):
        if len(self.weights) != len(FEATURES):
            raise ValueError("ranker feature count mismatch")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in (*self.weights, self.intercept)):
            raise ValueError("ranker coefficients must be finite numbers")

    def score(self, features):
        # Avoid BLAS thread overhead for this narrow matrix-vector product.
        return self.intercept + np.sum(features * np.asarray(self.weights), axis=1)


class CausalCache(replay.Cache):
    """Reuse Cache.resolve verbatim; replace only victim selection when learned.

    History is retained for evicted experts, bounded by the fixed expert domain.
    Snapshot hotness and last_used are usable; missing pre-snapshot gaps and
    token timestamps remain unknown rather than being reconstructed from future.
    """

    def __init__(self, snapshot, ranker=None, observer=None):
        self.ranker = ranker
        self.observer = observer
        self.last_token = np.full(DOMAIN, -1, dtype=np.int64)
        self.last_clock = np.zeros(DOMAIN, dtype=np.uint64)
        self.hotness_values = np.zeros(DOMAIN, dtype=np.float64)
        self.freq4 = np.zeros(DOMAIN, dtype=np.float64)
        self.freq64 = np.zeros(DOMAIN, dtype=np.float64)
        self.last_gap = np.zeros(DOMAIN, dtype=np.float64)
        self.mean_gap = np.zeros(DOMAIN, dtype=np.float64)
        self.seen = np.zeros(DOMAIN, dtype=np.uint64)
        self.present = np.zeros(DOMAIN, dtype=np.bool_)
        self.decisions = 0
        self.candidates_scored = 0
        self.ranking_seconds = 0.0
        super().__init__(snapshot, "current")
        for key, clock in self.resident.items():
            index = self.index(key)
            self.present[index] = True
            self.last_clock[index] = clock
        self._sync_hotness()

    @staticmethod
    def index(key):
        return key[0] * replay.MAX_EXPERT + key[1]

    @staticmethod
    def key(index):
        return divmod(int(index), replay.MAX_EXPERT)

    def _sync_hotness(self):
        self.hotness_values.fill(0)
        for key, value in self.hotness.items():
            self.hotness_values[self.index(key)] = value

    def _age(self, token):
        previous = self.decay_token
        super()._age(token)
        if self.decay_token != previous:
            self._sync_hotness()

    def _touch(self, key):
        super()._touch(key)
        index = self.index(key)
        previous = self.last_token[index]
        if previous >= 0:
            gap = self.token - int(previous)
            self.freq4[index] *= math.exp2(-gap / 4)
            self.freq64[index] *= math.exp2(-gap / 64)
            self.last_gap[index] = gap
            self.mean_gap[index] = gap if self.seen[index] == 1 else 0.75 * self.mean_gap[index] + 0.25 * gap
        self.freq4[index] += 1
        self.freq64[index] += 1
        self.seen[index] += 1
        self.last_token[index] = self.token
        self.last_clock[index] = self.clock
        self.hotness_values[index] = self.hotness[key]
        self.present[index] = True

    def features(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        observed = self.last_token[indices] >= 0
        age = np.where(observed, self.token - self.last_token[indices], 0)
        return np.column_stack((
            np.log1p((np.uint64(self.clock) - self.last_clock[indices]).astype(np.float64) / 256),
            np.log1p(self.hotness_values[indices]),
            np.log1p(self.freq4[indices] * np.exp2(-age / 4)),
            np.log1p(self.freq64[indices] * np.exp2(-age / 64)),
            np.log1p(age), np.log1p(self.last_gap[indices]),
            np.log1p(self.mean_gap[indices]), observed.astype(np.float64),
            (self.seen[indices] >= 2).astype(np.float64),
        ))

    def _evict(self, count, protected):
        if self.observer is not None and count:
            self.observer(self, count, protected)
        if self.ranker is None:
            victims = super()._evict(count, protected)
        elif count:
            started = time.perf_counter()
            protected_indices = np.asarray([self.index(key) for key in protected], dtype=np.int64)
            eligible = self.present.copy()
            eligible[protected_indices] = False
            indices = np.flatnonzero(eligible)
            if len(indices) < count:
                raise replay.TraceError("no unprotected resident victim for mandatory admission")
            scores = self.ranker.score(self.features(indices))
            # Lower score loses; exact score ties retain the audited LRU/key rule.
            # partition avoids sorting all residents on every miss set.
            cutoff = np.partition(scores, count - 1)[count - 1]
            shortlist = np.flatnonzero(scores <= cutoff)
            order = np.lexsort((indices[shortlist], self.last_clock[indices[shortlist]], scores[shortlist]))
            victims = tuple(self.key(index) for index in indices[shortlist[order[:count]]])
            for key in victims:
                del self.resident[key]
            self.decisions += 1
            self.candidates_scored += len(indices)
            self.ranking_seconds += time.perf_counter() - started
        else:
            victims = ()
        for key in victims:
            self.present[self.index(key)] = False
        return victims

    def compute(self):
        return {"eviction_decisions": self.decisions, "candidates_scored": self.candidates_scored,
                "ranking_seconds": self.ranking_seconds,
                "ranking_us_per_decision": 1e6 * self.ranking_seconds / self.decisions if self.decisions else 0,
                "history_array_bytes": sum(array.nbytes for array in (self.last_token, self.last_clock,
                    self.hotness_values, self.freq4, self.freq64, self.last_gap, self.mean_gap, self.seen, self.present)),
                "ranking_algorithm": "scan eligible residents, linear score, partition; LRU/key ties"}


def learned_replay(trace, ranker):
    """No future lookup or labels: cache receives only snapshot and current route."""
    started = time.perf_counter()
    cache = CausalCache(trace.snapshot, ranker)
    total = replay._metrics()
    layers = {}
    for route in trace.routes:
        hit_mask, victims = cache.resolve(route)
        replay._add(total, route, hit_mask, len(victims), trace.snapshot.expert_bytes)
        replay._add(layers.setdefault(route.layer, replay._metrics()), route, hit_mask, 0, trace.snapshot.expert_bytes)
        for victim_layer, _ in victims:
            layers.setdefault(victim_layer, replay._metrics())["evictions"] += 1
    total["tokens"] = trace.routes[-1].token - trace.routes[0].token + 1
    return {"policy": "learned_logistic", "aggregate": replay._finish(total),
            "per_layer": [{"layer": layer, **replay._finish(metrics)} for layer, metrics in sorted(layers.items())],
            "compute": {"replay_seconds": time.perf_counter() - started, **cache.compute()}}


def collect_training(trace, settings):
    """Sample current-policy at-risk residents. Only this function reads labels.

    Entire tail decision sets are omitted unless the whole label horizon was
    observed. Never-reused candidates in an uncensored set are negative labels.
    """
    settings.validate()
    started = time.perf_counter()
    next_use, after = replay.future_uses(trace.routes)
    rng = random.Random(settings.seed)
    rows, labels = [], []
    eviction_events = 0
    sampled_decisions = 0
    censored_decisions = 0
    index = 0
    final_token = trace.routes[-1].token

    def observe(cache, count, protected):
        nonlocal eviction_events, sampled_decisions, censored_decisions
        eviction_events += 1
        if (eviction_events - 1) % settings.sample_every:
            return
        if cache.token + settings.horizon_tokens > final_token:
            censored_decisions += 1
            return
        eligible = [key for key in cache.resident if key not in protected]
        keys = rng.sample(eligible, min(settings.candidates_per_sample, len(eligible)))
        keys.sort()
        rows.append(cache.features([cache.index(key) for key in keys]))
        for key in keys:
            future_index = next_use.get(key, len(trace.routes))
            labels.append(float(future_index < len(trace.routes)
                                and trace.routes[future_index].token <= cache.token + settings.horizon_tokens))
        sampled_decisions += 1

    cache = CausalCache(trace.snapshot, observer=observe)
    for index, route in enumerate(trace.routes):
        next_use.update(zip(route.keys, after[index]))
        cache.resolve(route)
    x = np.concatenate(rows) if rows else np.empty((0, len(FEATURES)), dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return x, y, {"samples": len(y), "positive_labels": int(y.sum()),
                  "sampled_decisions": sampled_decisions, "censored_decisions": censored_decisions,
                  "collection_seconds": time.perf_counter() - started,
                  "dataset_bytes": x.nbytes + y.nbytes}


def fit_ranker(x, y, settings):
    settings.validate()
    if x.ndim != 2 or x.shape != (len(y), len(FEATURES)) or not len(y):
        raise ValueError("training requires nonempty feature rows and matching labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("training requires finite features and binary labels")
    started = time.perf_counter()
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1
    design = np.column_stack((np.ones(len(y)), (x - mean) / scale))
    parameters = np.zeros(design.shape[1])
    positive_rate = float(y.mean())
    parameters[0] = math.log((positive_rate + 1e-6) / (1 - positive_rate + 1e-6))
    penalty = np.full(len(parameters), settings.l2)
    penalty[0] = 0

    def objective(coefficients):
        logits = np.sum(design * coefficients, axis=1)
        return float(np.mean(np.logaddexp(0, logits) - y * logits)
                     + 0.5 * np.sum(penalty * coefficients * coefficients))

    for iteration in range(settings.max_iterations):
        logits = np.sum(design * parameters, axis=1)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        residual = probabilities - y
        gradient = np.mean(design * residual[:, None], axis=0) + penalty * parameters
        weighted = design * (probabilities * (1 - probabilities))[:, None]
        hessian = design.T @ weighted / len(y) + np.diag(penalty + 1e-12)
        step = np.linalg.solve(hessian, gradient)
        before = objective(parameters)
        fraction = 1.0
        while fraction > 2 ** -20 and objective(parameters - fraction * step) > before:
            fraction *= 0.5
        parameters -= fraction * step
        if np.max(np.abs(fraction * step)) <= settings.tolerance:
            break
    raw_weights = parameters[1:] / scale
    intercept = float(parameters[0] - np.sum(raw_weights * mean))
    ranker = Ranker(tuple(float(value) for value in raw_weights), intercept)
    return ranker, {"fit_seconds": time.perf_counter() - started, "iterations": iteration + 1,
                    "mean_regularized_log_loss": objective(parameters),
                    "positive_label_rate": positive_rate, "samples": len(y),
                    "parameter_count": len(parameters)}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_record(trace):
    return {"source": trace.source, "sha256": file_hash(trace.source),
            "routes": len(trace.routes), "budget": trace.snapshot.budget,
            "expert_bytes": trace.snapshot.expert_bytes}


def freeze_model(ranker, settings, inputs, training):
    return {"version": VERSION, "kind": "linear_logistic_retention", "features": list(FEATURES),
            "weights": list(ranker.weights), "intercept": ranker.intercept,
            "settings": asdict(settings), "training_inputs": inputs, "training": training,
            "implementation_sha256": file_hash(__file__),
            "audited_replay_sha256": file_hash(replay.__file__),
            "inference": {"formula": "score = intercept + sum(weights[i] * features[i]); evict lowest score",
                          "probability": "sigmoid(score), unnecessary for ranking",
                          "tie_break": "oldest last_used, ascending layer, ascending expert",
                          "admission": "mandatory all selected experts; protect complete current route set",
                          "budget": "same exact expert-slot budget and initial residents as trace",
                          "history": "bounded per-expert state, retained after eviction; reset on snapshot",
                          "identifiers": "no expert/layer ID or workload/phase feature",
                          "label": "future reuse within horizon_tokens; train inputs only, censored tail excluded"},
            "holdout_policy": "Frozen before fresh traces; evaluate only, no refitting or hyperparameter tuning"}


def load_model(path):
    with open(path, encoding="utf-8") as source:
        artifact = json.load(source)
    if artifact.get("version") != VERSION or artifact.get("kind") != "linear_logistic_retention":
        raise ValueError("unsupported frozen ranker format")
    if artifact.get("features") != list(FEATURES):
        raise ValueError("frozen feature definitions differ from this implementation")
    if artifact.get("audited_replay_sha256") != file_hash(replay.__file__):
        raise ValueError("audited replay changed since model freeze")
    settings = Settings(**artifact["settings"])
    settings.validate()
    return Ranker(tuple(artifact["weights"]), artifact["intercept"]), artifact


def baselines(trace):
    results = []
    for policy in replay.POLICIES:
        started = time.perf_counter()
        result = replay.replay(trace, policy, audit=policy == "current")
        result["compute"] = {"replay_seconds": time.perf_counter() - started}
        results.append(result)
    return results


def compare(trace, ranker, baseline_results):
    result = learned_replay(trace, ranker)
    current = baseline_results[0]
    for policy in [*baseline_results, result]:
        replay._reductions(policy["aggregate"], current["aggregate"])
    return {"source": trace.source, "baseline_validated": True,
            "snapshot_budget": trace.snapshot.budget, "policies": [*baseline_results, result]}


def load_inputs(paths):
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("each workload trace must be supplied exactly once")
    # Canonical order makes fitting and leave-one-out splits input-order independent.
    traces = [replay.load_trace(path) for path in sorted(resolved)]
    return traces


def _miss_summary(evaluations):
    totals = dict.fromkeys((*replay.POLICIES, "learned_logistic"), 0)
    for evaluation in evaluations:
        for result in evaluation["policies"]:
            totals[result["policy"]] += result["aggregate"]["misses"]
    current = totals["current"]
    return {"aggregate_misses": totals,
            "learned_miss_reduction_pct": 100 * (current - totals["learned_logistic"]) / current if current else 0}


def train_experiment(traces, settings):
    if len(traces) < 2:
        raise ValueError("leave-one-workload-out training needs at least two distinct traces")
    started = time.perf_counter()
    # Audit every recorded bit/byte/clock before collecting any training data.
    references = [baselines(trace) for trace in traces]
    datasets = [collect_training(trace, settings) for trace in traces]
    folds = []
    records = [input_record(trace) for trace in traces]
    for held_out in range(len(traces)):
        train_indices = [index for index in range(len(traces)) if index != held_out]
        x = np.concatenate([datasets[index][0] for index in train_indices])
        y = np.concatenate([datasets[index][1] for index in train_indices])
        ranker, costs = fit_ranker(x, y, settings)
        folds.append({"validation_input": records[held_out],
                      "training_inputs": [records[index] for index in train_indices],
                      "weights": list(ranker.weights), "intercept": ranker.intercept,
                      "fit": costs, "evaluation": compare(traces[held_out], ranker, references[held_out])})
    x = np.concatenate([dataset[0] for dataset in datasets])
    y = np.concatenate([dataset[1] for dataset in datasets])
    ranker, costs = fit_ranker(x, y, settings)
    training = {"fit": costs, "collection": [{"source": trace.source, **dataset[2]}
                                            for trace, dataset in zip(traces, datasets)]}
    artifact = freeze_model(ranker, settings, records, training)
    report = {"experiment": "leave_one_workload_out", "settings": asdict(settings), "folds": folds,
              **_miss_summary(fold["evaluation"] for fold in folds),
              "final_training": training, "total_cpu_wall_seconds": time.perf_counter() - started,
              "numpy_version": np.__version__, "python_version": sys.version,
              "limits": ["Finite initial traces are not independent fresh-holdout evidence",
                         "Sampled teacher-cache training distribution differs from learned-cache residents",
                         "NumPy scan costs do not predict native implementation cost",
                         "Logical read savings do not establish serving speed or physical I/O savings"]}
    return artifact, report


def write_json(path, value, exclusive=False):
    with open(path, "x" if exclusive else "w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("traces", nargs="+")
        sub.add_argument("--model", required=True)
        sub.add_argument("--output", required=True)
        if command == "train":
            sub.add_argument("--horizon-tokens", type=int, default=16)
            sub.add_argument("--sample-every", type=int, default=32)
            sub.add_argument("--candidates-per-sample", type=int, default=32)
            sub.add_argument("--seed", type=int, default=20261010)
            sub.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        model_path, output_path = Path(args.model).resolve(), Path(args.output).resolve()
        inputs = {Path(path).resolve() for path in args.traces}
        if model_path == output_path or model_path in inputs or output_path in inputs:
            raise ValueError("model, report and input traces must have distinct paths")
        if args.command == "train" and model_path.exists():
            raise ValueError("refusing to overwrite a frozen model; choose a new artifact path")
        traces = load_inputs(args.traces)
        if args.command == "train":
            settings = Settings(horizon_tokens=args.horizon_tokens, sample_every=args.sample_every,
                                candidates_per_sample=args.candidates_per_sample, seed=args.seed, l2=args.l2)
            settings.validate()
            artifact, report = train_experiment(traces, settings)
            write_json(model_path, artifact, exclusive=True)
        else:
            ranker, artifact = load_model(model_path)
            report = {"experiment": "frozen_evaluation", "model_sha256": file_hash(model_path),
                      "training_inputs": artifact["training_inputs"],
                      "inputs": [input_record(trace) for trace in traces],
                      "evaluations": [compare(trace, ranker, baselines(trace)) for trace in traces],
                      "limits": ["Logical reads and Python CPU time do not establish serving throughput"]}
            report.update(_miss_summary(report["evaluations"]))
        report["model_sha256"] = file_hash(model_path)
        write_json(output_path, report)
        print(json.dumps({"model": str(model_path), "model_sha256": report["model_sha256"],
                          "report": str(output_path),
                          "learned_miss_reduction_pct": report["learned_miss_reduction_pct"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
