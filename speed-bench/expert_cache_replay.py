#!/usr/bin/env python3
"""Audit version-1 streaming expert-cache traces and compare same-budget policies.

Usage: python3 speed-bench/expert_cache_replay.py trace.jsonl trace.jsonl.gz
       python3 speed-bench/expert_cache_replay.py --format csv --output replay.csv TRACE...

Every recorded hit mask, logical pread byte count, and cache clock must match the
current-policy replay before any report is written. Policy hit rates are simulated;
recorded read/load wall times do not predict alternative-policy throughput. The
oracle uses the finite trace, equal-size experts, mandatory admission, and atomic
request sets protected from eviction. Snapshot state is shared by all policies.

least_stale_layer_fifo is a demand-only paper-inspired interpretation:
evict stale residents before current-cycle residents, then ascending layer and
FIFO enqueue order. Initial residents are stale, seeded by snapshot last_used.
All stale layers are eligible, including right-layer fallback. Warm-state and
fallback details are variant choices; it does not reproduce paper prefetching.
"""

import argparse
import csv
import gzip
import zlib
import heapq
import io
import json
import math
from pathlib import Path
import sys
from dataclasses import dataclass


MAX_LAYER = 80
MAX_EXPERT = 384
MAX_SELECTED = 8
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
DECAY_INTERVAL = 16
POLICIES = ("current", "least_stale_layer_fifo", "oracle")


class TraceError(ValueError):
    """Malformed, incompatible, incomplete, or unauditable trace."""


@dataclass(frozen=True)
class Snapshot:
    budget: int
    expert_bytes: int
    clock: int
    token: int
    decay_token: int
    resident: dict
    hotness: dict


@dataclass(frozen=True)
class Route:
    token: int
    layer: int
    ids: tuple
    hit_mask: int
    pread_bytes: int
    pread_ms: float
    load_ms: float
    clock: int
    line: int = 0

    @property
    def keys(self):
        return tuple((self.layer, expert) for expert in self.ids)


@dataclass(frozen=True)
class Trace:
    snapshot: Snapshot
    routes: tuple
    source: str


def _integer(value, name, minimum=0, maximum=UINT64_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise TraceError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _milliseconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise TraceError(f"{name} must be a finite nonnegative number")
    return float(value)


def _fields(record, required):
    missing = required - record.keys()
    extra = record.keys() - required
    if missing or extra:
        raise TraceError(f"invalid fields: missing={sorted(missing)}, extra={sorted(extra)}")


def _key(layer, expert):
    return (_integer(layer, "layer", maximum=MAX_LAYER - 1),
            _integer(expert, "expert", maximum=MAX_EXPERT - 1))


def _triples(value, name, maximum):
    if not isinstance(value, list):
        raise TraceError(f"{name} must be an array")
    result = {}
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise TraceError(f"{name} entries must be [layer, expert, value]")
        key = _key(row[0], row[1])
        if key in result:
            raise TraceError(f"duplicate {name} expert {key}")
        result[key] = _integer(row[2], name + " value", 1, maximum)
    return result


def _snapshot(record):
    _fields(record, {"event", "version", "budget", "expert_bytes", "clock", "token",
                     "decay_token", "decay_interval", "resident", "hotness"})
    if _integer(record["version"], "version") != 1:
        raise TraceError("unsupported trace version (expected 1)")
    if _integer(record["decay_interval"], "decay_interval") != DECAY_INTERVAL:
        raise TraceError("unsupported hotness decay interval (expected 16)")
    budget = _integer(record["budget"], "budget", 1, MAX_LAYER * MAX_EXPERT)
    expert_bytes = _integer(record["expert_bytes"], "expert_bytes", 1)
    clock = _integer(record["clock"], "clock")
    token = _integer(record["token"], "token", 1)
    decay_token = _integer(record["decay_token"], "decay_token", 1, token)
    if token - decay_token >= DECAY_INTERVAL:
        raise TraceError("snapshot token has not completed hotness decay")
    resident = _triples(record["resident"], "resident", clock)
    if len(resident) > budget:
        raise TraceError("initial resident set exceeds budget")
    if len(set(resident.values())) != len(resident):
        raise TraceError("resident last_used values must be unique serial cache clocks")
    hotness = _triples(record["hotness"], "hotness", UINT32_MAX)
    return Snapshot(budget, expert_bytes, clock, token, decay_token, resident, hotness)


def _route(record, snapshot, line):
    _fields(record, {"event", "token", "layer", "ids", "hit_mask", "pread_bytes",
                     "pread_ms", "load_ms", "clock"})
    token = _integer(record["token"], "token", 1)
    layer = _integer(record["layer"], "layer", maximum=MAX_LAYER - 1)
    ids = record["ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_SELECTED:
        raise TraceError("ids must contain 1..8 experts")
    ids = tuple(_integer(expert, "expert", maximum=MAX_EXPERT - 1) for expert in ids)
    if len(set(ids)) != len(ids):
        raise TraceError("duplicate requested expert")
    if len(ids) > snapshot.budget:
        raise TraceError("protected request set exceeds budget")
    if len(ids) * snapshot.expert_bytes > UINT64_MAX:
        raise TraceError("request byte count exceeds uint64")
    return Route(token, layer, ids,
                 _integer(record["hit_mask"], "hit_mask", maximum=(1 << len(ids)) - 1),
                 _integer(record["pread_bytes"], "pread_bytes"),
                 _milliseconds(record["pread_ms"], "pread_ms"),
                 _milliseconds(record["load_ms"], "load_ms"),
                 _integer(record["clock"], "clock"), line)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def parse_trace(lines, source="<stream>"):
    """Parse strictly ordered JSONL; snapshot is already aged for the first row."""
    snapshot = None
    routes = []
    ended = False
    clock = None
    first_layers = None
    token_layers = []
    for line_number, text in enumerate(lines, 1):
        try:
            if ended:
                raise TraceError("record after end footer")
            record = json.loads(text, object_pairs_hook=_object)
            if not isinstance(record, dict):
                raise TraceError("record must be a JSON object")
            event = record.get("event")
            if snapshot is None:
                if event != "snapshot":
                    raise TraceError("first record must be a snapshot")
                snapshot = _snapshot(record)
                clock = snapshot.clock
            elif event == "route":
                route = _route(record, snapshot, line_number)
                if not routes:
                    if route.token != snapshot.token:
                        raise TraceError("first route token differs from aged snapshot token")
                elif route.token == routes[-1].token:
                    if route.layer <= routes[-1].layer:
                        raise TraceError("layers must increase within each serial decode token")
                elif route.token == routes[-1].token + 1:
                    if first_layers is None:
                        first_layers = tuple(token_layers)
                    elif tuple(token_layers) != first_layers:
                        raise TraceError("serial layer sequence changed between tokens")
                    token_layers = []
                else:
                    raise TraceError("decode tokens must be contiguous and nondecreasing")
                token_layers.append(route.layer)
                if first_layers is not None and tuple(token_layers) != first_layers[:len(token_layers)]:
                    raise TraceError("serial layer sequence changed between tokens")
                clock += len(route.ids)
                if clock > UINT64_MAX:
                    raise TraceError("cache clock would overflow uint64")
                routes.append(route)
            elif event == "end":
                _fields(record, {"event", "routes"})
                if not routes:
                    raise TraceError("trace contains no routes")
                if _integer(record["routes"], "routes") != len(routes):
                    raise TraceError("footer route count differs from trace")
                ended = True
            else:
                raise TraceError(f"unexpected event {event!r}")
        except (TraceError, json.JSONDecodeError, OverflowError) as error:
            raise TraceError(f"{source}:{line_number}: {error}") from error
    if not ended:
        raise TraceError(f"{source}: incomplete trace: missing end footer")
    return Trace(snapshot, tuple(routes), str(source))


def load_trace(path):
    """Read plain or gzip JSONL, detecting gzip by its magic bytes."""
    try:
        with open(path, "rb") as raw:
            compressed = raw.read(2) == b"\x1f\x8b"
            raw.seek(0)
            stream = gzip.GzipFile(fileobj=raw) if compressed else raw
            with io.TextIOWrapper(stream, encoding="utf-8") as text:
                return parse_trace(text, str(path))
    except (OSError, EOFError, UnicodeError, zlib.error) as error:
        raise TraceError(f"{path}: cannot read trace: {error}") from error


def future_uses(routes):
    """Return initial and post-route next-use indices in O(requested experts)."""
    next_use = {}
    after = [None] * len(routes)
    never = len(routes)
    for index in range(len(routes) - 1, -1, -1):
        keys = routes[index].keys
        after[index] = tuple(next_use.get(key, never) for key in keys)
        for key in keys:
            next_use[key] = index
    return next_use, after


class Cache:
    """Heap eviction with lazy invalidation and bounded stale-entry storage."""

    def __init__(self, snapshot, policy, next_use=None, never=0):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy
        self.budget = snapshot.budget
        self.resident = dict(snapshot.resident)
        self.hotness = dict(snapshot.hotness)
        self.clock = snapshot.clock
        self.token = snapshot.token
        self.decay_token = snapshot.decay_token
        self.next_use = dict(next_use or {})
        self.never = never
        self.current_keys = set()
        self.fifo = dict(snapshot.resident)
        self.fifo_clock = snapshot.clock
        self._rebuild()

    def _entry(self, key):
        last_used = self.resident[key]
        if self.policy == "current":
            priority = (self.hotness.get(key, 0),)
        elif self.policy == "least_stale_layer_fifo":
            priority = (int(key in self.current_keys), key[0], self.fifo[key])
        else:
            priority = (-self.next_use.get(key, self.never),)
        return priority, last_used, key[0], key[1]

    def _rebuild(self):
        self.heap = [self._entry(key) for key in self.resident]
        heapq.heapify(self.heap)

    def _touch(self, key):
        self.clock += 1
        self.resident[key] = self.clock
        if self.policy == "least_stale_layer_fifo" and key not in self.current_keys:
            self.fifo_clock += 1
            self.fifo[key] = self.fifo_clock
            self.current_keys.add(key)
        heapq.heappush(self.heap, self._entry(key))

    def _age(self, token):
        if token == self.token:
            return
        self.token = token
        if self.policy == "least_stale_layer_fifo":
            previous = self.current_keys
            self.current_keys = set()
            # Only the previous cycle's accessed residents change queue class.
            for key in previous:
                if key in self.resident:
                    heapq.heappush(self.heap, self._entry(key))
        shifts = (token - self.decay_token) // DECAY_INTERVAL
        if shifts:
            self.hotness = {key: count >> shifts for key, count in self.hotness.items()
                            if count >> shifts}
            self.decay_token += shifts * DECAY_INTERVAL
            if self.policy == "current":
                self._rebuild()

    def _evict(self, count, protected):
        skipped = []
        victims = []
        while len(victims) < count:
            if not self.heap:
                raise TraceError("no unprotected resident victim for mandatory admission")
            entry = heapq.heappop(self.heap)
            key = (entry[2], entry[3])
            if key not in self.resident or entry != self._entry(key):
                continue
            if key in protected:
                skipped.append(entry)
                continue
            del self.resident[key]
            if self.policy == "least_stale_layer_fifo":
                self.current_keys.discard(key)
                del self.fifo[key]
            victims.append(key)
        for entry in skipped:
            heapq.heappush(self.heap, entry)
        return tuple(victims)

    def resolve(self, route, next_after=None):
        self._age(route.token)
        keys = route.keys
        for key in keys:
            self.hotness[key] = min(UINT32_MAX, self.hotness.get(key, 0) + 1)
        if self.policy == "oracle":
            if next_after is None or len(next_after) != len(keys):
                raise ValueError("oracle requires next-use data for every requested expert")
            self.next_use.update(zip(keys, next_after))
        hit_mask = 0
        misses = []
        # Every hit is peeked before any victim selection or miss installation.
        for index, key in enumerate(keys):
            if key in self.resident:
                hit_mask |= 1 << index
                self._touch(key)
            else:
                misses.append(key)
        victims = self._evict(max(0, len(self.resident) + len(misses) - self.budget),
                              set(keys))
        for key in misses:
            self._touch(key)
        if len(self.heap) > max(64, 3 * self.budget):
            self._rebuild()
        return hit_mask, victims


def _metrics():
    return {"routes": 0, "lookups": 0, "hits": 0, "misses": 0, "evictions": 0,
            "pread_bytes": 0}


def _add(metrics, route, hit_mask, evictions, expert_bytes):
    hits = hit_mask.bit_count()
    misses = len(route.ids) - hits
    metrics["routes"] += 1
    metrics["lookups"] += len(route.ids)
    metrics["hits"] += hits
    metrics["misses"] += misses
    metrics["evictions"] += evictions
    metrics["pread_bytes"] += misses * expert_bytes


def _finish(metrics):
    metrics["hit_rate"] = metrics["hits"] / metrics["lookups"] if metrics["lookups"] else 0.0
    metrics["miss_rate"] = metrics["misses"] / metrics["lookups"] if metrics["lookups"] else 0.0
    return metrics


def replay(trace, policy="current", audit=True):
    """Replay one policy; audit is applicable only to the recorded current policy."""
    if audit and policy != "current":
        raise ValueError("observed-state audit requires current policy")
    initial, after = future_uses(trace.routes) if policy == "oracle" else ({}, None)
    cache = Cache(trace.snapshot, policy, initial, len(trace.routes))
    total = _metrics()
    layers = {}
    for index, route in enumerate(trace.routes):
        hit_mask, victims = cache.resolve(route, after[index] if after is not None else None)
        if audit:
            expected_bytes = (len(route.ids) - hit_mask.bit_count()) * trace.snapshot.expert_bytes
            for field, observed, predicted in (("hit_mask", route.hit_mask, hit_mask),
                                               ("pread_bytes", route.pread_bytes, expected_bytes),
                                               ("clock", route.clock, cache.clock)):
                if observed != predicted:
                    raise TraceError(f"{trace.source}:{route.line}: route {index + 1} "
                                     f"token={route.token} layer={route.layer}: {field} mismatch "
                                     f"(observed {observed}, replay {predicted})")
        _add(total, route, hit_mask, len(victims), trace.snapshot.expert_bytes)
        layer = layers.setdefault(route.layer, _metrics())
        _add(layer, route, hit_mask, 0, trace.snapshot.expert_bytes)
        # Per-layer evictions are charged to the evicted expert's layer.
        for victim_layer, _ in victims:
            layers.setdefault(victim_layer, _metrics())["evictions"] += 1
    total["tokens"] = trace.routes[-1].token - trace.routes[0].token + 1
    return {"policy": policy, "aggregate": _finish(total),
            "per_layer": [{"layer": layer, **_finish(metrics)}
                          for layer, metrics in sorted(layers.items())]}


def _reductions(metrics, baseline):
    for metric, prefix in (("misses", "miss"), ("pread_bytes", "byte")):
        saved = baseline[metric] - metrics[metric]
        metrics[prefix + "_reduction"] = saved
        metrics[prefix + "_reduction_pct"] = 100 * saved / baseline[metric] if baseline[metric] else 0.0


def analyze_trace(trace, label=None):
    current = replay(trace)
    results = [current] + [replay(trace, policy, audit=False) for policy in POLICIES[1:]]
    baseline_layers = {row["layer"]: row for row in current["per_layer"]}
    observed_layers = {}
    for route in trace.routes:
        values = observed_layers.setdefault(route.layer, {"pread_ms": 0.0, "load_ms": 0.0})
        values["pread_ms"] += route.pread_ms
        values["load_ms"] += route.load_ms
    for result in results:
        _reductions(result["aggregate"], current["aggregate"])
        for row in result["per_layer"]:
            _reductions(row, baseline_layers.get(row["layer"], _metrics()))
    snapshot = trace.snapshot
    return {"label": label or Path(trace.source).name, "source": trace.source,
            "baseline_validated": True,
            "snapshot": {"budget": snapshot.budget, "expert_bytes": snapshot.expert_bytes,
                         "resident_entries": len(snapshot.resident),
                         "hotness_entries": len(snapshot.hotness), "clock": snapshot.clock,
                         "token": snapshot.token, "decay_token": snapshot.decay_token},
            "observed": {"pread_bytes": current["aggregate"]["pread_bytes"],
                         "pread_ms": sum(route.pread_ms for route in trace.routes),
                         "load_ms": sum(route.load_ms for route in trace.routes),
                         "per_layer": [{"layer": layer, **values}
                                       for layer, values in sorted(observed_layers.items())]},
            "policies": results}


CSV_FIELDS = ("label", "source", "baseline_validated", "budget", "expert_bytes", "policy",
              "scope", "layer", "routes", "tokens", "lookups", "hits", "misses", "hit_rate",
              "miss_rate", "evictions", "pread_bytes", "miss_reduction", "miss_reduction_pct",
              "byte_reduction", "byte_reduction_pct", "observed_pread_ms", "observed_load_ms")


def write_report(reports, output, format="json"):
    if format == "json":
        json.dump({"version": 1,
                   "policy_definitions": {
                       "current": "lowest decayed route hotness, then oldest cache use",
                       "least_stale_layer_fifo": ("demand-only layer-FIFO variant inspired by least-stale: "
                                       "stale before current-cycle residents, ascending layer, FIFO; "
                                       "all initial residents stale, FIFO seeded by last_used; "
                                       "includes stale-right fallback; no prefetch; not a paper reproduction"),
                       "oracle": "farthest next route use within the finite trace"},
                   "metric_semantics": {
                       "pread_bytes": "logical requested expert bytes, not physical SSD traffic",
                       "hit_rate": "simulated expert hit fraction; no throughput prediction",
                       "pread_ms": "observed blocking worker-read batch wall time",
                       "load_ms": "observed synchronous load wall time including preparation/install",
                       "reductions": "same initial resident state, budget, and routing; versus audited current"},
                   "traces": reports}, output, indent=2, allow_nan=False)
        output.write("\n")
        return
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for report in reports:
        common = {name: report[name] for name in ("label", "source", "baseline_validated")}
        common.update({name: report["snapshot"][name] for name in ("budget", "expert_bytes")})
        timing = {row["layer"]: row for row in report["observed"]["per_layer"]}
        for result in report["policies"]:
            for scope, metrics in [("aggregate", result["aggregate"])] + [
                    ("layer", row) for row in result["per_layer"]]:
                observed = report["observed"] if scope == "aggregate" else timing.get(metrics["layer"], {})
                row = {**common, "policy": result["policy"], "scope": scope, **metrics,
                       "observed_pread_ms": observed.get("pread_ms", 0.0),
                       "observed_load_ms": observed.get("load_ms", 0.0)}
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="complete version-1 JSONL or gzip traces")
    parser.add_argument("--label", action="append", help="workload label, once per trace in input order")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path, help="report path (default: stdout)")
    args = parser.parse_args(argv)
    if args.label and len(args.label) != len(args.traces):
        parser.error("--label must be supplied once per trace")
    try:
        reports = [analyze_trace(load_trace(path), args.label[index] if args.label else None)
                   for index, path in enumerate(args.traces)]
        if args.output:
            with args.output.open("w", encoding="utf-8", newline="") as output:
                write_report(reports, output, args.format)
        else:
            write_report(reports, sys.stdout, args.format)
    except (TraceError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
