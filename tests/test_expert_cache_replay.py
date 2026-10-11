import copy
import csv
from functools import lru_cache
import gzip
import importlib.util
import io
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/expert_cache_replay.py"
spec = importlib.util.spec_from_file_location("expert_cache_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def snapshot(budget=2, resident=None, hotness=None, clock=None, token=1, decay_token=1):
    resident = resident or []
    return {"event": "snapshot", "version": 1, "budget": budget, "expert_bytes": 100,
            "clock": clock if clock is not None else max((row[2] for row in resident), default=0),
            "token": token, "decay_token": decay_token, "decay_interval": 16,
            "resident": resident, "hotness": hotness or []}


def route(token, layer, ids, mask=0, clock=0, pread_bytes=None):
    return {"event": "route", "token": token, "layer": layer, "ids": list(ids),
            "hit_mask": mask, "pread_bytes": (len(ids) - mask.bit_count()) * 100
            if pread_bytes is None else pread_bytes,
            "pread_ms": 1.0 if mask != (1 << len(ids)) - 1 else 0.0,
            "load_ms": 2.0 if mask != (1 << len(ids)) - 1 else 0.0, "clock": clock}


def encode(records):
    return "".join(json.dumps(record) + "\n" for record in records)


def parsed(records):
    return replay.parse_trace(io.StringIO(encode(records)), "synthetic.jsonl")


def policy_trace():
    records = [snapshot(resident=[[0, 0, 1], [0, 1, 2]], hotness=[[0, 0, 10]])]
    records.extend(route(token, 0, [expert], clock=token + 2)
                   for token, expert in enumerate([2, 1, 2, 1], 1))
    return records + [{"event": "end", "routes": 4}]


def cache_from(record, policy="current"):
    return replay.Cache(replay._snapshot(record), policy)


def request(token, layer, ids):
    return replay.Route(token, layer, tuple(ids), 0, 0, 0.0, 0.0, 0)


def reference_records(seed=932):
    """Independent scan implementation used only to generate an audited fixture."""
    rng = random.Random(seed)
    initial = snapshot(9, [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]],
                       [[0, 0, 5], [0, 7, 100], [3, 2, replay.UINT32_MAX]])
    resident = {(layer, expert): age for layer, expert, age in initial["resident"]}
    hotness = {(layer, expert): count for layer, expert, count in initial["hotness"]}
    clock = initial["clock"]
    decay_token = initial["decay_token"]
    records = [initial]
    for token in range(1, 51):
        while token - decay_token >= 16:
            hotness = {key: value >> 1 for key, value in hotness.items() if value >> 1}
            decay_token += 16
        for layer in range(4):
            ids = rng.sample(range(8), 3)
            keys = [(layer, expert) for expert in ids]
            for key in keys:
                hotness[key] = min(replay.UINT32_MAX, hotness.get(key, 0) + 1)
            mask = 0
            missing = []
            for index, key in enumerate(keys):
                if key in resident:
                    mask |= 1 << index
                    clock += 1
                    resident[key] = clock
                else:
                    missing.append(key)
            protected = set(keys)
            while len(resident) + len(missing) > initial["budget"]:
                victim = min((key for key in resident if key not in protected),
                             key=lambda key: (hotness.get(key, 0), resident[key], key))
                del resident[victim]
            for key in missing:
                clock += 1
                resident[key] = clock
            records.append(route(token, layer, ids, mask, clock))
    return records + [{"event": "end", "routes": 200}]


class CurrentPolicyTests(unittest.TestCase):
    def test_hotness_precedes_age_and_nonresident_hotness_survives_eviction(self):
        cache = cache_from(snapshot(resident=[[0, 0, 1], [0, 1, 2]],
                                    hotness=[[0, 0, 20], [0, 1, 2], [0, 2, 100]]))
        mask, victims = cache.resolve(request(1, 0, [2]))
        self.assertEqual((mask, victims), (0, ((0, 1),)))
        self.assertEqual(cache.hotness[(0, 2)], 101)
        cache.resolve(request(2, 0, [1]))
        self.assertEqual(set(cache.resident), {(0, 1), (0, 2)})
        self.assertEqual(cache.hotness[(0, 0)], 20)
        self.assertEqual(cache.hotness[(0, 1)], 3)

    def test_equal_hotness_evicts_oldest_not_lowest_expert_id(self):
        cache = cache_from(snapshot(resident=[[0, 0, 9], [0, 1, 2]], clock=10))
        self.assertEqual(cache.resolve(request(1, 0, [2]))[1], ((0, 1),))
        self.assertEqual(cache.resident[(0, 2)], 11)

    def test_hit_peeks_precede_ordered_miss_install_and_batch_is_protected(self):
        cache = cache_from(snapshot(3, [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
                                    [[0, 2, 100]]))
        mask, victims = cache.resolve(request(1, 0, [3, 0, 4]))
        self.assertEqual(mask, 0b010)
        self.assertEqual(victims, ((0, 1), (0, 2)))
        self.assertEqual(cache.resident, {(0, 0): 4, (0, 3): 5, (0, 4): 6})
        self.assertEqual(cache.clock, 6)
        self.assertEqual(cache.resolve(request(2, 0, [4, 3, 0])), (0b111, ()))
        self.assertEqual(cache.resident, {(0, 0): 9, (0, 3): 8, (0, 4): 7})

    def test_snapshot_is_already_aged_and_boundary_decay_changes_victim(self):
        cache = cache_from(snapshot(resident=[[0, 0, 1], [0, 1, 2]],
                                    hotness=[[0, 0, 3], [0, 1, 2], [0, 7, 3]],
                                    token=17, decay_token=17))
        cache.resolve(request(17, 0, [0]))
        self.assertEqual(cache.hotness[(0, 0)], 4)
        self.assertEqual(cache.hotness[(0, 7)], 3)
        cache.resolve(request(32, 0, [0]))
        self.assertEqual(cache.hotness[(0, 1)], 2)
        # Two untouched counts become equal after aging, so age decides.
        other = cache_from(snapshot(resident=[[0, 0, 1], [0, 1, 2]],
                                    hotness=[[0, 0, 3], [0, 1, 2]], token=16))
        self.assertEqual(other.resolve(request(17, 0, [2]))[1], ((0, 0),))
        self.assertEqual(other.hotness[(0, 0)], 1)
        self.assertEqual(other.decay_token, 17)
        other.resolve(request(49, 0, [2]))
        self.assertEqual(other.decay_token, 49)
        self.assertNotIn((0, 0), other.hotness)

    def test_uint32_hotness_saturates(self):
        cache = cache_from(snapshot(resident=[[0, 0, 1]],
                                    hotness=[[0, 0, replay.UINT32_MAX]]))
        cache.resolve(request(1, 0, [0]))
        self.assertEqual(cache.hotness[(0, 0)], replay.UINT32_MAX)

    def test_current_audit_matches_independent_scan_across_layers_and_decay(self):
        trace = parsed(reference_records())
        result = replay.replay(trace)
        self.assertEqual(result["aggregate"]["routes"], 200)
        self.assertEqual(result["aggregate"]["tokens"], 50)
        self.assertEqual(result["aggregate"]["lookups"], 600)
        self.assertEqual(sum(row["misses"] for row in result["per_layer"]),
                         result["aggregate"]["misses"])
        self.assertEqual(sum(row["evictions"] for row in result["per_layer"]),
                         result["aggregate"]["evictions"])


class LeastStaleLayerFIFOTests(unittest.TestCase):
    policy = "least_stale_layer_fifo"

    def test_stale_right_layer_precedes_current_left_layer(self):
        cache = cache_from(snapshot(resident=[[0, 0, 1], [3, 0, 2]]), self.policy)
        cache.resolve(request(1, 0, [0]))
        self.assertEqual(cache.resolve(request(1, 1, [0]))[1], ((3, 0),))
        self.assertEqual(set(cache.resident), {(0, 0), (1, 0)})

    def test_layer_order_precedes_fifo_age_and_rollover_marks_current_stale(self):
        cache = cache_from(snapshot(resident=[[0, 0, 20], [2, 0, 1]]), self.policy)
        self.assertEqual(cache.resolve(request(1, 1, [0]))[1], ((0, 0),))
        cache.resolve(request(1, 2, [0]))
        self.assertEqual(cache.resolve(request(2, 0, [1]))[1], ((1, 0),))
        self.assertEqual(set(cache.resident), {(0, 1), (2, 0)})

    def test_fifo_preserves_order_on_repeat_current_hit_and_protects_batch(self):
        cache = cache_from(snapshot(3, [[0, 0, 1], [0, 1, 2], [0, 2, 3]]), self.policy)
        cache.resolve(request(1, 0, [1, 0, 2]))
        cache.resolve(request(1, 0, [1]))
        # Current fallback uses FIFO; protected 1 cannot be the victim.
        self.assertEqual(cache.resolve(request(1, 0, [1, 3]))[1], ((0, 0),))
        self.assertEqual(set(cache.resident), {(0, 1), (0, 2), (0, 3)})
        self.assertEqual(cache.resolve(request(2, 0, [3, 4]))[1], ((0, 1),))


class OracleTests(unittest.TestCase):
    def test_oracle_matches_exhaustive_atomic_victim_choices(self):
        # Exhaustive DP has no next-use ranking or production cache helpers.
        for universe_size, budget, length in ((3, 2, 4), (4, 2, 2), (4, 3, 2)):
            universe = tuple(range(universe_size))
            requests = [frozenset(group) for size in range(1, budget + 1)
                        for group in itertools.combinations(universe, size)]
            warm_sets = [frozenset(group) for size in range(budget + 1)
                         for group in itertools.combinations(universe, size)]
            for sequence in itertools.product(requests, repeat=length):
                @lru_cache(maxsize=None)
                def optimal(index, resident):
                    if index == len(sequence):
                        return 0
                    selected = sequence[index]
                    misses = len(selected - resident)
                    overflow = max(0, len(resident | selected) - budget)
                    return misses + min(
                        optimal(index + 1, (resident | selected) - frozenset(victims))
                        for victims in itertools.combinations(sorted(resident - selected), overflow))

                routes = tuple(request(index + 1, 0, sorted(selected))
                               for index, selected in enumerate(sequence))
                for warm in warm_sets:
                    state = replay._snapshot(snapshot(
                        budget, [[0, expert, index + 1] for index, expert in enumerate(sorted(warm))]))
                    trace = replay.Trace(state, routes, "exhaustive")
                    actual = replay.replay(trace, "oracle", audit=False)["aggregate"]["misses"]
                    self.assertEqual(actual, optimal(0, warm), (budget, sequence, warm))

    def test_request_order_does_not_change_oracle_misses_or_protection(self):
        selected = [(2, 0), (3, 1), (2, 3), (0, 1)]
        for permutations in itertools.product((False, True), repeat=len(selected)):
            routes = tuple(request(index + 1, 0, ids[::-1] if reverse else ids)
                           for index, (ids, reverse) in enumerate(zip(selected, permutations)))
            state = replay._snapshot(snapshot(resident=[[0, 0, 1], [0, 1, 2]]))
            result = replay.replay(replay.Trace(state, routes, "permutations"), "oracle", audit=False)
            self.assertEqual(result["aggregate"]["misses"], 6)

    def test_oracle_never_evicts_an_already_peeked_current_request(self):
        routes = (request(1, 0, [0, 1]), request(2, 0, [2]), request(3, 0, [1]))
        initial = replay._snapshot(snapshot(resident=[[0, 0, 1], [0, 2, 2]]))
        first, after = replay.future_uses(routes)
        cache = replay.Cache(initial, "oracle", first, len(routes))
        misses = 0
        for index, selected in enumerate(routes):
            mask, _ = cache.resolve(selected, after[index])
            misses += len(selected.ids) - mask.bit_count()
            self.assertTrue(set(selected.keys) <= cache.resident.keys())
            self.assertLessEqual(len(cache.resident), initial.budget)
        self.assertEqual(misses, 2)


class TraceAndReportingTests(unittest.TestCase):
    def test_warm_start_metrics_and_same_budget_reductions(self):
        report = replay.analyze_trace(parsed(policy_trace()), "workload")
        metrics = {row["policy"]: row["aggregate"] for row in report["policies"]}
        self.assertTrue(report["baseline_validated"])
        self.assertEqual(report["snapshot"]["resident_entries"], 2)
        self.assertEqual(report["observed"]["pread_bytes"], 400)
        self.assertEqual(report["observed"]["pread_ms"], 4)
        self.assertEqual(report["observed"]["load_ms"], 8)
        self.assertEqual(metrics["current"]["misses"], 4)
        for policy in ("least_stale_layer_fifo", "oracle"):
            self.assertEqual(metrics[policy]["misses"], 1)
            self.assertEqual(metrics[policy]["hits"], 3)
            self.assertEqual(metrics[policy]["miss_reduction"], 3)
            self.assertEqual(metrics[policy]["byte_reduction"], 300)
            self.assertEqual(metrics[policy]["miss_reduction_pct"], 75)
        self.assertEqual(metrics["current"]["byte_reduction"], 0)

    def test_evictions_are_attributed_to_victim_layer(self):
        records = [snapshot(1, [[2, 0, 1]]), route(1, 0, [0], clock=2),
                   {"event": "end", "routes": 1}]
        result = replay.replay(parsed(records))
        layers = {row["layer"]: row for row in result["per_layer"]}
        self.assertEqual(layers[0]["misses"], 1)
        self.assertEqual(layers[0]["evictions"], 0)
        self.assertEqual(layers[2]["lookups"], 0)
        self.assertEqual(layers[2]["evictions"], 1)
        self.assertEqual(result["aggregate"]["evictions"], 1)

    def test_rejects_malformed_truncated_and_incompatible_state(self):
        base = policy_trace()
        cases = [base[:-1], [], base + [base[-1]], base[1:]]
        for index, field, value in [
            (0, "version", 2), (0, "version", True), (0, "budget", 0),
            (0, "budget", 1), (0, "expert_bytes", 0), (0, "decay_interval", 8),
            (0, "decay_token", 0), (0, "token", 17),
            (0, "resident", [[0, 0, 1], [0, 0, 2]]),
            (0, "resident", [[0, 0, 1], [0, 1, 1]]),
            (0, "resident", [[0, 0, 0]]), (0, "resident", [[0, 0, 3]]),
            (0, "hotness", [[0, 0, 0]]), (0, "hotness", [[0, 0, 1 << 32]]),
            (0, "hotness", [[0, 0, 1], [0, 0, 2]]),
            (1, "ids", []), (1, "ids", [1, 1]), (1, "ids", [-1]),
            (1, "ids", [384]), (1, "ids", [True]), (1, "ids", list(range(9))),
            (1, "ids", [1, 2, 3]), (1, "layer", 80), (1, "hit_mask", 2),
            (1, "token", 2), (2, "token", 3), (2, "token", 1),
            (1, "pread_ms", -1), (1, "load_ms", float("nan")),
            (1, "load_ms", float("inf")), (1, "pread_bytes", -1),
            (-1, "routes", 3),
        ]:
            records = copy.deepcopy(base)
            records[index][field] = value
            cases.append(records)
        records = copy.deepcopy(base)
        records[1]["budget"] = 3
        cases.append(records)
        records = copy.deepcopy(base)
        del records[1]["clock"]
        cases.append(records)
        for records in cases:
            with self.subTest(records=records):
                with self.assertRaises(replay.TraceError):
                    parsed(records)
        for text in ('{"event": "snapshot"}\n{bad\n',
                     encode(base).replace('"budget": 2', '"budget": 2, "budget": 2'),
                     encode(base) + "\n", "[]\n"):
            with self.assertRaises(replay.TraceError):
                replay.parse_trace(io.StringIO(text))

    def test_rejects_serial_layer_sequence_changes_and_clock_overflow(self):
        records = [snapshot(), route(1, 0, [0], clock=1), route(1, 2, [0], clock=2),
                   route(2, 0, [0], mask=1, clock=3), route(2, 3, [0], clock=4),
                   {"event": "end", "routes": 4}]
        with self.assertRaises(replay.TraceError):
            parsed(records)
        records = [snapshot(clock=replay.UINT64_MAX), route(1, 0, [0]),
                   {"event": "end", "routes": 1}]
        with self.assertRaises(replay.TraceError):
            parsed(records)

    def test_observed_hit_bytes_and_clock_must_all_match(self):
        for field, value in (("hit_mask", 1), ("pread_bytes", 0), ("clock", 99)):
            records = copy.deepcopy(policy_trace())
            records[2][field] = value
            trace = parsed(records)
            with self.subTest(field=field), self.assertRaises(replay.TraceError):
                replay.analyze_trace(trace)

    def test_zero_miss_baseline_has_finite_zero_reductions(self):
        records = [snapshot(resident=[[0, 0, 1]]), route(1, 0, [0], mask=1, clock=2),
                   {"event": "end", "routes": 1}]
        report = replay.analyze_trace(parsed(records))
        for policy in report["policies"]:
            self.assertEqual(policy["aggregate"]["miss_reduction_pct"], 0)
            self.assertEqual(policy["aggregate"]["byte_reduction_pct"], 0)
            self.assertEqual(policy["aggregate"]["hit_rate"], 1)
        output = io.StringIO()
        replay.write_report([report], output)
        self.assertEqual(json.loads(output.getvalue())["traces"][0]["observed"]["pread_ms"], 0)

    def test_gzip_input_and_failed_audit_preserve_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain.jsonl"
            zipped = Path(directory) / "compressed.data"
            broken = Path(directory) / "broken.jsonl"
            plain.write_text(encode(policy_trace()))
            with gzip.open(zipped, "wt") as stream:
                stream.write(encode(policy_trace()))
            csv_path = Path(directory) / "report.csv"
            command = [sys.executable, str(SCRIPT), str(zipped),
                       "--format", "csv", "--output", str(csv_path)]
            subprocess.run(command, capture_output=True, text=True, check=True)
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            oracle = next(row for row in rows if
                          row["policy"] == "oracle" and row["scope"] == "aggregate")
            self.assertEqual(int(oracle["misses"]), 1)
            self.assertEqual(int(oracle["byte_reduction"]), 300)
            damaged = policy_trace()
            damaged[3]["clock"] += 1
            broken.write_text(encode(damaged))
            csv_path.write_text("keep-existing-output\n")
            failure = subprocess.run([sys.executable, str(SCRIPT), str(plain), str(broken),
                                      "--output", str(csv_path)], capture_output=True, text=True)
            self.assertEqual(failure.returncode, 2)
            self.assertEqual(failure.stdout, "")
            self.assertEqual(csv_path.read_text(), "keep-existing-output\n")
            raw = zipped.read_bytes()
            zipped.write_bytes(raw[:-4])
            with self.assertRaises(replay.TraceError):
                replay.load_trace(zipped)
            # A reserved DEFLATE block type raises zlib.error, not OSError.
            zipped.write_bytes(bytes.fromhex("1f8b08000000000000ff07") + b"\x00" * 8)
            with self.assertRaises(replay.TraceError):
                replay.load_trace(zipped)


if __name__ == "__main__":
    unittest.main()
