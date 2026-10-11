import copy
from dataclasses import replace
import gzip
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/expert_prefetch_replay.py"
spec = importlib.util.spec_from_file_location("expert_prefetch_replay", SCRIPT)
prefetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefetch)
replay = prefetch.replay


def snapshot(budget=8, resident=None, hotness=None):
    resident = resident or {}
    return replay.Snapshot(budget, 100, max(resident.values(), default=0), 1, 1,
                           resident, hotness or {})


def request(layer, ids, token=1):
    return replay.Route(token, layer, tuple(ids), 0, 0, 0.0, 0.0, 0)


def audited_trace(routes, initial=None, source="synthetic.jsonl"):
    initial = initial or snapshot()
    cache = replay.Cache(initial, "current")
    recorded = []
    for route in routes:
        mask, _ = cache.resolve(route)
        recorded.append(replace(route, hit_mask=mask, clock=cache.clock,
                                pread_bytes=(len(route.ids) - mask.bit_count()) * initial.expert_bytes))
    return replay.Trace(initial, tuple(recorded), source)


def fixture(seed=4, tokens=9, budget=8):
    rng = random.Random(seed)
    return audited_trace([request(layer, rng.sample(range(6), 2), token)
                          for token in range(1, tokens + 1) for layer in range(3)], snapshot(budget))


def model():
    artifact, _ = prefetch.fit([fixture()], [])
    return prefetch.Predictor(artifact)


def encode(trace):
    initial = trace.snapshot
    records = [{"event": "snapshot", "version": 1, "budget": initial.budget,
                "expert_bytes": initial.expert_bytes, "clock": initial.clock,
                "token": initial.token, "decay_token": initial.decay_token,
                "decay_interval": replay.DECAY_INTERVAL,
                "resident": [[*key, value] for key, value in initial.resident.items()],
                "hotness": [[*key, value] for key, value in initial.hotness.items()]}]
    records.extend({"event": "route", "token": route.token, "layer": route.layer,
                    "ids": list(route.ids), "hit_mask": route.hit_mask,
                    "pread_bytes": route.pread_bytes, "pread_ms": route.pread_ms,
                    "load_ms": route.load_ms, "clock": route.clock} for route in trace.routes)
    records.append({"event": "end", "routes": len(trace.routes)})
    return "".join(json.dumps(record) + "\n" for record in records)


def frozen_files(root, artifact=None):
    artifact = artifact or model().artifact
    model_path = root / "model.json.gz"
    prefetch.write_model(model_path, artifact)
    freeze = {"version": prefetch.VERSION, "settings": prefetch.SETTINGS,
              "training_inputs": artifact["training_inputs"],
              "model_sha256": prefetch.file_hash(model_path),
              "predictor_script_sha256": prefetch.file_hash(SCRIPT),
              "baseline_script_sha256": prefetch.file_hash(replay.__file__)}
    path = root / "freeze.json"
    prefetch.write_json(path, freeze)
    return path


class PredictorTests(unittest.TestCase):
    def test_conditional_scores_shrink_toward_training_target_priors(self):
        predictor = model()
        source = predictor.artifact["counts"][0]
        route = request(0, [0, 1])
        expected = []
        for expert in range(prefetch.EXPERTS):
            prior = source["target_counts"][expert] / source["events"]
            expected.append(sum((source["coactivation_counts"][identifier][expert] + 32 * prior) /
                                (source["source_counts"][identifier] + 32) for identifier in route.ids))
        ranking = sorted(range(prefetch.EXPERTS), key=lambda expert: (-expected[expert], expert))
        self.assertEqual(predictor.predict(route, 4), tuple((2, expert) for expert in ranking[:4]))
        for expert, value in enumerate(predictor.tables[0][200]):
            self.assertAlmostEqual(value, source["target_counts"][expert] / source["events"])

    def test_layer_wrap_uses_two_event_offset_without_cross_trace_pairs(self):
        first, second = fixture(tokens=4), fixture(seed=7, tokens=4)
        artifact, costs = prefetch.fit([first, second], [])
        self.assertEqual(costs["training_pairs"], 20)
        self.assertEqual([row["events"] for row in artifact["counts"]], [8, 6, 6])
        predictor = prefetch.Predictor(artifact)
        self.assertEqual(predictor.targets, {0: 2, 1: 0, 2: 1})
        self.assertEqual({layer for layer, _ in predictor.predict(request(2, [0]), 1)}, {1})

    def test_malformed_domain_parameters_and_layer_topology_fail(self):
        artifact = copy.deepcopy(model().artifact)
        artifact["settings"] = {**artifact["settings"], "prior_pseudoroutes": 31}
        with self.assertRaisesRegex(ValueError, "parameters"):
            prefetch.Predictor(artifact)
        with self.assertRaisesRegex(ValueError, "domain"):
            model().predict(request(0, [256]), 1)
        trace = fixture()
        routes = list(trace.routes)
        routes[1] = replace(routes[1], layer=7)
        with self.assertRaisesRegex(ValueError, "sequence"):
            prefetch.staging_replay(replace(trace, routes=tuple(routes)), model(), 1)


class SlotAndLifecycleTests(unittest.TestCase):
    def test_stage_never_evicts_any_member_of_complete_current_request(self):
        cache = prefetch.StagingCache(snapshot(3, {(0, 0): 1, (0, 1): 2, (0, 2): 3}), 2)
        route = request(0, [0, 3])
        cache.demand(route, 0)
        clock = cache.cache.clock
        loaded = cache.issue(((2, 0), (2, 1)), 0, set(route.keys))
        self.assertEqual(loaded, ((2, 0),))
        self.assertTrue(set(route.keys) <= cache.cache.resident.keys())
        self.assertEqual(cache.total["protection_skips"], 1)
        self.assertEqual(cache.cache.clock, clock)
        self.assertEqual(len(cache.cache.resident) + len(cache.staged), 3)
        self.assertEqual(cache.peak_slots, 3)
        cache.finish()
        self.assertEqual(cache.total["prefetch_read_bytes"], 100)
        self.assertEqual(cache.total["end_wastes"], 1)

    def test_promotion_is_one_load_and_one_normal_demand_clock_touch(self):
        cache = prefetch.StagingCache(snapshot(4), 1)
        cache.demand(request(0, [0]), 0)
        cache.issue(((2, 1),), 0, {(0, 0)})
        self.assertEqual(cache.cache.clock, 1)
        cache.demand(request(1, [7]), 1)
        mask, promoted = cache.demand(request(2, [1]), 2)
        self.assertEqual((mask, promoted), (1, ((2, 1),)))
        self.assertEqual(cache.cache.clock, 3)
        cache.demand(request(2, [1]), 3)
        cache.finish()
        self.assertEqual(cache.cache.clock, 4)
        self.assertEqual(cache.total["prefetch_loads"], 1)
        self.assertEqual(cache.total["useful_prefetch_loads"], 1)
        self.assertEqual(cache.total["exposed_demand_misses"], 2)
        self.assertEqual(cache.total["total_logical_reads"], 3)
        self.assertEqual(cache.total["useful_predictions"], 1)

    def test_wrong_prediction_expires_and_tail_prediction_is_charged(self):
        cache = prefetch.StagingCache(snapshot(5), 1)
        cache.demand(request(0, [0]), 0)
        cache.issue(((2, 1),), 0, {(0, 0)})
        cache.demand(request(1, [0]), 1)
        cache.demand(request(2, [2]), 2)
        self.assertNotIn((2, 1), cache.staged)
        self.assertEqual(cache.total["expiry_wastes"], 1)
        cache.issue(((1, 2),), 2, {(2, 2)})
        clock = cache.cache.clock
        cache.finish()
        self.assertEqual(cache.cache.clock, clock)
        self.assertEqual(cache.total["prefetch_loads"], 2)
        self.assertEqual(cache.total["prefetch_read_bytes"], 200)
        self.assertEqual(cache.total["wasted_prefetch_loads"], 2)
        self.assertEqual(cache.total["wasted_predictions"], 2)
        self.assertEqual(cache.total["end_predictions"], 1)
        self.assertEqual(cache.total["end_wastes"], 1)

    def test_pending_stage_yields_to_complete_demand_batch(self):
        cache = prefetch.StagingCache(snapshot(2), 1)
        cache.demand(request(0, [0]), 0)
        cache.issue(((2, 1),), 0, {(0, 0)})
        route = request(1, [0, 1])
        mask, _ = cache.demand(route, 1)
        self.assertEqual(mask, 0)
        self.assertEqual(set(cache.cache.resident), set(route.keys))
        self.assertEqual(cache.staged, {})
        self.assertEqual(cache.total["pressure_wastes"], 1)
        cache.finish()
        self.assertEqual(cache.peak_slots, 2)

    def test_resident_and_pending_predictions_do_not_reload_or_extend_expiry(self):
        cache = prefetch.StagingCache(snapshot(5), 2)
        cache.demand(request(0, [0]), 0)
        cache.issue(((0, 0), (2, 1)), 0, {(0, 0)})
        cache.demand(request(1, [0]), 1)
        cache.issue(((2, 1),), 1, {(1, 0)})
        self.assertEqual(cache.staged[(2, 1)].due, 2)
        self.assertEqual(cache.total["prefetch_loads"], 1)
        self.assertEqual(cache.total["resident_predictions"], 1)
        self.assertEqual(cache.total["pending_predictions"], 1)
        cache.demand(request(2, [2]), 2)
        self.assertNotIn((2, 1), cache.staged)
        cache.finish()
        self.assertEqual(cache.total["wasted_prefetch_loads"], 1)

    def test_all_widths_share_original_budget_and_have_complete_load_accounting(self):
        for width in prefetch.WIDTHS:
            trace = fixture(budget=20)
            states = []
            result = prefetch.staging_replay(trace, model(), width, states.append)
            self.assertEqual(result["staging_capacity"], 2 * width)
            self.assertLessEqual(result["peak_slots"], 20)
            self.assertLessEqual(result["peak_staging_slots"], 2 * width)
            for route, state in zip(trace.routes, states):
                self.assertTrue(set(route.keys) <= state["resident"].keys())
                self.assertLessEqual(len(state["resident"]) + len(state["staged"]), 20)
            totals = result["aggregate"]
            self.assertEqual(totals["prefetch_loads"], totals["useful_prefetch_loads"] + totals["wasted_prefetch_loads"])
            self.assertEqual(totals["predictions"], totals["useful_predictions"] + totals["wasted_predictions"])
            self.assertEqual(totals["total_logical_read_bytes"], 100 * totals["total_logical_reads"])
            self.assertEqual(result["final_clock"], 54)
            for field in totals:
                self.assertEqual(totals[field], sum(row[field] for row in result["per_layer"]))


class CausalityAndReportTests(unittest.TestCase):
    def test_changed_future_and_changed_footer_leave_prior_decisions_identical(self):
        predictor = model()
        frozen = json.dumps(predictor.artifact, sort_keys=True)
        original = fixture(seed=91)
        changed = list(original.routes)
        changed[5:] = [replace(route, ids=((route.ids[0] + 11) % 256,
                                         (route.ids[1] + 11) % 256)) for route in changed[5:]]
        altered = audited_trace(changed, original.snapshot)
        extended = audited_trace([*original.routes, request(0, [44, 45], 10),
                                  request(1, [44, 45], 10), request(2, [44, 45], 10)], original.snapshot)
        for width in prefetch.WIDTHS:
            reference, future, longer = [], [], []
            prefetch.staging_replay(original, predictor, width, reference.append)
            prefetch.staging_replay(altered, predictor, width, future.append)
            prefetch.staging_replay(extended, predictor, width, longer.append)
            self.assertEqual(reference[:5], future[:5])
            self.assertEqual(reference, longer[:len(reference)])
            self.assertEqual(reference[3]["predictions"], future[3]["predictions"])
        self.assertEqual(json.dumps(predictor.artifact, sort_keys=True), frozen)

    def test_baseline_audit_precedes_prediction_and_negative_traffic_is_reported(self):
        trace = fixture()
        invalid = replace(trace, routes=(replace(trace.routes[0], pread_bytes=1), *trace.routes[1:]))
        predictor = model()
        with mock.patch.object(predictor, "predict", side_effect=AssertionError("prediction ran")):
            with self.assertRaisesRegex(replay.TraceError, "pread_bytes mismatch"):
                prefetch.compare(invalid, predictor)
        # Entire training-domain requests stay resident; unseen holdout experts
        # force incorrect staging loads and cannot benefit from this predictor.
        trace = audited_trace([request(layer, [200], token)
                               for token in range(1, 4) for layer in range(3)], snapshot(20))
        result = prefetch.compare(trace, predictor)
        for row in result["staging"]:
            self.assertEqual(row["result"], "negative")
            self.assertLess(row["aggregate"]["logical_reads_saved"], 0)
            self.assertEqual(row["aggregate"]["useful_predictions"], 0)

    def test_audited_parser_is_reused_for_input(self):
        trace = fixture(budget=prefetch.CACHE_SLOTS)
        parsed = replay.parse_trace(io.StringIO(encode(trace)))
        self.assertEqual(replay.replay(parsed)["aggregate"]["misses"],
                         replay.replay(trace)["aggregate"]["misses"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.jsonl"
            path.write_text(encode(trace), encoding="utf-8")
            self.assertEqual(len(prefetch.load_inputs([path])[0].routes), 27)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                prefetch.load_inputs([path, path])

    def test_frozen_model_hash_checked_before_holdout_is_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = frozen_files(root)
            with open(root / "model.json.gz", "ab") as target:
                target.write(b"changed")
            with mock.patch.object(prefetch, "load_inputs") as inputs:
                with self.assertRaises(SystemExit) as failure:
                    prefetch.main(["evaluate", str(root / "unread.jsonl"), "--freeze", str(freeze),
                                   "--output", str(root / "evaluation.json")])
                self.assertEqual(failure.exception.code, 2)
                inputs.assert_not_called()

    def test_real_frozen_evaluate_cli_preserves_model_and_reports_each_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = frozen_files(root)
            trace_path = root / "fresh.jsonl.gz"
            with gzip.open(trace_path, "wt", encoding="utf-8") as target:
                target.write(encode(fixture(seed=31, budget=prefetch.CACHE_SLOTS)))
            original = (root / "model.json.gz").read_bytes()
            output = root / "evaluation.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "evaluate", str(trace_path),
                                     "--freeze", str(freeze), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["experiment"], "fresh_frozen_holdout")
            self.assertEqual([row["width"] for row in report["aggregate"]], [1, 2, 4])
            self.assertTrue(report["evaluations"][0]["baseline_validated"])
            self.assertEqual(report["inputs"][0]["sha256"], prefetch.file_hash(trace_path))
            self.assertEqual((root / "model.json.gz").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prefetch.write_model(root / "model.json.gz", model().artifact)


if __name__ == "__main__":
    unittest.main()
