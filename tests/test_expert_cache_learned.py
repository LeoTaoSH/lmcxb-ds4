import importlib.util
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/expert_cache_learned.py"
spec = importlib.util.spec_from_file_location("expert_cache_learned", SCRIPT)
learned = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = learned
spec.loader.exec_module(learned)
replay = learned.replay


def snapshot(budget=3, resident=None, hotness=None, token=1):
    resident = resident or {}
    return replay.Snapshot(budget, 100, max(resident.values(), default=0), token, token,
                           resident, hotness or {})


def request(token, ids, layer=0):
    return replay.Route(token, layer, tuple(ids), 0, 0, 0.0, 0.0, 0)


def zero_ranker():
    return learned.Ranker((0.0,) * len(learned.FEATURES), 0.0)


def fixture(seed=17, tokens=48, budget=3, layers=1):
    """Audited current-policy records with independent seeded route requests."""
    initial = snapshot(budget, {(0, 0): 1, (0, 1): 2, (0, 2): 3}, {(0, 0): 6})
    records = [{"event": "snapshot", "version": 1, "budget": budget,
                "expert_bytes": 100, "clock": initial.clock, "token": 1,
                "decay_token": 1, "decay_interval": 16,
                "resident": [[*key, value] for key, value in initial.resident.items()],
                "hotness": [[*key, value] for key, value in initial.hotness.items()]}]
    current = replay.Cache(initial, "current")
    rng = random.Random(seed)
    for token in range(1, tokens + 1):
        for layer in range(layers):
            ids = rng.sample(range(7), 2)
            route = request(token, ids, layer)
            mask, _ = current.resolve(route)
            records.append({"event": "route", "token": token, "layer": layer,
                            "ids": ids, "hit_mask": mask,
                            "pread_bytes": (len(ids) - mask.bit_count()) * 100,
                            "pread_ms": 0.0, "load_ms": 0.0, "clock": current.clock})
    records.append({"event": "end", "routes": tokens * layers})
    return records


def parsed(records, source="synthetic.jsonl"):
    return replay.parse_trace(io.StringIO("".join(json.dumps(row) + "\n" for row in records)), source)


class CausalFeatureTests(unittest.TestCase):
    def test_unknown_snapshot_history_uses_only_known_clock_and_hotness(self):
        cache = learned.CausalCache(snapshot(resident={(0, 0): 1, (0, 1): 2}, hotness={(0, 0): 15}))
        features = cache.features([0])[0]
        self.assertAlmostEqual(features[0], math.log1p(1 / 256))
        self.assertAlmostEqual(features[1], math.log1p(15))
        np.testing.assert_array_equal(features[2:], np.zeros(7))
        self.assertEqual(cache.last_token[0], -1)

    def test_exact_past_frequencies_gaps_and_decay(self):
        cache = learned.CausalCache(snapshot())
        cache.resolve(request(1, [0]))
        cache.resolve(request(5, [0]))
        cache.resolve(request(13, [0]))
        cache._age(17)
        features = cache.features([0])[0]
        freq4 = ((1 * 2 ** -1 + 1) * 2 ** -2 + 1) * 2 ** -1
        freq64 = ((1 * 2 ** (-4 / 64) + 1) * 2 ** (-8 / 64) + 1) * 2 ** (-4 / 64)
        self.assertAlmostEqual(features[1], math.log1p(1))  # integer 3 >> 1
        self.assertAlmostEqual(features[2], math.log1p(freq4))
        self.assertAlmostEqual(features[3], math.log1p(freq64))
        self.assertAlmostEqual(features[4], math.log1p(4))
        self.assertAlmostEqual(features[5], math.log1p(8))
        self.assertAlmostEqual(features[6], math.log1p(5))
        np.testing.assert_array_equal(features[7:], [1, 1])

    def test_history_survives_eviction_without_admission_shortcut(self):
        cache = learned.CausalCache(snapshot(1), zero_ranker())
        cache.resolve(request(1, [0]))
        cache.resolve(request(2, [1]))
        self.assertNotIn((0, 0), cache.resident)
        self.assertEqual(cache.seen[0], 1)
        mask, victims = cache.resolve(request(6, [0]))
        self.assertEqual((mask, victims), (0, ((0, 1),)))
        self.assertEqual(cache.seen[0], 2)
        self.assertEqual(cache.last_gap[0], 5)
        self.assertEqual(cache.mean_gap[0], 5)
        self.assertAlmostEqual(cache.freq4[0], 1 + 2 ** (-5 / 4))

    def test_equal_prefixes_produce_identical_features_and_decisions(self):
        # Different future suffixes are never supplied to either cache.
        first = learned.CausalCache(snapshot(), learned.Ranker(tuple(range(9)), -3))
        second = learned.CausalCache(snapshot(), learned.Ranker(tuple(range(9)), -3))
        prefixes = [request(1, [0, 1]), request(2, [2, 0]), request(3, [3, 1])]
        for route in prefixes:
            self.assertEqual(first.resolve(route), second.resolve(route))
            np.testing.assert_array_equal(first.features(np.arange(7)), second.features(np.arange(7)))
            self.assertEqual(first.resident, second.resident)

    def test_learned_evaluator_does_not_request_future_labels(self):
        trace = parsed(fixture())
        with mock.patch.object(replay, "future_uses", side_effect=AssertionError("future leakage")):
            result = learned.learned_replay(trace, zero_ranker())
        self.assertEqual(result["aggregate"]["lookups"], 96)

    def test_training_future_suffix_changes_labels_not_features(self):
        settings = learned.Settings(horizon_tokens=2, sample_every=1, candidates_per_sample=1)
        initial = snapshot(2, {(0, 0): 1, (0, 1): 2})
        first = replay.Trace(initial, (request(1, [2]), request(2, [0]), request(3, [2])), "first")
        second = replay.Trace(initial, (request(1, [2]), request(2, [5]), request(3, [2])), "second")
        # Sample key 0 at the first decision in both traces, using a fixed RNG.
        with mock.patch.object(random.Random, "sample", side_effect=lambda values, size: values[:size]):
            x1, y1, _ = learned.collect_training(first, settings)
            x2, y2, _ = learned.collect_training(second, settings)
        np.testing.assert_array_equal(x1[0], x2[0])
        self.assertEqual((y1[0], y2[0]), (1, 0))


class EvictionTests(unittest.TestCase):
    def test_full_set_protection_before_any_miss_admission(self):
        initial = snapshot(3, {(0, 0): 1, (0, 1): 2, (0, 2): 3})
        cache = learned.CausalCache(initial, zero_ranker())
        mask, victims = cache.resolve(request(1, [3, 0, 4]))
        self.assertEqual(mask, 0b010)
        self.assertEqual(victims, ((0, 1), (0, 2)))
        self.assertEqual(cache.resident, {(0, 0): 4, (0, 3): 5, (0, 4): 6})
        self.assertEqual(cache.resolve(request(2, [4, 3, 0])), (0b111, ()))
        self.assertEqual(len(cache.resident), cache.budget)

    def test_zero_ranker_is_lru_tie_with_stale_metadata_ignored(self):
        cache = learned.CausalCache(snapshot(2, {(0, 2): 1, (0, 0): 2}), zero_ranker())
        self.assertEqual(cache.resolve(request(1, [3]))[1], ((0, 2),))
        cache.resolve(request(2, [0]))
        self.assertEqual(cache.resolve(request(3, [2]))[1], ((0, 3),))
        np.testing.assert_array_equal(np.flatnonzero(cache.present), [0, 2])

    def test_higher_score_survives_and_intercept_is_irrelevant(self):
        weights = [0.0] * 9
        weights[1] = 1
        initial = snapshot(2, {(0, 0): 1, (0, 1): 2}, {(0, 0): 100, (0, 1): 1})
        first = learned.CausalCache(initial, learned.Ranker(tuple(weights), -1000))
        second = learned.CausalCache(initial, learned.Ranker(tuple(weights), 1000))
        self.assertEqual(first.resolve(request(1, [2])), second.resolve(request(1, [2])))
        self.assertEqual(set(first.resident), {(0, 0), (0, 2)})

    def test_teacher_reuses_current_audit_at_all_decay_boundaries(self):
        trace = parsed(fixture(tokens=80, layers=3, budget=8))
        teacher = learned.CausalCache(trace.snapshot)
        current = replay.Cache(trace.snapshot, "current")
        for route in trace.routes:
            self.assertEqual(teacher.resolve(route), current.resolve(route))
            self.assertEqual(teacher.resident, current.resident)
            self.assertEqual(teacher.hotness, current.hotness)
            self.assertEqual(teacher.clock, route.clock)
            self.assertEqual(set(teacher.resident), {teacher.key(index) for index in np.flatnonzero(teacher.present)})

    def test_scan_matches_independent_scalar_selection_and_exact_budget(self):
        rng = random.Random(411)
        weights = tuple(rng.uniform(-1, 1) for _ in range(9))
        ranker = learned.Ranker(weights, 0.13)
        cache = learned.CausalCache(snapshot(7), ranker)
        original = cache._evict

        def independently_checked(count, protected):
            entries = []
            for key in cache.resident:
                if key not in protected:
                    row = cache.features([cache.index(key)])[0]
                    score = 0.13 + sum(weight * value for weight, value in zip(weights, row))
                    entries.append((score, cache.resident[key], key))
            # Scalar summation can differ at the last bit; random weights avoid ties.
            expected = tuple(entry[2] for entry in sorted(entries)[:count])
            actual = original(count, protected)
            self.assertEqual(actual, expected)
            self.assertFalse(set(actual) & protected)
            return actual

        cache._evict = independently_checked
        for token in range(1, 90):
            for layer in range(3):
                ids = rng.sample(range(10), 3)
                cache.resolve(request(token, ids, layer))
                self.assertTrue(set((layer, expert) for expert in ids) <= cache.resident.keys())
                self.assertLessEqual(len(cache.resident), cache.budget)
                self.assertEqual(cache.present.sum(), len(cache.resident))


class TrainingTests(unittest.TestCase):
    def test_sampling_and_fitting_are_deterministic(self):
        trace = parsed(fixture(tokens=64))
        settings = learned.Settings(horizon_tokens=4, sample_every=2, candidates_per_sample=2)
        x1, y1, _ = learned.collect_training(trace, settings)
        x2, y2, _ = learned.collect_training(trace, settings)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        model1, stats1 = learned.fit_ranker(x1, y1, settings)
        model2, stats2 = learned.fit_ranker(x2, y2, settings)
        self.assertEqual(model1, model2)
        self.assertEqual(stats1["mean_regularized_log_loss"], stats2["mean_regularized_log_loss"])
        self.assertGreater(len(y1), 0)
        self.assertTrue(np.isfinite(model1.score(x1)).all())

    def test_right_censored_tail_has_no_negative_training_labels(self):
        initial = snapshot(2, {(0, 0): 1, (0, 1): 2})
        trace = replay.Trace(initial, tuple(request(token, [token + 1]) for token in range(1, 6)), "tail")
        x, y, stats = learned.collect_training(trace, learned.Settings(horizon_tokens=3, sample_every=1))
        self.assertEqual(stats["sampled_decisions"], 2)
        self.assertEqual(stats["censored_decisions"], 3)
        self.assertEqual(len(x), 4)
        np.testing.assert_array_equal(y, np.zeros(4))

    def test_linear_fit_learns_monotone_retention_and_unscales_features(self):
        x = np.zeros((80, 9))
        x[:, 0] = np.linspace(-10, 10, 80)
        y = (x[:, 0] > 0).astype(float)
        ranker, stats = learned.fit_ranker(x, y, learned.Settings())
        self.assertGreater(ranker.weights[0], 0)
        self.assertLess(ranker.score(x[:1])[0], ranker.score(x[-1:])[0])
        self.assertLess(stats["mean_regularized_log_loss"], math.log(2))
        self.assertTrue(all(math.isfinite(value) for value in ranker.weights))

    def test_invalid_data_settings_and_rankers_are_rejected(self):
        for kwargs in ({"sample_every": 0}, {"l2": float("nan")}, {"horizon_tokens": -1}):
            with self.assertRaises(ValueError):
                learned.Settings(**kwargs).validate()
        with self.assertRaises(ValueError):
            learned.fit_ranker(np.zeros((0, 9)), np.zeros(0), learned.Settings())
        with self.assertRaises(ValueError):
            learned.fit_ranker(np.zeros((2, 9)), np.array([0.0, 0.2]), learned.Settings())
        with self.assertRaises(ValueError):
            learned.Ranker((0.0,) * 8, 0.0)
        with self.assertRaises(ValueError):
            learned.Ranker((0.0,) * 9, float("inf"))


class ArtifactAndCliTests(unittest.TestCase):
    def write_fixture(self, root, name, seed):
        path = root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in fixture(seed)), encoding="utf-8")
        return path

    def test_leave_one_out_freeze_and_real_evaluate_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.write_fixture(root, "first.jsonl", 7)
            second = self.write_fixture(root, "second.jsonl", 9)
            model, report = root / "model.json", root / "report.json"
            command = [sys.executable, str(SCRIPT), "train", str(first), str(second),
                       "--model", str(model), "--output", str(report),
                       "--horizon-tokens", "4", "--sample-every", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifact = json.loads(model.read_text())
            validation = json.loads(report.read_text())
            self.assertEqual(len(artifact["weights"]), 9)
            self.assertEqual(artifact["features"], list(learned.FEATURES))
            self.assertEqual(validation["model_sha256"], learned.file_hash(model))
            self.assertEqual(len(validation["folds"]), 2)
            for fold in validation["folds"]:
                self.assertNotIn(fold["validation_input"]["sha256"],
                                 [item["sha256"] for item in fold["training_inputs"]])
                self.assertTrue(fold["evaluation"]["baseline_validated"])
                self.assertEqual({policy["policy"] for policy in fold["evaluation"]["policies"]},
                                 {*replay.POLICIES, "learned_logistic"})
            original_bytes = model.read_bytes()
            fresh = self.write_fixture(root, "fresh.jsonl", 24)
            evaluation = root / "evaluation.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "evaluate", str(fresh),
                                     "--model", str(model), "--output", str(evaluation)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(model.read_bytes(), original_bytes)
            self.assertEqual(json.loads(evaluation.read_text())["experiment"], "frozen_evaluation")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(model.read_bytes(), original_bytes)

    def test_trace_audit_failure_prevents_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.write_fixture(root, "first.jsonl", 7)
            records = fixture(9)
            records[1]["pread_bytes"] += 100
            invalid = root / "invalid.jsonl"
            invalid.write_text("".join(json.dumps(row) + "\n" for row in records))
            model = root / "model.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "train", str(first), str(invalid),
                                     "--model", str(model), "--output", str(root / "report.json")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pread_bytes mismatch", result.stderr)
            self.assertFalse(model.exists())

    def test_model_feature_drift_and_nonfinite_weights_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.json"
            artifact = learned.freeze_model(zero_ranker(), learned.Settings(), [], {})
            artifact["features"][0] = {"name": "wrong", "definition": "future"}
            learned.write_json(model, artifact)
            with self.assertRaisesRegex(ValueError, "feature definitions"):
                learned.load_model(model)
            artifact = learned.freeze_model(zero_ranker(), learned.Settings(), [], {})
            artifact["weights"][0] = "not a coefficient"
            learned.write_json(model, artifact)
            with self.assertRaisesRegex(ValueError, "finite numbers"):
                learned.load_model(model)

    def test_duplicate_paths_and_output_collision_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = self.write_fixture(root, "trace.jsonl", 7)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                learned.load_inputs([trace, trace])
            result = subprocess.run([sys.executable, str(SCRIPT), "train", str(trace),
                                     "--model", str(trace), "--output", str(root / "report.json")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("distinct paths", result.stderr)


if __name__ == "__main__":
    unittest.main()
