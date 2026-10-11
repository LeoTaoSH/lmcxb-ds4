import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/expert_activation_prefetch.py"
spec = importlib.util.spec_from_file_location("expert_activation_prefetch", SCRIPT)
activation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activation)
replay = activation.replay
prefetch = activation.prefetch


def make_capture(seed=11, tokens=24, layers=3, budget=8):
    rng = np.random.default_rng(seed)
    records = np.zeros(tokens * layers, dtype=activation.RECORD_DTYPE)
    initial = replay.Snapshot(budget, 100, 0, 1, 1, {}, {})
    cache = replay.Cache(initial, "current")
    requests = []
    # Features from the current event determine the target two events later.
    categories = rng.integers(0, 4, len(records))
    records["x"][:, :4] = np.eye(4)[categories]
    records["x"][:, 4:8] = rng.normal(size=(len(records), 4)) * 0.001
    for index in range(len(records)):
        token, layer = index // layers + 1, index % layers + 1
        category = int(categories[max(0, index - prefetch.LOOKAHEAD)])
        ids = (category, category + 4)
        route = replay.Route(token, layer, ids, 0, 0, 0.0, 0.0, 0)
        mask, _ = cache.resolve(route)
        requests.append(replace(route, hit_mask=mask, clock=cache.clock,
                                pread_bytes=(len(ids) - mask.bit_count()) * 100))
        records[index]["index"] = index
        records[index]["token"] = token
        records[index]["layer"] = layer
        records[index]["count"] = len(ids)
        records[index]["ids"][:2] = ids
    trace = replay.Trace(initial, tuple(requests), "synthetic.jsonl")
    return activation.Capture("synthetic", trace, Path("synthetic.bin"), records, 1234)


def real_shape_capture():
    capture = make_capture(tokens=4)
    cache = replay.Cache(capture.trace.snapshot, "current")
    routes = []
    for index, route in enumerate(capture.trace.routes):
        ids = tuple(range(8))
        route = replace(route, ids=ids)
        mask, _ = cache.resolve(route)
        routes.append(replace(route, hit_mask=mask, clock=cache.clock,
                              pread_bytes=(8 - mask.bit_count()) * 100))
        capture.records[index]["count"] = 8
        capture.records[index]["ids"] = ids
    capture.trace = replace(capture.trace, routes=tuple(routes))
    return capture


def encode_trace(trace):
    initial = trace.snapshot
    records = [{"event": "snapshot", "version": 1, "budget": initial.budget,
                "expert_bytes": initial.expert_bytes, "clock": initial.clock,
                "token": initial.token, "decay_token": initial.decay_token,
                "decay_interval": replay.DECAY_INTERVAL, "resident": [], "hotness": []}]
    records += [{"event": "route", "token": route.token, "layer": route.layer,
                 "ids": list(route.ids), "hit_mask": route.hit_mask,
                 "pread_bytes": route.pread_bytes, "pread_ms": 0.0,
                 "load_ms": 0.0, "clock": route.clock} for route in trace.routes]
    records.append({"event": "end", "routes": len(trace.routes)})
    return "".join(json.dumps(record) + "\n" for record in records)


def encode_sidecar(capture):
    return (activation.HEADER.pack(b"DS4ACT01", 1, 0x01020304, activation.DIM,
                                   256, 8, activation.RECORD_BYTES, capture.model_bytes) +
            capture.records.tobytes() + activation.FOOTER.pack(b"DS4END01", len(capture.records)))


class SidecarTests(unittest.TestCase):
    def load(self, data, trace=None):
        capture = real_shape_capture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.jsonl").write_text(encode_trace(trace or capture.trace))
            (root / "features.bin").write_bytes(data)
            loaded = activation.Capture.load("test", root / "trace.jsonl", root / "features.bin")
            return loaded.records.copy()

    def test_fixed_record_roundtrip_exact_f32_and_route_identity(self):
        capture = real_shape_capture()
        loaded = self.load(encode_sidecar(capture))

        np.testing.assert_array_equal(loaded, capture.records)

    def test_header_version_endianness_shape_and_footer_are_required(self):
        original = encode_sidecar(real_shape_capture())
        mutations = [b"BADMAGIC" + original[8:], original[:8] + struct.pack("<I", 2) + original[12:],
                     original[:12] + struct.pack("<I", 0x04030201) + original[16:],
                     original[:16] + struct.pack("<I", 4095) + original[20:],
                     original[:-16], original[:-8] + struct.pack("<Q", 13),
                     original + b"extra"]
        for data in mutations:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                self.load(data)

    def test_every_route_metadata_field_must_match(self):
        for field in ("index", "token", "layer", "count", "ids"):
            capture = real_shape_capture()
            capture.records[3][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(encode_sidecar(capture))

    def test_nonfinite_vectors_and_unauditable_baseline_fail(self):
        capture = real_shape_capture()
        capture.records["x"][4, 12] = np.nan
        with self.assertRaises(ValueError):
            self.load(encode_sidecar(capture))
        capture = real_shape_capture()
        routes = list(capture.trace.routes)
        routes[3] = replace(routes[3], pread_bytes=routes[3].pread_bytes + 1)
        with self.assertRaises(replay.TraceError):
            self.load(encode_sidecar(capture), replace(capture.trace, routes=tuple(routes)))


class TrainingAndCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.captures = [make_capture(seed=value, tokens=30) for value in (11, 12, 13)]
        cls.arrays, cls.costs = activation.fit(cls.captures, feature_count=8, candidates=(0.01, 0.1))
        cls.predictor = activation.Predictor(cls.arrays)

    def test_ridge_head_learns_current_activation_to_future_real_routes(self):
        capture = make_capture(seed=33, tokens=20)
        predictor = self.predictor.bind(capture)
        correct = 0
        for index, source in enumerate(capture.trace.routes[:-prefetch.LOOKAHEAD]):
            target = capture.trace.routes[index + prefetch.LOOKAHEAD]
            predictions = predictor.predict(source, 2)
            self.assertEqual({layer for layer, _ in predictions}, {target.layer})
            correct += len({expert for _, expert in predictions} & set(target.ids))
        self.assertGreater(correct / (2 * (len(capture.trace.routes) - 2)), 0.90)
        self.assertIn(self.costs["selected_ridge"], (0.01, 0.1))
        self.assertEqual(self.costs["training_pairs"], 3 * (30 * 3 - 2))

    def test_prediction_cannot_read_future_activation_or_routes(self):
        capture = make_capture(seed=33)
        bound = self.predictor.bind(capture)
        source = capture.trace.routes[2]
        before = bound.predict(source, 4)
        capture.records["x"][3:] = 100000
        routes = list(capture.trace.routes)
        routes[4] = replace(routes[4], ids=(200, 201))
        capture.trace = replace(capture.trace, routes=tuple(routes))
        self.assertEqual(bound.predict(source, 4), before)
        other = make_capture(seed=33)
        other.records["x"][2, :4] = np.roll(other.records["x"][2, :4], 1)
        self.assertNotEqual(self.predictor.bind(other).predict(source, 4), before)

    def test_training_pairs_do_not_cross_prompt_or_validation_boundary(self):
        first, second = make_capture(tokens=10), make_capture(seed=7, tokens=10)
        for layer in (1, 2, 3):
            x, ids, labels = activation.layer_samples([first, second], layer)
            expected = sum(route.layer == layer for route in first.trace.routes[:-2]) * 2
            self.assertEqual(len(x), expected)
            prefix = activation.layer_samples([first], layer, "prefix")
            suffix = activation.layer_samples([first], layer, "suffix")
            indices = [i for i, route in enumerate(first.trace.routes[:-2]) if route.layer == layer]
            self.assertEqual(len(prefix[0]), sum(first.trace.routes[i + 2].token < 9 for i in indices))
            self.assertEqual(len(suffix[0]), sum(first.trace.routes[i].token >= 9 for i in indices))
            self.assertEqual(labels.shape, (expected, 256))

    def test_variance_selection_and_normalization_use_only_supplied_training_data(self):
        x = np.zeros((4, activation.DIM))
        x[:, 9] = (0, 1, 2, 3)
        x[:, 4] = (0, 1, 2, 3)
        indices, mean, scale = activation.feature_stats(x, 2)
        np.testing.assert_array_equal(indices, (4, 9))
        np.testing.assert_allclose(mean, (1.5, 1.5))
        normalized = activation.design(x, [(0, 3)] * 4, indices, mean, scale)
        np.testing.assert_allclose(normalized[:, :2].mean(axis=0), 0)
        np.testing.assert_allclose(normalized[:, :2].std(axis=0), 1)
        self.assertEqual(normalized[0, 2], 1)
        self.assertEqual(normalized[0, 5], 1)

    def test_staging_reuses_full_accounting_and_never_changes_demand_clock(self):
        capture = make_capture(seed=33, tokens=10, budget=8)
        events = []
        result = prefetch.staging_replay(capture.trace, self.predictor.bind(capture), 4, events.append)
        self.assertLessEqual(result["peak_slots"], 8)
        self.assertEqual(result["final_clock"], sum(len(route.ids) for route in capture.trace.routes))
        row = result["aggregate"]
        self.assertEqual(row["total_logical_read_bytes"], row["total_logical_reads"] * 100)
        self.assertEqual(row["prefetch_loads"], row["useful_prefetch_loads"] + row["wasted_prefetch_loads"])
        for event, route in zip(events, capture.trace.routes):
            self.assertFalse(event["resident"].keys() & event["staged"].keys())
            self.assertLessEqual(len(event["resident"]) + len(event["staged"]), 8)
            self.assertTrue(set(route.keys).issubset(event["resident"]))

    def test_invalid_model_shapes_or_domains_fail(self):
        for key in ("scale_1", "weights_1", "indices_1"):
            arrays = copy.deepcopy(self.arrays)
            if key == "scale_1":
                arrays[key][0] = 0
            elif key == "weights_1":
                arrays[key] = arrays[key][:-1]
            else:
                arrays[key][0] = activation.DIM
            with self.subTest(key=key), self.assertRaises(ValueError):
                activation.Predictor(arrays)
        with self.assertRaises(ValueError):
            self.predictor.predict_current(self.captures[0].trace.routes[0], np.zeros(4), 1)

    def test_stable_expert_id_ties(self):
        self.assertEqual(tuple(activation.top_ids(np.zeros(256), 4)), (0, 1, 2, 3))


class FreezeTests(unittest.TestCase):
    def test_freeze_rejects_source_or_model_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = {"version": 1, "settings": activation.SETTINGS,
                      "predictor_script_sha256": "changed"}
            (root / "freeze.json").write_text(json.dumps(freeze))
            with self.assertRaises(ValueError):
                activation.load_frozen(root / "freeze.json")
            freeze.update({"predictor_script_sha256": prefetch.file_hash(SCRIPT),
                           "staging_script_sha256": prefetch.file_hash(prefetch.__file__),
                           "baseline_script_sha256": prefetch.file_hash(replay.__file__),
                           "model_sha256": "changed"})
            (root / "model.npz").write_bytes(b"altered")
            (root / "freeze.json").write_text(json.dumps(freeze))
            with self.assertRaises(ValueError):
                activation.load_frozen(root / "freeze.json")

    def test_original_prompt_hash_mismatch_rejects_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = activation.capture_commands(root)
            (root / "capture-commands.json").write_text(json.dumps(manifest))
            initial = {"model": activation.MODEL_NAME, "source_revision": activation.REVISION,
                       "cases": [{"workload": name, "prompt_sha256": "not-the-original"}
                                 for name in activation.TRAIN_NAMES]}
            real_load = json.load
            def load(source):
                if str(source.name).endswith("20261010/manifest.json"):
                    return initial
                return real_load(source)
            with mock.patch.object(activation.json, "load", side_effect=load):
                with self.assertRaises(ValueError):
                    activation.load_captures(root, "train")



if __name__ == "__main__":
    unittest.main()
