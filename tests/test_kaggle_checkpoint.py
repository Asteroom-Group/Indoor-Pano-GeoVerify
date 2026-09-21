"""Kaggle 場次停機與單一 JSON 進度檔：cache ZIP 只在特徵抽完後才打。"""
import csv
import importlib.util
import json
import os
import unittest
import zipfile

from _support import ROOT, temp_dir  # noqa: F401


def _load(name, rel):
    path = os.path.join(ROOT, *rel.split("/"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_pairs(repo, rows):
    d = os.path.join(repo, "Output", "roma_regions")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "_pairs.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house", "img_a", "img_b"])
        for row in rows:
            w.writerow(row)
    return path


def _touch_cache(repo, house, a, b):
    d = os.path.join(repo, "roma_match", "cache")
    os.makedirs(d, exist_ok=True)
    name = "%s_%s_%s_direct_roma_v2_std0.1_face512.npz" % (house, a, b)
    path = os.path.join(d, name)
    with open(path, "wb") as f:
        f.write(b"npz")
    return path


class TestSessionDeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kc = _load("kaggle_checkpoint_under_test", "environments/kaggle_checkpoint.py")

    def test_when_20_min_remain_of_11_5h_session_it_is_time_to_pack(self):
        start = 1_000_000.0
        deadline = start + 11.5 * 3600
        now = deadline - 20 * 60
        self.assertTrue(self.kc.due_to_pack(deadline, now, reserve_min=20))

    def test_when_an_hour_remains_it_is_not_yet_time_to_pack(self):
        start = 1_000_000.0
        deadline = start + 11.5 * 3600
        now = deadline - 3600
        self.assertFalse(self.kc.due_to_pack(deadline, now, reserve_min=20))

    def test_match_budget_is_session_remainder_minus_zip_reserve(self):
        now = 1_000_000.0
        deadline = now + 3 * 3600
        hours = self.kc.match_budget_hours(deadline, now, reserve_min=20)
        self.assertAlmostEqual(hours, 3.0 - 20 / 60.0, places=5)

    def test_match_budget_is_zero_once_inside_the_zip_reserve(self):
        now = 1_000_000.0
        deadline = now + 10 * 60
        self.assertEqual(self.kc.match_budget_hours(deadline, now, reserve_min=20), 0.0)


class TestProgressJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kc = _load("kaggle_checkpoint_progress", "environments/kaggle_checkpoint.py")

    def test_json_lists_pairs_and_done_keys_from_cache(self):
        with temp_dir() as d:
            repo = os.path.join(d, "repo")
            _write_pairs(repo, [("H", "a", "b"), ("H", "c", "e")])
            _touch_cache(repo, "H", "a", "b")
            payload = self.kc.make_progress(
                repo, last_completed_step="6 批次匹配（填滿快取）",
                next_step="6 批次匹配（填滿快取）", status="session_stop",
                reason="session_deadline", elapsed_h=11.4, deadline_h=11.5)
            self.assertEqual(payload["n_pairs"], 2)
            self.assertEqual(payload["n_done"], 1)
            self.assertEqual(payload["n_remaining"], 1)
            self.assertFalse(payload["features_complete"])
            self.assertEqual(payload["pairs"], [["H", "a", "b"], ["H", "c", "e"]])
            self.assertEqual(payload["done"], [["H", "a", "b"]])
            path = os.path.join(d, "roma_progress.json")
            self.kc.write_progress(path, payload)
            with open(path, encoding="utf-8") as f:
                got = json.load(f)
            self.assertEqual(got["schema_version"], 1)
            self.assertEqual(got["done"], [["H", "a", "b"]])

    def test_previous_done_keys_are_kept_even_without_cache_files(self):
        with temp_dir() as d:
            repo = os.path.join(d, "repo")
            _write_pairs(repo, [("H", "a", "b"), ("H", "c", "e")])
            prev = {"done": [["H", "a", "b"]]}
            payload = self.kc.make_progress(
                repo, last_completed_step="6", next_step="6",
                status="session_stop", reason="session_deadline",
                elapsed_h=1.0, deadline_h=11.5, previous=prev)
            self.assertEqual(payload["done"], [["H", "a", "b"]])
            self.assertEqual(payload["n_done"], 1)

    def test_features_complete_when_every_pair_is_done(self):
        with temp_dir() as d:
            repo = os.path.join(d, "repo")
            _write_pairs(repo, [("H", "a", "b")])
            _touch_cache(repo, "H", "a", "b")
            payload = self.kc.make_progress(
                repo, last_completed_step="6", next_step=None,
                status="complete", reason="finished",
                elapsed_h=4.0, deadline_h=11.5)
            self.assertTrue(payload["features_complete"])
            self.assertEqual(payload["n_remaining"], 0)


class TestResumeFromJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kc = _load("kaggle_checkpoint_resume", "environments/kaggle_checkpoint.py")

    def test_finds_progress_json_nested_under_kaggle_input(self):
        with temp_dir() as d:
            ds = os.path.join(d, "input", "roma-progress")
            os.makedirs(ds)
            with open(os.path.join(ds, "roma_progress.json"), "w", encoding="utf-8") as f:
                json.dump({"schema_version": 1, "pairs": [], "done": []}, f)
            found = self.kc.find_progress_json(os.path.join(d, "input"))
            self.assertTrue(found.endswith("roma_progress.json"))

    def test_uploading_only_the_json_restores_pairs_and_done_keys(self):
        with temp_dir() as d:
            ds = os.path.join(d, "input", "ckpt")
            os.makedirs(ds)
            src = {
                "schema_version": 1,
                "pairs": [["H", "a", "b"], ["H", "c", "e"]],
                "done": [["H", "a", "b"]],
                "n_done": 1,
                "n_pairs": 2,
            }
            with open(os.path.join(ds, "roma_progress.json"), "w", encoding="utf-8") as f:
                json.dump(src, f)
            repo = os.path.join(d, "Roma_Geo")
            found = self.kc.find_progress_json(os.path.join(d, "input"))
            payload = self.kc.restore_progress(repo, found)
            self.assertEqual(payload["done"], [["H", "a", "b"]])
            pairs_path = os.path.join(repo, "Output", "roma_regions", "_pairs.csv")
            self.assertTrue(os.path.isfile(pairs_path))
            with open(pairs_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["img_a"], "a")
            self.assertTrue(os.path.isfile(os.path.join(repo, "roma_progress.json")))
            self.assertFalse(os.path.isdir(os.path.join(repo, "roma_match", "cache")))

    def test_done_keys_skip_a_pair_even_when_cache_file_is_absent(self):
        skip = self.kc.done_key_set({"done": [["H", "imgA", "imgB"]]})
        self.assertTrue(self.kc.is_done_pair("H", "imgA.jpg", "imgB.jpg", skip))
        self.assertTrue(self.kc.is_done_pair("H", "imgB", "imgA", skip))
        self.assertFalse(self.kc.is_done_pair("H", "imgA", "other", skip))


class TestCacheZipOnlyWhenComplete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kc = _load("kaggle_checkpoint_zip", "environments/kaggle_checkpoint.py")

    def test_cache_zip_uses_stored_compression(self):
        with temp_dir() as d:
            repo = os.path.join(d, "repo")
            _touch_cache(repo, "H", "a", "b")
            dest = os.path.join(d, "roma_cache.zip")
            self.assertTrue(self.kc.pack_cache_zip(repo, dest))
            info = zipfile.ZipFile(dest).getinfo("roma_match/cache/"
                "H_a_b_direct_roma_v2_std0.1_face512.npz")
            self.assertEqual(info.compress_type, zipfile.ZIP_STORED)


class TestKaggleRunWiresDeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(ROOT, "environments", "kaggle_run.py")
        spec = importlib.util.spec_from_file_location("kaggle_run_deadline", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_session_deadline_is_11_and_a_half_hours(self):
        self.assertEqual(self.mod.SESSION_DEADLINE_H, 11.5)

    def test_batch_match_command_receives_progress_json(self):
        cmd = self.mod.batch_match_command("python", "auto", 0, 10.2, "/tmp/roma_progress.json")
        self.assertIn("environments/batch_match.py", cmd)
        self.assertIn("--time_budget_h 10.2", cmd)
        self.assertIn("--progress /tmp/roma_progress.json", cmd)

    def test_partial_stop_does_not_zip_cache(self):
        self.assertFalse(self.mod.should_zip_features({"features_complete": False}))
        self.assertTrue(self.mod.should_zip_features({"features_complete": True}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
