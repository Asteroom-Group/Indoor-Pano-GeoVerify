"""批次匹配的時間預算：平行路徑必須在開下一批之前停，不能整批超時後才醒。"""
import importlib.util
import os
import unittest

from _support import ROOT  # noqa: F401


def _load_batch_match():
    path = os.path.join(ROOT, "environments", "batch_match.py")
    spec = importlib.util.spec_from_file_location("batch_match_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestParallelChunkFitsRemainingBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_batch_match()

    def test_with_plenty_of_time_uses_the_nominal_chunk(self):
        n = self.mod.next_chunk_len(
            n_workers=2, n_left=5000, remaining_s=8 * 3600, sec_per_pair=12.0)
        self.assertEqual(n, 400)

    def test_with_five_minutes_left_does_not_start_a_400_pair_chunk(self):
        n = self.mod.next_chunk_len(
            n_workers=2, n_left=5000, remaining_s=5 * 60, sec_per_pair=12.0)
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 50)

    def test_when_budget_is_already_spent_returns_zero(self):
        n = self.mod.next_chunk_len(
            n_workers=2, n_left=5000, remaining_s=0, sec_per_pair=12.0)
        self.assertEqual(n, 0)

    def test_first_chunk_uses_a_conservative_prior_so_it_cannot_overshoot(self):
        n = self.mod.next_chunk_len(
            n_workers=2, n_left=5000, remaining_s=10 * 60, sec_per_pair=None)
        self.assertLess(n, 400)
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
