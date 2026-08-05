"""
RomaScoreCache — compact, sparse, edge-keyed store of aggregated RoMa match scores.

Purpose: run_roma_above_threshold.py runs RoMa matching only for the subset of pairs whose
coarse-filter score is above the per-fold LOO threshold, then discards the in-memory matches
(only the raw per-pair .npz correspondence cache in roma_match/cache/ survives). This cache
captures a small, aggregated, immediately-usable score per matched pair so downstream tools
(stitch_tool) can rank pairs by "how much RoMa evidence backs this connection" without re-running
RoMa or re-loading every raw correspondence file.

Design mirrors coarse_filter_experiment/block_match_v2/models/history/roma_evidence_extractor.py's
RomaEvidenceCache (same proven shape: unordered-pair key, flat parallel-array npz, zero-cost
"not verified" lookups) but is intentionally NOT that module - this one is scoped to a single
scalar-ish score record for ranking, not a fixed-order feature vector for the R1 refiner's MLP,
and it has no dependency on the (deprecated, broken-import) models/history/ tree.

Key: (house, min(imgA_basename, imgB_basename), max(imgA_basename, imgB_basename)) - undirected,
since RoMa match_direct(A, B) and match_direct(B, A) share the same underlying cache file (see
RoMaMatcher._get_cache_info's is_swapped handling) and describe one physical correspondence set.
"""
import os
import numpy as np


class RomaScoreCache:
    """Accumulable, npz-backed cache of per-pair aggregated RoMa scores."""

    def __init__(self):
        self._keys = {}   # (house, a, b) -> row index
        self._match_count = []   # list[int]
        self._conf_sum = []      # list[float]
        self._conf_mean = []     # list[float]

    @staticmethod
    def _basename(path_or_name):
        return path_or_name.replace("\\", "/").rsplit("/", 1)[-1]

    @staticmethod
    def make_key(house, img_a, img_b):
        a = RomaScoreCache._basename(img_a)
        b = RomaScoreCache._basename(img_b)
        lo, hi = (a, b) if a <= b else (b, a)
        return (house, lo, hi)

    def has(self, house, img_a, img_b):
        return self.make_key(house, img_a, img_b) in self._keys

    def put(self, house, img_a, img_b, match_count, conf_sum):
        """Stores/overwrites one pair's aggregated score. conf_mean is derived, not stored twice."""
        key = self.make_key(house, img_a, img_b)
        conf_mean = float(conf_sum) / match_count if match_count > 0 else 0.0
        if key in self._keys:
            idx = self._keys[key]
            self._match_count[idx] = int(match_count)
            self._conf_sum[idx] = float(conf_sum)
            self._conf_mean[idx] = conf_mean
        else:
            self._keys[key] = len(self._match_count)
            self._match_count.append(int(match_count))
            self._conf_sum.append(float(conf_sum))
            self._conf_mean.append(conf_mean)

    def get(self, house, img_a, img_b):
        """Returns {'match_count', 'conf_sum', 'conf_mean'} or None if this pair was never scored."""
        idx = self._keys.get(self.make_key(house, img_a, img_b))
        if idx is None:
            return None
        return {
            "match_count": self._match_count[idx],
            "conf_sum": self._conf_sum[idx],
            "conf_mean": self._conf_mean[idx],
        }

    def num_edges(self):
        return len(self._match_count)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        houses = np.array([k[0] for k in self._keys], dtype=object)
        a_names = np.array([k[1] for k in self._keys], dtype=object)
        b_names = np.array([k[2] for k in self._keys], dtype=object)
        rows = np.array([self._keys[k] for k in self._keys], dtype=np.int64)
        order = np.argsort(rows)
        np.savez_compressed(
            path,
            houses=houses[order], a_names=a_names[order], b_names=b_names[order],
            match_count=np.asarray(self._match_count, dtype=np.int32)[rows[order]],
            conf_sum=np.asarray(self._conf_sum, dtype=np.float32)[rows[order]],
            conf_mean=np.asarray(self._conf_mean, dtype=np.float32)[rows[order]],
        )

    @classmethod
    def load(cls, path):
        cache = cls()
        if not os.path.exists(path):
            return cache
        d = np.load(path, allow_pickle=True)
        houses, a_names, b_names = d["houses"], d["a_names"], d["b_names"]
        match_count, conf_sum, conf_mean = d["match_count"], d["conf_sum"], d["conf_mean"]
        for idx in range(len(houses)):
            key = (str(houses[idx]), str(a_names[idx]), str(b_names[idx]))
            cache._keys[key] = len(cache._match_count)
            cache._match_count.append(int(match_count[idx]))
            cache._conf_sum.append(float(conf_sum[idx]))
            cache._conf_mean.append(float(conf_mean[idx]))
        return cache
