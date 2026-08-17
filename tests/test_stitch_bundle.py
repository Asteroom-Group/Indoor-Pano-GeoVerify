"""
工具載入對外交付檔的相容層測試。

`export_public.py` 的交付檔可以直接放進產出包給工具載入——工具渲染得出來，即為
「交付內容足夠」的直接檢驗。要成立有三個前提，本檔逐一驗證：

  1. 包內的 `<house>.json` 是交付檔本身，未經改寫（改寫過就不算檢驗了）。
  2. 伺服器讀取時把幾何證據欄位對映回工具沿用的內部名，並標明無 GT 真值。
  3. 每對內嵌的共視區域能經 `/api/regions` 供應，且側別（swap）判定正確。

第 3 點最需要測：交付檔的區域側別已正規化為「a 側 = 該對的 i 圖」，查詢時哪張圖當
上半部才決定 swap。判反了的症狀是遮罩上下顛倒而畫面仍然合理，不會有錯誤訊息。
"""
import json
import os
import sys
import unittest

from _support import ROOT, quiet, temp_dir, write_json  # noqa: F401（匯入時補 sys.path）

# stitch_tool 內部彼此以同層模組名互相匯入（house_dirs 等），需補進 sys.path。
_TOOL = os.path.join(ROOT, "stitch_tool")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

import pack_bundle                                   # noqa: E402
import server                                        # noqa: E402（匯入會建立 state/，見模組說明）

HOUSE = "DollhouseTask_00000"
IMG = ["aaa.jpg", "bbb.jpg", "ccc.jpg"]

REGION = {"n": 120, "frac_of_inliers": 0.9,
          "a": {"polygon": [[1.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
                "area_frac": 0.11, "az_range": [0.0, 30.0]},
          "b": {"polygon": [[5.0, 6.0], [7.0, 8.0], [5.0, 8.0]],
                "area_frac": 0.22, "az_range": [10.0, 50.0]}}


def public_house():
    """一份對外交付檔（欄位取自 export_public.py 的實際輸出）。"""
    return {
        "house": HOUSE, "n_images": 3, "width": 8192, "height": 4096,
        "images": list(IMG),
        "az_convention": "az(x)=((x/W)*360-180)%360 ; 0deg = image center",
        "doors": [],
        "pairs": [
            {"i": 0, "j": 1, "prob": 0.95, "az_ij": 116.39, "az_ji": 209.37,
             "az_source": "geo", "has_geometry": True, "n_inlier": 1508,
             "inlier_rate": 0.7264, "conf_used": 0.65, "regions": [REGION]},
            {"i": 0, "j": 2, "prob": 0.23, "az_ij": 12.0, "az_ji": 190.0,
             "az_source": "coarse", "has_geometry": False},
        ],
    }


def internal_house():
    """對照組：流程內部的房子資料（工具原本讀的那一種）。"""
    return {
        "house": HOUSE, "n_images": 3, "width": 8192, "height": 4096,
        "images": list(IMG), "doors": [], "n_gt_connected_pairs": 1,
        "pairs": [{"i": 0, "j": 1, "logit": 2.89, "prob": 0.95,
                   "az_ij": 116.39, "az_ji": 209.37, "az_source": "geo",
                   "gt_connected": True, "geo_posed": 1, "geo_n_inlier": 1508}],
    }


class TestIsPublicHouse(unittest.TestCase):
    def test_detects_public_file(self):
        self.assertTrue(server.is_public_house(public_house()))

    def test_rejects_internal_file(self):
        self.assertFalse(server.is_public_house(internal_house()))

    def test_no_pairs_is_not_public(self):
        self.assertFalse(server.is_public_house({"house": HOUSE, "pairs": []}))
        self.assertFalse(server.is_public_house({}))


class TestPublicHouseToTool(unittest.TestCase):
    def setUp(self):
        self.src = public_house()
        self.out = server.public_house_to_tool(self.src)

    def test_geometry_fields_are_renamed(self):
        p = self.out["pairs"][0]
        self.assertEqual(p["geo_n_inlier"], 1508)
        self.assertEqual(p["geo_inlier_rate"], 0.7264)
        self.assertEqual(p["geo_conf_used"], 0.65)
        self.assertEqual(p["geo_posed"], 1)
        for k in ("n_inlier", "inlier_rate", "conf_used", "has_geometry"):
            self.assertNotIn(k, p)

    def test_mapping_covers_every_public_geometry_field(self):
        """對映表須涵蓋交付檔實際會出現的欄位，漏一個就會在工具上默默消失。"""
        import export_public
        renamed = set(server.PUBLIC_TO_TOOL)
        geo_fields = {v for k, v in export_public.PAIR_FIELDS.items() if k.startswith("geo_")}
        self.assertEqual(geo_fields, renamed)

    def test_has_geometry_becomes_integer_flag(self):
        """前端把 geo_posed 當 0/1 用（排序與徽章），布林要轉成整數。"""
        vals = [p.get("geo_posed") for p in self.out["pairs"]]
        self.assertEqual(vals, [1, 0])
        for v in vals:
            self.assertIsInstance(v, int)

    def test_passthrough_fields_are_untouched(self):
        p = self.out["pairs"][0]
        for k in ("i", "j", "prob", "az_ij", "az_ji", "az_source"):
            self.assertEqual(p[k], self.src["pairs"][0][k])

    def test_marks_absence_of_ground_truth(self):
        """交付檔不含真值；工具據 has_gt=false 顯示「無 GT 真值」而非誤以為疊層失效。"""
        self.assertIs(self.out["has_gt"], False)
        self.assertEqual(self.out["n_gt_connected_pairs"], 0)

    def test_regions_are_moved_out_of_house_payload(self):
        """區域改由 /api/regions 供應，維持前端既有取用方式，也避免一次回傳全部多邊形。"""
        for p in self.out["pairs"]:
            self.assertNotIn("regions", p)

    def test_input_is_not_mutated(self):
        self.assertIn("regions", self.src["pairs"][0])
        self.assertIn("has_geometry", self.src["pairs"][0])
        self.assertNotIn("has_gt", self.src)

    def test_house_level_fields_survive(self):
        for k in ("house", "n_images", "width", "height", "images", "az_convention"):
            self.assertEqual(self.out[k], self.src[k])


class TestEmbeddedRegions(unittest.TestCase):
    """`/api/regions` 從交付檔取區域，含側別判定。"""

    def setUp(self):
        self.d = public_house()

    def test_forward_query_does_not_swap(self):
        r = server.embedded_regions(self.d, "aaa", "bbb")
        self.assertIsNotNone(r)
        self.assertFalse(r["swap"])
        self.assertEqual(r["regions"], [REGION])

    def test_reverse_query_sets_swap(self):
        """查詢把 j 圖當上半部時必須回 swap=true，前端據此對調兩側。"""
        r = server.embedded_regions(self.d, "bbb", "aaa")
        self.assertTrue(r["swap"])
        self.assertEqual(r["regions"], [REGION], "regions 本身不對調，只回旗標")

    def test_image_sizes_come_from_house_level(self):
        """前端用 img_size_* 換算多邊形座標；同房影像尺寸相同，交付檔只記一次。"""
        r = server.embedded_regions(self.d, "aaa", "bbb")
        self.assertEqual(r["img_size_a"], [8192, 4096])
        self.assertEqual(r["img_size_b"], [8192, 4096])

    def test_pair_without_regions_returns_none(self):
        self.assertIsNone(server.embedded_regions(self.d, "aaa", "ccc"))

    def test_unknown_images_return_none(self):
        self.assertIsNone(server.embedded_regions(self.d, "aaa", "zzz"))

    def test_tolerates_extensions_in_image_list(self):
        """images 帶副檔名、查詢參數不帶，比對前須先去除。"""
        self.assertIsNotNone(server.embedded_regions(self.d, "aaa", "bbb"))

    def test_empty_house(self):
        self.assertIsNone(server.embedded_regions({}, "aaa", "bbb"))


class TestPackHousePublic(unittest.TestCase):
    """交付檔模式的打包：包內房子資料必須是交付檔本身。"""

    def _pack(self, with_images=True):
        d = temp_dir().__enter__()
        pub_dir = os.path.join(d, "public")
        ds_dir = os.path.join(d, "datasets", HOUSE)
        out_dir = os.path.join(d, "bundles")
        os.makedirs(ds_dir, exist_ok=True)
        write_json(os.path.join(pub_dir, HOUSE + ".json"), public_house())
        if with_images:
            for fn in IMG:
                with open(os.path.join(ds_dir, fn), "wb") as f:
                    f.write(b"jpeg-bytes-" + fn.encode())
        with quiet():
            ok = pack_bundle.pack_house_public(HOUSE, pub_dir, os.path.join(d, "datasets"),
                                               out_dir, copy=True)
        return d, pub_dir, out_dir, ok

    def test_house_json_is_byte_identical(self):
        """未經改寫是這條路徑的前提：改寫過就不再是「下游收到的那一份」。"""
        d, pub_dir, out_dir, ok = self._pack()
        try:
            self.assertTrue(ok)
            src = os.path.join(pub_dir, HOUSE + ".json")
            dst = os.path.join(out_dir, HOUSE, HOUSE + ".json")
            with open(src, "rb") as f1, open(dst, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_images_are_packed_and_no_regions_dir(self):
        d, _, out_dir, _ = self._pack()
        try:
            hd = os.path.join(out_dir, HOUSE)
            self.assertEqual(sorted(os.listdir(os.path.join(hd, "images"))), sorted(IMG))
            self.assertFalse(os.path.isdir(os.path.join(hd, "regions")),
                             "交付檔的區域已內嵌，不應另建 regions/")
            self.assertFalse(os.path.exists(os.path.join(hd, "semantic.json")))
            self.assertFalse([f for f in os.listdir(hd) if "HOTSPOT" in f],
                             "交付檔模式不得帶入真值")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_house_is_reported_not_raised(self):
        with temp_dir() as d, quiet():
            self.assertFalse(pack_bundle.pack_house_public(
                "NoSuchHouse", d, d, os.path.join(d, "out"), copy=True))


class TestRoundTrip(unittest.TestCase):
    """打包 → 伺服器讀取，兩步串起來的結果須與前端預期一致。"""

    def test_packed_file_loads_as_tool_data(self):
        with temp_dir() as d:
            pub_dir = os.path.join(d, "public")
            out_dir = os.path.join(d, "bundles")
            write_json(os.path.join(pub_dir, HOUSE + ".json"), public_house())
            with quiet():
                pack_bundle.pack_house_public(HOUSE, pub_dir, os.path.join(d, "ds"),
                                              out_dir, copy=True)
            with open(os.path.join(out_dir, HOUSE, HOUSE + ".json"), encoding="utf-8") as f:
                packed = json.load(f)
            self.assertTrue(server.is_public_house(packed))
            tool = server.public_house_to_tool(packed)
            self.assertIs(tool["has_gt"], False)
            self.assertEqual(tool["pairs"][0]["geo_n_inlier"], 1508)
            r = server.embedded_regions(packed, "aaa", "bbb")
            self.assertEqual(r["regions"][0]["n"], 120)

    def test_internal_house_is_served_unchanged(self):
        """既有的內部資料包不受影響：不是交付檔就原樣回傳。"""
        d = internal_house()
        self.assertFalse(server.is_public_house(d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
