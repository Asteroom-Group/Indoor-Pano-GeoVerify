"""
[模組用途]
Roma_Geo 的**唯一對外入口**：給兩張 ERP 全景圖，回傳幾何驗證結果
（共視區域 + 通道方位角 + 可信度）。

    from roma_geo import verify_pair, to_public

    res = verify_pair("a.jpg", "b.jpg")        # 完整結果（schema wp1-step3-v2）
    out = to_public(res)                        # 只留對外契約欄位

內部分兩層，本模組把它們串起來：

    L1  RoMa v2 匹配      roma_match.matcher.RoMaMatcher   GPU，約 2.6 s/對
    L2  幾何驗證          pose_regions.analyze_pair        純 CPU numpy，約 0.19 s/對

預設參數與正式批次 `pose_regions/batch_pose_from_cache.py` 一致
（conf 0.65 / retry 0.3 / max_pts 4000 / thresh 0.5° / iters 2000 / seed 0 /
stratified / lat_max 60）—— `Output/roma_regions` 的既有產出皆以這組參數產生。

[設定來源]
本模組是**函式介面**：除了選配的 `cache_dir`（未指定時才回落到環境變數
`ROMA_GEO_CACHE`），一切行為由傳入的參數決定，不讀 `ROMA_GEO_DATASETS` /
`ROMA_GEO_MATRICES` / `ROMA_GEO_OUTPUT` / `PAIR_THRESHOLD`。那些環境變數屬於
`environments/` 與 `pose_regions/` 的**批次管線**，是部署設定，不是本介面的契約。
被包成服務時同一行程可用不同參數並存。

[效能]
建構 matcher 要載入 torch + CUDA + 模型權重，約 6.4 s。處理多對時用
`verify_pairs()`，或自己 `m = get_matcher()` 後傳進 `matcher=m` 以重用。
單次 `verify_pair()` 會重用模組層的單例（以參數組合為鍵）。

[CLI]
    venv\\Scripts\\python.exe roma_geo.py a.jpg b.jpg [--public] [--out res.json]
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

__all__ = ["verify_pair", "verify_pairs", "to_public", "region_to_public", "get_matcher",
           "DEFAULTS", "SCHEMA_VERSION"]

#: 與 batch_pose_from_cache.py 的 argparse 預設逐項相同。
DEFAULTS = {
    "conf_min": 0.65,
    "retry_conf": 0.3,
    "max_pts": 4000,
    "thresh_deg": 0.5,
    "iters": 2000,
    "seed": 0,
    "sampler": "stratified",
    "lat_max": 60.0,       # 天底(腳架帶)/天頂(天花板)前濾角；schema v2 起預設開啟
}

#: 對外契約欄位（見 README「模組邊界」）。
PUBLIC_FIELDS = ("schema_version", "house", "img_a", "img_b",
                 "img_size_a", "img_size_b", "conf_used", "inlier_rate")

#: 共視區域的對外欄位。捨去的兩項：`mean_conf` 是匹配內部的品質數字，
#: `az_width_deg` 可由 `az_range` 兩端相減得到。整房交付路徑
#: （`export_public.attach_regions`）共用下方的 `region_to_public()`，
#: 兩條對外路徑因此不會各自漂移。
PUBLIC_REGION_FIELDS = ("n", "frac_of_inliers")
PUBLIC_REGION_SIDE_FIELDS = ("polygon", "area_frac", "az_range")

_matchers = {}


def _lazy():
    from pose_regions import analyze_pair
    from pose_regions.batch_pose_from_cache import SCHEMA_VERSION as SV
    from roma_match.matcher import RoMaMatcher
    return analyze_pair, SV, RoMaMatcher


try:
    from pose_regions.batch_pose_from_cache import SCHEMA_VERSION
except Exception:                                     # pragma: no cover
    SCHEMA_VERSION = "wp1-step3-v2"


def house_of(path):
    """從影像路徑推房子名（與 roma_match.matcher._get_cache_info 同規則）。"""
    for part in str(path).replace("\\", "/").split("/"):
        if "DollhouseTask" in part or "direct_match" in part:
            return part
    return "unknown"


def get_matcher(model_type="roma_v2", device=None, use_cache=True, cache_dir=None):
    """取得（或建立）共用的 RoMaMatcher 單例。

    conf_threshold 固定 0.1，與快取的底床一致；分層由 analyze_pair 自行處理。

    `cache_dir` 為 None 時沿用 `RoMaMatcher` 的解析順序
    （環境變數 `ROMA_GEO_CACHE` > `roma_match/cache/`）。單例以參數組合為鍵，
    不同 `cache_dir` 各自持有一份，同一行程內可並存。
    """
    key = (model_type, device, bool(use_cache), cache_dir)
    m = _matchers.get(key)
    if m is None:
        _, _, RoMaMatcher = _lazy()
        m = RoMaMatcher(device=device, conf_threshold=0.1, use_cache=use_cache,
                        model_type=model_type, cache_dir=cache_dir)
        _matchers[key] = m
    return m


def verify_pair(img_a, img_b, matcher=None, mode="direct", model_type="roma_v2",
                device=None, use_cache=True, cache_dir=None, house=None,
                keep_inliers=True, **params):
    """對一組 ERP 全景圖做幾何驗證。

    Args:
        img_a, img_b: ERP 全景圖路徑。
        matcher: 重用既有的 RoMaMatcher；None 則使用模組層單例。
        mode: "direct"（預設，生產路徑）或 "core"（RoMaScan 分塊，貴約 5.6 倍）。
        use_cache: 是否讀寫匹配快取的 npz（純加速，不影響結果）。
        cache_dir: 快取目錄。None 則依環境變數 `ROMA_GEO_CACHE` > `roma_match/cache/`。
            本函式除此之外不讀任何環境設定，其餘一律由參數決定。
        house: 房子名；None 則從路徑推。
        keep_inliers: False 則不回傳 inliers（2000 點，佔 JSON 體積大宗）。
        **params: 覆寫 DEFAULTS 中的任一項（一般不需要，見模組 docstring）。

    Returns:
        dict — schema wp1-step3-v2 的完整結果，另含 house/img_a/img_b/timing。
        無共視證據時 pose 為 None、flags.no_covisibility 為 True
        （這是「沒有共視證據」，不等於「兩張圖不連通」）。
    """
    bad = set(params) - set(DEFAULTS)
    if bad:
        raise TypeError("未知參數 %s；可覆寫的只有 %s" % (sorted(bad), sorted(DEFAULTS)))
    p = dict(DEFAULTS, **params)

    import numpy as np
    analyze_pair, _, _ = _lazy()

    m = matcher if matcher is not None else get_matcher(model_type, device, use_cache, cache_dir)

    t0 = time.perf_counter()
    if mode == "direct":
        matches, info = m.match_direct(img_a, img_b)
    elif mode == "core":
        matches, info = m.match_core(img_a, img_b)
    else:
        raise ValueError("mode 必須是 direct 或 core，收到 %r" % (mode,))
    t_match = time.perf_counter() - t0

    # 兩張圖尺寸可能不同，分別量。
    W, H = info["w"], info["h"]
    Wb, Hb = _size(img_b, W, H)

    arr = np.asarray(matches, dtype=float)
    if arr.size == 0:
        arr = np.empty((0, 5), dtype=float)

    t0 = time.perf_counter()
    res = analyze_pair(arr, W, H, Wb, Hb,
                       conf_min=p["conf_min"], max_pts=p["max_pts"],
                       thresh_deg=p["thresh_deg"], iters=p["iters"],
                       seed=p["seed"], sampler=p["sampler"],
                       retry_conf=p["retry_conf"], lat_max=p["lat_max"])
    t_geo = time.perf_counter() - t0

    res["house"] = house if house is not None else house_of(img_a)
    res["img_a"] = os.path.splitext(os.path.basename(str(img_a)))[0]
    res["img_b"] = os.path.splitext(os.path.basename(str(img_b)))[0]
    res["timing"] = {"match_s": round(t_match, 4),
                     "verify_s": round(t_geo, 4),
                     "total_s": round(t_match + t_geo, 4)}
    if not keep_inliers:
        res.pop("inliers", None)
    return res


def _size(path, default_w, default_h):
    """讀 B 圖尺寸；讀不到就沿用 A 的（同房內尺寸相同）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[0], im.size[1]
    except Exception:
        return default_w, default_h


def verify_pairs(pairs, progress=False, **kw):
    """對多組圖做幾何驗證，全程重用同一個 matcher。

    Args:
        pairs: [(img_a, img_b), ...]
        progress: True 則每對印一行進度。
    Yields:
        dict —— 逐對 yield，方便邊跑邊寫檔而不必全部留在記憶體。
    """
    m = kw.pop("matcher", None) or get_matcher(
        kw.get("model_type", "roma_v2"), kw.get("device"),
        kw.get("use_cache", True), kw.get("cache_dir"))
    for i, (a, b) in enumerate(pairs, 1):
        res = verify_pair(a, b, matcher=m, **kw)
        if progress:
            posed = "有位姿" if res.get("pose") else "無位姿"
            print("[%d/%d] %s %s · %d 區域 · %.2fs"
                  % (i, len(pairs), res["img_a"][:8], posed,
                     len(res.get("regions") or []), res["timing"]["total_s"]), flush=True)
        yield res


def region_to_public(r):
    """把單一共視區域裁成對外欄位（見 `PUBLIC_REGION_FIELDS`）。

    整房交付路徑併入區域時亦呼叫本函式，兩條對外路徑的區域欄位因此必然一致。
    """
    out = {k: r[k] for k in PUBLIC_REGION_FIELDS if k in r}
    for side in ("a", "b"):
        g = r.get(side) or {}
        out[side] = {k: g[k] for k in PUBLIC_REGION_SIDE_FIELDS if k in g}
    return out


def to_public(res):
    """裁成對外契約：共視區域 + 通道方位角 + 可信度。

    不含 R / t / cam_b_in_a / inliers / quality。區域多邊形只保留幾何。
    """
    pose = res.get("pose") or {}
    out = {k: res.get(k) for k in PUBLIC_FIELDS}
    out["az_ab_deg"] = pose.get("az_ab_deg")
    out["az_ba_deg"] = pose.get("az_ba_deg")
    out["no_covisibility"] = bool((res.get("flags") or {}).get("no_covisibility"))
    out["regions"] = [region_to_public(r) for r in (res.get("regions") or [])]
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Roma_Geo 單對幾何驗證")
    ap.add_argument("img_a")
    ap.add_argument("img_b")
    ap.add_argument("--mode", default="direct", choices=["direct", "core"])
    ap.add_argument("--conf_min", type=float, default=DEFAULTS["conf_min"])
    ap.add_argument("--no_cache", action="store_true", help="不讀寫 RoMa 匹配快取")
    ap.add_argument("--public", action="store_true", help="只輸出對外契約欄位")
    ap.add_argument("--out", help="寫入 JSON 檔（預設印到 stdout）")
    a = ap.parse_args(argv)

    res = verify_pair(a.img_a, a.img_b, mode=a.mode, conf_min=a.conf_min,
                      use_cache=not a.no_cache)
    payload = to_public(res) if a.public else res
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("已寫入 %s（%.1f KB）" % (a.out, len(text) / 1024))
        print("  %s · %d 區域 · az_ab %s · %.2fs"
              % ("有位姿" if res.get("pose") else "無位姿",
                 len(res.get("regions") or []),
                 (res.get("pose") or {}).get("az_ab_deg"), res["timing"]["total_s"]))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
