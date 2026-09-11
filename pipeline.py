"""逐步呼叫的處理流程。每一步吃明確的輸入路徑，不依賴工作目錄或環境變數。"""
import csv
import json
import os
import sys

from config import Settings
from errors import RomaGeoError

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class Degradation:
    """可選輸入缺席時的降級紀錄，呼叫端可從 input_name 辨識缺了哪一項。"""

    def __init__(self, input_name, condition, fallback):
        self.input_name = input_name
        self.condition = condition
        self.fallback = fallback

    def as_dict(self):
        return {"input": self.input_name, "condition": self.condition, "fallback": self.fallback}


class StepResult:
    def __init__(self, name, outputs=None, degradations=None, skipped=False, message=""):
        self.name = name
        self.outputs = outputs or {}
        self.degradations = list(degradations or [])
        self.skipped = skipped
        self.message = message

    @property
    def degraded(self):
        return bool(self.degradations)


class PipelineResult:
    def __init__(self, house, steps, degradations):
        self.house = house
        self.steps = steps
        self.degradations = degradations


def _add_path(rel):
    p = os.path.join(_ROOT, rel) if rel else _ROOT
    if p not in sys.path:
        sys.path.insert(0, p)


def _read_pairs_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["house"], r["img_a"], r["img_b"]) for r in csv.DictReader(f)]


def _write_pairs_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house", "img_a", "img_b", "score"])
        for h, x, y, s in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            w.writerow([h, x, y, "" if s is None else round(float(s), 4)])
    return path


def _list_images(datasets_root, house):
    _add_path("stitch_tool")
    from house_dirs import list_house_jpgs
    return list_house_jpgs(datasets_root, house)


def _uniq_degradations(items):
    out, seen = [], set()
    for d in items:
        if d.input_name in seen:
            continue
        seen.add(d.input_name)
        out.append(d)
    return out


def _has_semantic(semantic_root, house):
    if not semantic_root or not os.path.isdir(semantic_root):
        return False
    tid = "".join(ch for ch in house if ch.isdigit())
    candidates = [os.path.join(semantic_root, house)]
    if tid:
        candidates.append(os.path.join(semantic_root, tid))
    for d in candidates:
        if not os.path.isdir(d):
            continue
        if any(n.endswith("_mask.png") for n in os.listdir(d)):
            return True
    return False


def ingest_connections(settings, input_dir=None, out_dir=None):
    """上游連通 JSON → 連通矩陣。目錄缺席時略過並標記降級。"""
    src = input_dir or settings.connections_dir()
    dest = out_dir or settings.matrices_root()
    if not src or not os.path.isdir(src):
        return StepResult(
            "ingest_connections",
            skipped=True,
            degradations=[Degradation(
                "upstream_connectivity", "absent",
                "略過連通輸出轉矩陣，後續改以影像兩兩配對",
            )],
        )
    import ingest_connections as ic
    houses = ic.convert_connections(src, dest)
    return StepResult("ingest_connections", outputs={"matrices_dir": dest, "houses": houses})


def select_pairs(settings, houses, out_csv=None, threshold=None, pool="mean"):
    """連通矩陣或影像清單 → 配對 CSV。矩陣缺席時改以該房全部影像兩兩配對。"""
    if not houses:
        raise RomaGeoError("未指定房子", input_name="房子名", condition="houses 為空")
    houses = list(houses)
    mdir = settings.matrices_root()
    ds = settings.datasets_root()
    degradations = []
    rows = []
    have_matrix, missing_matrix = [], []
    for h in houses:
        npy = os.path.join(mdir, h + ".npy")
        js = os.path.join(mdir, h + "_images.json")
        if os.path.isfile(npy) and os.path.isfile(js):
            have_matrix.append(h)
        else:
            missing_matrix.append(h)

    pair_source = "matrix"
    if have_matrix:
        _add_path("environments")
        import select_pairs as sp
        collected = sp.collect(mdir, have_matrix, pool)
        thr = settings.pair_threshold() if threshold is None else threshold
        rows.extend(r for r in collected if r[3] >= thr)

    img_n = 0
    for h in missing_matrix:
        task_dir, names = _list_images(ds, h)
        if not names:
            raise RomaGeoError(
                "找不到可用的全景影像",
                input_name="全景影像",
                condition="房子 %s 在 %s 底下沒有 jpg，也沒有連通矩陣" % (h, ds),
            )
        if len(names) < 2:
            raise RomaGeoError(
                "至少需要兩張全景影像才能配對",
                input_name="全景影像",
                condition="房子 %s 只有 %d 張圖" % (h, len(names)),
            )
        stems = [os.path.splitext(n)[0] for n in names]
        for i in range(len(stems)):
            for j in range(i + 1, len(stems)):
                rows.append((h, stems[i], stems[j], None))
                img_n += 1
    if missing_matrix:
        degradations.append(Degradation(
            "upstream_connectivity", "absent",
            "改以影像兩兩配對（%d 對）" % img_n,
        ))
        pair_source = "images" if not have_matrix else "mixed"

    out_csv = out_csv or os.path.join(settings.regions_root(), "_pairs.csv")
    _write_pairs_csv(out_csv, rows)
    return StepResult(
        "select_pairs",
        outputs={"pairs_csv": out_csv, "pair_source": pair_source, "n_pairs": len(rows)},
        degradations=degradations,
    )


def match_pairs(settings, pairs_csv, matcher=None, mode="direct"):
    """對配對清單做 RoMa 匹配，寫入快取。matcher 由呼叫端傳入時不自行載入模型。"""
    from pose_regions import find_house_image
    if not os.path.isfile(pairs_csv):
        raise RomaGeoError(
            "檔案不存在：%s" % pairs_csv,
            input_name="配對清單",
            condition="請先呼叫 select_pairs，或傳入既有的 pairs_csv",
        )
    rows = list(csv.DictReader(open(pairs_csv, encoding="utf-8", newline="")))
    if matcher is None:
        from roma_geo import get_matcher
        matcher = get_matcher(cache_dir=settings.cache_root())
    ds = settings.datasets_root()
    n_done = n_fail = 0
    errors = []
    for r in rows:
        pa = find_house_image(r["house"], r["img_a"], datasets_root=ds)
        pb = find_house_image(r["house"], r["img_b"], datasets_root=ds)
        if not (pa and pb):
            missing = r["img_a"] if not pa else r["img_b"]
            raise RomaGeoError(
                "找不到 %s.jpg" % missing,
                input_name="全景影像",
                condition="房子 %s 的資料夾裡沒有這張圖（datasets=%s）" % (r["house"], ds),
            )
        try:
            if mode == "direct":
                matcher.match_direct(pa, pb)
            else:
                matcher.match_core(pa, pb)
            n_done += 1
        except RomaGeoError:
            raise
        except Exception as e:
            n_fail += 1
            errors.append("%s: %s" % (type(e).__name__, e))
    if n_done == 0 and n_fail:
        raise RomaGeoError(
            "；".join(errors[:3]),
            input_name="匹配",
            condition="全部 %d 對都失敗" % n_fail,
        )
    return StepResult("match_pairs", outputs={"n_done": n_done, "n_fail": n_fail, "cache_dir": settings.cache_root()})


def estimate_poses(settings, pairs=None, pairs_csv=None, resume=False, limit=0):
    """匹配快取 → 逐對幾何驗證 JSON 與 _summary.csv。"""
    from pose_regions.batch_pose_from_cache import run_pose_batch
    if pairs is None and pairs_csv:
        pairs = _read_pairs_csv(pairs_csv)
    n = run_pose_batch(
        cache_dir=settings.cache_root(),
        out_dir=settings.regions_root(),
        datasets_root=settings.datasets_root(),
        pairs=pairs,
        limit=limit,
        resume=resume,
        on_missing="error" if pairs else "skip",
    )
    return StepResult(
        "estimate_poses",
        outputs={"regions_dir": settings.regions_root(), "n_pairs": n},
        message="wrote %d" % n,
    )


def evaluate_ranking(settings, summary_csv=None):
    """排序評估。沒有標註真值時略過。"""
    _add_path("stitch_tool")
    from house_dirs import resolve_task_dir
    from gt_hotspot import find_hotspot_files
    summary_csv = summary_csv or os.path.join(settings.regions_root(), "_summary.csv")
    if not os.path.isfile(summary_csv):
        return StepResult(
            "evaluate_ranking",
            skipped=True,
            degradations=[Degradation(
                "ground_truth", "absent",
                "沒有 _summary.csv，略過排序評估",
            )],
        )
    houses = sorted({r["house"] for r in csv.DictReader(open(summary_csv, encoding="utf-8"))})
    has_gt = False
    for h in houses:
        task_dir = resolve_task_dir(settings.datasets_root(), h)
        if task_dir and find_hotspot_files(task_dir):
            has_gt = True
            break
    if not has_gt:
        return StepResult(
            "evaluate_ranking",
            skipped=True,
            degradations=[Degradation(
                "ground_truth", "absent",
                "略過排序評估（需要 HOTSPOT 標註）",
            )],
        )
    return StepResult("evaluate_ranking", outputs={"summary_csv": summary_csv})


def prepare_house(settings, houses, oof_dir=None):
    """影像（可加連通矩陣與幾何驗證 CSV）→ 內部房子 JSON。"""
    _add_path("stitch_tool")
    import prepare_house_data as phd
    from house_dirs import resolve_task_dir
    from gt_hotspot import find_hotspot_files

    oof_dir = oof_dir if oof_dir is not None else settings.matrices_root()
    summary_path = os.path.join(settings.regions_root(), "_summary.csv")
    ranking_path = os.path.join(settings.regions_root(), "_ranking_eval.csv")
    summary_lookup = phd.load_pair_csv(summary_path) if os.path.isfile(summary_path) else None
    ranking_lookup = phd.load_pair_csv(ranking_path) if os.path.isfile(ranking_path) else None
    out_dir = settings.house_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    degradations = []
    last_path = None
    for h in houses:
        npy = os.path.join(oof_dir, h + ".npy") if oof_dir else ""
        if not (npy and os.path.isfile(npy)):
            degradations.append(Degradation(
                "upstream_connectivity", "absent",
                "改由影像清單與幾何驗證結果組房",
            ))
        task_dir = resolve_task_dir(settings.datasets_root(), h)
        if not task_dir or not find_hotspot_files(task_dir):
            degradations.append(Degradation(
                "ground_truth", "absent",
                "gt_* 為空值，不把缺席當成不連通",
            ))
        data = phd.build_house(
            h, oof_dir, ranking_lookup=ranking_lookup, summary_lookup=summary_lookup,
            datasets_root=settings.datasets_root(),
        )
        if data is None:
            raise RomaGeoError(
                "無法組出房子資料",
                input_name="全景影像",
                condition="房子 %s 沒有連通矩陣，也找不到影像" % h,
            )
        path = os.path.join(out_dir, h + ".json")
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        last_path = path
    return StepResult(
        "prepare_house",
        outputs={"house_json": last_path, "house_data_dir": out_dir},
        degradations=_uniq_degradations(degradations),
    )


def prepare_semantic(settings, houses):
    """語意圖 → 語意門 JSON。沒有語意圖時略過。"""
    if not any(_has_semantic(settings.semantic_root(), h) for h in houses):
        return StepResult(
            "prepare_semantic",
            skipped=True,
            degradations=[Degradation(
                "semantic_maps", "absent",
                "略過語意門抽取",
            )],
        )
    return StepResult("prepare_semantic", outputs={"semantic_root": settings.semantic_root()})


def export_public_houses(settings, houses=None, with_regions=True):
    """內部房子 JSON → 對外交付子集。"""
    import export_public as ep
    written = ep.export_houses(
        in_dir=settings.house_data_dir(),
        out_dir=settings.public_dir(),
        regions_dir=settings.regions_root() if with_regions else None,
        with_regions=with_regions,
        datasets_dir=settings.datasets_root(),
        houses=houses,
    )
    out = {"public_dir": settings.public_dir()}
    if written:
        out["public_json"] = written[0]
    return StepResult("export_public", outputs=out)


def run_house(house, settings, matcher=None, run_match=True):
    """一間房從頭跑完：挑配對、匹配、幾何驗證、組房、匯出。可選輸入缺席時降級並記錄。"""
    steps = []
    degradations = []

    r = select_pairs(settings, [house])
    steps.append(r)
    degradations.extend(r.degradations)
    pairs_csv = r.outputs["pairs_csv"]

    if run_match:
        r = match_pairs(settings, pairs_csv, matcher=matcher)
        steps.append(r)

    r = estimate_poses(settings, pairs_csv=pairs_csv)
    steps.append(r)

    r = evaluate_ranking(settings)
    steps.append(r)
    degradations.extend(r.degradations)

    r = prepare_house(settings, [house])
    steps.append(r)
    degradations.extend(r.degradations)

    r = prepare_semantic(settings, [house])
    steps.append(r)
    degradations.extend(r.degradations)

    r = export_public_houses(settings, houses=[house])
    steps.append(r)

    return PipelineResult(house, steps, _uniq_degradations(degradations))
