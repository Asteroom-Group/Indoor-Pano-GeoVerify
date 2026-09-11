"""語意門抽取與通道方位矯正。矯正只改輸出一，不改相對位姿。"""
import glob
import json
import os

CONTAIN_FRAC = 0.8
MERGE_GAP_DEG = 1.0


def az_overlap(a0, a1, b0, b1):
    """兩個可過縫角區間的重疊角度。區間以 [a0, a1] 表示。"""
    def norm(s, e):
        w = (e - s) % 360.0
        return s % 360.0, w
    s1, w1 = norm(a0, a1)
    s2, w2 = norm(b0, b1)
    best = 0.0
    for sh in (-360.0, 0.0, 360.0):
        lo = max(s1, s2 + sh)
        hi = min(s1 + w1, s2 + sh + w2)
        best = max(best, hi - lo)
    return max(0.0, best)


def region_az_span(side):
    """從共視區域一側取出方位區間 (az0, az1, width_deg)。"""
    if not side:
        return None
    ar = side.get("az_range")
    if ar and len(ar) >= 2:
        a0, a1 = float(ar[0]), float(ar[1])
        w = (a1 - a0) % 360.0
        if w <= 0:
            w = abs(a1 - a0)
        return a0, a1, w
    poly = side.get("polygon")
    if not poly or len(poly) < 3:
        return None
    xs = [p[0] for p in poly]
    # 無影像寬時只能用像素差近似；呼叫端應提供 az_range。
    return None


def choose_door_correction(az, side_regions, doors, contain=CONTAIN_FRAC):
    """取與共視區域重疊度最高的門；共視區域有 contain 以上落在該門內時改用門中心。

    回傳 (方位, 是否矯正)。
    """
    if not doors or not side_regions:
        return az, False
    best = None
    for side in side_regions:
        span = region_az_span(side)
        if span is None:
            continue
        r0, r1, rw = span
        if rw <= 0:
            continue
        for d in doors:
            ov = az_overlap(r0, r1, d["az0"], d["az1"])
            frac = ov / rw
            if best is None or frac > best[0]:
                best = (frac, d)
    if best is None or best[0] < contain:
        return az, False
    return float(best[1]["az"]) % 360.0, True


def invert_pose(R, t):
    """x_b = R x_a + t 的逆：x_a = Rᵀ x_b - Rᵀ t。t 保持單位長。"""
    import numpy as np
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    Ri = R.T
    ti = -R.T @ t
    n = float(np.linalg.norm(ti))
    if n:
        ti = ti / n
    return Ri.tolist(), ti.tolist()


def azimuth_from_pose(R, t):
    """A 幀中 B 相機方位（度，0~360）。cam_b_in_a = -Rᵀ t。"""
    import math
    import numpy as np
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    cb = -R.T @ t
    return math.degrees(math.atan2(cb[0], cb[2])) % 360.0


def _house_tid(house):
    return "".join(ch for ch in house if ch.isdigit())


def extract_house_doors(semantic_root, house, merge_gap_deg=MERGE_GAP_DEG):
    """對一間房的語意圖跑門抽取。回傳 {image_stem: [door, ...]}。"""
    from semantic_doors.semantic_door_eval import extract_doors
    tid = _house_tid(house)
    dirs = [os.path.join(semantic_root, house)]
    if tid:
        dirs.append(os.path.join(semantic_root, tid))
    doors = {}
    seen = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for mp in sorted(glob.glob(os.path.join(d, "*_mask.png"))):
            img = os.path.basename(mp)[:-len("_mask.png")]
            if img in seen:
                continue
            seen.add(img)
            doors[img] = extract_doors(mp, merge_gap_deg)
    return doors


def apply_semantic_correction(pub, doors_by_image):
    """就地矯正 pub['pairs'] 的 az_ij / az_ji。回傳矯正對數。

    只改通道方位。R / t 不動。
    """
    names = pub.get("images") or []
    n = 0
    for p in pub.get("pairs") or []:
        p["semantic_corrected"] = False
        if p.get("az_source") != "geo":
            continue
        regions = p.get("regions") or []
        if not regions:
            continue
        try:
            ni, nj = names[p["i"]], names[p["j"]]
        except (IndexError, KeyError, TypeError):
            continue
        si = os.path.splitext(ni)[0]
        sj = os.path.splitext(nj)[0]
        sides_a = [r.get("a") or {} for r in regions]
        sides_b = [r.get("b") or {} for r in regions]
        corr_i = corr_j = False
        if "az_ij" in p and si in doors_by_image:
            new_az, hit = choose_door_correction(p["az_ij"], sides_a, doors_by_image[si])
            if hit:
                p["az_ij"] = round(new_az, 2) % 360.0
                corr_i = True
        if "az_ji" in p and sj in doors_by_image:
            new_az, hit = choose_door_correction(p["az_ji"], sides_b, doors_by_image[sj])
            if hit:
                p["az_ji"] = round(new_az, 2) % 360.0
                corr_j = True
        if corr_i or corr_j:
            p["semantic_corrected"] = True
            n += 1
    return n


def load_doors_json(path):
    if not path or not os.path.isfile(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    return data.get("doors") or {}
