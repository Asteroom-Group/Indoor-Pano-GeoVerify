"""
Kaggle 場次停機與單一 JSON 進度檔。

Kaggle GPU 工作階段約 12 小時硬殺。到點先寫 roma_progress.json 再停。
下一輪把這份 JSON 掛進資料集即可接續：已完成的配對會跳過，不再重抽。

cache / regions 等大檔等到特徵全部抽完才打成 ZIP，中途不打包，輸出比較整齊。
JSON 只記配對清單與已完成的識別碼，不含 npz 本體。
"""
import csv
import glob
import json
import os
import shutil
import zipfile

PROGRESS_JSON = "roma_progress.json"
CACHE_ZIP = "roma_cache.zip"
SCHEMA_VERSION = 1

_CACHE_REL = os.path.join("roma_match", "cache")
_PAIRS_REL = os.path.join("Output", "roma_regions", "_pairs.csv")


def due_to_pack(deadline_unix, now, reserve_min=20):
    """True：已進入打包預留時間，不該再開新的重活。"""
    return now >= deadline_unix - reserve_min * 60


def match_budget_hours(deadline_unix, now, reserve_min=20, cap_h=0):
    """批次匹配還能跑幾小時。0 表示該停了，不要把 0 傳給 --time_budget_h（那代表不限）。"""
    left = deadline_unix - now - reserve_min * 60
    hours = max(0.0, left / 3600.0)
    if cap_h and cap_h > 0:
        hours = min(hours, float(cap_h))
    return hours


def cache_dir(repo):
    return os.path.join(repo, _CACHE_REL)


def pairs_csv(repo):
    return os.path.join(repo, _PAIRS_REL)


def pair_id(house, a, b):
    """穩定的配對識別：房子 + 兩側檔名去副檔名。"""
    a = os.path.splitext(os.path.basename(str(a)))[0]
    b = os.path.splitext(os.path.basename(str(b)))[0]
    return [str(house), a, b]


def pair_key(pid):
    return tuple(pid)


def is_done_pair(house, a, b, skip_keys):
    pid = pair_id(house, a, b)
    key = pair_key(pid)
    if key in skip_keys:
        return True
    return (key[0], key[2], key[1]) in skip_keys


def done_key_set(payload):
    keys = set()
    for item in (payload or {}).get("done") or []:
        if len(item) != 3:
            continue
        keys.add(pair_key(pair_id(*item)))
    return keys


def load_pairs_rows(repo):
    path = pairs_csv(repo)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_pairs_csv(repo, pairs):
    dest = pairs_csv(repo)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["house", "img_a", "img_b"])
        for item in pairs:
            if len(item) != 3:
                continue
            w.writerow(pair_id(*item))
    return dest


def _cache_names(repo):
    cd = cache_dir(repo)
    if not os.path.isdir(cd):
        return []
    return [n for n in os.listdir(cd) if n.endswith(".npz")]


def _cache_covers(names, house, a, b):
    ab = "%s_%s_%s_" % (house, a, b)
    ba = "%s_%s_%s_" % (house, b, a)
    return any(n.startswith(ab) or n.startswith(ba) for n in names)


def done_from_cache(repo, pairs):
    names = _cache_names(repo)
    found = []
    for item in pairs:
        if len(item) != 3:
            continue
        pid = pair_id(*item)
        if _cache_covers(names, *pid):
            found.append(pid)
    return found


def _merge_done(pairs, *groups):
    order = [pair_id(*p) for p in pairs if len(p) == 3]
    allowed = {pair_key(p) for p in order}
    seen = set()
    out = []
    for group in groups:
        for item in group or []:
            if len(item) != 3:
                continue
            pid = pair_id(*item)
            key = pair_key(pid)
            rev = (key[0], key[2], key[1])
            if key not in allowed and rev not in allowed:
                continue
            canon = key if key in allowed else rev
            if canon in seen:
                continue
            seen.add(canon)
            out.append(list(canon))
    return out


def make_progress(repo, last_completed_step, next_step, status, reason,
                  elapsed_h, deadline_h, previous=None):
    previous = previous or {}
    rows = load_pairs_rows(repo)
    if rows:
        pairs = [pair_id(r["house"], r["img_a"], r["img_b"]) for r in rows]
    else:
        pairs = [pair_id(*p) for p in (previous.get("pairs") or []) if len(p) == 3]
    done = _merge_done(pairs, previous.get("done"), done_from_cache(repo, pairs))
    n_pairs = len(pairs)
    n_done = len(done)
    n_remaining = max(0, n_pairs - n_done)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "last_completed_step": last_completed_step,
        "next_step": next_step,
        "elapsed_h": round(float(elapsed_h), 3),
        "session_deadline_h": deadline_h,
        "n_pairs": n_pairs,
        "n_done": n_done,
        "n_remaining": n_remaining,
        "features_complete": n_pairs > 0 and n_remaining == 0,
        "pairs": pairs,
        "done": done,
        "artifacts": {
            "progress_json": PROGRESS_JSON,
            "cache_zip": CACHE_ZIP,
        },
    }


def write_progress(path, payload):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def find_progress_json(input_root):
    if not os.path.isdir(input_root):
        return None
    hits = [
        h for h in sorted(glob.glob(os.path.join(input_root, "**", PROGRESS_JSON),
                                    recursive=True))
        if os.path.isfile(h)
    ]
    return hits[0] if hits else None


def usable_progress(payload):
    """有配對清單或已完成識別碼才算能接續；空物件從頭跑。"""
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("pairs") or payload.get("done"))


def read_progress_file(path):
    """讀 JSON。空檔、壞檔、非物件都回 None。"""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        payload = json.loads(text)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_resume(input_root):
    """回傳 (路徑, payload)。沒有檔 → (None, None)；空的或壞的 → (路徑, None)。"""
    path = find_progress_json(input_root)
    if not path:
        return None, None
    payload = read_progress_file(path)
    if not usable_progress(payload):
        return path, None
    return path, payload


def restore_progress(repo, json_path):
    """把進度 JSON 拷進 repo，並還原 _pairs.csv。回傳 payload。"""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    os.makedirs(repo, exist_ok=True)
    shutil.copy2(json_path, os.path.join(repo, PROGRESS_JSON))
    pairs = payload.get("pairs") or []
    if pairs:
        write_pairs_csv(repo, pairs)
    return payload


def pack_cache_zip(repo, dest):
    """特徵全部抽完後才呼叫。npz 已壓縮，用 ZIP_STORED。"""
    cd = cache_dir(repo)
    if not os.path.isdir(cd):
        return None
    files = [os.path.join(cd, n) for n in sorted(os.listdir(cd)) if n.endswith(".npz")]
    if not files:
        return None
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as z:
        for p in files:
            z.write(p, "roma_match/cache/" + os.path.basename(p),
                    compress_type=zipfile.ZIP_STORED)
    return dest
