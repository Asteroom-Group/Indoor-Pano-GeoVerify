"""
[腳本用途]
拼接工具的本機伺服器 (沿用 panorama_viewer 的作法並擴充)。
提供房子清單、連通性資料、ERP 影像、以及人工標註進度的存讀。

[用法]
  python stitch_tool/server.py            # 預設 http://localhost:8010
  python stitch_tool/server.py --port 8010

需先跑過 prepare_house_data.py 以產生 data/*.json。
"""
import os
import re
import sys
import json
import glob
import socket
import argparse
import mimetypes
from urllib.parse import unquote
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from house_dirs import resolve_task_dir as _find_house_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)            # 專案根目錄
DATA_DIR = os.path.join(SCRIPT_DIR, "data")       # prepare_house_data.py 產物
STATE_DIR = os.path.join(SCRIPT_DIR, "state")     # 人工標註進度
SEMANTIC_DIR = os.path.join(SCRIPT_DIR, "semantic")  # prepare_semantic_doors.py 產物（任務8）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config import datasets_root                   # noqa: E402（要先補 sys.path）

DATASETS = datasets_root()                         # 可用 ROMA_GEO_DATASETS 覆寫，見 config.py

os.makedirs(STATE_DIR, exist_ok=True)

_SAFE = re.compile(r"^[A-Za-z0-9_\-.]+$")         # 房子名 / 檔名白名單，擋路徑穿越
_img_dir_cache: dict[str, str] = {}

# ---- RoMa 即時比對（/api/roma_match）：延遲載入單例 ----
# RoMaMatcher 建構會載入 torch + CUDA + 模型權重，成本不小；一般標註工作完全用不到它，
# 所以只在第一次真的點了「RoMa 比對」鈕時才 import/建構，且只建構一次（快取的圖對之後秒回）。
_roma_matcher = None
_roma_matcher_error = None


def _get_roma_matcher():
    global _roma_matcher, _roma_matcher_error
    if _roma_matcher is not None:
        return _roma_matcher
    if _roma_matcher_error is not None:
        raise RuntimeError(_roma_matcher_error)
    try:
        # 將專案根目錄加入 sys.path。
        if ROOT_DIR not in sys.path:
            sys.path.append(ROOT_DIR)
        from roma_match.matcher import RoMaMatcher
        _roma_matcher = RoMaMatcher(conf_threshold=0.5, use_cache=True, model_type="roma_v2")
        return _roma_matcher
    except Exception as e:
        _roma_matcher_error = f"RoMa matcher unavailable ({e}). Check RoMa weights / CUDA setup."
        raise RuntimeError(_roma_matcher_error)


def _resolve_task_dir(house: str):
    if house in _img_dir_cache:
        return _img_dir_cache[house]
    d = _find_house_dir(DATASETS, house)
    if d:
        _img_dir_cache[house] = d
    return d


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=SCRIPT_DIR, **k)

    def end_headers(self):
        # 本地開發工具：JS/HTML/CSS 一律 no-store。瀏覽器對 ES module 的快取很兇，
        # 改完程式常常連 Ctrl+F5 都帶不動子模組（歷史痛點，通用規則有記載）——從源頭關掉。
        p = self.path.split("?", 1)[0].lower()
        if p == "/" or p.endswith((".js", ".html", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---- helpers ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "":
            self.path = "/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)

        if path == "/api/houses":
            idx_path = os.path.join(DATA_DIR, "_index.json")
            if os.path.exists(idx_path):
                return self._json(json.load(open(idx_path, encoding="utf-8")))
            # 沒有 _index.json 就即時掃描 data/*.json
            houses = []
            for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
                if os.path.basename(p).startswith("_"):
                    continue
                try:
                    d = json.load(open(p, encoding="utf-8"))
                    houses.append({"house": d["house"], "n_images": d["n_images"],
                                   "n_pairs": len(d["pairs"]),
                                   "n_gt_connected_pairs": d.get("n_gt_connected_pairs", 0)})
                except Exception:
                    pass
            return self._json(houses)

        m = re.match(r"^/api/house/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            fp = os.path.join(DATA_DIR, f"{house}.json")
            if not os.path.exists(fp):
                return self._err(404, "house data not found; run prepare_house_data.py")
            return self._json(json.load(open(fp, encoding="utf-8")))

        m = re.match(r"^/api/state/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            fp = os.path.join(STATE_DIR, f"{house}.json")
            if os.path.exists(fp):
                return self._json(json.load(open(fp, encoding="utf-8")))
            return self._json({})

        m = re.match(r"^/images/([^/]+)/([^/]+)$", path)
        if m:
            house, fname = unquote(m.group(1)), unquote(m.group(2))
            if not (_SAFE.match(house) and _SAFE.match(fname)):
                return self._err(400, "bad path")
            return self._serve_image(house, fname)

        # GET /api/regions/<house>?a=<imgA檔名>&b=<imgB檔名>
        # 讀 WP1 Step3 的共視區域 JSON（Output/roma_regions/{house}_{a}_{b}.json，
        # schema wp1-step3-v1/v2，見 pose_regions/README.md）。檔名只存 A_B 一個方向；
        # 若命中反序檔則回 swap:true，前端把 regions 的 a/b 兩側對調使用。
        m = re.match(r"^/api/regions/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            from urllib.parse import parse_qs
            q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            a = os.path.splitext((q.get("a") or [""])[0])[0]
            b = os.path.splitext((q.get("b") or [""])[0])[0]
            if not (a and b and _SAFE.match(a) and _SAFE.match(b)):
                return self._err(400, "bad image name")
            rdir = os.path.join(ROOT_DIR, "Output", "roma_regions")
            p_ab = os.path.join(rdir, f"{house}_{a}_{b}.json")
            p_ba = os.path.join(rdir, f"{house}_{b}_{a}.json")
            if os.path.exists(p_ab):
                d = json.load(open(p_ab, encoding="utf-8")); d["swap"] = False
            elif os.path.exists(p_ba):
                d = json.load(open(p_ba, encoding="utf-8")); d["swap"] = True
            else:
                return self._err(404, "此對沒有區域 JSON（先跑 pose_regions/batch_pose_from_cache.py）")
            d.pop("inliers", None)   # 遮罩用不到內點明細，省流量（每檔可達數百 KB）
            return self._json(d)

        # GET /api/semantic_doors/<house>：子計畫一門語意圖抽取結果（任務8，見
        # prepare_semantic_doors.py / stitch_tool/semantic/<house>.json）。缺檔（該房沒有語意圖，
        # 目前 22/57）回 404，前端顯示「此房無語意圖」而不觸發校正。
        m = re.match(r"^/api/semantic_doors/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            fp = os.path.join(SEMANTIC_DIR, f"{house}.json")
            if not os.path.exists(fp):
                return self._err(404, "此房無語意圖（未跑過 prepare_semantic_doors.py 或該房沒有語意資料）")
            return self._json(json.load(open(fp, encoding="utf-8")))

        # 其餘交給靜態檔 (index.html 等)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        m = re.match(r"^/api/state/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b"{}"
                obj = json.loads(raw.decode("utf-8"))
            except Exception as e:
                return self._err(400, f"bad json: {e}")
            fp = os.path.join(STATE_DIR, f"{house}.json")
            json.dump(obj, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
            return self._json({"ok": True, "saved": os.path.basename(fp)})

        m = re.match(r"^/api/roma_match/([^/]+)$", path)
        if m:
            house = unquote(m.group(1))
            if not _SAFE.match(house):
                return self._err(400, "bad house name")
            try:
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                return self._err(400, f"bad json: {e}")
            return self._do_roma_match(house, body)

        return self._err(404, "no such endpoint")

    def _do_roma_match(self, house, body):
        """POST /api/roma_match/<house> {img_a, img_b, conf?} -> {matches, w, h, count}.
        img_a/img_b are image basenames (as in house JSON's `images` list). Runs RoMa direct
        matching (reuses/produces the SAME roma_match/cache/*.npz files as the offline
        run_roma_above_threshold.py batch script, so repeat clicks on an already-matched pair
        are instant). Never raises into the request handler - errors come back as JSON so a
        missing GPU/model doesn't take down the rest of the annotation tool."""
        img_a, img_b = body.get("img_a"), body.get("img_b")
        if not img_a or not img_b:
            return self._err(400, "img_a and img_b required")
        if not (_SAFE.match(img_a) and _SAFE.match(img_b)):
            return self._err(400, "bad image name")
        task_dir = _resolve_task_dir(house)
        if task_dir is None:
            return self._err(404, "house image dir not found")
        path_a, path_b = os.path.join(task_dir, img_a), os.path.join(task_dir, img_b)
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            return self._err(404, "image(s) not found")

        try:
            conf = float(body.get("conf", 0.5))
        except (TypeError, ValueError):
            conf = 0.5

        try:
            matcher = _get_roma_matcher()
        except Exception as e:
            return self._err(503, str(e))

        matcher.conf_threshold = conf
        try:
            matches, info = matcher.match_direct(path_a, path_b)
        except Exception as e:
            return self._err(500, f"RoMa matching failed: {e}")

        return self._json({
            "matches": matches, "w": info.get("w"), "h": info.get("h"), "count": len(matches),
        })

    def _serve_image(self, house, fname):
        task_dir = _resolve_task_dir(house)
        if task_dir is None:
            return self._err(404, "house image dir not found")
        fp = os.path.join(task_dir, fname)
        if not os.path.isfile(fp):
            return self._err(404, "image not found")
        ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
        try:
            size = os.path.getsize(fp)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            with open(fp, "rb") as f:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        # 安靜些，只印非影像請求
        if "/images/" not in (self.path or ""):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    n_data = len(glob.glob(os.path.join(DATA_DIR, "*.json")))
    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        ip = get_ip()
        print("=" * 60)
        print("  拼接標註工具 (Stitch Labeler) 已啟動")
        print(f"  房子資料檔: {n_data} 個 (data/*.json)")
        print(f"  本機:   http://localhost:{args.port}")
        print(f"  區網:   http://{ip}:{args.port}")
        print("  Ctrl+C 結束")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止。")


if __name__ == "__main__":
    main()
