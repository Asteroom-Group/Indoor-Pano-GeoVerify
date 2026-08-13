"""
[腳本用途]
把流程各階段的產出，組成一個「自足的房子產出包（bundle）」，供標註/檢視工具直接載入。
工具因此不再依賴完整的資料集樹，改為「使用時手動選取一或數棟房子的產出包」——面向未知資料。

    <bundle_root>/<house>/
        <house>.json          幾何流程整合後的房子資料（連通、通道方位、共視統計）
        images/*.jpg          該房 ERP 全景影像（隨包攜帶，工具不再回頭讀資料集）
        regions/*.json        共視區域（每對一檔）
        semantic.json         （可選）門語意輔助資料
        <house>_HOTSPOT.json  （可選）標註真值；帶了工具才顯示 GT 對照，沒帶就是純預測檢視

[路徑彈性]
輸入各段路徑、輸出包根目錄、要打包哪些房子，全部可由參數指定；預設值取自 config 與既有產出位置。

[影像處理]
預設以硬連結節省空間（同磁碟時等效）；要做成可搬到別台機器的可攜包，加 --copy 實體複製。
"""
import argparse
import glob
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from config import datasets_root, regions_root                 # noqa: E402
from house_dirs import resolve_task_dir                        # noqa: E402


def _place(src, dst, copy):
    """把單一檔案放到目的地：copy=實體複製；否則硬連結（失敗時退回複製）。"""
    if os.path.exists(dst):
        os.remove(dst)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pack_house(house, data_dir, regions_dir, semantic_dir, datasets, out_root, copy):
    hj = os.path.join(data_dir, f"{house}.json")
    if not os.path.exists(hj):
        print(f"[skip] {house}: 找不到房子資料 {hj}")
        return False
    hd = json.load(open(hj, encoding="utf-8"))
    dst = os.path.join(out_root, house)
    os.makedirs(os.path.join(dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst, "regions"), exist_ok=True)

    _place(hj, os.path.join(dst, f"{house}.json"), copy=True)   # 房子資料一律實體複製（小、且是核心）

    # 影像：依房子資料的 images 清單，從資料集資料夾取。
    task_dir = resolve_task_dir(datasets, house)
    n_img = n_img_miss = 0
    if task_dir:
        for fn in hd.get("images", []):
            src = os.path.join(task_dir, fn)
            if os.path.exists(src):
                _place(src, os.path.join(dst, "images", fn), copy); n_img += 1
            else:
                n_img_miss += 1
    else:
        n_img_miss = len(hd.get("images", []))

    # 共視區域：本房所有 <house>_<a>_<b>.json。
    n_reg = 0
    for rp in glob.glob(os.path.join(regions_dir, f"{house}_*.json")):
        _place(rp, os.path.join(dst, "regions", os.path.basename(rp)), copy=True); n_reg += 1

    # 門語意（可選）。
    sem = os.path.join(semantic_dir, f"{house}.json")
    has_sem = os.path.exists(sem)
    if has_sem:
        _place(sem, os.path.join(dst, "semantic.json"), copy=True)

    # 標註真值（可選）：帶了工具才顯示 GT 對照。
    has_gt = False
    if task_dir:
        hs = glob.glob(os.path.join(task_dir, "*_HOTSPOT.json"))
        for h in hs:
            _place(h, os.path.join(dst, os.path.basename(h)), copy=True); has_gt = True

    print(f"[ok] {house}: 影像 {n_img}(缺 {n_img_miss}) · 區域 {n_reg} · "
          f"語意 {'有' if has_sem else '無'} · GT {'有' if has_gt else '無'}")
    return True


def main():
    ap = argparse.ArgumentParser(description="流程產出 → 自足房子產出包")
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "data"),
                    help="房子資料（<house>.json）目錄")
    ap.add_argument("--regions_dir", default=None,
                    help="共視區域目錄（預設 config.regions_root()）")
    ap.add_argument("--semantic_dir", default=os.path.join(SCRIPT_DIR, "semantic"),
                    help="門語意目錄")
    ap.add_argument("--datasets", default=None,
                    help="影像與 HOTSPOT 來源資料集根（預設 config.datasets_root()）")
    ap.add_argument("--out", required=True, help="輸出包根目錄 <bundle_root>")
    ap.add_argument("--houses", nargs="*", default=None, help="指定房子；預設 data_dir 內全部")
    ap.add_argument("--copy", action="store_true",
                    help="影像實體複製（可攜包）；預設硬連結（本機省空間）")
    a = ap.parse_args()

    regions_dir = a.regions_dir or regions_root()
    datasets = a.datasets or datasets_root()
    houses = a.houses or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(a.data_dir, "*.json"))
        if not os.path.basename(p).startswith("_"))
    os.makedirs(a.out, exist_ok=True)

    ok = 0
    for h in houses:
        ok += pack_house(h, a.data_dir, regions_dir, a.semantic_dir, datasets, a.out, a.copy)
    print(f"\n打包 {ok}/{len(houses)} 房 → {a.out}")


if __name__ == "__main__":
    main()
