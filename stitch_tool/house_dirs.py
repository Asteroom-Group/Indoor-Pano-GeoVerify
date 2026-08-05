"""
房屋影像資料夾路徑解析（stitch_tool 專用、獨立實作，不依賴 coarse_filter_experiment）。

datasets/ 底下的房屋資料夾可能被同名巢狀重複任意層，例如：
  flat:   datasets/<house>/img.jpg
  double: datasets/<house>/<house>/img.jpg
  triple: datasets/<house>/<house>/<house>/img.jpg
resolve_task_dir() 由淺到深持續往下找同名子資料夾，直到找到含有 *.jpg 的那一層，
或超過 max_depth 才放棄回傳 None。
"""
import os
import glob


def resolve_task_dir(data_root: str, house: str, max_depth: int = 5):
    current = os.path.join(data_root, house)
    if not os.path.isdir(current):
        return None
    depth = 0
    while depth <= max_depth:
        if glob.glob(os.path.join(current, "*.jpg")):
            if depth > 1:
                print(f"[house_dirs] '{house}' 巢狀 {depth} 層才找到 -> {current}")
            return current
        nested = os.path.join(current, house)
        if not os.path.isdir(nested):
            return None
        current = nested
        depth += 1
    return None
