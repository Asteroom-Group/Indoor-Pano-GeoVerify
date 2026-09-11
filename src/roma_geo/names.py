"""從路徑取出房子名與影像識別碼。"""
import os

_HOUSE_MARKERS = ("DollhouseTask", "direct_match")


def house_of(path):
    """從影像路徑取出房子名。

    路徑中含 DollhouseTask 或 direct_match 的目錄名時沿用該段，與既有快取檔名一致。
    其餘路徑取影像所在目錄名，未登錄的房子也能寫出可分辨的快取與結果。
    路徑只剩檔名時回傳 unknown。
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    for part in parts:
        if any(m in part for m in _HOUSE_MARKERS):
            return part
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def image_stem(path):
    """影像檔名去掉副檔名。"""
    return os.path.splitext(os.path.basename(str(path)))[0]
