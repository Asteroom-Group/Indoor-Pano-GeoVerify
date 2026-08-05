"""
[模組用途]
專案層級的路徑設定 —— 目前只有「資料集根目錄」這一項。

資料集的擺放位置依部署環境而異，所有讀取資料集的地方統一經由本模組解析。

解析優先序：
  1. 環境變數 `ROMA_GEO_DATASETS`（支援 ~ 展開；相對路徑會轉成絕對路徑）
  2. 專案根目錄底下的 `datasets/`

[用法]
  from config import datasets_root, semantic_root
  img_dir = os.path.join(datasets_root(), house)

換路徑時不必改任何程式碼：

  PowerShell:  $env:ROMA_GEO_DATASETS = "D:\panorama_data"
  bash:        export ROMA_GEO_DATASETS=/mnt/data/panorama

  檢查目前解析到哪裡：  python config.py
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#: 覆寫資料集根目錄的環境變數名。pose_regions / roma_match 各自讀同一個變數名
#: 而不 import 本模組（維持可獨立匯入），改名時三處一併修改。
ENV_DATASETS = "ROMA_GEO_DATASETS"

#: 預設位置（未設環境變數時）
DEFAULT_DATASETS = os.path.join(PROJECT_ROOT, "datasets")

#: 連通性矩陣（粗篩輸出）來源的環境變數名。多個版本可並存於 `matrices/` 底下。
ENV_MATRICES = "ROMA_GEO_MATRICES"

#: 目前使用的版本目錄名。要換版本時改這裡，或設 ENV_MATRICES 指向任意路徑，
#: 或直接用 prepare_house_data.py 的 --oof_dir（優先權最高）。
MATRICES_VERSION = "cos_joint_l4"
DEFAULT_MATRICES_ROOT = os.path.join(PROJECT_ROOT, "matrices")


def datasets_root() -> str:
    """ERP 全景資料集根目錄，底下是 `<house>/.../*.jpg` 與 `*_HOTSPOT.json`。"""
    p = os.environ.get(ENV_DATASETS)
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return DEFAULT_DATASETS


def semantic_root() -> str:
    """門語意圖：`<datasets>/semantic/<taskid>/<image>_mask.png`。

    目前與全景圖置於同一棵目錄樹下。若語意圖改由其他位置提供，只需修改本函式，
    不必更動 semantic_doors/ 與 stitch_tool/ 的呼叫端。
    """
    return os.path.join(datasets_root(), "semantic")


def matrices_version_root() -> str:
    """連通性矩陣的版本目錄，底下是各矩陣變體的子目錄（`refined_matrices/` 等）。

    解析優先序：環境變數 `ROMA_GEO_MATRICES` > `matrices/<MATRICES_VERSION>/`。
    """
    p = os.environ.get(ENV_MATRICES)
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(DEFAULT_MATRICES_ROOT, MATRICES_VERSION)


def matrices_root(variant: str = "refined") -> str:
    """某一變體的矩陣目錄（`<house>.npy` / `<house>_images.json` / `<house>_doormap.npy`）。

    variant：`refined`（正式流程）或 `oof`（僅 covis_ranking_eval.py 使用）。
    版本目錄底下沒有子目錄時，回退為版本目錄本身。

    優先序：`prepare_house_data.py --oof_dir` > 環境變數 > 預設值。
    格式契約見 `matrices/README.md`。
    """
    base = matrices_version_root()
    sub = os.path.join(base, "%s_matrices" % variant)
    if os.path.isdir(sub):
        return sub
    return base          # 相容：.npy 直接放在版本目錄底下


def matrices_versions() -> list:
    """列出 `matrices/` 底下可用的版本目錄名。"""
    try:
        return sorted(d for d in os.listdir(DEFAULT_MATRICES_ROOT)
                      if os.path.isdir(os.path.join(DEFAULT_MATRICES_ROOT, d)))
    except OSError:
        return []


def describe() -> str:
    src = "環境變數 %s" % ENV_DATASETS if os.environ.get(ENV_DATASETS) else "預設（專案內 datasets/）"
    msrc = ("環境變數 %s" % ENV_MATRICES if os.environ.get(ENV_MATRICES)
            else "預設（matrices/%s）" % MATRICES_VERSION)
    d, s, m = datasets_root(), semantic_root(), matrices_root()
    n_npy = len([f for f in os.listdir(m) if f.endswith(".npy")]) if os.path.isdir(m) else 0
    variants = [x for x in ("refined", "oof")
                if os.path.isdir(os.path.join(matrices_version_root(), "%s_matrices" % x))]
    return (
        "datasets_root : %s\n"
        "  來源        : %s\n"
        "  存在        : %s\n"
        "semantic_root : %s\n"
        "  存在        : %s\n"
        "matrices_root : %s\n"
        "  來源        : %s\n"
        "  存在        : %s（%d 個 .npy）\n"
        "  變體        : %s\n"
        "  可用版本    : %s"
        % (d, src, os.path.isdir(d), s, os.path.isdir(s),
           m, msrc, os.path.isdir(m), n_npy,
           variants or "（無子目錄，.npy 直接置於版本目錄）",
           matrices_versions() or "（無）")
    )


if __name__ == "__main__":
    print(describe())
