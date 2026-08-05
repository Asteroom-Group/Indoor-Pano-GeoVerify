"""
[腳本用途]
環境對齊驗收：把本機實際版本與目標配置逐項比對，並做三個
實跑驗證（CUDA kernel、romav2 匯入、幾何核心 selftest）。

[用法]
  ..\venv\Scripts\python.exe environments\check_env.py

驗收標準：TARGET 表格全部 OK，且三個實跑驗證全部 PASS。
同一份腳本可在其他機器執行以確認環境一致：arch 一項顯示各自 GPU 的
compute capability，其餘欄位應完全相同。
"""
import platform
import subprocess
import sys
from pathlib import Path

# 目標配置
TARGET = {
    "python": "3.11.0",
    "torch": "2.10.0+cu130",
    "cuda": "13.0",
    "numpy": "1.26.4",
    "cv2": "4.11.0",
    "sklearn": "1.8.0",
    "PIL": "12.0.0",
}

OK, BAD, WARN = "OK  ", "MISMATCH", "WARN"


def _ver(mod_name, attr="__version__"):
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, attr))
    except Exception as e:  # noqa: BLE001 - 缺套件也要照常印出來
        return f"<{type(e).__name__}>"


def row(label, actual, target, minor_only=False):
    """minor_only：只比 major.minor（Python patch 版差異不影響數值結果，不當成落差）。"""
    if minor_only:
        hit = actual.split(".")[:2] == target.split(".")[:2]
        status = OK if hit else BAD
    else:
        hit = actual == target
        status = OK if hit else BAD
    print(f"  [{status:8}] {label:<12} {actual:<20} (目標 {target})")
    return hit


def main():
    print("=" * 68)
    print("  Roma_Geo 環境對齊驗收")
    print("=" * 68)

    print(f"\nOS          : {platform.system()} {platform.release()} {platform.machine()}")

    print("\n--- 版本對齊 ---")
    allok = True
    allok &= row("python", platform.python_version(), TARGET["python"], minor_only=True)

    try:
        import torch
    except ImportError:
        print("  [FATAL   ] torch 未安裝 —— 先跑 setup_env.ps1")
        return 1

    allok &= row("torch", torch.__version__, TARGET["torch"])
    allok &= row("cuda build", str(torch.version.cuda), TARGET["cuda"])
    allok &= row("numpy", _ver("numpy"), TARGET["numpy"])
    allok &= row("opencv", _ver("cv2"), TARGET["cv2"])
    allok &= row("sklearn", _ver("sklearn"), TARGET["sklearn"])
    allok &= row("pillow", _ver("PIL"), TARGET["PIL"])

    print("\n--- GPU ---")
    if not torch.cuda.is_available():
        print("  [FATAL   ] torch.cuda.is_available() = False")
        return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    print(f"  GPU         : {name}  capability {cap} -> {sm}")
    print(f"  arch_list   : {arch_list}")

    # wheel 未涵蓋本機 GPU 的 arch 時，is_available() 仍回 True，但一發 kernel
    # 就出 no kernel image。
    #
    # 不做字面比對：CUDA 的二進位相容規則是「為 sm_X.y 編的 cubin 可在 sm_X.z
    # （z >= y）的裝置上執行」，同 major 且 minor 不高於裝置即可。
    covering = []
    for a in arch_list:
        if not a.startswith("sm_"):
            continue
        d = a[3:]
        if not d.isdigit():
            continue
        maj, mnr = int(d[:-1]), int(d[-1])   # sm_89 -> (8,9) · sm_120 -> (12,0)
        if maj == cap[0] and mnr <= cap[1]:
            covering.append(a)

    if sm in arch_list:
        print(f"  [{OK:8}] wheel 直接含 {sm}")
    elif covering:
        print(f"  [{OK:8}] wheel 未直接含 {sm}，但 {covering[-1]} 的 cubin 向上相容（同 major，minor 較低）")
    else:
        print(f"  [FATAL   ] wheel 不涵蓋 {sm} —— 這顆 wheel 在這張卡上跑不動")
        allok = False

    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        print(f"  driver      : {drv}")
    except Exception as e:  # noqa: BLE001
        print(f"  driver      : <查詢失敗 {type(e).__name__}>")

    print("\n--- 實跑驗證 ---")
    passed = 0

    # 1) 真的發一次 CUDA kernel（is_available() 為 True 但 kernel 炸掉是本次對齊的核心風險）
    try:
        a = torch.randn(512, 512, device="cuda")
        val = float((a @ a).sum().item())
        torch.cuda.synchronize()
        print(f"  [PASS    ] CUDA kernel 實跑（512x512 matmul, sum={val:.1f}）")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL    ] CUDA kernel: {type(e).__name__}: {e}")

    # 2) romav2 匯入
    try:
        import romav2  # noqa: F401
        print("  [PASS    ] romav2 匯入")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL    ] romav2 匯入: {type(e).__name__}: {e}")

    # 3) 幾何核心 selftest（純 numpy，合成資料，不吃 GPU）
    root = Path(__file__).resolve().parent.parent
    selftest = root / "pose_regions" / "estimate_relative_pose.py"
    if selftest.exists():
        r = subprocess.run(
            [sys.executable, str(selftest), "--selftest"],
            capture_output=True, text=True, cwd=str(root), timeout=300,
        )
        tail = (r.stdout or r.stderr).strip().splitlines()
        if r.returncode == 0:
            print("  [PASS    ] pose_regions --selftest")
            for line in tail[-3:]:
                print(f"             {line}")
            passed += 1
        else:
            print(f"  [FAIL    ] pose_regions --selftest (exit {r.returncode})")
            for line in tail[-8:]:
                print(f"             {line}")
    else:
        print(f"  [SKIP    ] pose_regions 尚未搬入 ({selftest})")

    print("\n" + "=" * 68)
    verdict = "全部通過" if (allok and passed >= 3) else "有項目未通過（見上）"
    print(f"  結果：版本對齊 {'一致' if allok else '有落差'} · 實跑 {passed}/3 · {verdict}")
    print("=" * 68)
    return 0 if (allok and passed >= 3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
