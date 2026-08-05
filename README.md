# Roma_Geo — 幾何驗證模組

多房間室內全景影像的幾何驗證模組（子計畫二）。輸入一對 ERP 全景圖，輸出兩張圖之間
經幾何驗證的共視區域與通道方位角，供其他子計畫與標註工具使用。

## 模組邊界

```
輸入   (img_a, img_b)                      ← 由連通性粗篩挑出的候選配對
      [+ 門語意圖 <image>_mask.png]        ← 選配，直接讀檔

輸出   regions[].a.polygon / .b.polygon    ← 共識區域（correspondence 的視覺形式）
      regions[].area_frac / az_range
      pose.az_ab_deg / az_ba_deg           ← 通道方位角
      inlier_rate / conf_used              ← 可信度
      flags.no_covisibility                ← 無共視證據

內部   R、t、cam_b_in_a、inliers[]、quality{}、RoMa cache npz
```

R/t 僅供內部折算方位角使用，不列入對外欄位。
RoMa 匹配快取（`*.npz`）為暫存檔，可重新產生，不屬於交付內容。

輸出格式沿用既有的 per-pair JSON，`schema_version: wp1-step3-v2`，
schema 全文見 `pose_regions/README.md`。

## 環境

目標配置：

| 項目 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.10.0+cu130 |
| numpy | 1.26.4 |
| opencv-python | 4.11.0 |
| scikit-learn | 1.8.0 |
| pillow | 12.0.0 |


建置與驗收：

```powershell
cd environments
.\setup_env.ps1          # 建 venv + 裝套件 + 自動跑 check_env.py
```

```powershell
..\venv\Scripts\python.exe environments\check_env.py     # 單獨重驗
```

`check_env.py` ：逐項比對版本、確認 wheel 的 `arch_list` 含本機 GPU 的 arch、
實跑三個驗證（CUDA kernel、romav2 匯入、幾何核心 selftest）。同一份腳本可在其他機器
執行以確認環境一致：arch 一項顯示各自 GPU 的 compute capability，其餘欄位應完全相同。

## 資料集路徑
資料集的擺放位置依部署環境而異，所有讀取資料集的地方統一經由 `config.py` 解析。

```powershell
$env:ROMA_GEO_DATASETS = "D:\panorama_data"     # 換路徑不必改任何程式碼
```
```bash
python config.py     # 檢查目前解析到哪裡、存不存在
```

解析優先序 = 環境變數 `ROMA_GEO_DATASETS` > 專案根目錄的 `datasets/`。
子計畫一的門語意圖走 `config.semantic_root()`（目前是 `<datasets>/semantic/`）；
若日後語意圖改由別處提供，只要改那一個函式。


## 現況

| 模組 | 內容 |
|---|---|
| `roma_geo.py` | **對外入口**：`verify_pair()` / `verify_pairs()` / `to_public()` |
| `config.py` | 資料集路徑單一來源 |
| `core_roma_match/` | 匹配核心（RoMaScan pipeline，9 個 .py） |
| `roma_match/` | `RoMaMatcher` 封裝（matcher.py + utils_cubemap.py） |
| `pose_regions/` | 幾何驗證（純 numpy，可獨立匯入、匯入零副作用） |
| `semantic_doors/` | 子計畫一門語意圖的門抽取 + 參考快取 |
| `stitch_tool/` | 標註工具介面 |
| `Output/roma_regions/` | 共視區域 JSON 7,069 份 + 排序/方位角 CSV |
| `matrices/<版本>/` | 上游粗篩的連通性矩陣（`prepare_house_data.py` 的輸入，可換版本） |
| `environments/` | 建置、驗收、煙霧測試、計時基準 |

對外 API 的 I/O 規格與各項工具的用法見 **[API.md](API.md)**。
GT（連通標註、通道方位、同房判定）一律讀 `datasets/` 底下的 `*_HOTSPOT.json`，
