# API 與工具規格

本文件描述 Roma_Geo 對外提供的程式介面（I/O 規格）與隨附工具的用法。
模組總覽與環境需求見 [README.md](README.md)。

---

## 1. 對外 API

唯一入口為根目錄的 `roma_geo` 模組。

```python
from roma_geo import verify_pair, verify_pairs, to_public, get_matcher
```

### 1.1 `verify_pair(img_a, img_b, **kwargs) -> dict`

對一組 ERP 全景圖做幾何驗證，回傳完整結果。

**輸入**

| 參數 | 型別 | 預設 | 說明 |
|---|---|---|---|
| `img_a`, `img_b` | `str` | 必填 | ERP 全景圖路徑。兩張圖尺寸可不同 |
| `matcher` | `RoMaMatcher` | `None` | 重用既有 matcher；`None` 時使用模組層單例 |
| `mode` | `str` | `"direct"` | `direct`（生產路徑）或 `core`（RoMaScan 分塊，約貴 5.6 倍） |
| `model_type` | `str` | `"roma_v2"` | 匹配模型版本。目前只支援 `roma_v2` |
| `device` | `str` | `None` | `None` 時自動選 cuda／cpu |
| `use_cache` | `bool` | `True` | 是否讀寫匹配快取的 npz。純加速，不影響結果 |
| `cache_dir` | `str` | `None` | 快取目錄。`None` 時依環境變數 `ROMA_GEO_CACHE` > `roma_match/cache/`。本函式除此之外不讀任何環境設定 |
| `house` | `str` | `None` | 房子名；`None` 時從路徑推導 |
| `keep_inliers` | `bool` | `True` | `False` 時不回傳 `inliers`（可省下 JSON 體積大宗） |

另可覆寫下列演算法參數（一般不需要，見 §1.5）：
`conf_min` `retry_conf` `max_pts` `thresh_deg` `iters` `seed` `sampler` `lat_max`。
傳入未定義的參數會 `raise TypeError`。

**輸出**：`dict`。欄位全文見
[`pose_regions/README.md`](pose_regions/README.md) 的「per-pair JSON schema」。主要欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `schema_version` | `str` | 產出格式識別字串，供批次續跑判斷是否需重算 |
| `house` / `img_a` / `img_b` | `str` | 房子名與兩張圖的檔名（不含副檔名） |
| `img_size_a` / `img_size_b` | `[W, H]` | 原圖尺寸，**per-house 不同**（5376×2688 ~ 8192×4096） |
| `params` | `dict` | 實際使用的演算法參數 |
| `n` | `dict` | `raw` / `conf` / `used` / `inlier` / `lat_drop` 各階段點數 |
| `conf_used` | `float\|null` | 自適應後實際採用的 conf 門檻（0.65／0.3／null） |
| `inlier_rate` | `float\|null` | `n.inlier / n.used` |
| `pose` | `dict\|null` | `R` `t` `cam_b_in_a` `az_ab_deg` `az_ba_deg` `rot_angle_deg` `baseline_txz` |
| `quality` | `dict` | 對極殘差、純旋轉殘差、方位楔形寬度與覆蓋度 |
| `flags` | `dict` | `low_matches` `no_covisibility` `degenerate_rot_only` `doorway_wedge` |
| `regions` | `list` | 共視區域，依點數降冪 |
| `inliers` | `list` | 內點明細 `[x_a, y_a, x_b, y_b, conf]`，最多 2000 點 |
| `timing` | `dict` | `match_s` / `verify_s` / `total_s` |

`pose` 為 `null` 且 `flags.no_covisibility` 為 `true` 代表找不到足夠的共視證據，
不等於「兩張圖不連通」——它們可能相連但視野不重疊。

**方位角規約**：`az(x) = ((x / W) * 360 - 180) mod 360`，0° = 影像中央，±180° = 接縫。
`pose.az_ab_deg` 的範圍是 `(-180, 180]`。

**位姿規約**：`x_b = R · x_a + t`，`t` 為單位向量（**尺度未知**），
`cam_b_in_a = -Rᵀt` = B 相機在 A 座標系中的方向。

### 1.2 `to_public(res) -> dict`

把 `verify_pair()` 的結果裁成對外契約。捨去 `R` / `t` / `cam_b_in_a` / `inliers` /
`quality` / `params` / `n` / `flags` 等內部量。

```json
{
  "schema_version": "wp1-step3-v2",
  "house": "DollhouseTask_58553",
  "img_a": "08ddb590-a6df-409d-850f-e45f59efa7a0",
  "img_b": "08ddb590-ad9f-4647-8dab-ccfd0b14c8da",
  "img_size_a": [8192, 4096],
  "img_size_b": [8192, 4096],
  "conf_used": 0.65,
  "inlier_rate": 0.6769,
  "az_ab_deg": 178.6,
  "az_ba_deg": -174.61,
  "no_covisibility": false,
  "regions": [
    {
      "n": 499,
      "frac_of_inliers": 0.841,
      "a": { "polygon": [[3318.8, 2907.9], [3370.7, 1370.0], "…"],
             "area_frac": 0.0475, "az_range": [-34.2, 9.0] },
      "b": { "polygon": [[7859.2, 1676.8], [7879.7, 1600.0], "…"],
             "area_frac": 0.015,  "az_range": [165.4, 192.6] }
    }
  ]
}
```

`regions[].a` / `.b` 為同一塊實體空間分別在 A、B 兩張圖上的凸包多邊形，
座標為**原圖 ERP 像素**。跨接縫的區域其 `x` 可能超出 `[0, W)` 以保持多邊形連續，
繪製時自行取模。`area_frac` 為多邊形面積佔全圖比例（ERP 像素面積，極區有失真）。

### 1.3 `verify_pairs(pairs, progress=False, **kwargs)`

多組圖的產生器版本，全程重用同一個 matcher。

```python
for res in verify_pairs([(a1, b1), (a2, b2)], progress=True):
    ...
```

建構 matcher 需載入 torch、CUDA 與模型權重（約 6.4 秒），多對處理用這個或自行傳入
`matcher=` 以重用。逐對 `yield`。

### 1.4 `get_matcher(model_type, device, use_cache, cache_dir) -> RoMaMatcher`

取得模組層共用的 matcher 單例。`conf_threshold` 固定 0.1，由 `analyze_pair`
自行做自適應分層（0.65 → 0.3）。

單例以「參數組合」為鍵（含 `cache_dir`），因此同一行程內可並存多組設定，
包成服務時不必擔心行程層狀態互相干擾。

### 1.5 演算法參數

`roma_geo.DEFAULTS` 必須與 `pose_regions/batch_pose_from_cache.py` 的 argparse
預設逐項相同 —— `Output/roma_regions/` 的既有產出皆以這組參數產生。

| 參數 | 值 | 說明 |
|---|---|---|
| `conf_min` | 0.65 | 主要 conf 門檻 |
| `retry_conf` | 0.3 | 點數不足時的降級門檻 |
| `max_pts` | 4000 | 超過即抽樣，加速 RANSAC |
| `thresh_deg` | 0.5 | 對極內點角度門檻 |
| `iters` | 2000 | RANSAC 迭代數 |
| `seed` | 0 | RANSAC 亂數種子 |
| `sampler` | `stratified` | conf 漸進池 + 方位分桶 |
| `lat_max` | 60.0 | 天底／天頂前濾角 |

`environments/smoke_test.py` 內含這組參數與參考 JSON 的逐項比對。

### 1.6 CLI

```bash
python roma_geo.py a.jpg b.jpg --public --out result.json
```

`--mode` `--conf_min` `--no_cache` `--public` `--out`。省略 `--out` 時輸出到 stdout。

### 1.7 效能

單對端到端以「RoMa 匹配」為主、「幾何驗證」佔比很小；模型建構為一次性成本，
多對處理務必重用 matcher。驗證時間呈雙峰：有位姿的對較久，無共視證據的對提早退出。

逐項數字與外推耗時見 `environments/timing_torch210_n120.md`（由
`environments/local_timing_bench.py` 產生，換機器請重跑）。

---

## 2. 工具

所有指令以專案根目錄為工作目錄，`<py>` 代表 `venv\Scripts\python.exe`。

### 2.1 標註工具（`stitch_tool/`）

瀏覽器介面，用於檢視配對、共視區域遮罩、裁切對與通道方位標示。

```bash
<py> stitch_tool/prepare_house_data.py      # 產生 data/*.json（先跑）
```

```bash
<py> stitch_tool/server.py --port 8010      # 啟動，開 http://localhost:8010
```

HTTP 介面：

| 路由 | 說明 |
|---|---|
| `GET /api/houses` | 房子清單 |
| `GET /api/house/<house>` | 單一房子的配對與方位資料 |
| `GET /api/regions/<house>?a=<圖A>&b=<圖B>` | 共視區域 JSON（命中反序檔時回 `swap: true`） |
| `GET /api/semantic_doors/<house>` | 子計畫一語意門抽取結果（無語意圖回 404） |
| `GET /api/state/<house>` · `POST /api/state/<house>` | 人工標註進度存讀 |
| `POST /api/roma_match/<house>` | 即時 RoMa 比對（延遲載入模型） |
| `GET /images/<house>/<檔名>` | ERP 影像 |

詳細操作見 [`stitch_tool/使用說明.md`](stitch_tool/使用說明.md)。

### 2.2 `stitch_tool/prepare_house_data.py`

把上游的連通性矩陣整理成前端可讀的 house JSON。

- 輸入：`matrices/<版本>/`（見 [`matrices/README.md`](matrices/README.md)）、
  `datasets/` 的 ERP 圖與 HOTSPOT、`Output/roma_regions/` 的兩份 CSV
- 輸出：`stitch_tool/data/<house>.json` 與 `_index.json`
- 換矩陣版本：`--oof_dir` > 環境變數 `ROMA_GEO_MATRICES` > `config.MATRICES_VERSION`
- 有位姿的 pair 會以幾何驗證的 `az_ab/az_ba` 覆寫粗篩方位，並記於 `az_source` 欄位

### 2.3 `stitch_tool/prepare_semantic_doors.py`

把子計畫一的門語意圖抽取結果轉存成工具可讀的格式。

```bash
<py> stitch_tool/prepare_semantic_doors.py             # 讀現成快取轉檔，秒級
<py> stitch_tool/prepare_semantic_doors.py --refresh   # 重跑抽取，約 25 分鐘
```

輸出 `stitch_tool/semantic/<house>.json`。只有具語意圖的房子會產出，其餘前端顯示 404。

### 2.4 `pose_regions/batch_pose_from_cache.py`

全量批次：讀 RoMa 匹配快取，產生 per-pair JSON 與彙總 CSV。

```bash
<py> pose_regions/batch_pose_from_cache.py --out Output/roma_regions
```

輸入為 `roma_match/cache/` 底下符合
`{house}_{imgA}_{imgB}_direct_roma_v2_std0.1_face512.npz` 命名的檔案，其餘略過。
GT（連通標註、通道方位、同房判定）讀 `datasets/` 的 `*_HOTSPOT.json`。

### 2.5 `pose_regions/estimate_relative_pose.py`

單對診斷工具，輸出內點疊圖與俯視圖。

```bash
<py> pose_regions/estimate_relative_pose.py --cache_npz <npz> --out pose_out
<py> pose_regions/estimate_relative_pose.py --selftest     # 幾何核心自我驗證
```

### 2.6 `pose_regions/render_region_overlay.py`

把共視區域 JSON 畫成疊圖。

### 2.7 環境工具（`environments/`）

| 腳本 | 用途 |
|---|---|
| `setup_env.ps1` | 建立 venv 並安裝全部相依，結束時自動驗收 |
| `check_env.py` | 三項實跑驗證（CUDA kernel／romav2 匯入／幾何 selftest）決定通過與否；版本對照與 GPU arch 涵蓋僅供參考 |
| `smoke_test.py` | 端到端功能測試，與參考 JSON 逐欄對照 |
| `local_timing_bench.py` | 單對耗時基準，決定性抽樣，可跨環境比較 |

用法見 [`environments/README.md`](environments/README.md)。
