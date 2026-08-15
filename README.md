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

輸出格式沿用既有的 per-pair JSON，欄位全文見 `pose_regions/README.md`。

## 兩種使用方式

| | 函式介面 | 批次管線 |
|---|---|---|
| 進入點 | `roma_geo.verify_pair()` / `verify_pairs()` | `environments/`、`pose_regions/` 的 CLI |
| 設定來源 | **只有傳入的參數** | 環境變數 + CLI 參數 |
| 對象 | 被其他程式呼叫、包成 API | 整批離線計算與評估 |
| 資料範圍 | 一對圖 | 整個資料集 |

`verify_pair()` 不讀 `ROMA_GEO_DATASETS` / `ROMA_GEO_MATRICES` / `ROMA_GEO_OUTPUT` /
`PAIR_THRESHOLD`；唯一會回落到環境變數的是快取目錄，且可用 `cache_dir=` 明確指定。
因此同一個行程內可以用不同設定並存，包成服務時不必擔心行程層狀態互相干擾。

## 檔案交界一覽

與其他子計畫、與批次執行環境之間**只透過檔案往來，不互相 import**。
每一項的位置都可設定：

| 方向 | 內容 | 位置 | 可設定 |
|---|---|---|---|
| 輸入 ← 上游粗篩 | 連通性矩陣 `<house>.npy` + `_images.json` | `matrices/<版本>/` | `ROMA_GEO_MATRICES` |
| 輸入 ← 子計畫一 | 門語意圖 `<image>_mask.png` | `<datasets>/semantic/` | 跟隨 `ROMA_GEO_DATASETS` |
| 輸入 | ERP 全景圖與 `*_HOTSPOT.json` | `datasets/` | `ROMA_GEO_DATASETS` |
| 產出 → 下游 | per-pair 幾何驗證 JSON | `Output/roma_regions/` | `ROMA_GEO_OUTPUT` |
| 產出 → 下游 | `_summary.csv`、`_ranking_eval.csv` | 同上 | 同上 |
| 產出 → 標註工具 | house JSON、語意門 JSON | `stitch_tool/data/`、`semantic/` | 工具本機狀態，不外露 |
| 內部暫存 | RoMa 匹配快取 `*.npz` | `roma_match/cache/` | `ROMA_GEO_CACHE` |

⚠ `*_HOTSPOT.json` 是**資料集自帶的 GT 標註**，僅供評估對照。**外來資料通常沒有這個檔**，
此時 `gt_*` / `same_room` 欄位為空值（不是 0），幾何輸出不受影響；
排序評估（`covis_ranking_eval.py`）需要 GT 才能執行，無 GT 時會略過並提示。

上游的格式契約、方向解碼與已知陷阱見 **[matrices/README.md](matrices/README.md)**
與 **[upstream_direction.py](upstream_direction.py)**。

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

`check_env.py` 實跑三項驗證：CUDA kernel、romav2 匯入、幾何核心 selftest。
**三項都過即代表這個環境可用**——判準是跑不跑得動，不是版本相等。

上表的版本是開發與驗證時所用的參考配置。`check_env.py` 會列出對照，
與本機不同的欄位標為 `DIFF`，但不影響結論;它在實跑失敗時是第一線線索。

## 測試

```powershell
..\venv\Scripts\python.exe tests\run_all.py
```

單一進入點，分三層：單元測試（無前提，數秒）、交付物檢查（掃描既有產出是否符合契約）、
端到端煙霧測試（需 GPU 與資料集）。缺少前提的層級回報略過而非失敗，因此在沒有 GPU 或
資料集的機器上仍可執行前兩層。離開碼 0 表示沒有任何一項失敗。

第二層可單獨用於**檢驗別處產生的資料**：收到其他機器、其他版本或上游的檔案時，
先確認格式相符再接。

```powershell
..\venv\Scripts\python.exe tests\check_deliverables.py --regions_dir <路徑>
```

各層涵蓋範圍與判準見 [`tests/README.md`](tests/README.md)。只用標準函式庫的 `unittest`，
不需要額外安裝測試框架。

## 路徑與門檻設定

**所有會隨部署環境改變的位置都由 `config.py` 解析，呼叫端不寫死路徑。**

| 項目 | 環境變數 | 預設 | 用途 |
|---|---|---|---|
| 全景資料集 | `ROMA_GEO_DATASETS` | `datasets/` | 上游輸入 |
| 門語意圖 | （跟隨資料集） | `<datasets>/semantic/` | 子計畫一輸入 |
| 連通性矩陣 | `ROMA_GEO_MATRICES` | `matrices/<版本>/` | **上游交界點**：粗篩模型的輸出 |
| 挑配對門檻 | `ROMA_GEO_PAIR_THRESHOLD` | `config.PAIR_THRESHOLD` | 決定哪些配對要算 |
| 產出根目錄 | `ROMA_GEO_OUTPUT` | `Output/` | 幾何驗證產物（批次於其他機器執行時指向掛載位置） |
| 匹配快取 | `ROMA_GEO_CACHE` | `roma_match/cache/` | 暫存、可重新產生（全量數百 MB） |

```powershell
$env:ROMA_GEO_DATASETS = "D:\panorama_data"     # 換路徑不必改任何程式碼
```
```bash
python config.py     # 檢查以上每一項目前解析到哪裡、存不存在、各有幾個檔
```

子計畫一的門語意圖經由 `config.semantic_root()` 解析；若日後改由其他位置提供，僅需修改該函式。
`stitch_tool/data/` 與 `stitch_tool/state/` 未納入上表：前者由 `prepare_house_data.py`
產生、後者為標註進度，兩者皆屬工具的本機狀態。

## 完整鏈條

從上游連通輸出到可瀏覽／視覺化，各步驟的方法與產出：

1. 上游連通模型對每間房的輸出（每對全景影像的連通機率 ＋ 對方所在的視角方位）
   → `ingest_connections.py` → 連通矩陣 `matrices/<版本>/`
2. 連通矩陣 → `environments/select_pairs.py` → 待幾何驗證的配對清單（連通機率 ≥ 0.5）
3. 配對清單 → `environments/batch_match.py`（RoMa 密集匹配）→ 密集對應點 correspondence
4. correspondence → `pose_regions/batch_pose_from_cache.py` → 共視區域的幾何 ＋ 通道方位
   （＋ `pose_regions/covis_ranking_eval.py` → 排序用的內點統計；完整欄位見 `pose_regions/README.md`）
5. 以上整合 → `stitch_tool/prepare_house_data.py` → 每間房的瀏覽資料
   （＋ `stitch_tool/prepare_semantic_doors.py` → 門語意輔助）
6. → `stitch_tool/pack_bundle.py` → 自足的房子產出包（含影像，供工具離線載入、可手動選房）
7. → 啟動伺服器 `stitch_tool/server.py` → 視覺化 correspondence、共識區域、通道方位

交付給下游時：步驟 5 的每間房資料 → `export_public.py` → 對外子集（房間連接圖 ＋ 邊方向
＋ 方位來源 ＋ 幾何證據強度，加 `--with_regions` 併入共視區域），評估與排序用的內部欄位不外流。

單獨驗一對：`roma_geo.verify_pair(img_a, img_b)` → 共視區域 ＋ 通道方位 ＋ correspondence（＝步驟 3–4 的單對版）。

方向規則：存在足夠 correspondence 時以其幾何驗證結果為準；correspondence 不足時退回使用 coarse filter 的輸出。

### 挑配對門檻

矩陣 ch0 的分數 ≥ 門檻的配對才送進 RoMa。三種設法，優先序由高到低：

```bash
python environments/select_pairs.py --threshold 0.0      # 單次執行
```
```powershell
$env:ROMA_GEO_PAIR_THRESHOLD = "0.0"                     # 本次工作階段
```

或改 `config.py` 的 `PAIR_THRESHOLD`（專案預設，現值 0.0；`candidate_val` 的 ch0 是 logit，連通機率 ≥ 0.5 ⇔ logit ≥ 0.0）。

單位由矩陣決定，不做 sigmoid，**換上游來源必須重新校準**。不確定新來源的尺度時，
`select_pairs.py --inspect` 會列出分數範圍與各門檻選出的配對數，不寫檔。

格式契約與已知陷阱見 **[matrices/README.md](matrices/README.md)**。


## 現況

| 模組 | 內容 |
|---|---|
| `roma_geo.py` | **對外入口**：`verify_pair()` / `verify_pairs()` / `to_public()` |
| `ingest_connections.py` | 上游連通輸出（每對機率＋視角方位 JSON）→ 連通矩陣的低耦合 adapter |
| `export_public.py` | 每間房的內部資料 → 對外交付子集（含輸出前的內部欄位自檢） |
| `config.py` | 路徑與門檻的單一來源（六項，見上表） |
| `core_roma_match/` | 匹配核心（RoMaScan pipeline） |
| `roma_match/` | `RoMaMatcher` 封裝、多 GPU 平行匹配、RoMa 聚合分數快取 |
| `pose_regions/` | 幾何驗證（純 numpy，可獨立匯入、匯入零副作用） |
| `semantic_doors/` | 子計畫一門語意圖的門抽取 + 參考快取 |
| `stitch_tool/` | 標註/檢視工具（面向未知資料：`pack_bundle.py` 打包自足產出包，`server.py` 讀包根目錄，可手動選房、GT 隨包內是否含 HOTSPOT） |
| `Output/roma_regions/` | 逐 pair 共視區域 JSON + 排序/方位角 CSV |
| `matrices/<版本>/` | 上游粗篩的連通性矩陣（`prepare_house_data.py` 的輸入，可換版本） |
| `environments/` | 建置、環境驗收、跨環境一致性、計時基準、**挑配對**（`select_pairs.py`）、批次匹配 |
| `tests/` | 測試總入口（`run_all.py`）：單元測試、交付物契約檢查、端到端煙霧測試 |

對外 API 的 I/O 規格與各項工具的用法見 **[API.md](API.md)**。
GT（連通標註、通道方位、同房判定）一律讀 `datasets/` 底下的 `*_HOTSPOT.json`。
外來資料沒有這個檔時，GT 欄位為空值，幾何輸出不受影響（見「檔案交界一覽」）。
