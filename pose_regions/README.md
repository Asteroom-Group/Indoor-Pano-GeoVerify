# pose_regions — 相對位姿與共視區域(RoMa 匹配的幾何驗證層)

從 RoMa 的 ERP 全景匹配點估計兩張圖之間的相對位姿(R、t 方向;尺度未知),
並以對極幾何過濾不合理匹配、萃取共視區域多邊形。不需要 GT pose / 深度 / layout。
WP1(RoMaV2 有意義 correspondence,供子計畫三拼接)以本資料夾為核心逐步擴充。

> **2026-07-23 模組化改名:`pose_experiment/` → `pose_regions/`**(git mv,歷史保留)。
> 現在是可直接匯入的套件 — Kaggle 或即時 RoMa 匹配流程皆可
> `from pose_regions import analyze_pair`(核心純 numpy、匯入零副作用、cwd 無關);
> CLI 用法全部不變(只是資料夾名換掉)。公開 API 見 `__init__.py`。
> 歷史文件/報告中的 `pose_experiment/` 路徑一律讀作 `pose_regions/`。

## 檔案

| 檔案 | 用途 |
|---|---|
| `estimate_relative_pose.py` | 單對主程式:三種輸入 → RANSAC essential → R/t、內點 overlay、top-down 圖 |
| `batch_pose_from_cache.py` | **Step 2 全量批次**:7,069 cache pair → per-pair JSON + `_summary.csv` + 報告 |
| `batch_pose_report.md` | 全量批次自動報告(耗時、inlier 率分佈、GT 方位交叉誤差、旗標統計) |
| `spike_cache_consistency.py` | WP1 Step 1 spike:驗證 cache npz 座標約定(可重跑) |
| `spike_cache_consistency_report.md` | spike 執行結果(2026-07-22,總判定 PASS) |
| `covis_ranking_eval.py` | 共視/幾何訊號當**連通性排序分數**的判別力評估;產生 `_ranking_eval.csv` |
| `pose_out_cache_demo/` | `--cache_npz` 端到端範例輸出(不入 git;重生:見上方 A 用法) |
| `covis_ranking_eval.py` | 共視/幾何訊號當**連通性排序分數**的判別力與上限評估(見下方章節) |
| `covis_ranking_report.md` | 上述評估結果(2026-07-26) |

## 三種輸入

```bash
# A) 快取 npz(推薦;免 GPU、免匯出。圖自動從 datasets/ 找,--img_a/--img_b 可省略)
venv/Scripts/python.exe pose_regions/estimate_relative_pose.py \
    --cache_npz roma_match/cache/DollhouseTask_..._face512.npz --out pose_out

# B) 單對 live 跑 RoMa(需 GPU venv)
python estimate_relative_pose.py --match --img_a a.jpg --img_b b.jpg --mode core --conf 0.65

# C) export_correspondences.py 匯出的 CSV
python estimate_relative_pose.py --csv a_to_b.csv --img_a a.jpg --img_b b.jpg

# 幾何核心自我驗證(合成資料;目前 PASS)
python estimate_relative_pose.py --selftest
```

`--cache_npz` 補充:
- `--img_a/--img_b` 給反了會自動交換座標(同 `roma_match/matcher.py` 讀快取的 is_swapped 行為),R/t 隨之取逆(已驗證:R 恰為轉置)。
- `--conf` 未指定時預設 **0.65**(cache 底床 0.1 的低段大多是雜訊)。

## cache npz 座標約定(WP1 Step 1 已驗證,2026-07-22)

`roma_match/cache/` 共 **7,069 檔**,檔名全數符合:

```
{house}_{imgA}_{imgB}_direct_roma_v2_std0.1_face512.npz
```

- 內容:`matches` (K,5) = **(x_a, y_a, x_b, y_b, conf)**,座標 = **原圖 ERP 像素**(A 圖前兩欄、B 圖後兩欄)。
- 座標為**連續值**(RoMaV2 內部 512×512 warp 雙線性取樣後放大回原圖;非格點量化)。
- conf 底床 **0.1**(存完整結果供不同門檻共享;上層讀取時再過濾)。
- **原圖尺寸依房子而異**(7 種:5376×2688 ~ 8192×4096)——批次處理(Step 2)換算球面射線時必須用**該房實際 W×H**,不能寫死 8192×4096。

### 驗證方法與結果(`spike_cache_consistency.py`)

抽 3 對(3 棟房、字母序首/中/尾、各取 conf≥0.65 點數最多 pair),
「cache 讀出」vs「同對 live 重跑 RoMa(繞過快取)」比對。
RoMaV2 的 `sample()` 是隨機抽樣(5000 點/方向),兩次 draw 不會逐點相同,故比的是**約定與幾何**:

1. **range/scale**:兩邊座標皆在 [0,W]×[0,H],max 比值 0.99~1.02(若 cache 是 512 處理空間,比值會是 ~16 倍)。
2. **NN 距離**:cache→live 最近鄰中位數 11~31 px(8192 寬圖;同約定的兩次隨機 draw 就該是這個量級,約定錯會是數千 px)。
3. **對極交叉驗證**:cache 點 fit 的 E 拿去算 live 點內點率(反向亦然),與自身內點率幾乎相同(例:0.510→0.514)。
4. **全量掃描**:7,069 檔座標全數落在該房原圖範圍內,超界 0 檔。

**總判定 PASS** — 明細見 `spike_cache_consistency_report.md`。
程式碼佐證:私有核心 `matchers.py` 的 `to_pixel_coordinates(matches, ha, wa, hb, wb)` 收的就是原圖尺寸。

## RANSAC 取樣策略(`--sampler`,單對與批次共用)

8 點法每次迭代抽 8 點 minimal sample(E 只有 5 自由度,存在 5 點法,但線性 8 點法在點數
充足時是標準務實選擇;輸入是單位球面射線,天然良態,不需 Hartley 正規化)。兩種取樣:

- **`stratified`(預設)**:(1) **conf 漸進池**(PROSAC-lite)——點按 RoMa conf 降冪排序,
  取樣池從 top 25% 逐塊(每 256 迭代)放寬到 100%,高 conf 點先被抽到;
  (2) **方位分桶**——每個 sample 的 8 點在 A 側方位 12 桶(30°)間輪替,提升空間分散度/
  條件數;只在「有點的桶」之間輪替,門洞 pair 支撐窄不會被強迫散開。
  conf 只影響「先抽誰」;內點判定與最終 refit 都對全部點做,不被 conf 偏見污染。
- **`uniform`**:完全均勻隨機(對照/重現舊數據用)。

兩種取樣結束後都做 **LO 式迭代 refit**:全內點重擬合 → 重算內點,循環至內點集不再變大
(只在變好時接受,最多 3 輪)。

A/B(170 對,分層 hi/mid/lo inlier 率,LO refit 後):總耗時無差;內點數 hi 帶 +2、
mid/lo 帶 −10~−16(uniform 的 max-consensus 會多吸門檻邊緣點);**外部精度(az 誤差
vs GT hotspot)stratified 明顯較好:中位 11.3° vs 13.8°、p90 51.5° vs 63.0°** ——
內點「數」是 RANSAC 自家目標,az 是外部檢核,以後者為準。conf 收益有限的原因:
conf≥0.65 門檻已先吃掉大部分 conf 訊號;分桶/conf 池的效果主要反映在方向精度。
selftest 兩種取樣都 PASS。

## nadir 前濾(`--lat_filter`,批次限定;2026-07-23 A/B 實驗)

天底腳架帶(lat<−60°)在每張全景長一樣 → 系統性跨圖假匹配,過對極驗證但無語意
(baseline 主區污染:0.65 級 6.8% / 0.3 撈回級 23.2%)。`--lat_filter 60` 在 RANSAC 前
丟掉「任一側 |lat|>60°」的匹配點(天底+天頂對稱處理),丟點數記入 JSON `n.lat_drop`、
參數記入 `params.lat_max`。全量 A/B(`nadir_ab_report.md`):az 誤差不退反進(0.3 級
中位 19.45°→14.47°、跨房 p90 77°→54°)、nadir 主區歸零、假 degenerate 減半(343→166,
腳架同位假「純旋轉」)、inlier 分離 4.2×→6.8×;覆蓋率 54.9%→47.3%,流失者追查多為
假證據 pair(誠實拒絕)。**已採用:`--lat_filter 60` 為預設**(設 0 關閉=舊行為),
schema 升 `wp1-step3-v2`,正式 `Output/roma_regions/` 已全量重生(3,345 對位姿,
與實驗跑逐位一致;60621 示範 pair 完好);實驗對比目錄已刪除(可由本 flag 重生)。

## 輸出與判讀

- `inliers_overlay.jpg`:A|B 並排,綠=對極內點、紅=外點(半透明)。
- `topdown.png`:內點三角化俯視散點 + 兩相機位置(尺度自由)。
- stdout:內點率、R、t(單位向量)、`median rot-only err`
  (內點若能被純旋轉解釋 → 疑似近零基線,t 方向不可靠,會印警告)。

## Step 2:全量批次 + 品質指標(`batch_pose_from_cache.py`)

```bash
venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py             # 全量 7,069 對
venv/Scripts/python.exe pose_regions/batch_pose_from_cache.py --limit 20  # 冒煙
# --resume 跳過已有 JSON 續跑;--conf/--max_pts/--thresh_deg/--iters 可調
```

輸出三份:
1. `Output/roma_regions/{house}_{imgA}_{imgB}.json` — per-pair 結果(下方 schema;**不入 git**,約 20 分鐘可重生)
2. `Output/roma_regions/_summary.csv` — 每 pair 一列指標總表(入 git,供驗收與出圖)
3. `batch_pose_report.md` — 耗時與分佈自動報告

### 自適應 conf(Step 3 補,`--retry_conf`,預設 0.3)

基準 conf 0.65 需 ≥30 點才進 RANSAC;**點數 <30 或內點 <8 時自動降到 0.3 重試**
(需 ≥8 點),實際門檻記入 `conf_used`(0.65 / 0.3 / null)。兩級都湊不出 8 個內點
才標 `no_covisibility`。動機:穿門窄楔形 pair 在 0.65 常只剩 <20 點(位姿覆蓋率是
拼接可用性的關鍵)。

實測(2026-07-22 backfill):位姿覆蓋率 38.6% → **54.9%**;0.3 級撈回 1,681 對
(GT 連通 290、其中跨房 202 = 門洞 case 主要受益者)。**撈回層品質重尾**:az 誤差
中位 19.4° 但 p90 142°——下游必須用 `conf_used` + inlier 率 + 旗標一起 gate,
0.3 級位姿不可直接當 0.65 級用。語意提醒:`no_covisibility` = 無共視證據,
**不是**「不連通」——幾何驗證驗的是共視+幾何一致,不是可通行。

### per-pair JSON schema(`schema_version: wp1-step3-v2`)

> 版本沿革:`wp1-step2-v1` = pose/quality/flags/inliers/gt;`wp1-step3-v1` 新增
> `conf_used`、`flags.no_covisibility`、`regions`(共視區域);**`wp1-step3-v2`(2026-07-23)
> = nadir 前濾預設開啟**,新增 `params.lat_max`、`n.lat_drop`(正式目錄已全量重生,
> 位姿 3,345 對 = 47.3%;A/B 依據見 `nadir_ab_report.md`)。

```jsonc
{
  "schema_version": "wp1-step3-v2",
  "house": "DollhouseTask_58472_NoOutdoor",
  "img_a": "<uuid>", "img_b": "<uuid>",        // cache 檔名的 A/B 順序
  "img_size_a": [8192, 4096], "img_size_b": [8192, 4096],   // 原圖 W,H(per-house 不同!)
  "params": {"conf_min": 0.65, "retry_conf": 0.3, "max_pts": 4000, "thresh_deg": 0.5,
             "ransac_iters": 2000, "seed": 0, "sampler": "stratified",
             "lat_max": 60.0},                  // nadir 前濾角(v2 起;0/缺 = 未濾)
  "n": {"raw": 8146, "conf": 2134, "used": 2134, "inlier": 1474,
        "lat_drop": 312},                       // 前濾丟點數(v2 起)
  "conf_used": 0.65,                            // 自適應實際門檻:0.65 / 0.3 / null(無位姿)
  "inlier_rate": 0.6907,                        // n.inlier / n.used;無位姿時 pose=null
  "pose": {                                     // null = low_matches 或內點 <8
    "R": [[...3x3...]],                         // bearing 約定:x_b 幀 = R·x_a 幀 + t(尺度自由)
    "t": [x, y, z],                             // 單位向量;= A 相機在 B 幀的方向
    "cam_b_in_a": [x, y, z],                    // -R^T t = B 相機在 A 幀的方向
    "az_ab_deg": -139.1,                        // A 幀中望向 B 的方位角(ERP 經度,0=圖中央,±180=接縫)
    "az_ba_deg": 40.2,                          // B 幀中望向 A 的方位角
    "rot_angle_deg": 12.3, "baseline_txz": 0.998
  },
  "quality": {
    "sampson_med_deg": 0.11, "sampson_p90_deg": 0.32,   // 內點對極(重投影)殘差統計
    "rot_only_med_deg": 20.6,                   // 內點被純旋轉解釋的中位殘差(近零基線偵測)
    "az_wedge_a_deg": 109.3, "az_wedge_b_deg": 95.0,    // 包住 80% 內點方位的最窄圓弧
    "az_cov_a": 0.42, "az_cov_b": 0.39          // 方位覆蓋度:有內點的 10° bin 佔比
  },
  "flags": {
    "low_matches": false,          // conf0.65 級點數 < 30(品質註記;自適應後非跳過原因)
    "no_covisibility": false,      // 兩級 conf 都湊不出 ≥8 內點(=無共視證據,非「不連通」)
    "degenerate_rot_only": false,  // rot_only_med < 1.5×thresh_deg → t 方向不可靠
    "doorway_wedge": false         // 非退化 且 inlier_rate≥0.3 且 n_inlier≥30 且兩側楔形≤75°
  },                               //(閾值為暫定,五類抽查驗收後可調)
  "regions": [                     // 共視區域(Step 3):內點 6D 聯合 DBSCAN 聚類,依 n 降冪
    {"n": 1347, "frac_of_inliers": 0.965, "mean_conf": 0.872,
     "a": {"polygon": [[x,y], ...],   // A 側凸包頂點,原圖像素;過縫 pair x 可能超出 [0,W)
                                       // 以保多邊形連續,繪製時自行 mod W
           "area_frac": 0.2148,        // 多邊形面積 / (W*H)(ERP 像素面積,極區有失真)
           "az_range": [-81.7, 141.1], "az_width_deg": 222.8},   // 方位範圍(全跨度)
     "b": { /* B 側同構 */ }}
  ],                               // 聚類參數:eps=兩側各約12°(鏈接距離)、min_pts=max(8, 2%);
                                   // 門 pair 預期單一門洞區,同房/大環預期大覆蓋或多區。
                                   // ⚠ 區域「代表方位」請用 pose.az_ab,勿用區域中心(已測負結果)
  "gt": {"label": 1, "same_room": true, "az_ab_deg": -132.19, "az_ba_deg": 43.5},
                                   // label:metadata 連通標註;same_room:HOTSPOT.json 的
                                   // (floor,Roomidx) 相同;az:GT hotspot 方位(直接相鄰才有)。
                                   // ⚠ hotspot 語意 = 「通道位置」,只有同房 pair 才近似指向對方
                                   // 相機;但直接相鄰 pair 的 B 通常就在通道正後方,兩語意方位
                                   // 高度重合(實測同房誤差中位 9.1° / 跨房 10.7°),故 err_az_*
                                   // 當一致性檢核,非嚴格位姿精度。僅供驗收,非計算輸入
  "inliers": [[x_a, y_a, x_b, y_b, conf], ...], // 原圖像素,0.1px 取整,最多 2000 點(seed 0 抽樣)
  "time_ms": 833.7
  // "regions": Step 3 將加入 A/B 側共視多邊形遮罩
}
```

`_summary.csv` 欄位 = 上述指標攤平(`house,img_a,img_b,n_*,inlier_rate,conf_used,az_*,sampson_*,rot_only_med_deg,az_wedge_*,az_cov_*,flag_*,n_regions,region1_frac,region1_az_width_a,gt_label,same_room,gt_az_*,err_az_*,time_ms`);
`err_az_ab` = 位姿方位與 GT hotspot 方位的圓角誤差(全量,stratified+LO:同房中位 9.5°、跨房 10.7°、退化對照 ~86°)。

注意:LO refit 會讓 FP pair 的虛假模型也多吸一些門檻邊緣點(label=0 inlier 率中位
0.186→0.212),inlier 率的真/假分離比從 ~2.4 降至 ~2.1 — 仍清楚分離;判 pair 好壞
請綜合 inlier 率+旗標+az 一致性,不要單看 inlier 率。

附帶一個已測的**負結果**(供 Step 3 設計參考):「內點方位楔形中心」對 hotspot 方位的誤差
中位 39°~62°(即使限定 doorway_wedge 旗標子集也 39°)——外觀叢集中心**不是**好的通道方位
估計;對極位姿的 t 方向(az_ab)才是。

## Step 4:overlay 渲染器(`render_region_overlay.py`,WP2 模式 B 參考 + 驗收抽查)

```bash
# 單對 on-demand
venv/Scripts/python.exe pose_regions/render_region_overlay.py --json Output/roma_regions/XXX.json
# 驗收抽樣批次:6 類(doorway/同房/跨房/0.3撈回/多區/過縫)各 N 張,決定性選擇
venv/Scripts/python.exe pose_regions/render_region_overlay.py --sample 3
```

輸出 `Output/roma_regions/overlays/*.jpg`(gitignore;A 上 B 下堆疊、region 兩側同色
半透明 + 邊框 + 編號、az_ab/az_ba 黃線、GT hotspot 青色虛線、資訊列)。按工作計畫
「overlay 不全量產出」,只做抽樣/on-demand(全量 7,069 張會吃爆磁碟)。

**WP2 模式 B 過縫繪製約定(已在本渲染器驗證)**:polygon x 可能超出 [0,W) 以保連續;
繪製時同一多邊形平移 −W / 0 / +W 各畫一次,超界自然裁掉即可。

**已知品質問題(2026-07-22 抽查發現,量化後留給驗收裁決)**:主區重心落在 nadir 帶
(仰角 < −60°,即腳架/地面盲區)的比例 = **0.65 級 6.8%、0.3 撈回級 23.2%**——腳架
在每張全景長一樣,是系統性假匹配熱點,0.3 撈回層污染尤重。候選對策:RANSAC 前濾掉
|lat|>60° 的 match(但會同時砍掉真地板匹配,需 A/B 驗證後再定),或至少在 region
加 nadir_frac 欄供下游過濾。下游短期先用「主區重心緯度」自行 gate。

## 共視訊號當排序分數:判別力與上限(`covis_ranking_eval.py`,2026-07-26)

```bash
python pose_regions/covis_ranking_eval.py          # 約 1 分鐘,純 CPU
```

回答「幾何驗證後的共視證據能不能把真正連通的 pair 排到前面」。族群 = `_summary.csv`
全 7,069 對(coarse filter R90 存活者),GT 用 `datasets/*_HOTSPOT.json` 完整鄰接重建
(不用 `metadata/*.csv`,那份只覆蓋 68%),另算 BFS topology distance。輸出
`covis_ranking_report.md` + `Output/roma_regions/_ranking_eval.csv`(每 pair 一列,
含 coarse logit —— 矩陣本身在 gitignore 內,這份 CSV 是唯一入庫的分數快照)。

結論摘要(完整表格見報告):

- 幾何 n_inlier 單獨 AP **0.650** > RoMa 匹配數 0.618 > coarse refined 0.596 > 基準率 0.256;
  **與 coarse 組合 AP 0.711(免訓練 rank 平均)/ 0.713(house-5fold logistic OOF)**,
  同 R≥0.90 下 FP 2,991 → 2,443(−18%)。組合增益不是擬合假象。
- 但**上限沒被打破**:d1 vs d3+ 幾何 AP 0.787(強),d1 vs d2 只有 0.758 vs coarse 0.749;
  限縮到兩邊都有位姿時 0.800 反而略低於 coarse 0.807。R90 被移除的 874 個 FP 中
  **75% 是 d3 以上**,殘餘 FP 的 d2 佔比由 37% **升到 42%** —— 誤差被濃縮到
  「看得到、走不到」那道牆上,不是穿過它。
- 覆蓋率:僅 87.4% 正例有 inlier、82.6% 有位姿 ⇒ **只能當特徵,不能當獨立過濾器**。
- 注意:coarse 在此族群的 AP 被截斷(族群本來就由它挑出),不可與全矩陣評測相比,
  也不可讀成「幾何贏過模型」。

## WP1 後續(見 進度報告/7-22/工作計畫.md)

- ~~Step 3:自適應 conf + 共視區域萃取~~(2026-07-22 完成,schema `wp1-step3-v1`)。
- ~~Step 4:overlay 資產~~(2026-07-22 完成,抽樣+on-demand 渲染器)。
- 未排工項:同房統一尺度(每對 |t|=1,跨 pair 平移長度不可比;需尺度傳播或 HSLoc_3D 錨定)。
- 已知事實:live 重跑 3~10 s/對(RTX 4060);cache 批次路徑純 CPU,毫秒級/對;**per-house 圖尺寸不同(7 種,5376×2688 ~ 8192×4096)**。
