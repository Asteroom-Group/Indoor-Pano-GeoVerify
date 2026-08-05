# matrices/ — 連通性矩陣

連通性粗篩的輸出。本專案據此決定哪些配對送進 RoMa 幾何驗證，
並在標註工具中顯示粗篩的連通機率。

## 版本並存

一個子目錄一個版本，可以同時存在：

```
matrices/
├── cos_joint_l4/            ← 目前使用
│   ├── refined_matrices/    ← 正式流程讀這個
│   └── oof_matrices/        ← out-of-fold 矩陣，僅 covis_ranking_eval.py 使用
├── <其他版本>/
└── <其他版本>/
```

`config.matrices_root("refined")` / `matrices_root("oof")` 解析到對應子目錄。
版本目錄底下沒有子目錄、`.npy` 直接置於其中時，回退為版本目錄本身。

切換版本，優先序由高到低：

```bash
python stitch_tool/prepare_house_data.py --oof_dir matrices/<版本>
```

```powershell
$env:ROMA_GEO_MATRICES = "D:\somewhere\matrices_v2"
```

或改 `config.py` 的 `MATRICES_VERSION`。

檢查目前解析到哪一版：

```bash
python config.py
```

## 格式契約

新來源必須符合這份契約，否則 `prepare_house_data.py` 會 skip 或算錯。每間房三個檔：

| 檔名 | 形狀 | 內容 |
|---|---|---|
| `<house>.npy` | `(N, N, 3)` | ch0 = 連通 logit（**對稱**）· ch1 = sin(az) · ch2 = cos(az) |
| `<house>_images.json` | 長度 N | 影像檔名，順序與矩陣的 row/col 對齊 |
| `<house>_doormap.npy` | `(N, 8)` | 每張圖的門方位 8-bin aux logits |

`<house>` 必須與 `datasets/` 底下的資料夾名逐字一致（例：`DollhouseTask_58553`、
`DollhouseTask_58472_NoOutdoor`——`_NoOutdoor` 後綴時有時無）。

### 方位角規約

本專案：`az(x) = ((x / W) * 360 - 180) mod 360`，**0° = 影像中央**，±180° = 接縫。

⚠ 子計畫一的 `segmentation/aligner.py` 用 `az = (x / W) * 360`（0° = 影像左緣），
**與本專案差 180°**。接新來源前先確認對方採用哪一種規約。

### 方位通道為選配

有位姿的 pair，`prepare_house_data.py` 會以 `Output/roma_regions/_summary.csv` 的
`az_ab_deg` / `az_ba_deg` 覆寫 `ch1/ch2` 推得的方位，並記在 `az_source` 欄位；
沒有位姿的 pair 才沿用矩陣的方位。

因此新版本的矩陣不提供方位通道也可以接，但 `ch1/ch2` 的欄位要在，填 0 即可。

## 目前版本：cos_joint_l4

`refined_matrices/` 171 檔（57 間房 × 3），`oof_matrices/` 172 檔。

`oof_matrices/` 只有 `pose_regions/covis_ranking_eval.py` 使用。正式流程只讀
`refined_matrices/`，新版本不提供 oof 也能接，排序評估報告會少一項對照。
