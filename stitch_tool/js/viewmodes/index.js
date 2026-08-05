/* =========================================================================
   viewmodes/index.js — 檢視模式清單(唯一的「增/停用模式」修改點)
   每個 import 一行 = 一個模式(檔案內自行 registerViewMode)。
   新增模式:寫好 mode-xxx.js 後在這裡加一行 import。
   停用模式:註解/刪除該行,其餘模式與既有功能不受影響。
   ========================================================================= */

import './mode-3d.js';      // 3D 球面(基底層,行為不變)
import './mode-erp.js';     // 2D ERP 對比(既有,行為不變;沿用 btn-erp / E 鍵)
import './mode-crop.js';    // 模式 A:透視裁切對(WP2 首輪;次輪補對望/同向)
// import './mode-morph.js'; // 模式 C:動態 morph — 2026-07-22 停用(使用者親測:特徵 warp 中間態無語意,溝通失敗);檔案保留不刪
// mode-regions 已改為 ERP 內疊層開關(btn-erp-regions / erp.js)，不再當獨立 viewmode
// import './mode-regions.js';
import './mode-walk.js';    // 模式 D:走一遍(WP2 次輪;zoom+交叉淡化,兩端朝行進方向)

export { initViewModes, setViewMode, getViewMode, toggleViewMode } from './registry.js';
