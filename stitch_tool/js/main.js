/* =========================================================================
   main.js — 組裝與啟動（boot）
   1) 初始化 three.js 場景與各 UI 訂閱者（minimap/compass/queue/info/erp/scene3d）。
   2) 綁定 UI 與快捷鍵。
   3) 載入房子清單、選房子（讀 house data + state），emit 'house-loaded' 觸發所有重繪。
   4) 連接跨模組事件（reject-confirmed）。
   模組間避免評估期循環：低層不 import 高層；跨層互動走 events 或此處注入。
   ========================================================================= */

import { on, emit } from './events.js';
import { fetchHouses, fetchHouseData, fetchState } from './api.js';
import { setHouseName, setHouseData, normalizeState } from './state.js';
import { nav, gotoCurrentPair } from './nav.js';
import { initThree, initScene3d } from './scene3d.js';
import { initMinimap } from './minimap.js';
import { initCompass } from './compass.js';
import { initQueue, resetStatFilter } from './queue.js';
import { initInfo } from './info.js';
import { initErp, initRecoPanel } from './erp.js';
// import { initRoma } from './roma.js';  // RoMa 即時比對 — 2026-08-05 停用(會議拍板:共識區域
// 取代 raw RoMa 顯示;按鈕/roma-conf-input 已從 index.html 移除;js/roma.js 與後端
// /api/roma_match、離線 cache 保留不動)
import { initViewModes } from './viewmodes/index.js';   // 檢視模式插件系統(WP2)
import { clearMode } from './modes.js';
import { rejectConfirmed } from './actions.js';
import { initBatchPanel } from './batch-panel.js';
import { wireUI, applyUiToggles, syncThresholdUI } from './ui.js';
import { toast } from './util.js';

/**
 * 載入房子清單填入下拉，並選定測試房子（或第一個）。
 */
function loadHousesList(){
  fetchHouses().then(list => {
    const sel = document.getElementById('house-select'); if(!sel) return;
    sel.innerHTML = '';
    list.forEach(h => {
      const o = document.createElement('option'); o.value = h.house;
      o.textContent = h.house + '  (' + h.n_images + 'img / ' + h.n_pairs + 'pr)';
      o.dataset.meta = h.n_gt_connected_pairs;
      sel.appendChild(o);
    });
    const want = 'DollhouseTask_58472_NoOutdoor';
    if(list.some(h => h.house === want)) sel.value = want;
    else if(list[0]) sel.value = list[0].house;
    sel.onchange = () => selectHouse(sel.value);
    if(sel.value) selectHouse(sel.value);
  }).catch(e => toast('讀取房子清單失敗: ' + e));
}

/**
 * 選一間房子：重置導覽/模式/過濾，讀 house data 與 state，emit 'house-loaded'。
 * [bug 1 修復點之一] 'house-loaded' 一到，minimap/queue/compass 立即重繪，不再空白。
 * @param {string} name
 */
function selectHouse(name){
  setHouseName(name);
  nav.curPair = null; nav.curImage = -1; nav.viewFrom = null; nav.reciprocalKey = null;
  nav.erpTopImg = -1; nav.erpBotImg = -1;
  clearMode();
  resetStatFilter();
  emit('view-changed');   // 清空目前 markers/資訊由 loadImageView 之後接手

  const iName = document.getElementById('i-name');
  if(iName) iName.textContent = '載入資料…';

  Promise.all([
    fetchHouseData(name),
    fetchState(name)
  ]).then(([hd, st]) => {
    setHouseData(hd);
    normalizeState(st);
    applyUiToggles();
    syncThresholdUI();
    const meta = document.getElementById('house-meta');
    if(meta) meta.textContent = hd.n_images + ' 圖 · ' + hd.pairs.length + ' 對 · GT連通 ' + hd.n_gt_connected_pairs;

    emit('house-loaded');   // minimap/queue 訂閱重繪（根治選房後空白）
    gotoCurrentPair();      // 進入第一個 pending（會 loadImageView + renderCompass）
  }).catch(e => toast('讀取房子資料失敗: ' + e));
}

/** 啟動。 */
(function boot(){
  if(typeof THREE === 'undefined'){
    const el = document.getElementById('i-name');
    if(el) el.textContent = 'three.js 載入失敗';
    return;
  }
  // 場景與訂閱者
  initThree();
  initScene3d();
  initMinimap();
  initCompass();
  initQueue();
  initInfo();
  initErp();
  // 需求 2/3：綁定批量否決面板 與 unconn 推薦面板的按鈕（否則面板可開、按鈕無反應）
  initBatchPanel();
  initRecoPanel();
  // initRoma();  // 2026-08-05 停用，見上方 import 註記
  initViewModes();   // 生成新模式按鈕 + 模式互斥(既有 3D/ERP 行為不變)

  // UI 綁定
  wireUI();

  // 跨模組事件：右鍵綠圈 -> 重新否決
  on('reject-confirmed', ({ i, j }) => rejectConfirmed(i, j));

  // 載入房子
  loadHousesList();
})();
