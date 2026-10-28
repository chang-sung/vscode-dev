/* ========== 전역 상태/유틸 ========== */
let shelves = [];
let itemDT = null; // 전역
let $imageWrap, $tableWrap; // .parts-search-image / .parts-search-table 컨테이너 캐시
let globalSearchState = {
    key: "",     // "type|term"
    units: [],   // ['IN LFT', 'PA TR', ...]
    buckets: {}, // { 'IN LFT': [rows...], ... }
    idx: 0       // 다음에 보여줄 unit 인덱스
};

const selectedItemKeys = new Set();
const __searchAuth = { isPrivileged: false };

// 인라인 수정 컨텍스트 (다중 행 편집)
let searchEditCtxs = null;

function setAdminMode(isPrivileged) {
  // 조회 탭 전역 권한 상태 반영 (있으면)
  __searchAuth.isPrivileged = !!isPrivileged;

  // 👉 관리자/구매담당자만 '파츠 편집' 버튼 보이기
  enforceEditButtonVisibility();

  // (선택) 편집 패널 버튼 상태 갱신
  refreshButtonsByStateSearch?.();
}

async function checkClientAuthority() {
    try {
        const res = await fetch('/check-client-authority-parts');
        const data = await res.json();

        // 관리자 또는 구매담당자면 true
        const isPrivileged = (data.authority === '관리자' || data.authority === '구매담당자');
        setAdminMode(isPrivileged);

        // ✅ 현재 사용자 이름 전역 저장
        window.__currentUserName = String(data.userName || '').trim();

        // 상단 사용자 표시 그대로 사용
        if (!document.getElementById('user-name-display')) {
        const userNameContainer = document.createElement('div');
        userNameContainer.id = 'user-name-display';
        const userNameBox = document.createElement('div');
        userNameBox.id = 'user-name-box';
        userNameBox.innerText = `${data.authority} 모드 ( ${data.userName} )`;
        userNameContainer.appendChild(userNameBox);
        document.body.prepend(userNameContainer);
        }
    } catch (e) {
        console.error('Authority check failed:', e);
        setAdminMode(false);
    }
}

function enforceEditButtonVisibility() {
  const wrap = document.querySelector('.search-wrapper');
  if (!wrap) return;
  const btn = wrap.querySelector('#add-parts-btn');
  const box = wrap.querySelector('.edit-btn-container');
  if (!btn) return;

  if (__searchAuth?.isPrivileged) {
    btn.style.display = '';
  } else {
    btn.style.display = 'none';
    if (box) box.style.display = 'none';
  }
}


// === [검색 테이블] 버튼 상태/모드 관리 ===
function getButtonsInSearchWrap() {
    const wrap = document.querySelector('.search-wrapper');
    return {
        btnAdd:    wrap?.querySelector('#edit-add-btn-3'),
        btnDel:    wrap?.querySelector('#edit-delete-btn-3'),
        btnEdit:   wrap?.querySelector('#edit-btn-3'),
        btnCancel: wrap?.querySelector('#edit-cancel-btn-3'),
    };
}

function hasDraftRowNowSearch() {
    return $('#item-table tbody tr.draft-row').length > 0;
}

function isEditingNowSearch() {
    return Array.isArray(searchEditCtxs) && searchEditCtxs.length > 0;
}

/**
 * mode: 'idle' | 'draft' | 'editing'
 * - draft / editing: Add/Delete/Edit 비활성, Cancel 활성
 * - idle: 모두 활성 (Essential과 동일하게 Cancel도 활성)
 */
function setButtonsForModeSearch(mode) {
    const { btnAdd, btnDel, btnEdit, btnCancel } = getButtonsInSearchWrap();
    if (!btnAdd || !btnDel || !btnEdit || !btnCancel) return;

    if (mode === 'draft' || mode === 'editing') {
        btnAdd.disabled = true;
        btnDel.disabled = true;
        btnEdit.disabled = true;
        btnCancel.disabled = false;
    } else {
        btnAdd.disabled = false;
        btnDel.disabled = false;
        btnEdit.disabled = false;
        btnCancel.disabled = false;
    }
}

/** 현재 상태 읽어 버튼 자동 갱신 */
function refreshButtonsByStateSearch() {
    if (hasDraftRowNowSearch()) return setButtonsForModeSearch('draft');
    if (isEditingNowSearch())   return setButtonsForModeSearch('editing');
    return setButtonsForModeSearch('idle');
}

// 선택 상태와 하이라이트 클래스를 즉시 지우는 함수
function clearItemSelection() {
    selectedItemKeys.clear();

    const tbl = document.getElementById('item-table');
    if (tbl) {
        const $tbl = $(tbl);
        $tbl.find('tbody tr.selected').removeClass('selected');
    }
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** 하이라이트 박스 active 토글 */
function updateActiveHighlight(name) {
    // 현재 화면에 렌더된 모든 박스에서 active 제거
    document.querySelectorAll('.highlight-box').forEach(box => {
        box.classList.toggle('active', box.dataset.shelfName === name);
    });
}

/** 선택(드롭다운/클릭 공용 진입점): 드롭다운 값 동기화 + 하이라이트 강조 + 테이블 fetch */
function setActiveShelf(name) {
    if (!name) return;
    // 🔵 선택 상태/클래스 즉시 제거 (플리커 방지)
    clearItemSelection();

    const select = document.getElementById('search-unit-select');
    if (select && select.value !== name) {
        select.value = name;
    }
    updateActiveHighlight(name);
    fetchShelfInfo(name);
}

// ── 레이아웃 상태 ───────────────────────────────
let __currentLayout = 'PLM_S'; // 현재 렌더된 설비 레이아웃

// 각 레이아웃이 가진 Unit 이름 집합(이미지 상의 박스 이름 기준)
const LAYOUT_UNITS = {
    PLM_S: new Set([
        'IN LFT','LD PA TR','LD TO TR','PA PRE ALIGN','ATT LD TR','ATT STG','ATT ULD TR','INSP STG',
        'ATT INSP','ULD TO TR','ULD PA TR','OUT LFT','MAG TR','FILM SPY TR','2SHEET STG',
        'FILM LD TR','SET TBL','FILM DET','PA DET', 'PA BUF TR'
    ]),
    PLM_T: new Set([
        'PA_LOADER','PA_BUFFER','PA_TURNOVER','PA_SHUTTLE','PA_TRANSFER','PA_LAMI_SHT',
        'FILM_LIFT','FILM_LOADER','FILM_SHUTTLE','FILM_TANSFER','FILM_LAMI_SHT','LAMI_TRANSFER',
        'LAMI_SHUTTLE','LAMI_TRUNOVER','INSPECT_TR','PA_OUT_SHUTTLE','OK_TR'
    ]),
    ACV_Y: new Set(['CHAMBER'])
};

// Unit 이름으로 어느 레이아웃인지 추정
function detectLayoutByUnit(unitName = '') {
    if (LAYOUT_UNITS.PLM_S.has(unitName)) return 'PLM_S';
    if (LAYOUT_UNITS.PLM_T.has(unitName)) return 'PLM_T';
    if (LAYOUT_UNITS.ACV_Y.has(unitName)) return 'ACV_Y';
    return null; // 모르면 현재 레이아웃 유지
}

// 해당 Unit에 맞춰 레이아웃을 전환(필요할 때만)
function ensureLayoutForUnit(unitName) {
    const layout = detectLayoutByUnit(unitName);
    if (!layout || layout === __currentLayout) return;

    const eqpSel = document.getElementById('eqp-division-select');
    if (eqpSel) eqpSel.value = layout;

    if (layout === 'PLM_S') img_PLM_S();
    else if (layout === 'PLM_T') img_PLM_T();
    else if (layout === 'ACV_Y') img_ACV_Y();

    __currentLayout = layout;
}

/* ========== 이미지 박스  ============ */
function img_PLM_S() {
    shelves = [
        {name:'IN LFT', x1:83, y1:200, x2:148, y2:348},
        {name:'PA TR', x1:155, y1:255, x2:219, y2:295},
        {name:'LD TO TR', x1:225, y1:255, x2:360, y2:295},
        {name:'PRE ALIGN', x1:367, y1:255, x2:504, y2:295},
        {name:'ATT LD TR', x1:437, y1:200, x2:645, y2:241},
        {name:'ATT STG', x1:651, y1:29, x2:788, y2:249},
        {name:'ATT ULD TR', x1:510, y1:29, x2:646, y2:70},
        {name:'INSP STG', x1:494, y1:83, x2:559, y2:123},
        {name:'ATT INSP', x1:384, y1:83, x2:452, y2:70},
        {name:'PA BUF TR', x1:400, y1:95, x2:452, y2:100},

        {name:'ULD TO TR', x1:296, y1:75, x2:362, y2:123},
        {name:'ULD PA TR', x1:155, y1:83, x2:290, y2:130},
        {name:'OUT LFT', x1:83, y1:30, x2:148, y2:123},
        {name:'MAG TR', x1:154, y1:405, x2:219, y2:176},
        {name:'FILM SPY TR', x1:225, y1:352, x2:362, y2:446},
        {name:'2SHEET STG', x1:367, y1:352, x2:504, y2:393},
        {name:'FILM LD TR', x1:510, y1:352, x2:646, y2:393},
        {name:'SET TBL', x1:651, y1:254, x2:788, y2:393},
        {name:'FILM DET', x1:580, y1:300, x2:646, y2:340},
        {name:'PA DET', x1:580, y1:252, x2:646, y2:293},
    ];
    updatePage('PLM S社 부품 검색', 'css/01_parts/base_s.jpg');
    populateSearchUnitSelect(); // ✅ 드롭다운 채우기
    __currentLayout = 'PLM_S'; // ✅ 추가
}

function img_PLM_T() {
    shelves = [
        {name:'PA_LOADER', x1:83, y1:200, x2:148, y2:348},
        {name:'PA_BUFFER', x1:155, y1:255, x2:219, y2:295},
        {name:'PA_TURNOVER', x1:225, y1:255, x2:360, y2:295},
        {name:'PA_SHUTTLE', x1:367, y1:255, x2:504, y2:295},
        {name:'PA_TRANSFER', x1:437, y1:200, x2:645, y2:241},
        {name:'PA_LAMI_SHT', x1:651, y1:29, x2:788, y2:249},
        {name:'FILM_LIFT', x1:510, y1:29, x2:646, y2:70},
        {name:'FILM_LOADER', x1:494, y1:83, x2:559, y2:123},
        {name:'FILM_SHUTTLE', x1:384, y1:83, x2:452, y2:70},
        {name:'FILM_TANSFER', x1:296, y1:75, x2:362, y2:123},
        {name:'FILM_LAMI_SHT', x1:155, y1:83, x2:290, y2:130},
        {name:'LAMI_TRANSFER', x1:155, y1:83, x2:290, y2:130},
        {name:'LAMI_SHUTTLE', x1:155, y1:83, x2:290, y2:130},
        {name:'LAMI_TRUNOVER', x1:155, y1:83, x2:290, y2:130},
        {name:'INSPECT_TR', x1:155, y1:83, x2:290, y2:130},
        {name:'PA_OUT_SHUTTLE', x1:155, y1:83, x2:290, y2:130},
        {name:'OK_TR', x1:155, y1:83, x2:290, y2:130},

    ];
    updatePage('PLM T社 부품 검색', 'css/01_parts/base_t.jpg');
    populateSearchUnitSelect(); // ✅ 드롭다운 채우기
    __currentLayout = 'PLM_T'; // ✅ 추가
}

function img_ACV_Y() {
    shelves = [
        {name:'CHAMBER', x1:83, y1:200, x2:148, y2:348},

    ];
    updatePage('ACV Y社 부품 검색', 'css/01_parts/base_y.jpg');
    populateSearchUnitSelect(); // ✅ 드롭다운 채우기
    __currentLayout = 'ACV_Y'; // ✅ 추가
}

function ensureTable() {
    // 표 컨테이너 내부에 표가 없으면 생성
    if (!$tableWrap) return;
    if ($tableWrap.querySelector('#item-table')) return;

    $tableWrap.innerHTML = `
        <h2 id="parts-title">[ Parts 정보 ]</h2>
        <div class="update-time">Loading update time...</div>
        <div class="edit-btn-wrapper">
            <button id="add-parts-btn" class="edit-group-btn" style="display:none">파츠 편집</button>
            
            <div class="edit-btn-container" style="display:none">
                <button id="edit-add-btn-3" class="edit-group-btn">추가</button>
                <button id="edit-delete-btn-3" class="edit-group-btn">삭제</button>
                <button id="edit-btn-3" class="edit-group-btn">수정</button>
                <button id="edit-cancel-btn-3" class="edit-group-btn">취소</button>
            </div>
        </div>
        

        <table id="item-table" class="item-table">
              <colgroup>
                <col style="width:40px">
                <col style="width:80px">
                <col style="width:140px">
                <col style="width:290px">
                <col style="width:80px">
                <col style="width:50px">
                <col style="width:50px">
                <col style="width:50px">
              </colgroup>
              <thead>
                <tr class="top-group">
                    <th rowspan="2">축</th>
                    <th rowspan="2">부품</th>
                    <th rowspan="2">Q-Code</th>
                    <th rowspan="2">업체P/N</th>
                    <th rowspan="2">Maker</th>
                    <th colspan="2">재고</th>      <!-- ✅ 라인/EMS를 묶는 그룹 헤더 -->
                    <th rowspan="2">사용<br>이력</th>
                </tr>
                <tr class="sub-head">
                    <th>EMS</th>       <!-- 하위 헤더 2 -->
                    <th>라인</th>      <!-- 하위 헤더 1 -->
                </tr>
            </thead>
            <tbody id="item-table-body">
                <tr class="no-data">
                <td colspan="8" style="text-align:center; padding:20px; color:#888;">
                    ※ 유닛을 선택하세요
                </td>
                </tr>
            </tbody>
        </table>
    `;
}

function renderHighlightsInto(container, imgEl) {
    // 기존 제거
    container.querySelectorAll('.highlight-box').forEach(el => el.remove());
    let mapEl = container.querySelector('map#parts-map');
    if (!mapEl) {
        mapEl = document.createElement('map');
        mapEl.name = 'parts-map';
        mapEl.id = 'parts-map';
        container.appendChild(mapEl);
    }
    mapEl.innerHTML = '';

    shelves.forEach(shelf => {
        // 이미지맵
        const area = document.createElement('area');
        area.shape = 'rect';
        area.coords = `${shelf.x1},${shelf.y1},${shelf.x2},${shelf.y2}`;
        area.alt = shelf.name;
        area.dataset.shelfName = shelf.name;
        // 🔗 이미지맵 클릭도 드롭다운과 동기화
        area.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveShelf(shelf.name);
        });
        mapEl.appendChild(area);

        // 하이라이트 박스
        const box = document.createElement('div');
        box.className = 'highlight-box';
        box.dataset.shelfName = shelf.name; // 🔗 동기화용 데이터 속성
        box.style.position = 'absolute';
        box.style.left   = `${shelf.x1}px`;
        box.style.top    = `${shelf.y1}px`;
        box.style.width  = `${shelf.x2 - shelf.x1}px`;
        box.style.height = `${shelf.y2 - shelf.y1}px`;

        // 🔗 박스 클릭 시 드롭다운 동기화 + 테이블 갱신
        box.addEventListener('click', () => setActiveShelf(shelf.name));

        container.appendChild(box);
    });

    imgEl.useMap = '#parts-map';
}

function renderImageAndMap(imagePath) {
    if (!$imageWrap) return;
    $imageWrap.innerHTML = `
        <div class="image-container">
        <img src="${imagePath}" alt="Parts" class="parts-image">
        </div>
    `;
    const container = $imageWrap.querySelector('.image-container');
    const imgEl = container.querySelector('.parts-image');

    // 이미지 로드 후 하이라이트 그리기 (이미지 크기 확보)
    if (imgEl.complete) {
        renderHighlightsInto(container, imgEl);
    } else {
        imgEl.addEventListener('load', () => renderHighlightsInto(container, imgEl), { once: true });
    }
}

function updatePage(title, imagePath) {

    const titleEl = document.querySelector('.search-wrapper h2');
    if (titleEl) titleEl.textContent = title;
    
    renderImageAndMap(imagePath); // 이미지/맵 렌더
    ensureTable(); // 표 스켈레톤 보장
    enforceEditButtonVisibility()
}

/** item-table을 DataTables로 초기화 */
function initItemDataTable(rows) {
    const $tbl = $("#item-table");

    if (!itemDT) {
        itemDT = $tbl.DataTable({
        data: rows,
        columns: [
            { data: "SUB_Unit"},
            { data: "Parts_Name"},
            { data: "Q_Code"},
            { data: "Spec" },
            { data: "Maker" },
            {
            data: "EMS재고",
            createdCell: (td, val) => td.classList.add((val|0) === 0 ? "zero-stock" : "non-zero-stock")
            },
            {
            data: "라인재고", 
            createdCell: (td, val) => td.classList.add((val|0) === 0 ? "zero-stock" : "non-zero-stock")
            },
            { data: "UsageCount" }
        ],
        // ✅ 각 행에 식별 키 부여 (Q_Code|Spec)
        createdRow: (row, data) => {
            const key = `${data.Q_Code ?? ''}|${data.Spec ?? ''}`;
            row.setAttribute('data-key', key);
        },
        autoWidth: false,
        order: [[0,"asc"],[1,"asc"]],
        scrollY: "420px",
        scrollX: false,
        scrollCollapse: true,
        paging: false,
        searching: true,
        fixedHeader: true,
        orderCellsTop: true,
        info: false,
        dom: "t"
        });
    } else {
        itemDT.clear().rows.add(rows).draw(false);
        itemDT.columns.adjust();
    }

    // ✅ 행 선택 토글 바인딩 (복원은 안함: redraw/재검색 시 선택 해제)
    bindDTSelectable($tbl, {
        selectedSet: selectedItemKeys,
        getKey: (tr) => tr.getAttribute('data-key'),
        onChange: () => {}     // 필요시 콜백 사용

    });

    wireOutsideDeselect($tbl, {
        selectedSet: selectedItemKeys,
        preserveSelectors: [
            'button',
            '.search-btn',
            '.edit-group-btn',
            '#search-btn',
            '#add-parts-btn',
            '#edit-add-btn-3',
            '#edit-delete-btn-3',
            '#edit-btn-3',
            '#edit-cancel-btn-3'
        ]
    });
}

/* ========== 데이터 통신 ========== */
async function fetchShelfInfo(shelfName) {
    if (!$tableWrap) return;
    ensureTable();

    const titleEl = document.getElementById('parts-title');
    const tbody   = document.getElementById('item-table-body');

    try {
        const resp = await fetch(`/parts/essential_list/${encodeURIComponent(shelfName)}`);
        if (!resp.ok) throw new Error(`서버 오류: ${resp.status}`);
        const data = await resp.json();

        // 테이블 제목
        titleEl.textContent = `[ Parts 정보 ] - ${shelfName}`;

        // 데이터가 없을 때
        if (!data || data.length === 0) {
            // DataTables가 이미 있다면 DT 방식으로 비우고 종료
            if (itemDT) {
                itemDT.clear().draw(false);
            }
            if (tbody) {
                tbody.innerHTML = `
                <tr class="no-data">
                    <td colspan="8" style="text-align:center; padding:20px; color:#888;">
                    선택한 유닛에 해당하는 데이터가 없습니다.
                    </td>
                </tr>`;
            }
            return;
        }

        // 1) 정렬 (축 기준)
        data.sort((a, b) => String(a.SUB_Unit).localeCompare(String(b.SUB_Unit)));

        // 2) Q-Code 유니크 추출
        const codes = [...new Set(data.map(d => d.Q_Code).filter(Boolean))];

        // 3) 서버에 배치 카운트 요청
        let repairMap = {};
        if (codes.length > 0) {
            const r = await fetch('/parts/repair_counts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ codes })
            });
            if (!r.ok) throw new Error(`repair_counts 서버 오류: ${r.status}`);
            const json = await r.json();
            repairMap = json?.counts || {};
        }

        // 4) DataTables 행 데이터 구성 + 교체 이력 매핑
        const rows = data.map(item => ({
            SUB_Unit:    item.SUB_Unit ?? '',
            Parts_Name:  item.Parts_Name ?? '',
            Q_Code:      item.Q_Code ?? '',
            Spec:        item.Spec ?? '',
            Maker:       item.Maker ?? '',
            라인재고:     item.라인재고 ?? 0,
            EMS재고:     item.EMS재고 ?? 0,
            UsageCount: repairMap[item.Q_Code] ?? 0   // ✅ 추가
        }));

        // 5) DataTables 초기화
        initItemDataTable(rows); 

    } catch (err) {
        console.error('선반 정보를 가져오는 중 오류:', err);
        // 오류 시 안전하게 안내문 표시
        if (itemDT) { itemDT.clear().draw(false); }
        if (tbody) {
            tbody.innerHTML = `
            <tr class="no-data">
                <td colspan="8" style="text-align:center; padding:20px; color:red;">
                데이터를 불러오는 중 오류가 발생했습니다.
                </td>
            </tr>`;
        }
    }
}


/* ========== Unit 드롭다운 채우기 ========== */
function populateSearchUnitSelect() {
    const select = document.getElementById('search-unit-select');
    if (!select) return;

    // 기존 옵션 초기화
    select.innerHTML = '<option value="">Unit 구분</option>';

    // shelves 배열에서 name만 추출해 option 생성
    shelves.forEach(shelf => {
        const opt = document.createElement('option');
        opt.value = shelf.name;
        opt.textContent = shelf.name;
        select.appendChild(opt);
    });
}

async function renderSearchResultForUnit(unitName, rowsForUnit) {
    clearItemSelection();

    // ✅ 먼저 레이아웃 동기화
    ensureLayoutForUnit(unitName);

    const titleEl = document.getElementById('parts-title');
    const unitSel = document.getElementById('search-unit-select');

    if (titleEl) titleEl.textContent = `[ Parts 정보 ] - ${unitName}`;
    if (unitSel) unitSel.value = unitName;

    updateActiveHighlight(unitName);

    if (!itemDT) {
        initItemDataTable(rowsForUnit);
    } else {
        itemDT.clear().rows.add(rowsForUnit).draw(false);
        itemDT.columns.adjust();
    }
}

// ── scripts_parts_search.js 하단 DOMContentLoaded 내부 바깥 또는 상단 util 영역 ──
async function runSearchClick() {
    const typeSel = document.getElementById('search-type-select');
    const inputEl = document.getElementById('search-input');
    const btn     = document.getElementById('search-btn');

    const type = (typeSel?.value === 'pn-search') ? 'pn' : 'qcode';
    const term = (inputEl?.value || '').trim();

    if (!term) { 
        if (typeof resetItemTable === 'function') resetItemTable();
        return; 
    }

    const nextKey = `${type}|${term}`;

    // 새 검색 시작 전 선택 즉시 제거(플리커 방지)
    if (globalSearchState.key !== nextKey) {
        clearItemSelection();
    }

    // 같은 검색어/타입으로 다시 눌렀을 때 → 다음 Unit 로테이션
    if (globalSearchState.key === nextKey && globalSearchState.units.length > 0) {
        const u = globalSearchState.units[globalSearchState.idx];

        ensureLayoutForUnit(u); // 로테이션 대상 Unit에 맞춰 이미지도 전환

        await renderSearchResultForUnit(u, globalSearchState.buckets[u] || []);
        globalSearchState.idx = (globalSearchState.idx + 1) % globalSearchState.units.length;
        return;
    }

    // 새 검색: 로딩 표시
    const tbody = document.getElementById('item-table-body');
    if (tbody) {
        tbody.innerHTML = `
        <tr class="no-data"><td colspan="8" style="text-align:center; padding:20px; color:#666;">
            검색 중…
        </td></tr>`;
    }
    if (btn) { btn.disabled = true; btn.textContent = '조회중…'; }

    try {
        // 1) 전역 검색 호출
        const r = await fetch('/parts/use_search', {
        method: 'POST',
        headers: { 'Content-Type':'application/json' },
        body: JSON.stringify({ type, term })
        });
        if (!r.ok) throw new Error('use_search api error');
        let { rows = [] } = await r.json(); // [{Unit, SUB_Unit, Parts_Name, Q_Code, Spec, Maker, 라인재고, EMS재고}]

        // 2) 사용이력 카운트 부착
        const codes = [...new Set(rows.map(x => x.Q_Code).filter(Boolean))];
        if (codes.length) {
        const rr = await fetch('/parts/repair_counts', {
            method: 'POST',
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify({ codes })
        });
        if (rr.ok) {
            const { counts = {} } = await rr.json();
            rows = rows.map(x => ({ ...x, UsageCount: counts[x.Q_Code] ?? 0 }));
        }
        }

        // 3) Unit별 그룹핑
        const buckets = rows.reduce((acc, r) => {
        const u = r.Unit || r.SUB_Unit || '';
        if (!u) return acc;
        (acc[u] ||= []).push(r);
        return acc;
        }, {});
        const units = Object.keys(buckets);

        // 4) 결과 없음 처리
        if (units.length === 0) {
            if (itemDT) itemDT.clear().draw(false);
            if (tbody) {
                const safe = escapeHtml(term);
                tbody.innerHTML = `
                    <tr class="no-data">
                    <td colspan="8" style="text-align:center; padding:20px; color:#888;">
                        "${safe}"에 해당하는 데이터가 없습니다.
                    </td>
                    </tr>`;
            }
            updateActiveHighlight(null);
            const unitSel = document.getElementById('search-unit-select');
            if (unitSel) unitSel.value = '';
            const titleEl = document.getElementById('parts-title');
            if (titleEl) titleEl.textContent = `[ Parts 정보 ]`;
            return;
        }

        // 5) 상태 저장
        globalSearchState = { key: nextKey, units, buckets, idx: 0 };

        // 6) 첫 Unit 렌더
        const firstUnit = units[0];
        ensureLayoutForUnit(firstUnit); // ✅ 추가
        await renderSearchResultForUnit(firstUnit, buckets[firstUnit]);
        globalSearchState.idx = (units.length > 1) ? 1 : 0; // 다음 인덱스 준비

    } catch (e) {
        console.error('[search] error:', e);
        if (itemDT) itemDT.clear().draw(false);
        if (tbody) {
        tbody.innerHTML = `
            <tr class="no-data"><td colspan="8" style="text-align:center; padding:20px; color:#d00;">
            검색 중 오류가 발생했습니다.
            </td></tr>`;
        }
        // 제목/하이라이트 리셋
        const titleEl = document.getElementById('parts-title');
        if (titleEl) titleEl.textContent = `[ Parts 정보 ]`;
        updateActiveHighlight(null);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '조회'; }
    }
}

function resetItemTable() {

    // 먼저 선택 해제
    clearItemSelection();

    const titleEl = document.getElementById('parts-title');
    const tbody   = document.getElementById('item-table-body');

    // 하이라이트 해제
    updateActiveHighlight(null);

    // DataTables 비우기 (있으면)
    if (itemDT) itemDT.clear().draw(false);

    // 안내문 표시
    if (tbody) {
        tbody.innerHTML = `
        <tr class="no-data">
            <td colspan="8" style="text-align:center; padding:20px; color:#888;">
            ※ 유닛을 선택하세요
            </td>
        </tr>`;
    }

    // 제목 원복
    if (titleEl) titleEl.textContent = `[ Parts 정보 ]`;
}

/* ========== 초기화 ========== */
document.addEventListener('DOMContentLoaded', () => {
    // 컨테이너 캐시
    checkClientAuthority();
    $imageWrap = document.querySelector('.parts-search-image');
    $tableWrap = document.querySelector('.parts-search-table');
    const searchBtn = document.getElementById('search-btn');
    const searchInp = document.getElementById('search-input');

    searchBtn?.addEventListener('click', runSearchClick);
    searchInp?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') runSearchClick();
    });

    // 드롭다운 바인딩
    const eqpSel = document.getElementById('eqp-division-select');
    if (eqpSel) {
        eqpSel.addEventListener('change', () => {
        if (eqpSel.value === 'PLM_S') img_PLM_S();
        else if (eqpSel.value === 'PLM_T') img_PLM_T();
        else if (eqpSel.value === 'ACV_Y') img_ACV_Y();
        });


        // 초기 선택 (기본 PLM_S)
        if (!eqpSel.value) {
            eqpSel.value = 'PLM_S';
        }
        if (eqpSel.value === 'PLM_S') img_PLM_S();
        else if (eqpSel.value === 'PLM_T') img_PLM_T();
        else if (eqpSel.value === 'ACV_Y') img_ACV_Y();

        __currentLayout = 'PLM_S'; // ✅ 안전하게 초기화
    }

    const unitSel = document.getElementById('search-unit-select');
    if (unitSel) {
        unitSel.addEventListener('change', () => {
        const name = unitSel.value?.trim();
        if (name) setActiveShelf(name);
        else resetItemTable();
        });
    }

});


/* ============================== 테이블 편집 (행추가, 행삭제, 수정, 취소)  ============================== */
// 드래프트는 동시에 1행만
function makeDraftInput(name, value = '', { type = 'text', placeholder = '' } = {}) {
    const v = String(value ?? '');
    return `
        <input class="dt-input"
        name="${escapeHtml(name)}"
        type="${escapeHtml(type)}"
        value="${escapeHtml(v)}"
        placeholder="${escapeHtml(placeholder)}"
        style="width:100%;box-sizing:border-box;">
    `;
}

// 현재 테이블에서 '드래프트' 행 추가
function addDraftRowInSearch() {
    if (!itemDT) return;

    const $tbody = $('#item-table tbody');
    if ($tbody.find('tr.draft-row').length > 0) {
        alert('이미 추가 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
        return;
    }

    // 테이블 컬럼 순서: SUB_Unit, Parts_Name, Q_Code, Spec(P/N), Maker, EMS재고, 라인재고, UsageCount
    const draft = {
        SUB_Unit: '', Parts_Name: '', Q_Code: '', Spec: '', Maker: '',
        EMS재고: 0, 라인재고: 0, UsageCount: 0,
        _key: '|', _draft: true
    };

    const rowApi = itemDT.row.add(draft);
    itemDT.draw(false);
    const node = rowApi.node();
    node.classList.add('draft-row');
    node.classList.remove('selected');

    const cells = node.cells || node.children;
    cells[0].innerHTML = makeDraftInput('SUB_Unit', '');
    cells[1].innerHTML = makeDraftInput('Parts_Name', '');
    cells[2].innerHTML = makeDraftInput('Q_Code', '');
    cells[3].innerHTML = makeDraftInput('Spec', '');
    cells[4].innerHTML = makeDraftInput('Maker', '');
    cells[5].textContent = '0';
    cells[6].textContent = '0';
    cells[7].textContent = '0';

    // 헤더 바로 아래로
    $('#item-table tbody').prepend(node);
    setButtonsForModeSearch('draft');

    // ✅ 드래프트행 Q_Code 입력 시 대문자 자동 변환
    $(node).find('input.dt-input[name="Q_Code"], input.dt-input[name="SUB_Unit"]')
        .on('input', function () { this.value = this.value.toUpperCase(); });
}

// 드래프트 수집
function collectDraftRowsInSearch() {
    const rows = [];
    $('#item-table tbody tr.draft-row').each(function () {
        const obj = {};
        $(this).find('input.dt-input').each(function () {
            const n = this.name;
            let v = (this.value ?? '').trim();
            if (n === 'Q_Code' || n === 'SUB_Unit') v = v.toUpperCase();
            obj[n] = v;
        });
        rows.push(obj);
    });
    return rows;
}

// 드래프트 제거
function removeAllDraftRowsInSearch() {
    if (!itemDT) return;
    $('#item-table tbody tr.draft-row').each(function () {
        itemDT.row(this).remove();
    });
    itemDT.draw(false);
}

// 필수값 검증 (조회 테이블 기준)
function validateSearchRow(r, rowNode) {
    const required = ['SUB_Unit', 'Parts_Name', 'Q_Code', 'Spec', 'Maker'];
    const missing = required.filter(k => !String(r[k] ?? '').trim());
    if (missing.length) {
        if (rowNode) {
        const el = rowNode.querySelector(`input[name="${missing[0]}"]`);
        if (el) el.focus();
        }
        return `필수 항목 누락: ${missing.join(', ')}`;
    }
    return null;
}

// 인라인 수정 전환
function rowToInputsInSearch(rowNode) {
    const data  = itemDT.row(rowNode).data();
    const cells = rowNode.cells || rowNode.children;
    if (!cells || !data) return;

    cells[0].innerHTML = makeDraftInput('SUB_Unit', data.SUB_Unit);
    cells[1].innerHTML = makeDraftInput('Parts_Name', data.Parts_Name);
    cells[2].innerHTML = makeDraftInput('Q_Code', data.Q_Code);
    cells[3].innerHTML = makeDraftInput('Spec', data.Spec);
    cells[4].innerHTML = makeDraftInput('Maker', data.Maker);

    cells[5].textContent = String(data.EMS재고 ?? 0);
    cells[6].textContent = String(data.라인재고 ?? 0);
    cells[7].textContent = String(data.UsageCount ?? 0);

    rowNode.classList.add('editing-row');

    // ✅ 수정행 Q_Code 입력 시 대문자 자동 변환
    $(rowNode).find('input.dt-input[name="Q_Code"], input.dt-input[name="SUB_Unit"]')
        .on('input', function () { this.value = this.value.toUpperCase(); });
}

// 인라인 입력 수집
function collectInputsInSearch(rowNode) {
    const obj = {};
    $(rowNode).find('input.dt-input').each(function () {
        const n = this.name;
        let v = (this.value ?? '').trim();
        if (n === 'Q_Code' || n === 'SUB_Unit') v = v.toUpperCase();
        obj[n] = v;
    });
    return obj;
}

// 인라인 수정 취소/복원
function restoreRowFromDataInSearch(rowNode, data) {
    itemDT.row(rowNode).data({
        SUB_Unit: data.SUB_Unit ?? '',
        Parts_Name: data.Parts_Name ?? '',
        Q_Code: data.Q_Code ?? '',
        Spec: data.Spec ?? '',
        Maker: data.Maker ?? '',
        EMS재고: Number(data.EMS재고 ?? 0),
        라인재고: Number(data.라인재고 ?? 0),
        UsageCount: Number(data.UsageCount ?? 0),
        _key: `${data.Q_Code ?? ''}|${data.Spec ?? ''}`
    });
    itemDT.draw(false);
}

function currentEqNameForEssential() {
    const eqpSel = document.getElementById('eqp-division-select');
    const v = eqpSel?.value || 'PLM_S';
    return (v === 'ACV_Y') ? 'ACV' : 'PLM';
}

function currentUnitForEssential() {
    const unitSel = document.getElementById('search-unit-select');
    const nameFromDropdown = unitSel?.value?.trim();
    // 드래프트/수정 시 제목에서 가져올 수도 있음
    if (nameFromDropdown) return nameFromDropdown;
    const titleEl = document.getElementById('parts-title');
    const t = titleEl?.textContent || '';
    const m = t.match(/\-\s(.+)$/); // "[ Parts 정보 ] - UNIT"
    return m ? m[1] : '';
}

/* ========== 이벤트 배선: 조회 탭 전용 ========== */
function wireSearchEditEvents() {
    if (wireSearchEditEvents.__wired) return;
    wireSearchEditEvents.__wired = true;

    // (섹션 전용) 수정 버튼 직접 핸들: 선택 1행 → 인라인 편집
    const searchWrap = document.querySelector('.search-wrapper');
    const editBtn = searchWrap?.querySelector('#edit-btn-3');
    if (editBtn && !editBtn.__wired) {
        editBtn.__wired = true;
        editBtn.addEventListener('click', () => {
            if (!itemDT) return;
            if (isEditingNowSearch()) {
            // 이미 편집 모드라면 → 저장 트리거는 상단 "저장" 버튼이 처리
            alert('이미 편집 중입니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
            }

            const nodes = itemDT.rows('.selected').nodes().toArray();
            if (nodes.length === 0) {
            alert('수정할 행을 선택해 주세요.');
            return;
            }

            const ctxs = [];
            let skippedDraft = 0;

            for (const rowNode of nodes) {
            if (rowNode.classList.contains('draft-row')) { skippedDraft++; continue; }
            const rowApi  = itemDT.row(rowNode);
            const before  = { ...rowApi.data() };
            const beforeKey = { 
                Q_Code: before.Q_Code, 
                Spec: before.Spec,
                EQ_Unit: currentUnitForEssential()   // ← 현재 유닛을 where에 포함
            };
            rowToInputsInSearch(rowNode);
            ctxs.push({ rowNode, rowApi, before, beforeKey });
            }

            if (ctxs.length === 0) {
            let msg = '편집 가능한 행이 없습니다.';
            if (skippedDraft) msg += `\n- 드래프트 제외: ${skippedDraft}건`;
            alert(msg);
            return;
            }

            searchEditCtxs = ctxs;
            setButtonsForModeSearch('editing');
            if (skippedDraft) alert(`선택 ${nodes.length}건 중 ${ctxs.length}건만 편집 모드로 전환했습니다.\n- 드래프트 제외: ${skippedDraft}건`);
        });
            
    }

    // 행추가
    document.addEventListener('parts-edit:add-row', () => {
        const wrap = document.querySelector('.search-wrapper');
        if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;
        if (isEditingNowSearch()) {
        alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
        return;
        }
        clearItemSelection();
        addDraftRowInSearch();
    });

    // 저장 (드래프트 → INSERT x2, 인라인수정(다중) → UPDATE x2)
    // ----------------------------------------------------------------------------
    document.addEventListener('parts-edit:save', async () => {
        const wrap = document.querySelector('.search-wrapper');
        if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

        try {
            // 1) 인라인 수정(다중) UPDATE
            if (Array.isArray(searchEditCtxs) && searchEditCtxs.length > 0) {
                // 1-0) 모든 편집행 입력 수집 & 검증
                const updates = []; // [{ctx, updated, before, beforeKey, newQ, newPN, beforeQ, beforePN}]
                for (const ctx of searchEditCtxs) {
                    const updated = collectInputsInSearch(ctx.rowNode);
                    const err = validateSearchRow({ ...ctx.before, ...updated }, ctx.rowNode);
                    if (err) {
                    alert(err);
                    document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: 'VALIDATION', message: err } }));
                    return;
                    }
                    const before   = ctx.before;
                    const beforeQ  = before.Q_Code || '';
                    const beforePN = before.Spec   || '';
                    const newQ     = (updated.Q_Code ?? before.Q_Code) || '';
                    const newPN    = (updated.Spec   ?? before.Spec)   || '';
                    updates.push({ ctx, updated, before, beforeKey: ctx.beforeKey, newQ, newPN, beforeQ, beforePN });
                }

                const EQ_Unit = currentUnitForEssential();
                const EQ_Name = currentEqNameForEssential();

                // 1-1) 먼저 parts_use_list UPDATE를 모두 수행
                for (const u of updates) {
                    const usePayload = {
                    where: u.beforeKey, // {Q_Code, Spec, EQ_Unit}
                    data: {
                        SUB_Unit:   u.updated.SUB_Unit   ?? u.before.SUB_Unit,
                        Parts_Name: u.updated.Parts_Name ?? u.before.Parts_Name,
                        Q_Code:     u.newQ,
                        Spec:       u.newPN,
                        Maker:      u.updated.Maker      ?? u.before.Maker,
                        EQ_Name:    EQ_Name,
                        EQ_Unit:    EQ_Unit
                    }
                    };
                    const rUse = await fetch('/parts_use_list/update', {
                    method: 'PUT',
                    headers: { 'Content-Type':'application/json' },
                    body: JSON.stringify(usePayload)
                    });
                    if (!rUse.ok) throw new Error(await rUse.text());
                }

                // 1-2) essential 보정
                //     Q_Code가 변경된 항목들만 새 Q 리스트 추출 → exists 1회 호출
                const changedQs = [...new Set(updates.filter(u => u.newQ !== u.beforeQ).map(u => u.newQ))];
                let existsMap = {};
                if (changedQs.length) {
                    try {
                    const ex = await fetch('/parts_essential_list/exists', {
                        method: 'POST',
                        headers: { 'Content-Type':'application/json' },
                        body: JSON.stringify({ qcodes: changedQs })
                    });
                    if (ex.ok) { const { exists = {} } = await ex.json(); existsMap = exists; }
                    } catch {}
                }

                for (const u of updates) {
                    if (u.newQ !== u.beforeQ) {
                    let targetExists = !!existsMap[u.newQ];
                    if (!targetExists) {
                        // 업데이트 시도 → 없으면 insert
                        const essPayload = {
                        where: { Q_Code: u.beforeQ, P_N: u.beforePN, EQ_Unit: EQ_Unit },
                        data: {
                            Parts_Name: u.updated.Parts_Name ?? u.before.Parts_Name,
                            Q_Code:     u.newQ,
                            P_N:        u.newPN,
                            EQ_Name:    EQ_Name,
                            EQ_Unit:    EQ_Unit
                        }
                        };
                        let needInsert = true;
                        try {
                        const rEssUpd = await fetch('/parts_essential_list/update', {
                            method: 'PUT',
                            headers: { 'Content-Type':'application/json' },
                            body: JSON.stringify(essPayload)
                        });
                        if (rEssUpd.ok) {
                            let json = null;
                            try { json = await rEssUpd.json(); } catch {}
                            const affected = json?.affected ?? json?.affectedRows ?? 0;
                            if (affected > 0) needInsert = false;
                        }
                        } catch { needInsert = true; }
                        if (needInsert) {
                        const rEssIns = await fetch('/parts_essential_list', {
                            method: 'POST',
                            headers: { 'Content-Type':'application/json' },
                            body: JSON.stringify({
                            EQ_Name: EQ_Name, EQ_Unit: EQ_Unit,
                            Parts_Name: u.updated.Parts_Name ?? u.before.Parts_Name,
                            Q_Code: u.newQ, P_N: u.newPN, Etc: null
                            })
                        });
                        if (!rEssIns.ok) throw new Error(await rEssIns.text());
                        }
                    }
                    } else {
                    const essPayload = {
                        where: { Q_Code: u.beforeQ, P_N: u.beforePN, EQ_Unit: EQ_Unit },
                        data: {
                        Parts_Name: u.updated.Parts_Name ?? u.before.Parts_Name,
                        P_N:        u.newPN,
                        EQ_Name:    EQ_Name,
                        EQ_Unit:    EQ_Unit
                        }
                    };
                    await fetch('/parts_essential_list/update', {
                        method: 'PUT',
                        headers: { 'Content-Type':'application/json' },
                        body: JSON.stringify(essPayload)
                    });
                    }
                }

                // 1-3) 로컬 테이블 갱신 (각 행 복원)
                for (const u of updates) {
                    const newRow = {
                    SUB_Unit:   u.updated.SUB_Unit   ?? u.before.SUB_Unit,
                    Parts_Name: u.updated.Parts_Name ?? u.before.Parts_Name,
                    Q_Code:     u.newQ,
                    Spec:       u.newPN,
                    Maker:      u.updated.Maker      ?? u.before.Maker,
                    EMS재고:    u.before.EMS재고 ?? 0,
                    라인재고:    u.before.라인재고 ?? 0,
                    UsageCount: u.before.UsageCount ?? 0,
                    };
                    restoreRowFromDataInSearch(u.ctx.rowNode, newRow);
                }
                searchEditCtxs = null;
            }
            

            // 2) 드래프트 INSERT (양쪽 테이블)
            const drafts = collectDraftRowsInSearch();
            const draftNodes = Array.from(document.querySelectorAll('#item-table tbody tr.draft-row'));
            if (drafts.length) {
            for (const d of drafts) {
                const err = validateSearchRow(d);
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: 'VALIDATION', message: err } }));
                if (err) return alert(err);
            }
            const EQ_Unit = currentUnitForEssential();
            const EQ_Name = currentEqNameForEssential();

            // use_list INSERT
            for (const d of drafts) {
                const useRow = {
                SUB_Unit: d.SUB_Unit, Parts_Name: d.Parts_Name,
                Q_Code: d.Q_Code, Spec: d.Spec, Maker: d.Maker,
                EQ_Name, EQ_Unit
                };
                const r = await fetch('/parts_use_list', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(useRow)
                });
                if (!r.ok) throw new Error(await r.text());
            }

            // essential INSERT (exists 체크)
            const qcodes = [...new Set(drafts.map(d => d.Q_Code).filter(Boolean))];
            let existsMap = {};
            if (qcodes.length) {
                const ex = await fetch('/parts_essential_list/exists', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify({ qcodes })
                });
                if (ex.ok) { const { exists = {} } = await ex.json(); existsMap = exists; }
            }
            for (const d of drafts) {
                if (existsMap[d.Q_Code]) continue;
                const essRow = {
                Parts_Name: d.Parts_Name, Q_Code: d.Q_Code, P_N: d.Spec, EQ_Unit, EQ_Name
                };
                const r = await fetch('/parts_essential_list', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(essRow)
                });
                if (!r.ok) throw new Error(await r.text());
            }

            // ✅ 드래프트를 '일반 행'으로 전환 (로컬)
            drafts.forEach((d, i) => {
                const node = draftNodes[i];
                if (!node) return;
                const newRow = {
                SUB_Unit: d.SUB_Unit,
                Parts_Name: d.Parts_Name,
                Q_Code: d.Q_Code,
                Spec: d.Spec,
                Maker: d.Maker,
                EMS재고: 0,
                라인재고: 0,
                UsageCount: 0
                };
                itemDT.row(node).data(newRow);
                node.classList.remove('draft-row', 'selected');
            });
            itemDT.draw(false);
            }

            // ✅ 전체 리프레시 없이 완료
            alert('저장되었습니다.');
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));
            refreshButtonsByStateSearch();
            
        } catch (e) {
            console.error(e);
            alert('Q-Code 값이 중복입니다.\n' + (e.message || e));
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: e?.message } }));
        }
    });


    // 삭제 (선택된 행 → parts_use_list 에서만 삭제)
    document.addEventListener('parts-edit:delete-row', async () => {
        const wrap = document.querySelector('.search-wrapper');
        if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;
        if (!itemDT) return;

        // 1) 드래프트가 선택되어 있으면 로컬 제거만
        const $tbody = $('#item-table tbody');
        const $draftSel = $tbody.find('tr.draft-row.selected');
        if ($draftSel.length) {
            $draftSel.each(function () { itemDT.row(this).remove(); });
            itemDT.draw(false);
        }

        // 2) 실제 삭제 대상(서버) 다시 읽기
        const selectedData = itemDT.rows('.selected').data().toArray();
        const selectedNodes = itemDT.rows('.selected').nodes().toArray(); // ← 나중에 로컬 삭제용
        if (!selectedData.length) return alert('삭제할 행을 선택해 주세요.');
        if (isEditingNowSearch()) return alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');

        // 드래프트가 아닌 실제 행만 서버 삭제 대상으로 구성
        const serverRows = selectedData
            .filter(r => !r._draft)
            .map(r => ({
                Q_Code: r.Q_Code,
                Spec: r.Spec,
                EQ_Unit: r.EQ_Unit || currentUnitForEssential?.() || currentSelectedUnit // 현재 선택된 유닛값
            }));
        if (!serverRows.length) {
            // 전부 드래프트였으면 여기서 종료(이미 로컬 삭제 처리됨)
            return;
        }
        if (!confirm(`선택한 ${serverRows.length}건을 삭제할까요? (사용 목록만 삭제)`)) return;

        

        try {
            // 3) 서버 삭제
            const resp = await fetch('/parts_use_list/delete', {
            method: 'POST',
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify({ rows: serverRows })
            });
            if (!resp.ok) throw new Error(await resp.text());

            // 4) 전체 재검색(runSearchClick) 없이, 선택된 행만 로컬에서 제거
            itemDT.rows(selectedNodes).remove().draw(false);

            alert('삭제되었습니다.');
            refreshButtonsByStateSearch(); // 버튼 상태 갱신
        } catch (e) {
            console.error(e);
            alert('삭제 중 오류가 발생했습니다.\n' + (e.message || e));
        }
    });

    // 취소 (드래프트 제거/수정 복원)
    document.addEventListener('parts-edit:cancel', () => {
        const wrap = document.querySelector('.search-wrapper');
        if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

        removeAllDraftRowsInSearch();

        if (Array.isArray(searchEditCtxs) && searchEditCtxs.length) {
            for (const ctx of searchEditCtxs) {
            restoreRowFromDataInSearch(ctx.rowNode, ctx.before);
            }
            searchEditCtxs = null;
        }
        // ✅ 취소 → 현재 상태로 버튼 갱신
        refreshButtonsByStateSearch();
    });
}

// 초기화 시점에 이벤트 배선 보장
document.addEventListener('DOMContentLoaded', () => {
  // ensureTable() 가 최초 렌더링 때 호출되어 버튼 DOM이 생김
  // 여기서는 이벤트 배선을 1회 보장
  wireSearchEditEvents();
  refreshButtonsByStateSearch();
});