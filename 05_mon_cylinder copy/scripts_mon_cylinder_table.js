/* ============================ 공통 유틸 ============================ */
const SENTINEL = 2147483647;
const selectedCylRows = new Set(); // 실린더 테이블 행 선택 상태

function formatValue(value) {
    if (value === SENTINEL || value === null || value === undefined) return '_';
    const num = Number(value);
    if (!Number.isFinite(num)) return '_';
    return (num / 100).toFixed(2);  // ✅ 화면 표기(/100)
}

async function fetchSnapshot() {
    const resp = await fetch('/api/cylinder_tact');
    if (!resp.ok) throw new Error('HTTP error');

    const json = await resp.json();
    if (!json?.success) throw new Error('API failed');

    //console.log('📌 Spec 값:', json.specs);
    //console.log('📌 Spec_Max 값:', json.specs_Max);
    return json; // { success, data, specs, specs_Max }
}

/* ============================ 전역(스펙 캐시) ============================ */

// FRONT→DB 매핑 유틸(없으면 frontKey 그대로)
function toDbKey(frontKey) {
    return (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
}

/* ============================ 테이블 구조 보장 ============================ */
function ensureTableStructure(table) {
    if (!table) throw new Error('table(#cylinder-table-container) not found');

    if (!table.querySelector('tbody')) {
      const tbody = document.createElement('tbody');
      tbody.id = 'cylinder-table-body';
      table.appendChild(tbody);
    }
    return table.querySelector('tbody');
  }

// 필터링 함수
function getCylinderKeysByGroup(targetGroup) {
    const isAll = targetGroup === 'ALL';

    return isAll
      ? Object.keys(cylinder)                                 // 전체
      : Object.keys(cylinder).filter(k => cylinder[k]?.group === targetGroup); // 특정 그룹
}

/* ============================ 헤더 빌드 ============================ */
function buildCylinderHeader(specs = {}, specs_Max = {}, opts = {}) {
    const { editMode = false, targetGroup = 'ALL' } = opts;
    const table = document.getElementById('cylinder-table-container');
    if (!table) return;

    // reset thead
    let thead = table.querySelector('thead');
    if (thead) thead.remove();
    thead = document.createElement('thead');
    table.prepend(thead);

    const cylKeys = getCylinderKeysByGroup(targetGroup);

    if (cylKeys.length === 0) {
        console.warn('선택된 그룹에 해당하는 유닛이 없습니다:', targetGroup);
        // 필요 시: cylKeys = Object.keys(cylinder);
    }

    // SHORT/LONG 존재 여부를 "축 이름"으로 판단
    const unitHasShortLong = (c) => (c?.unit_count === 4);

    const hasSecondRow = cylKeys.some(k => unitHasShortLong(cylinder[k]));

    // 전역 컬럼 시퀀스
    window.__CYL_DB_COLS = [];

    // 1행/2행
    const tr1 = document.createElement('tr');
    const tr2 = document.createElement('tr');

    // 좌측 고정 셀은 hasSecondRow 확정 후 rowSpan 설정
    const thType = document.createElement('th');
    thType.className = 'type-col';
    thType.textContent = 'Type';
    thType.rowSpan = hasSecondRow ? 5 : 4;

    const thEq = document.createElement('th');
    thEq.className = 'eq-name';
    thEq.textContent = '설비';
    thEq.rowSpan = hasSecondRow ? 5 : 4;

    const thWT = document.createElement('th');
    thWT.className = 'write-time';
    thWT.textContent = 'WriteTime';
    thWT.rowSpan = hasSecondRow ? 3 : 2;

    tr1.appendChild(thType);  // ✅ 추가
    tr1.appendChild(thEq);
    tr1.appendChild(thWT);

    // 유닛 헤더
    cylKeys.forEach(k => {
        const c = cylinder[k];
        if (!c) return;

        const thUnit = document.createElement('th');
        thUnit.textContent = c.name;
        thUnit.colSpan = c.unit_count; // 2 또는 4

        const hasSL = unitHasShortLong(c);

        if (hasSecondRow) {
        if (hasSL) {
            // 2행에 SHORT/LONG
            const thShort = document.createElement('th');
            thShort.colSpan = 2; thShort.textContent = 'SHORT';
            const thLong = document.createElement('th');
            thLong.colSpan = 2; thLong.textContent = 'LONG';
            tr2.appendChild(thShort);
            tr2.appendChild(thLong);
        } else {
            // 다른 유닛들 때문에 2행이 존재하므로, 2칸 유닛은 1~2행 병합
            thUnit.rowSpan = 2;
        }
        }
        // (hasSecondRow가 false면 tr2 자체가 없으니 rowSpan 주지 않음)

        tr1.appendChild(thUnit);

        // 컬럼 시퀀스(3행/바디/SPEC 공용)
        Object.values(c.pos || {}).forEach(p => window.__CYL_DB_COLS.push(p.name));
    });

    thead.appendChild(tr1);
    if (hasSecondRow) thead.appendChild(tr2);

    // 3행: DN/UP/FWD/BWD
    const tr3 = document.createElement('tr');
    const labelFromPos = (posName) => {
        const last = String(posName || '').trim().split('_').pop()?.toUpperCase();
        if (last === 'DOWN' || last === 'DN') return 'DN';
        if (last === 'UP') return 'UP';
        if (last === 'FWD') return 'FWD';
        if (last === 'BWD') return 'BWD';
        return last || '';
    };
    
    cylKeys.forEach(k => {
        const c = cylinder[k];
        Object.values(c.pos || {}).forEach(pos => {
            const th = document.createElement('th');

            // ✅ 이 컬럼이 FDC 대상(DB 컬럼명)이라면 아이콘 추가
            const dbKey = toDbKey(pos?.name); // 이미 있는 유틸: FRONT→DB 변환
            if (FDC_UNITS.includes(dbKey)) {
                th.classList.add('has-fdc');     // 기준 컨테이너
            }

            // 라벨
            const label = document.createElement('span');
            label.textContent = labelFromPos(pos?.name);
            th.appendChild(label);
            tr3.appendChild(th);
        });
    });
    thead.appendChild(tr3);

    // 4/5행: SPEC 상/하한
    const tr4 = document.createElement('tr'); tr4.classList.add('spec-row-upper');
    const tr5 = document.createElement('tr'); tr5.classList.add('spec-row-lower');

    const thSpecLabel = document.createElement('th');
    thSpecLabel.rowSpan = 2;
    thSpecLabel.classList.add('spec-title-cell');
    thSpecLabel.innerHTML = `
        <div class="spec-cell-inner">
        <div class="spec-label-left">SPEC</div>
        <div class="spec-label-right">
            <div style="color:#cc4444;">상한Limit</div>
            <div style="color:#0077cc;">하한Limit</div>
        </div>
        </div>`;
    tr4.appendChild(thSpecLabel);

    (window.__CYL_DB_COLS || []).forEach(frontKey => {
        const dbKey = toDbKey(frontKey);
        const vU = formatValue(specs_Max?.[dbKey]);
        const thU = document.createElement('th');
        thU.classList.add('spec-upper');
        thU.innerHTML = editMode
        ? `<input class="spec-input spec-input-upper" data-key="${dbKey}" type="text" value="${vU === '_' ? '' : vU}">`
        : vU;
        tr4.appendChild(thU);
    });

    (window.__CYL_DB_COLS || []).forEach(frontKey => {
        const dbKey = toDbKey(frontKey);
        const vL = formatValue(specs?.[dbKey]);
        const thL = document.createElement('th');
        thL.classList.add('spec-lower');
        thL.innerHTML = editMode
        ? `<input class="spec-input spec-input-lower" data-key="${dbKey}" type="text" value="${vL === '_' ? '' : vL}">`
        : vL;
        tr5.appendChild(thL);
    });

    thead.appendChild(tr4);
    thead.appendChild(tr5);
}

/* ============================ 스펙 임계값 준비(/100) ============================ */
function prepareSpecThresholds(specs = {}, specs_Max = {}) {
    window.__SPEC_LOWER_DISPLAY = {};
    window.__SPEC_UPPER_DISPLAY = {};

    (window.__CYL_DB_COLS || []).forEach(frontKey => {
        const dbKey = toDbKey(frontKey);

        const lowNum = Number(specs?.[dbKey]);
        const upNum  = Number(specs_Max?.[dbKey]);

        window.__SPEC_LOWER_DISPLAY[dbKey] =
        Number.isFinite(lowNum) && lowNum !== SENTINEL ? (lowNum / 100) : null;

        window.__SPEC_UPPER_DISPLAY[dbKey] =
        Number.isFinite(upNum) && upNum !== SENTINEL ? (upNum / 100) : null;
    });
}

/* ============================ 하이라이트 적용 ============================ */

function highlightCellBySpec(td, rawValueFromDB, dbKey) {
    td.classList.remove('is-over-upper', 'is-under-lower', 'cell-critical');
    delete td.dataset.abnormal;

    const rawNum = Number(rawValueFromDB);
    if (!Number.isFinite(rawNum) || rawNum === SENTINEL || rawValueFromDB == null) return;

    const valDisp = rawNum / 100;
    const upper = window.__SPEC_UPPER_DISPLAY?.[dbKey];
    const lower = window.__SPEC_LOWER_DISPLAY?.[dbKey];
    const EPS = 1e-9;

    // SPEC 여부
    if (upper != null && valDisp > upper + EPS) {
        td.classList.add('is-over-upper');
        td.dataset.abnormal = 'upper';
    } else if (lower != null && valDisp < lower - EPS) {
        td.classList.add('is-under-lower');
        td.dataset.abnormal = 'lower';
    }

    // 중요위치 여부(항상 별도로 추가)
    if (CRITICAL_UNITS.some(unit => dbKey.includes(unit))) {
        td.classList.add('cell-critical');
    }
}
/* ============================ 바디 빌드 ============================ */
function buildRows(tbody, data, valueKeys) {
    tbody.innerHTML = '';
    const rows = (data || []).filter(r => r.EQ_Name !== 'spec' && r.EQ_Name !== 'spec_Max');

    rows.forEach(row => {
        const tr = document.createElement('tr');

        const rowKey = String(row.EQ_Name ?? '') + '|' + String(row.WriteTime ?? '');
        tr.setAttribute('data-key', rowKey);

        
        const rawEqName = row.EQ_Name ?? '';
        const displayEqName = EqNameUtils.transform(rawEqName);

        // ✅ Type 컬럼
        const tdType = document.createElement('td');
        tdType.classList.add('type-col');

        const t = window.getEqSpecType ? window.getEqSpecType(rawEqName) : "phone";
        tdType.textContent = (t === "foldable") ? "Foldable" : "Phone";

        tr.appendChild(tdType);

        // ✅ EQ_Name
        const tdEq = document.createElement('td');
        tdEq.textContent = displayEqName;
        tdEq.classList.add('eq-name');
        tr.appendChild(tdEq);

        // ✅ WriteTime
        const tdTime = document.createElement('td');
        tdTime.textContent = row.WriteTime ?? '';
        tdTime.classList.add('write-time');
        tr.appendChild(tdTime);

        valueKeys.forEach(frontKey => {
        const dbKey = toDbKey(frontKey);
        const td = document.createElement('td');

        if (!(dbKey in row)) {
            console.warn(`[DEBUG] ${row.EQ_Name} missing key: ${dbKey}`);
        }

        const raw = row[dbKey];                 // DB 원본 스케일
        td.dataset.raw = (raw ?? '');       
        td.textContent = formatValue(raw);  // 화면 표기(/100)
        highlightCellBySpec(td, raw, dbKey);    // ✅ 하이라이트


        td.dataset.eq = row.EQ_Name; // 인폼 식별자 심기
        td.dataset.unit = dbKey; // 인폼 식별자 심기

        tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });
}

/* ============================ DataTables ============================ */

function syncFdcHeaderClasses() {
  const heads = document.querySelectorAll(
    '#cylinder-table-container thead, .dataTables_scrollHead thead, .fixedHeader-floating thead'
  );
  heads.forEach(h => {
    h.querySelectorAll('th').forEach(th => {
      if (th.querySelector('img.fdc-icon')) th.classList.add('has-fdc');
    });
  });
}

function finalizeCylinderTable() {
    if ($.fn.DataTable.isDataTable("#cylinder-table-container")) {
        $("#cylinder-table-container").DataTable().destroy();
    }
    const dataTable = $("#cylinder-table-container").DataTable({
        scrollY: '610px',
        scrollX: true,
        scrollCollapse: true,
        paging: false,
        searching: true,
        fixedHeader: true,
        orderCellsTop: true,
        info: false,
        dom:'t'
    });

    const $table = $('#cylinder-table-container'); // ✅ 바인딩 대상

    // ✅ 공용 선택 바인딩(Shift 구간, Ctrl/⌘ 토글 지원)
    bindDTSelectable($table, {
        selectedSet: selectedCylRows,
        getKey: (tr) => tr.getAttribute('data-key'),
        // 필요하면 draw 후에도 선택 유지하려면 true로:
        restoreOnDraw: false,
        // 선택 변경시 필요한 후속 처리 있으면 연결 (선택만 바뀌면 요약엔 영향 X)
        onChange: () => {}

    });

    // ✅ 테이블 바깥 클릭 시 전체 해제(버튼 제외)
    wireOutsideDeselect($table, { 
        selectedSet: selectedCylRows, 
        preserveSelectors: ['button', '.status-btn'] 
    });

    
    syncFdcHeaderClasses(); // 헤더 복제 이후 기준 클래스/아이콘 기준 동기화
    renderEqStatusFromDataTable(dataTable); //  최초 렌더 시 요약 갱신
    wireAdvancedFilters(dataTable); // ✅ 체크박스 고급 필터 연결

    // ✅ 검색/필터/정렬 등으로 draw될 때마다 요약 갱신
    $('#cylinder-table-container').off('draw.dataTable.__status').on('draw.dataTable.__status', function () {
        renderEqStatusFromDataTable(dataTable);
    });

  return dataTable; // 필요 시 renderTable()에서 받을 수 있게
}

/* ============================ 전체 렌더 ============================ */
async function renderTable(targetGroup = (window.__ACTIVE_GROUP || 'ALL')) {
    const table = document.getElementById('cylinder-table-container');
    const loader = document.getElementById('loading-overlay');

    try {
        if (loader) loader.style.display = 'flex';

        if ($.fn.DataTable.isDataTable("#cylinder-table-container")) {
        $("#cylinder-table-container").DataTable().destroy();
        }

        const { data, specs, specs_Max } = await fetchSnapshot();
        const editMode = !!window.__specEditMode;

        // ✅ 선택된 그룹으로 헤더 구성
        buildCylinderHeader(specs, specs_Max, { editMode, targetGroup });

        // 스펙 캐시(/100)
        prepareSpecThresholds(specs, specs_Max);

        // 바디 렌더
        const tbody = ensureTableStructure(table);
        const clean = (data || []).filter(r => r.EQ_Name !== 'spec' && r.EQ_Name !== 'spec_Max');

        // 헤더에서 만든 시퀀스 그대로 사용
        let valueKeys = Array.isArray(window.__CYL_DB_COLS) ? window.__CYL_DB_COLS.slice() : [];
        if (!valueKeys.length) valueKeys = Object.keys(COLUMN_MAPPING);

        buildRows(tbody, clean, valueKeys);
        const dataTable = finalizeCylinderTable();

        applyFiltersToDataTable();
        renderEqStatusFromDataTable(dataTable);

        if (window.__loadAndPaintCylinderInforms) {
            window.__loadAndPaintCylinderInforms();
        }

    } catch (e) {
        console.error('Failed to render cylinder table:', e);
        const tbody = document.getElementById('cylinder-table-body');
        if (tbody) {
        const fallbackCols = 2 + (window.__CYL_DB_COLS?.length || 20);
        tbody.innerHTML = `<tr><td colspan="${fallbackCols}">데이터 로드 실패</td></tr>`;
        }
    } finally {
        if (loader) loader.style.display = 'none';
    }
}

/* ============================ 버튼 클릭 바인딩 ============================ */

document.addEventListener('DOMContentLoaded', () => {
    const container = document; // 버튼이 동적으로 생겨도 동작하도록 위임

    // 초기 활성화: 저장된 그룹 or ALL
    const initialGroup = window.__ACTIVE_GROUP || 'ALL';
    const initialBtn = document.querySelector(`.bay-group-btn[data-group="${initialGroup}"]`)
                        || document.querySelector('.bay-group-btn');
    if (initialBtn) initialBtn.classList.add('active');

    // 클릭 시 active 토글 + 렌더
    container.addEventListener('click', (e) => {
        const btn = e.target.closest('.bay-group-btn');
        if (!btn) return;

        // 모든 버튼 active 제거 후 선택만 활성화
        document.querySelectorAll('.bay-group-btn.active')
        .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // 선택 그룹 저장 & 렌더
        const selectedGroup = btn.getAttribute('data-group');
        window.__ACTIVE_GROUP = selectedGroup;
        renderTable(selectedGroup);
    });
});


/* ============================ 모니터링 요약 집계 ============================ */

// td가 SPEC 초과/미만인지 구분
function isUpperOutlierCell(td) {
    return td.dataset.abnormal === 'upper' || td.classList.contains('is-over-upper');
}
function isLowerOutlierCell(td) {
    return td.dataset.abnormal === 'lower' || td.classList.contains('is-under-lower');
}
function isCriticalCell(td) {
    return td.classList.contains('cell-critical') || td.dataset.isCritical === '1';
}


// ✅ 하한 미만(빠름) — 전체 위치 기준
function countLowerOutliersFromNodes(rowNodes) {
    let lowerCount = 0;
    rowNodes.forEach(tr => {
        const tds = Array.from(tr.cells).slice(3);
        tds.forEach(td => { if (isLowerOutlierCell(td)) lowerCount++; });
    });
    return lowerCount;
}
// ✅ 상한 초과(느림) — 전체 위치 기준 (기존과 동일)
function countUpperOutliersFromNodes(rowNodes) {
    let upperCount = 0;
    rowNodes.forEach(tr => {
        const tds = Array.from(tr.cells).slice(3);
        tds.forEach(td => { if (isUpperOutlierCell(td)) upperCount++; });
    });
    return upperCount;
}

// ✅ 중요위치 중 하한 미만(빠름)
function countImportantLowerOutliersFromNodes(rowNodes) {
    let cnt = 0;
    rowNodes.forEach(tr => {
        const tds = Array.from(tr.cells).slice(3);
        tds.forEach(td => {
        if (isCriticalCell(td) && isLowerOutlierCell(td)) cnt++;
        });
    });
    return cnt;
}

// ✅ 중요위치 중 상한 초과(느림)
function countImportantUpperOutliersFromNodes(rowNodes) {
    let cnt = 0;
    rowNodes.forEach(tr => {
        const tds = Array.from(tr.cells).slice(3);
        tds.forEach(td => {
        if (isCriticalCell(td) && isUpperOutlierCell(td)) cnt++;
        });
    });
    return cnt;
}


// ✅ DataTable 기준으로 요약 렌더
function renderEqStatusFromDataTable(dataTable) {
    if (!dataTable) return;

    // 현재 검색/필터 적용된 행 DOM 노드들
    const rowNodes = dataTable.rows({ search: 'applied' }).nodes().toArray();

    const totalCount = rowNodes.length;

    // 이상 설비(행) = 행의 데이터셀 중 하나라도 이상치가 있으면
    let abnormalCount = 0;
    rowNodes.forEach(tr => {
        const tds = Array.from(tr.cells).slice(3);
        const hasOutlier = tds.some(td => isUpperOutlierCell(td) || isLowerOutlierCell(td));
        if (hasOutlier) abnormalCount++;
    });

    const normalCount = Math.max(0, totalCount - abnormalCount);
    const normalRate = totalCount === 0 ? 0 : Math.round((normalCount / totalCount) * 100);

    const impLower = countImportantLowerOutliersFromNodes(rowNodes);
    const impUpper = countImportantUpperOutliersFromNodes(rowNodes);
    const impTotal = impLower + impUpper;

    const lowerCount = countLowerOutliersFromNodes(rowNodes);
    const upperCount = countUpperOutliersFromNodes(rowNodes);

    const box = document.querySelector('.spec_mon_group');
    if (!box) return;

    // ✅ summary-card 방식
    box.innerHTML = `
        <div class="summary-cards">
            <div class="summary-card st-total" data-card="total">
                <div class="label">전체설비</div>
                <div class="value">${totalCount.toLocaleString()}</div>
            </div>

            <div class="summary-card st-abn" data-card="abnormal">
                <div class="label">이상설비</div>
                <div class="value">${abnormalCount.toLocaleString()}</div>
            </div>

            <div class="summary-card st-rate" data-card="tttm">
                <div class="label">TTTM</div>
                <div class="value">${normalRate.toLocaleString()}%</div>
            </div>

            <div class="summary-card st-imp summary-click" data-card="critical">
                <div class="label">중요 이상치</div>
                <div class="sub">Fast ${impLower} / Slow ${impUpper}</div>
                <div class="value">${impTotal.toLocaleString()}</div>
            </div>

            <div class="summary-card st-fast summary-click" data-card="lower">
                <div class="label">이상치</div>
                <div class="sub">Fast</div>
                <div class="value">${lowerCount.toLocaleString()}</div>
            </div>

            <div class="summary-card st-slow summary-click" data-card="upper">
                <div class="label">이상치</div>
                <div class="sub">Slow</div>
                <div class="value">${upperCount.toLocaleString()}</div>
            </div>
        </div>
    `;

    // ✅ 카드 클릭 시 기존 “점프” 기능 재사용 (원치 않으면 이 블록 삭제)
    box.querySelector('.summary-cards')?.addEventListener('click', (e) => {
        const card = e.target.closest('.summary-card');
        if (!card) return;

        const type = card.dataset.card;
        if (type === 'critical') jumpToNext('critical');
        else if (type === 'lower') jumpToNext('lower');
        else if (type === 'upper') jumpToNext('upper');
    });
}


// thead의 SPEC 두 줄만 값 갱신
function updateSpecRowsDisplay(specs = {}, specs_Max = {}) {
  const { activeHead: thead, mirrorHead } = getHeads();
  if (!thead) return;

  const heads = mirrorHead ? [thead, mirrorHead] : [thead];
  const fronts = (window.__CYL_DB_COLS || []).slice();

  heads.forEach(h => {
    const trUpper = h.querySelector('tr.spec-row-upper');
    const trLower = h.querySelector('tr.spec-row-lower');
    if (!trUpper || !trLower) return;

    const upperCells = Array.from(trUpper.querySelectorAll('th')).slice(1); // 라벨 제외
    const lowerCells = Array.from(trLower.querySelectorAll('th'));

    upperCells.forEach((th, i) => {
      const dbKey = toDbKey(fronts[i]);
      th.textContent = formatValue(specs_Max?.[dbKey]);
    });
    lowerCells.forEach((th, i) => {
      const dbKey = toDbKey(fronts[i]);
      th.textContent = formatValue(specs?.[dbKey]);
    });
  });
}

// 스펙 캐시 갱신 후 바디 전체 재하이라이트
function rehighlightBodyBySpec(dt) {
  const meta = getColumnMeta(dt); // colIdx→frontKey/dbKey 매핑
  if (!meta.length) return;

  // 표시 중인 모든 행 노드
  const rowNodes = dt.rows({ search: 'applied' }).nodes().toArray();
  rowNodes.forEach(tr => {
    meta.forEach(({ colIdx, dbKey }) => {
      const td = tr.cells[colIdx];
      if (!td) return;
      const raw = td.dataset.raw !== undefined && td.dataset.raw !== '' ? Number(td.dataset.raw) : NaN;
      // raw가 없다면 표시값에서 역산(표시는 /100 적용) – fallback
      const fallbackRaw = Number.isFinite(raw) ? raw : Math.round((parseFloat(td.textContent) || 0) * 100);
      highlightCellBySpec(td, fallbackRaw, dbKey);
    });
  });

  // 열 필터(중요/이상치) 재적용 + 요약 갱신
  applyColumnFilters(dt);
  renderEqStatusFromDataTable(dt);
}

// SPEC만 소프트 리프레시 (스크롤 유지)
async function softRefreshSpecsAndStyles() {
  try {
    const dt = $("#cylinder-table-container").DataTable();

    // 1) 최신 SPEC만 재조회
    const { specs, specs_Max } = await fetchSnapshot();

    // 2) 캐시 갱신
    prepareSpecThresholds(specs, specs_Max);

    // 3) thead의 SPEC 라인 값만 갱신
    updateSpecRowsDisplay(specs, specs_Max);

    // 4) 바디 하이라이트만 재계산
    rehighlightBodyBySpec(dt);

    // 5) 레이아웃 보정
    dt.columns.adjust().draw(false);
  } catch (e) {
    console.error('softRefreshSpecsAndStyles failed:', e);
    // 문제가 있으면 최후의 수단으로 전체 렌더 (기존 동작)
    await renderTable(window.__ACTIVE_GROUP || 'ALL');
  }
}


/* ============================ 초기 진입 ============================ */
document.addEventListener("DOMContentLoaded", () => {
    checkClientAuthority();
    renderTable();
});