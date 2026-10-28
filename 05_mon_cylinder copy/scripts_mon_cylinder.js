let __specEditMode = false;
let __originalSpecs = {};     // 하한 캐시
let __originalSpecsMax = {};  // 상한 캐시

/* ============ 버튼 바인딩 ============ */

// 권한 체크 및 사용자 정보 표시
async function checkClientAuthority() {
    try {
        const response = await fetch('/check-client-authority');
        const data = await response.json();

        window.currentUserInfo = data;   // ✅ 전역 저장

        setAdminMode(data.isAdmin);

        const userNameContainer = document.createElement('div');
        userNameContainer.id = 'user-name-display';

        const userNameBox = document.createElement('div');
        userNameBox.id = 'user-name-box';
        userNameBox.innerText = `${data.authority} 모드 ( ${data.userName} )`;

        userNameContainer.appendChild(userNameBox);
        document.body.prepend(userNameContainer);
    } catch (error) {
        console.error('Error checking client authority:', error);
    }
}

function setAdminMode(isAdmin) {
    // 기존 버튼 (예: 공지사항 작성 버튼)
    const writeButton = document.getElementById('notice-write-button');
    if (writeButton) writeButton.style.display = isAdmin ? 'inline-block' : 'none';

    // ✅ SPEC 변경 버튼도 관리자일 때만 표시
    const specEditBtn = document.getElementById('toggle-spec-edit-btn');
    if (specEditBtn) specEditBtn.style.display = isAdmin ? 'inline-block' : 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('toggle-spec-edit-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        if (!__specEditMode) {
            // 편집 모드 진입
            try {
            cacheOriginalSpecs();
            enterSpecEditMode();
            btn.textContent = '저장';
            __specEditMode = true;
            } catch (e) {
            console.error(e);
            alert('편집 모드 진입 중 오류가 발생했습니다.');
            __specEditMode = false;
            btn.textContent = 'SPEC 변경';
            }
        } else {
            // 저장 처리
            try {
            const invalid = validateUpperLower();
            if (invalid) return;

            // ✅ 변경된 스펙 및 로그 수집
            const { specsLowerDB, specsUpperDB, logs } = collectChangedSpecsOnly();

            if (Object.keys(specsLowerDB).length === 0 && Object.keys(specsUpperDB).length === 0) {
                alert('변경된 값이 없습니다.');
                __specEditMode = false;
                btn.textContent = 'SPEC 변경';
                exitEditModeToPlainText();
                return;
            }

            // ✅ logs 그대로 전달
            await saveSpecsToServer(specsLowerDB, specsUpperDB, logs);

            __specEditMode = false;
            btn.textContent = 'SPEC 변경';
            await softRefreshSpecsAndStyles(); // 최신 스펙으로 재렌더
            } catch (e) {
            console.error(e);
            alert('SPEC 저장 중 오류가 발생했습니다. 콘솔을 확인하세요.');
            }
        }
    });
});

function getHeads() {
    const table = document.getElementById('cylinder-table-container');
    // DataTables가 있으면 보이는 헤더는 scrollHead 안에 있음
    const dtHead = document.querySelector('.dataTables_scrollHead thead');
    const realHead = table?.querySelector('thead') || null;

    // 보이는 헤더 우선
    const activeHead = dtHead || realHead;
    // 미러(동일하게 복제해줄 대상) — 둘 다 있으면 둘 다 업데이트
    const mirrorHead = (dtHead && realHead && dtHead !== realHead) ? realHead : null;

    return { activeHead, mirrorHead };
}

/* ============ 기존 SPEC 값 캐시 ============ */
function cacheOriginalSpecs() {
    __originalSpecs = {};
    __originalSpecsMax = {};

    const { activeHead: thead } = getHeads();
    if (!thead) return;

    const specRows = thead.querySelectorAll('tr.spec-row-upper, tr.spec-row-lower');
    if (specRows.length < 2) return;
    const [trUpper, trLower] = specRows;

    const frontKeys = window.__CYL_DB_COLS?.slice() || [];
    const upperCells = Array.from(trUpper.querySelectorAll('th')).slice(1); // SPEC 라벨 제외
    const lowerCells = Array.from(trLower.querySelectorAll('th'));

    upperCells.forEach((cell, i) => {
      const frontKey = frontKeys[i];
      const dbKey = (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
      const num = parseFloat(cell.textContent);
      __originalSpecsMax[dbKey] = isNaN(num) ? null : num;
    });

    lowerCells.forEach((cell, i) => {
      const frontKey = frontKeys[i];
      const dbKey = (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
      const num = parseFloat(cell.textContent);
      __originalSpecs[dbKey] = isNaN(num) ? null : num;
    });
}

/* ============ 편집 모드 진입 (헤더 셀 → input 교체) ============ */
function enterSpecEditMode() {
    // DataTable 복제 헤더 제거
    const { activeHead: thead, mirrorHead } = getHeads();
    if (!thead) return;

    const specRows = thead.querySelectorAll('tr.spec-row-upper, tr.spec-row-lower');
    if (specRows.length < 2) return;
    const [trUpper, trLower] = specRows;

    const frontKeys = window.__CYL_DB_COLS?.slice() || [];
    const upperCells = Array.from(trUpper.querySelectorAll('th')).slice(1);
    const lowerCells = Array.from(trLower.querySelectorAll('th'));

    upperCells.forEach((th, i) => {
      const frontKey = frontKeys[i];
      const dbKey = (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
      const currentText = th.textContent.trim();
      th.innerHTML = `<input class="spec-input upper" type="number" step="0.01"
                        data-bound="upper" data-db-key="${dbKey}"
                        value="${currentText === '_' ? '' : currentText}">`;
    });

    lowerCells.forEach((th, i) => {
      const frontKey = frontKeys[i];
      const dbKey = (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
      const currentText = th.textContent.trim();
      th.innerHTML = `<input class="spec-input lower" type="number" step="0.01"
                        data-bound="lower" data-db-key="${dbKey}"
                        value="${currentText === '_' ? '' : currentText}">`;
    });
}

/* ============ 보기 모드 텍스트로 복귀 (변경 없이 취소 시) ============ */
function exitEditModeToPlainText() {
    const { activeHead: thead, mirrorHead } = getHeads();
    if (!thead) return;

    const specRows = thead.querySelectorAll('tr.spec-row-upper, tr.spec-row-lower');
    if (specRows.length < 2) return;
    const [trUpper, trLower] = specRows;

    const upperCells = Array.from(trUpper.querySelectorAll('th')).slice(1);
    const lowerCells = Array.from(trLower.querySelectorAll('th'));

    upperCells.forEach((th) => {
      const input = th.querySelector('input');
      if (input) th.textContent = input.value || '_';
    });
    lowerCells.forEach((th) => {
      const input = th.querySelector('input');
      if (input) th.textContent = input.value || '_';
    });

    // 편의상 DataTable 재생성
    finalizeCylinderTable();
}

/* ============ 상/하한 교차 검증 ============ */
function validateUpperLower() {
    const { activeHead: thead } = getHeads();
    if (!thead) return false;

    const upperInputs = Array.from(thead.querySelectorAll('input.spec-input.upper'));
    const lowerInputs = Array.from(thead.querySelectorAll('input.spec-input.lower'));
    if (upperInputs.length !== lowerInputs.length) return false;

    for (let i = 0; i < upperInputs.length; i++) {
      const u = parseFloat(upperInputs[i].value);
      const l = parseFloat(lowerInputs[i].value);
      if (Number.isFinite(u) && Number.isFinite(l) && u < l) {
        alert(`⚠️ 상한값은 하한값보다 작을 수 없습니다. (index ${i + 1})`);
        return true;
      }
    }
    return false;
}

/* ============ 변경분만 수집 (SCALE 기본 1) ============ */
function collectChangedSpecsOnly() {
  const { activeHead: thead } = getHeads();
  if (!thead) return { specsLowerDB: {}, specsUpperDB: {}, logs: [] };

  const upperInputs = thead.querySelectorAll('input.spec-input.upper');
  const lowerInputs = thead.querySelectorAll('input.spec-input.lower');

  const specsUpperDB = {};
  const specsLowerDB = {};
  const changeLogs = []; // ✅ 추가: 로그 배열

  // 현재 선택된 설비 (없으면 'ALL')
  const selectedEq = (document.getElementById('eqp-select')?.value || '').trim() || 'ALL';
  const SCALE_DB = 100; // DB 저장 스케일

  // 상한 변경 수집
  upperInputs.forEach(input => {
    const dbKey = input.dataset.dbKey;
    const raw = input.value.trim();
    if (raw === '') return;
    const next = Number(raw);
    if (!Number.isFinite(next)) return;

    const prev = __originalSpecsMax[dbKey];   // 화면 스케일(실수)
    if (prev === next) return;                // 변경 없음

    specsUpperDB[dbKey] = Math.round(next * SCALE_DB);

    const frontKey = toFrontKey(dbKey);
    changeLogs.push({
      EQ_Name: selectedEq,
      Unit: frontKey,
      Previous_spec: prev,   // 화면 스케일 그대로
      Next_spec: next,       // 화면 스케일 그대로
      Bound: 'upper',
      Comment: `Spec 변경 : ${prettyUnit(frontKey)} ${boundKoLabel('upper')} ${prev ?? '_'}→${next}으로 변경`
    });
  });

  // 하한 변경 수집
  lowerInputs.forEach(input => {
    const dbKey = input.dataset.dbKey;
    const raw = input.value.trim();
    if (raw === '') return;
    const next = Number(raw);
    if (!Number.isFinite(next)) return;

    const prev = __originalSpecs[dbKey];      // 화면 스케일(실수)
    if (prev === next) return;                // 변경 없음

    specsLowerDB[dbKey] = Math.round(next * SCALE_DB);

    const frontKey = toFrontKey(dbKey);
    changeLogs.push({
      EQ_Name: selectedEq,
      Unit: frontKey,
      Previous_spec: prev,
      Next_spec: next,
      Bound: 'lower',
      Comment: `Spec 변경 : ${prettyUnit(frontKey)} ${boundKoLabel('lower')} ${prev ?? '_'}→${next}으로 변경`
    });
  });

  return { specsLowerDB, specsUpperDB, logs: changeLogs };
}

/* ============ 서버 저장 (배치 PUT) ============ */
async function saveSpecsToServer(specsLowerDB, specsUpperDB, logs = []) {
    const body = { specs: specsLowerDB, specs_Max: specsUpperDB, logs };

    const resp = await fetch('/api/cylinder_tact/specs', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    if (!resp.ok) {
        const raw = await resp.text().catch(() => '');
        throw new Error('SPEC_SAVE_FAILED: ' + raw);
    }

    const json = await resp.json();
    if (!json.success) {
        throw new Error(json.error || 'SPEC_SAVE_FAILED');
    }
}

/* ============ ESC로 취소 ============ */
document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && __specEditMode) {
      e.preventDefault();
      __specEditMode = false;

      const btn = document.getElementById("toggle-spec-edit-btn");
      if (btn) btn.textContent = "SPEC 변경";

      // 변경사항 반영 없이 보기 모드로
      exitEditModeToPlainText();
    }
});

/* ============================ 필터링/드롭다운 유틸 ============================ */

// Shift 옵션 채우기
function populateShiftDropdown() {
    const sel = document.querySelector('#shift-select');
    if (!sel) return;
    sel.innerHTML = '';

    const optAll = document.createElement('option');
    optAll.value = '';            // 전체(미선택)
    optAll.textContent = 'Shift 전체';
    sel.appendChild(optAll);

    // 값은 a/b/c/d 로 두고, 라벨은 A조/B조/...
    const items = [
      {v:'a', label:'A조'},
      {v:'b', label:'B조'},
      {v:'c', label:'C조'},
      {v:'d', label:'D조'},
    ];
    items.forEach(({v, label}) => {
      const op = document.createElement('option');
      op.value = v;
      op.textContent = label;
      sel.appendChild(op);
    });
}

// Shift에 맞춰 설비 옵션 채우기 (미선택이면 전체 eqpList)
function populateEqpDropdown(shiftKey = '') {
    const sel = document.querySelector('#eqp-select');
    if (!sel) return;

    const allEqps = Array.isArray(window.eqpList) ? window.eqpList.slice() : [];
    const byShift = (shiftKey && window.shiftGroups?.[shiftKey]) ? window.shiftGroups[shiftKey] : allEqps;

    const list = (byShift || []).slice().sort((a,b) => a.localeCompare(b));
    sel.innerHTML = '';

    const optAll = document.createElement('option');
    optAll.value = '';
    optAll.textContent = 'EQP 전체';
    sel.appendChild(optAll);

    list.forEach(eq => {
        const op = document.createElement('option');
        op.value = eq;
        op.textContent = eq;
        sel.appendChild(op);
    });
}

// DataTables에 필터 적용 (EQ_Name은 1열(0-인덱스)임)
function applyFiltersToDataTable() {
    const dt = $.fn.DataTable.isDataTable("#cylinder-table-container")
        ? $("#cylinder-table-container").DataTable()
        : null;
    if (!dt) return;

    const shiftKey = (document.querySelector('#shift-select')?.value || '').trim();
    const selEqp   = (document.querySelector('#eqp-select')?.value   || '').trim();
    const selType  = (document.querySelector('#type-select')?.value  || 'all').trim();

    /* ---------------- EQ / Shift 필터 ---------------- */

    if (selEqp) {
        const exact = '^' + selEqp.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$';
        dt.column(1).search(exact, true, false);  // EQ_Name은 1번
    } 
    else if (shiftKey && window.shiftGroups?.[shiftKey]?.length) {
        const group = '^(?:' + window.shiftGroups[shiftKey]
            .map(e => e.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
            .join('|') + ')$';
        dt.column(1).search(group, true, false);
    } 
    else {
        dt.column(1).search('', false, false);
    }

    /* ---------------- Type 필터 ---------------- */

    if (selType !== 'all') {
        const exactType = '^' + selType.charAt(0).toUpperCase() + selType.slice(1) + '$';
        dt.column(0).search(exactType, true, false);  // Type은 0번
    } else {
        dt.column(0).search('', false, false);
    }

    dt.draw(false);
    renderEqStatusFromDataTable(dt);
}

function initShiftFromQuery() {
    const sp = new URLSearchParams(location.search);
    const shift = (sp.get('shift') || '').toLowerCase();

    // 유효한 값만 허용
    const valid = ['a', 'b', 'c', 'd', 'all'];
    if (!valid.includes(shift)) return;

    const select = document.getElementById('shift-select');
    if (!select) return;

    // 드롭다운 선택 적용
    select.value = shift;

    // 내부 상태 갱신
    currentEqFilterGroup = shift;
    lastAppliedEqGroup = null; // eqp-select 초기화 유도

}

document.addEventListener('DOMContentLoaded', () => {
    // 드롭다운 채우기
    populateShiftDropdown();
    populateEqpDropdown('');  // 전체

    // 바인딩: Shift 바뀌면 Eqp 목록 재구성 + 필터 적용
    const shiftSel = document.querySelector('#shift-select');
    const eqpSel   = document.querySelector('#eqp-select');
    const typeSel = document.querySelector('#type-select');

    if (shiftSel) {
        shiftSel.addEventListener('change', () => {
        const key = shiftSel.value || '';
        populateEqpDropdown(key);

        // 설비 선택 초기화 후 필터 적용
        if (eqpSel) eqpSel.value = '';
        applyFiltersToDataTable();
        });
    }

    if (eqpSel) {
        eqpSel.addEventListener('change', () => {
        applyFiltersToDataTable();
        });
    }

    if (typeSel) {
        typeSel.addEventListener('change', () => {
            applyFiltersToDataTable();
        });
    }

    initShiftFromQuery();
});

/* ============================ 고급 필터(체크박스) ============================ */

// 열 메타 (colIdx → { frontKey, dbKey, unit })
function getColumnMeta(dt) {
    // 캐시
    if (window.__CYL_COLMETA && window.__CYL_COLMETAVersion === (window.__CYL_DB_COLS || []).join('|')) {
        return window.__CYL_COLMETA;
    }

    const meta = [];
    const fronts = (window.__CYL_DB_COLS || []).slice(); // frontKey들 (축들)
    // colIdx 0: EQ, 1: WriteTime → 2부터 축
    for (let i = 0; i < fronts.length; i++) {
        const frontKey = fronts[i];
        const dbKey = (window.FRONT_TO_DB && window.FRONT_TO_DB[frontKey]) || frontKey;
        // UNIT: 마지막 '_' 이전까지 (예: ATT_ULD_TR_DOWN → ATT_ULD_TR)
        const unit = frontKey.includes('_') ? frontKey.split('_').slice(0, -1).join('_') : frontKey;
        meta.push({ colIdx: i + 2, frontKey, dbKey, unit });
    }

    window.__CYL_COLMETA = meta;
    window.__CYL_COLMETAVersion = fronts.join('|');
    return meta;
}

// 중요 유닛 Set (베이스 유닛으로 변환)
function getCriticalUnitBases() {
    const crits = (window.CRITICAL_UNITS || []).map(k => {
        const front = k; // 이미 frontKey 기준
        const unit = front.includes('_') ? front.split('_').slice(0, -1).join('_') : front;
        return unit;
    });
    return new Set(crits);
}

// 현재 적용된 행 기준: 열별 이상치 존재 여부 계산
function computeOutlierUnits(dt, meta, restrictUnits = null) {
    const outUnits = new Set();
    if (!meta?.length) return outUnits;

    // 열을 훑되, 이상이 발견되면 해당 열의 'unit'을 집합에 추가
    meta.forEach(({ colIdx, unit }) => {
      if (restrictUnits && !restrictUnits.has(unit)) return; // 중요유닛 제한

      const nodes = dt.column(colIdx, { search: 'applied' }).nodes(); // ✅ DataTables API 사용
      for (let i = 0; i < nodes.length; i++) {
        const td = nodes[i];
        if (!td) continue;
        if (td.dataset.abnormal === 'upper' || td.classList.contains('is-over-upper') ||
            td.dataset.abnormal === 'lower' || td.classList.contains('is-under-lower')) {
          outUnits.add(unit); // ← 열이 아니라 '유닛'을 추가!
          break;
        }
      }
    });

    return outUnits;
}

// 열 가시성 일괄 적용 (mask: colIdx -> true/false)
function applyColumnVisibilityMask(dt, mask) {
    const colCount = dt.columns().count();
    for (let col = 2; col < colCount; col++) {
        const vis = mask[col] !== false; // 기본 true
        dt.column(col).visible(vis, false); // draw 지연
    }
    dt.columns.adjust().draw(false);
}

// 체크박스 상태 읽기
function getFilterToggles() {
    const abnormalOnly = !!document.getElementById('abnormal-checkbox')?.checked;
    const outlierOnly  = !!document.getElementById('outlier-checkbox')?.checked;
    const criticalOnly = !!document.getElementById('important-checkbox')?.checked;
    return { abnormalOnly, outlierOnly, criticalOnly };
}

// 행 필터(이상설비만) - 한 번만 등록
function registerRowFilterOnce() {
    if (window.__ROW_FILTER_REGISTERED) return;
    $.fn.dataTable.ext.search.push(function (settings, data, dataIndex) {
        const { abnormalOnly } = getFilterToggles();
        if (!abnormalOnly) return true; // 필터 꺼짐 → 통과

        const api = new $.fn.dataTable.Api(settings);
        const tr = api.row(dataIndex).node();
        if (!tr) return true;

        // EQ, WriteTime 제외, 데이터셀 중 하나라도 이상치가 있으면 통과
        const tds = Array.from(tr.cells).slice(2);
        return tds.some(td =>
        td.dataset.abnormal === 'upper' || td.classList.contains('is-over-upper') ||
        td.dataset.abnormal === 'lower' || td.classList.contains('is-under-lower')
        );
    });
    window.__ROW_FILTER_REGISTERED = true;
}

// 열 필터(중요축만 / 이상치만) 적용
function applyColumnFilters(dt) {
    const { outlierOnly, criticalOnly } = getFilterToggles();
    const meta = getColumnMeta(dt);
    const colCount = dt.columns().count();

    // 기본: 모두 보이게
    const mask = Array(colCount).fill(true);

    // 중요 유닛 Set 준비 (베이스 유닛: ATT_ULD_TR 등)
    const criticalBases = criticalOnly ? getCriticalUnitBases() : null;

    // outlierOnly가 활성화면, (중요축만 보기 켠 상태라면 그 유닛으로 제한해서) 이상치 열 집합 계산
    const outUnits = outlierOnly ? computeOutlierUnits(dt, meta, criticalBases) : null;

    meta.forEach(({ colIdx, unit }) => {
      const passCritical = criticalOnly ? criticalBases.has(unit) : true;
      const passOutlier  = outlierOnly  ? outUnits.has(unit)      : true;
      mask[colIdx] = passCritical && passOutlier;
    });

    applyColumnVisibilityMask(dt, mask);
}

// 체크박스 변화 → 필터 재적용
function bindFilterCheckboxHandlers() {
    const ids = ['abnormal-checkbox', 'outlier-checkbox', 'important-checkbox'];
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('change', () => {
        const dt = $("#cylinder-table-container").DataTable();
        // 행 필터(이상설비만) 먼저 적용 후 draw
        dt.draw(false);
        // 열 필터(중요/이상치) 적용
        applyColumnFilters(dt);
        // 요약 갱신
        renderEqStatusFromDataTable(dt);
        });
    });
}

// DataTable 생성 직후에 한 번 호출
function wireAdvancedFilters(dataTable) {
    registerRowFilterOnce();
    bindFilterCheckboxHandlers();
    // 최초 상태도 반영 (체크박스가 미리 켜져 있을 수 있음)
    dataTable.draw(false);
    applyColumnFilters(dataTable);
}


/* ============================ 이상치 점프 네비 ============================ */

// 순환 인덱스 상태
let __navIdx = { critical: 0, lower: 0, upper: 0 };

// 현재 적용된 검색/필터 상태를 반영하여, 조건에 맞는 셀 수집
function collectAbnormalCellsByType(type) {
    const dt = $.fn.DataTable.isDataTable("#cylinder-table-container")
        ? $("#cylinder-table-container").DataTable()
        : null;
    if (!dt) return [];

    const rows = dt.rows({ search: "applied" }).nodes().toArray();
    const list = [];

    rows.forEach(tr => {
        // EQ, WriteTime 제외
        const tds = Array.from(tr.cells).slice(2);

        tds.forEach(td => {
        const isCritical =
            td.classList.contains("cell-critical") || td.dataset.isCritical === "1";
        const over = isUpperOutlierCell(td);  // SPEC 초과(느림)
        const under = isLowerOutlierCell(td); // SPEC 미만(빠름)

        if (type === "critical") {
            if (isCritical && (over || under)) list.push(td);
        } else if (type === "lower") {
            if (under) list.push(td);
        } else if (type === "upper") {
            if (over) list.push(td);
        }
        });
    });

    return list;
}

// 스크롤바 있는 DataTables 영역 기준으로 해당 셀로 스크롤
function scrollToCellInWrapper(td) {
    const wrapper = document.querySelector(".dataTables_scrollBody");
    if (!wrapper || !td) return;

    const cellRect = td.getBoundingClientRect();
    const wrapRect = wrapper.getBoundingClientRect();

    const top =
        wrapper.scrollTop + (cellRect.top - wrapRect.top) - 80;  // 여백
    const left =
        wrapper.scrollLeft + (cellRect.left - wrapRect.left) - 80;

    wrapper.scrollTo({
        top: Math.max(0, top),
        left: Math.max(0, left),
        behavior: "smooth",
    });

    td.classList.add("scroll-focus");
    setTimeout(() => td.classList.remove("scroll-focus"), 1500);
}

// 타입별 다음 셀로 점프
function jumpToNext(type) {
    const cells = collectAbnormalCellsByType(type);
    if (cells.length === 0) {
        // 필요 시 토스트로 교체 가능
        alert(type === "critical" ? "중요 이상치가 없습니다."
            : type === "lower"   ? "SPEC 미만(빠름) 이상치가 없습니다."
                                : "SPEC 초과(느림) 이상치가 없습니다.");
        return;
    }

    const key = type === "critical" ? "critical" : (type === "lower" ? "lower" : "upper");
    const idx = __navIdx[key] % cells.length;
    scrollToCellInWrapper(cells[idx]);
    __navIdx[key] = (idx + 1) % cells.length;
}

// 버튼 클릭 위임
document.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-abnormal-critical, .btn-abnormal-important, .btn-abnormal-general");
    if (!btn) return;

    if (btn.classList.contains("btn-abnormal-critical")) {
        jumpToNext("critical");     // 중요 위치 + (미만/초과) 모두
    } else if (btn.classList.contains("btn-abnormal-important")) {
        jumpToNext("lower");        // SPEC 미만(빠름)
    } else if (btn.classList.contains("btn-abnormal-general")) {
        jumpToNext("upper");        // SPEC 초과(느림)
    }
});

// 테이블이 다시 그려질 때(검색/정렬/필터 등) 인덱스 리셋
const tableElement = document.getElementById('cylinder-table-container');

if (tableElement) {
    tableElement.addEventListener('draw.dt', function () {
        // 인덱스 초기화
        __navIdx = { critical: 0, lower: 0, upper: 0 };
    });
}


/* ---------------------------
   공용: DataTables 행 선택 토글 바인딩
--------------------------- */
function bindDTSelectable($table, {
    selectedSet,
    getKey,
    restoreOnDraw = false,
    onChange
    } = {}) {
    if (!$table || typeof getKey !== 'function') return;

    const tableEl = $table[0];
    const $tbody  = $table.find('tbody');

    // 🔹 (추가) Shift 클릭 시 텍스트 블럭 생성 방지
    $tbody
        .off('mousedown.__rowselect')
        .on('mousedown.__rowselect', 'tr', function (e) {
        if (e.shiftKey && !$(e.target).is('input,textarea,select,button,a')) {
            e.preventDefault();              // 텍스트 드래그 방지
        }
        });

    // (기존) anchor 저장
    const getLastAnchor = () => (tableEl.__lastAnchorIndex ?? null);
    const setLastAnchor = (idx) => { tableEl.__lastAnchorIndex = (typeof idx === 'number' ? idx : null); };

    const indexOfRow = (tr) => $tbody.find('tr').index(tr);
    const rowsBetween = (a, b) => {
        const $rows = $tbody.find('tr');
        const i1 = Math.min(a, b), i2 = Math.max(a, b);
        const arr = [];
        for (let i = i1; i <= i2; i++) arr.push($rows[i]);
        return arr;
    };

    $tbody
        .off('click.__rowselect')
        .on('click.__rowselect', 'tr', function (e) {
        if ($(e.target).is('input,textarea,select,button,a')) return;

        const key = getKey(this);
        if (!key) return;

        const isCtrl  = !!(e.ctrlKey || e.metaKey);
        const isShift = !!e.shiftKey;

        // 🔹 (추가) 혹시 생긴 브라우저 텍스트 선택 즉시 해제
        if (isShift && window.getSelection) {
            const sel = window.getSelection();
            if (sel && sel.removeAllRanges) sel.removeAllRanges();
        }

        const $rows = $tbody.find('tr');
        const wasSelected = this.classList.contains('selected');
        const currIdx = indexOfRow(this);
        let changed = false;

        // ---- Shift 구간 선택 ----
        if (isShift) {
            const last = getLastAnchor();

            if (last == null) {
            if (!isCtrl) {
                $rows.removeClass('selected');
                if (selectedSet) selectedSet.clear();
            }
            if (!wasSelected || isCtrl) {
                this.classList.add('selected');
                if (selectedSet) selectedSet.add(key);
                changed = true;
            }
            setLastAnchor(currIdx);
            if (onChange && changed) onChange(selectedSet);
            return;
            }

            const range = rowsBetween(last, currIdx);
            if (!isCtrl) {
            $rows.removeClass('selected');
            if (selectedSet) selectedSet.clear();
            }
            for (const tr of range) {
            const k = getKey(tr);
            if (!k) continue;
            if (!tr.classList.contains('selected')) {
                tr.classList.add('selected');
                if (selectedSet) selectedSet.add(k);
                changed = true;
            }
            }
            setLastAnchor(currIdx);
            if (onChange && changed) onChange(selectedSet);
            return;
        }

        // ---- Ctrl/⌘ 토글 ----
        if (isCtrl) {
            if (wasSelected) {
            this.classList.remove('selected');
            if (selectedSet) selectedSet.delete(key);
            } else {
            this.classList.add('selected');
            if (selectedSet) selectedSet.add(key);
            }
            setLastAnchor(currIdx);
            if (onChange) onChange(selectedSet);
            return;
        }

        // ---- 기본(단일) ----
        const selectedCount = $rows.filter('.selected').length;
        if (selectedCount === 1 && wasSelected) {
            this.classList.remove('selected');
            if (selectedSet) selectedSet.delete(key);
            setLastAnchor(null);
            if (onChange) onChange(selectedSet);
            return;
        }

        $rows.removeClass('selected');
        if (selectedSet) selectedSet.clear();
        this.classList.add('selected');
        if (selectedSet) selectedSet.add(key);
        setLastAnchor(currIdx);
        if (onChange) onChange(selectedSet);
        });

    // redraw 복원
    if (restoreOnDraw && selectedSet) {
        $table.off('draw.__rowselect').on('draw.__rowselect', function () {
        $table.find('tbody tr').each(function () {
            const k = getKey(this);
            if (k && selectedSet.has(k)) this.classList.add('selected');
            else this.classList.remove('selected');
        });
        });
    } else {
        $table.off('draw.__rowselect');
    }
}

function wireOutsideDeselect($table, {
        selectedSet,
        preserveSelectors = ['button']  // 버튼 클릭 시 유지
} = {}) {
    if (!$table || !$table.length) return;
    const tableEl = $table[0];
    if (tableEl.__outsideDeselectWired) return; // 중복 방지
    tableEl.__outsideDeselectWired = true;

    const handler = (e) => {
        // 1) 테이블 내부 클릭이면 무시
        if (tableEl.contains(e.target)) return;

        // 2) 화이트리스트(버튼 등) 클릭이면 유지
        for (const sel of preserveSelectors) {
        if (sel && e.target.closest?.(sel)) return;
        }

        // 3) 선택 해제
        const $rows = $($table).find('tbody tr.selected');
        if ($rows.length === 0) return;
        $rows.removeClass('selected');
        if (selectedSet?.clear) selectedSet.clear();
    };

    // document 전역 바인딩(한 번만)
    document.addEventListener('mousedown', handler);
    // 필요시 정리용 핸들 저장
    tableEl.__outsideDeselectCleanup = () => {
        document.removeEventListener('mousedown', handler);
        tableEl.__outsideDeselectWired = false;
    };
}