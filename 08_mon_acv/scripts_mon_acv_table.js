// ✅ 상세 모니터링 원본 데이터 저장 + 필터 핸들러 중복 바인딩 방지
let __monRawData = [];
let __monFilterHandlersBound = false;
let __monExcelBound = false;
const selectedMonRows = new Set();

// ✅ 현재 드롭다운 필터 읽기
function getCurrentMonFilters() {
    const shiftSel  = document.getElementById('shift-select');
    const eqSel     = document.getElementById('eq-group-select');
    const recipeSel = document.getElementById('recipe-select');

    const shiftKey  = (shiftSel?.value || 'all').toLowerCase();
    const eqVal     = eqSel?.value || 'all';
    const recipeVal = recipeSel?.value || 'all';
    return { shiftKey, eqVal, recipeVal };
}

// ✅ 원본 → 필터 적용(Shift → 설비 순)
function getFilteredMonData(opts = {}) {
    const { shiftKey, eqVal, recipeVal } = getCurrentMonFilters();
    let list = Array.isArray(__monRawData) ? [...__monRawData] : [];

    // 1) Shift
    if (shiftKey !== 'all' && window.shiftGroups?.[shiftKey]) {
        const allowed = new Set(window.shiftGroups[shiftKey]);
        list = list.filter(row => allowed.has(row.EQ_Name));
    }

    // 2) Recipe
    if (!opts.ignoreRecipe && recipeVal && recipeVal !== 'all') {
        list = list.filter(row => (row.recipe ?? '').trim() === recipeVal);
    }

    // 3) EQ
    if (!opts.ignoreEq && eqVal && eqVal !== 'all') {
        list = list.filter(row => row.EQ_Name === eqVal);
    }

    return list;
}

// ✅ 상세 테이블 렌더(정렬 이벤트 포함)
function renderMonTable(data) {
    const table = document.getElementById("mon-inline-table");
    if (!table) return;

    // 0) 기존 DataTable 인스턴스가 있으면 먼저 완전히 제거
    if ($.fn.DataTable && $.fn.dataTable.isDataTable(table)) {
        $(table).DataTable().clear().destroy();
        // DataTables가 붙였던 인라인 스타일/클래스 초기화(렌더 꼬임 방지)
        table.removeAttribute('style');
        table.className = 'mon-inline-table'; // 필요 시 기존 클래스 유지

    }

    // 1) 헤더/바디를 새로 구성
    const thead = table.querySelector("thead") || table.createTHead();
    const tbody = table.querySelector("tbody") || table.createTBody();

    thead.innerHTML = `
        <tr>
        <th data-sort="eq">설비</th>
        <th data-sort="text">Recipe</th>
        <th data-sort="text">Status</th>
        <th data-sort="num">미세 가압 Count</th>
        <th data-sort="num">미세 가압 Time</th>
        <th data-sort="num">가압 시간</th>
        <th data-sort="num">배기 시간</th>
        <th data-sort="num">Packing ON</th>
        <th data-sort="num">Packing OFF</th>
        </tr>
    `;
    
    // ✅ 고유키 생성기: EQ_Name + recipe (recipe가 없으면 빈문자)
    const makeKey = (r) => `${r.EQ_Name ?? ''}__${(r.recipe ?? '').trim()}`;

    const rowsHtml = (data || []).map(row => {
        const key = makeKey(row);
        // ✅ 선택 복원: 기존에 선택돼 있던 키면 'selected' 클래스 부여
        const selClass = selectedMonRows.has(key) ? ' class="selected"' : '';
        return `
        <tr data-key="${key}"${selClass}>
            <td>${row.EQ_Name ?? '-'}</td>
            <td>${row.recipe ?? '-'}</td>
            <td></td>
            <td>${row.max_micro_press_cnt ?? '-'}</td>
            <td>${row.max_micro_press_time ?? '-'}</td>
            <td>${row.max_press_time ?? '-'}</td>
            <td>${row.max_exhaust_time ?? '-'}</td>
            <td>${row.max_pack_on ?? '-'}</td>
            <td>${row.max_pack_off ?? '-'}</td>
        </tr>
        `;
    }).join('');
    tbody.innerHTML = rowsHtml;

    const $table = $(table);
    // 2) DataTable 재초기화
    $(table).DataTable({
        paging: false,
        searching: false,
        info: false,
        autoWidth:false,
        deferRender: true,
        scrollX: false,
        scrollY: '635px',
        scrollCollapse:true,
        fixedHeader: true,
        columnDefs: [
            { targets: 0, type: 'eq', width: '100px' },
            { targets: 1, width: '150px' },
            { targets: 2, width: '100px' },
            { targets: [3,4,5,6,7,8], type: 'num', width: '130px' }
        ],
        language: { emptyTable: '데이터가 없습니다.' }
    });

    // ✅ 공용 선택 바인딩(Shift 범위, Ctrl/⌘ 토글)
    bindDTSelectable($table, {
        selectedSet: selectedMonRows,
        getKey: (tr) => tr.getAttribute('data-key'),
        // DataTables 내부 draw 시에는 복원할 필요가 크지 않음(필터는 외부 드롭다운으로 전체 리렌더)
        restoreOnDraw: false,
        onChange: () => { /* 필요 시 선택 변화 후 처리 */ }
    });

    // ✅ 테이블 영역 밖 클릭 시 전체 선택 해제 (버튼 클릭은 보존)
    wireOutsideDeselect($table, {
        selectedSet: selectedMonRows,
        preserveSelectors: ['button', '.mon-inline-toolbar']
    });
}

// eq 정렬 타입 유지
if ($.fn && $.fn.dataTable) {
  $.fn.dataTable.ext.type.order['eq-pre'] = function (d) {
    const cleaned = String(d || '').replace(/\D/g, '');
    const n = parseInt(cleaned, 10);
    return isNaN(n) ? -Infinity : n;
  };
}

/**  #acv-mon-container에 렌더링 */
function openSummaryPopup() {
    showWrapper('monitor');
    const container = document.getElementById("acv-mon-table");
    if (!container) return;

    container.innerHTML = `
        <div class="mon-inline-toolbar">
            <table id="mon-inline-table" class="mon-inline-table" style="width:100%;">
                <thead></thead>
                <tbody></tbody>
            </table>
        </div>
    `;

    showLoading();

    return Promise.resolve()
        .then(() => fetch('/get-acv-summary'))
        .then(res => {
            if (!res.ok) throw new Error(`ACV summary 조회 실패 (HTTP ${res.status})`);
            return res.json();
        })
        .then(data => {
        __monRawData = Array.isArray(data) ? data : [];

        // 초기: Shift=all 기준으로 레시피/설비 후보 구성 후 테이블 렌더
        populateRecipeSelectByEq();
        populateEqSelectByRecipe();
        renderMonTable(getFilteredMonData());

        if (!__monFilterHandlersBound) {
            const shiftSelect  = document.getElementById('shift-select');
            const recipeSelect = document.getElementById('recipe-select');
            const eqSelect     = document.getElementById('eq-group-select');

            // 1) Shift 변경 → Recipe 후보 재구성 → EQ 후보도 재구성 → 테이블(Shift 기준) 렌더
            shiftSelect?.addEventListener('change', () => {
                // Shift 바뀌면 하위 선택 초기화
                recipeSelect && (recipeSelect.value = 'all');
                eqSelect && (eqSelect.value = 'all');

                populateRecipeSelectByEq();
                populateEqSelectByRecipe();
                renderMonTable(getFilteredMonData()); // Shift만 반영된 테이블
            });

            // 2) Recipe 변경 → EQ 후보 재구성 → 테이블(Shift+Recipe) 렌더
            recipeSelect?.addEventListener('change', () => {
                // Recipe 바뀌면 EQ 초기화
                populateRecipeSelectByEq();
                populateEqSelectByRecipe();
                renderMonTable(getFilteredMonData()); // Shift+Recipe 반영

            });

            // 3) EQ 변경 → 테이블(Shift+Recipe+EQ) 렌더
            eqSelect?.addEventListener('change', () => {
                populateRecipeSelectByEq();
                populateEqSelectByRecipe();
                renderMonTable(getFilteredMonData());
       
            });

            __monFilterHandlersBound = true;
        }
        })
        .finally(() => {
            hideLoading();
        })
        .catch(err => {
            console.error('❌ 미세가압 Summary 불러오기 실패:', err);
            alert('데이터를 불러오지 못했습니다.');
        });
}

function setupMonExcelExport() {
    const btn = document.getElementById('export-mon-excel-btn');
    if (!btn || __monExcelBound) return;

    // 중복 바인딩 방지
    btn.removeEventListener('click', handleMonExcelExport);
    btn.addEventListener('click', handleMonExcelExport);
    __monExcelBound = true;
}

// ===== 엑셀 저장 핸들러 =====
function handleMonExcelExport() {
    // 설비 효율 탭에서는 현재 상세 표를 저장
    const efficiencyWrapper = document.querySelector('.acv-effi-wrapper');
    if (efficiencyWrapper && efficiencyWrapper.style.display !== 'none') {
        exportEfficiencyDetailExcel();
        return;
    }

    // 1) 테이블 존재 확인
    const table = document.getElementById('mon-inline-table');
    if (!table) {
        alert('모니터링 테이블이 없습니다. 모니터링 탭에서 데이터를 먼저 불러오세요.');
        return;
    }

    // 2) 워크북/시트 생성
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.table_to_sheet(table);
    XLSX.utils.book_append_sheet(wb, ws, "상세정보");

    // 3) 파일명 구성 (eq / shift / 날짜범위 포함)
    const { shiftKey, eqVal } = (typeof getCurrentMonFilters === 'function')
        ? getCurrentMonFilters()
        : { shiftKey: 'all', eqVal: 'all' };

    const eqTag    = (eqVal && eqVal !== 'all') ? `_${eqVal}` : '';
    const shiftTag = (shiftKey && shiftKey !== 'all') ? `_Shift-${String(shiftKey).toUpperCase()}` : '';

    // date-range: "YYYY-MM-DD ~ YYYY-MM-DD" → "_YYYYMMDD-YYYYMMDD"
    const rangeInput = document.getElementById('date-range')?.value?.trim() || '';
    let rangeTag = '';
    if (rangeInput.includes('~')) {
        const [s, e] = rangeInput.split('~').map(v => v.trim().replaceAll('-', ''));
        if (s && e) rangeTag = `_${s}-${e}`;
    }

    const now = new Date();
    const ts = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}`;

    const filename = `미세가압_Count${eqTag}${shiftTag}${rangeTag}_${ts}.xlsx`;

    // 4) 저장
    XLSX.writeFile(wb, filename);
}