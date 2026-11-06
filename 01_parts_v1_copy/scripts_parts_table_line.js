// ---------------------------
// 라인 파츠 모듈 (parts_line_list) — DataTables 버전
// ---------------------------
const PartsLine = (() => {
    let state = {
        items: [],
        isFetching: false,
        isAdmin: false,
        inited: false,
        container: null,
        lastFilters: {} // 최근 전역 필터 저장
    };

    // DT 인스턴스
    let lineDT = null;
    const selectedPNs = new Set(); // 선택 상태 저장용 (행 식별: P_N)
    const __lineAuth = { isPrivileged: false };

    // 인라인 수정 컨텍스트 (단일 행만 편집)
    // { rowNode, rowApi, before, beforeKey: {Q_Code, EQ_Unit, P_N} }
    let editCtxs = [];

    // === [라인 파츠] 버튼 상태/모드 관리 ===
    function getButtonsInLineWrap() {
    const wrap = document.querySelector('.line-parts-wrapper');
    return {
        btnAdd:    wrap?.querySelector('#edit-add-btn-2'),
        btnDel:    wrap?.querySelector('#edit-delete-btn-2'),
        btnEdit:   wrap?.querySelector('#edit-btn-2'),
        btnCancel: wrap?.querySelector('#edit-cancel-btn-2'),
    };
    }

    function hasDraftRowNowLine() {
        const $tbody = $(state.container).find('.line-table tbody');
        return $tbody.find('tr.draft-row').length > 0;
    }

    function isEditingNowLine() {
        return editCtxs.length > 0;
    }

    /** mode: 'idle' | 'draft' | 'editing'
     *  - idle:    아무것도 아님 → [추가/삭제/수정] 가능, [취소] 비활성
     *  - draft:   드래프트 존재 → [추가/삭제/수정] 비활성, [취소] 활성
     *  - editing: 수정 진행 중 → [추가/삭제/수정] 비활성, [취소] 활성
     */
    function setButtonsForModeLine(mode) {
        const { btnAdd, btnDel, btnEdit, btnCancel } = getButtonsInLineWrap();
        if (!btnAdd || !btnDel || !btnEdit || !btnCancel) return;

        if (mode === 'draft' || mode === 'editing') {
            btnAdd.disabled = true;
            btnDel.disabled = true;
            btnEdit.disabled = true;
            btnCancel.disabled = false; // 편집/추가 중엔 취소만 활성
        } else {
            btnAdd.disabled = false;
            btnDel.disabled = false;
            btnEdit.disabled = false;
            btnCancel.disabled = false;  // 평소엔 취소 비활성
        }
    }

    /** 현재 상태를 읽어 버튼 모드 자동 결정 */
    function refreshButtonsByStateLine() {
    if (hasDraftRowNowLine()) return setButtonsForModeLine('draft');
    if (isEditingNowLine())   return setButtonsForModeLine('editing');
    return setButtonsForModeLine('idle');
    }

    function setAdminMode(isPrivileged) {
        state.isAdmin = !!isPrivileged;
        __lineAuth.isPrivileged = !!isPrivileged;

        renderTable(state.items);         // 모드 반영
        refreshButtonsByStateLine();      // 내부 버튼 상태 갱신
        enforceLineEditButtonVisibility();// 메인 버튼 표시/숨김
    }

    async function checkClientAuthority() {
        try {
            const res = await fetch('/check-client-authority-parts');
            const data = await res.json();

            const isPrivileged = (data.authority === '관리자' || data.authority === '구매담당자');
            setAdminMode(isPrivileged);

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
            setAdminMode(false); // 실패 시 비권한
        }
    }

    function enforceLineEditButtonVisibility() {
        const wrap = document.querySelector('.line-parts-wrapper');
        if (!wrap) return;
        const mainBtn = wrap.querySelector('#add-parts-btn');      // 파츠 편집 메인 버튼
        const panel   = wrap.querySelector('.edit-btn-container');  // 내부 버튼 패널 (열려있으면 닫아줌)

        if (!mainBtn) return;

        if (__lineAuth.isPrivileged) {
            mainBtn.style.display = '';
        } else {
            mainBtn.style.display = 'none';
            if (panel) panel.style.display = 'none';
        }
    }

    // ✅ 외부에서 전달되는 전역 필터 적용
    function searchWithFilters(params) {
        // scripts_parts.js에서 전달: {EQ_Type, Parts_Name, EQ_Unit, Q_Code, P_N}
        state.lastFilters = {
        eqType: params?.EQ_Type ?? '',
        parts:  params?.Parts_Name ?? '',
        unit:   params?.EQ_Unit ?? '',
        qcode:  params?.Q_Code ?? '',
        pn:     params?.P_N ?? ''
        };
        selectedPNs.clear();
        renderTable(state.items); // 필터 적용해 재렌더
        refreshButtonsByStateLine();
    }

    function buildUI(container) {
        container.innerHTML = `
        <div class="table-wrap">
            <table class="line-table display compact nowrap" style="width:100%">
            <colgroup>
                <col style="width:80px">
                <col style="width:150px">
                <col style="width:150px">
                <col style="width:400px">
                <col style="width:400px">
                <col style="width:80px">
                <col style="width:120px">
                <col style="width:260px">
            </colgroup>
            <thead>
                <tr>
                <th>구분</th>
                <th>파츠 분류</th>
                <th>Q-Code</th>
                <th>Unit</th>
                <th>업체P/N</th>
                <th>Count</th>
                <th>Stoker</th>
                <th>비고</th>
                </tr>
            </thead>
            <tbody></tbody>
            </table>
        </div>
        `;
    }

    function applyFilters(list) {
        return filterItemsByGlobalFilters(list, state.lastFilters);
    }

    function finalizeLineTable(rowsData) {
        const $table = $(state.container).find('.line-table');

        const dtData = (rowsData || []).map(it => ({
        EQ_Name: it.EQ_Name ?? '',
        Parts_Name: it.Parts_Name ?? '',
        Q_Code: it.Q_Code ?? '',
        EQ_Unit: it.EQ_Unit ?? '',
        P_N: it.P_N ?? '',
        Count: Number(it.Count ?? 0),
        Stoker: it.Stoker ?? '',
        Etc: it.Etc ?? '',
        _pn: it.P_N ?? '' // 행 식별용
        }));

        // 이미 초기화되어 있으면 갱신
        if ($.fn.DataTable.isDataTable($table)) {
        lineDT.clear().rows.add(dtData).draw(false);
        return;
        }

        // 최초 초기화
        lineDT = $table.DataTable({
        data: dtData,
        columns: [
            { data: 'EQ_Name'    },
            { data: 'Parts_Name' },
            { data: 'Q_Code'     },
            { data: 'EQ_Unit'    },
            { data: 'P_N',
            createdCell: (td) => td.setAttribute('data-col','P_N')
            },
            { data: 'Count',  className: 'dt-right',
            createdCell: (td) => td.setAttribute('data-col','Count')
            },
            { data: 'Stoker', 
            createdCell: (td) => td.setAttribute('data-col','Stoker')
            },
            { data: 'Etc',   
            createdCell: (td) => td.setAttribute('data-col','Etc')
            }
        ],
        // 요청한 DataTables 설정
        scrollY: '610px',
        scrollX: false,
        scrollCollapse: true,
        paging: false,
        searching: true,   // dom:'t'라 기본 검색창은 숨김
        fixedHeader: true,
        orderCellsTop: true,
        info: false,
        dom: 't',
        order: [[5, 'desc']], // 숫자 컬럼 기본 정렬 예시: Count 내림차순
        createdRow: (row, data) => {
            // 인라인 편집/선택용 키 부여
            Array.from(row.cells).forEach(td => td.setAttribute('data-pn', data._pn));
            // (선택 복원은 사용하지 않음; 필요시 bindDTSelectable에서 restoreOnDraw: true로 전환)
        }
        });

        // ✅ 행 선택 토글(다중 선택, 재그림 복원 OFF)
        bindDTSelectable($table, {
            selectedSet: selectedPNs,
            getKey: (tr) => tr.querySelector('td')?.getAttribute('data-pn'),
            singleSelect: false,
            restoreOnDraw: false
        });

        wireOutsideDeselect($table, {
            selectedSet: selectedPNs,
            preserveSelectors: [
                'button', '.edit-group-btn',
                '#add-parts-btn', '#edit-add-btn-2', '#edit-delete-btn-2', '#edit-btn-2', '#edit-cancel-btn-2',
                '.dataTables_length', '.dataTables_filter', '.dataTables_paginate', '.dataTables_info',
                '.dtfh-floatingparent', '.dtfh-floatingheader'
            ]
        });
    }

    function renderTable(items) {
        if (!state.container) return;
        finalizeLineTable(applyFilters(items));
    }

    async function fetchItems() {
        if (state.isFetching) return;
        state.isFetching = true;
        try {
        const res = await fetch('/parts');
        state.items = await res.json();

        // 공통(상단 필터)에게 최신 데이터 전달 → 드롭다운 채움 폴백용
        document.dispatchEvent(new CustomEvent('parts:data-loaded', {
            detail: { which: 'line', items: state.items }
        }));

        if (state.container) renderTable(state.items);
        } finally {
        state.isFetching = false;
        }
    }

    async function init(container) {
        if (state.inited) return;
        state.container = container;
        buildUI(container);
        enforceLineEditButtonVisibility();   // UI 생성 직후 1차 적용
        
        await checkClientAuthority();
        await fetchItems();

        // ========== 편집 버튼 이벤트 수신 ==========

        wireEditEvents();
        state.inited = true;
        refreshButtonsByStateLine();
    }

    /* ============================== 테이블 편집 (행추가, 행삭제, 수정, 취소)  ============================== */

    function inputHTML(name, val = '', type = 'text') {
        const v = String(val ?? '');
        return `
        <input class="dt-input" name="${escapeHtml(name)}"
            type="${escapeHtml(type)}" value="${escapeHtml(v)}"
            style="width:100%;box-sizing:border-box;">
        `;
    }

    function rowToInputs(rowNode) {
        const data = lineDT.row(rowNode).data();
        const cells = rowNode.cells || rowNode.children;
        if (!cells || !data) return;

        cells[0].innerHTML = inputHTML('EQ_Name', data.EQ_Name);
        cells[1].innerHTML = inputHTML('Parts_Name', data.Parts_Name);
        cells[2].innerHTML = inputHTML('Q_Code', data.Q_Code);
        cells[3].innerHTML = inputHTML('EQ_Unit', data.EQ_Unit);
        cells[4].innerHTML = inputHTML('P_N', data.P_N);
        cells[5].innerHTML = inputHTML('Count', data.Count, 'number');
        cells[6].innerHTML = inputHTML('Stoker', data.Stoker);
        cells[7].innerHTML = inputHTML('Etc', data.Etc);

        rowNode.classList.add('editing-row');

        // ✅ Q-Code 입력 시 대문자 자동 변환
        $(rowNode).find('input.dt-input[name="Q_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function () { this.value = this.value.toUpperCase(); });
    }

    function collectInputs(rowNode) {
        const obj = {};
        $(rowNode)
        .find('input.dt-input')
        .each(function () {
            const n = this.name;
            let v = this.value ?? '';
            if (n === 'Count') {
                const num = Number(v);
                v = Number.isFinite(num) ? num : 0;
            } else {
             v = v.trim();
            }
            if (n === 'Q_Code' || n === 'EQ_Name') v = v.toUpperCase();
            obj[n] = v;
        });
        return obj;
    }

    function restoreRowFromData(rowNode, data) {
        // DataTables에 데이터 다시 세팅 → draw(false)로 복원
        lineDT.row(rowNode).data({
        EQ_Name: data.EQ_Name ?? '',
        Parts_Name: data.Parts_Name ?? '',
        Q_Code: data.Q_Code ?? '',
        EQ_Unit: data.EQ_Unit ?? '',
        P_N: data.P_N ?? '',
        Count: Number(data.Count ?? 0),
        Stoker: data.Stoker ?? '',
        Etc: data.Etc ?? '',
        _pn: data.P_N ?? ''
        });
        lineDT.draw(false);
    }


    function makeInput(name, value = '', { type = 'text', placeholder = '' } = {}) {
        const v = String(value ?? '');
        return `
        <input class="dt-input"
            name="${escapeHtml(name)}"
            type="${escapeHtml(type)}"
            value="${escapeHtml(v)}"
            placeholder="${escapeHtml(placeholder)}">
        `;
    }

    // 드래프트 행 추가: 헤더 바로 아래로 1행
    
    function addDraftRow() {
        if (!lineDT) return;

        // 이미 드래프트가 있으면 추가 금지
        const $tbody = $(state.container).find('.line-table tbody');
        if ($tbody.find('tr.draft-row').length > 0) {
            alert('이미 추가 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
        }

        const draftData = {
            EQ_Name: '', Parts_Name: '', Q_Code: '', EQ_Unit: '',
            P_N: '', Count: 0, Stoker: '', Etc: '',
            _pn: '', _draft: true
        };

        // 1) 데이터 추가 + draw
        const rowApi = lineDT.row.add(draftData);
        lineDT.draw(false);

        // 2) 추가된 행을 드래프트로 마킹하고 헤더 바로 아래로 이동
        const node = rowApi.node();
        node.classList.add('draft-row');
        node.classList.remove('selected');
        

        // 3) 인풋으로 변환
        const cells = node.cells || node.children;
        cells[0].innerHTML = makeInput('EQ_Name', '');
        cells[1].innerHTML = makeInput('Parts_Name', '');
        cells[2].innerHTML = makeInput('Q_Code', '');
        cells[3].innerHTML = makeInput('EQ_Unit', '');
        cells[4].innerHTML = makeInput('P_N', '');
        cells[5].innerHTML = makeInput('Count', '0', { type: 'number' });
        cells[6].innerHTML = makeInput('Stoker', '');
        cells[7].innerHTML = makeInput('Etc', '');

        $tbody.prepend(node); // ★ 헤더 아래로 강제 이동

        // ✅ 드래프트 모드
        setButtonsForModeLine('draft');

        // ✅ 드래프트행 Q-Code 입력 시 대문자 자동 변환
        $(node).find('input.dt-input[name="Q_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function () { this.value = this.value.toUpperCase(); });
    }

    // 드래프트 행 → 객체로 수집
    function collectDraftRows() {
        const rows = [];
        const $table = $(state.container).find('.line-table');
        $table.find('tbody tr.draft-row').each(function () {
        const obj = {};
        $(this)
            .find('input.dt-input')
            .each(function () {
            const name = this.name;
            let val = this.value?.trim() ?? '';
            if (name === 'Count') {
                const num = Number(val);
                val = Number.isFinite(num) ? num : 0;
            }
            if (name === 'Q_Code' || name === 'EQ_Name') {
                val = String(val).toUpperCase();
            }
            obj[name] = val;
            });
        rows.push(obj);
        });
        return rows;
    }

    // 드래프트 정리(취소 시 호출)
    function removeAllDraftRows() {
        if (!lineDT) return;
            const $table = $(state.container).find('.line-table');
            $table.find('tbody tr.draft-row').each(function () {
            lineDT.row(this).remove();
        });

        lineDT.draw(false);
        // ✅ 현재 상태에 맞춰 버튼 갱신
        refreshButtonsByStateLine();
    }

    // 간단 검증
    function validateRow(r) {
        const required = ['EQ_Name', 'Parts_Name', 'Q_Code', 'EQ_Unit', 'P_N'];
        const missing = required.filter((k) => !String(r[k] ?? '').trim());
        if (missing.length) {
        return `필수 항목 누락: ${missing.join(', ')}`;
        }
        return null;
    }

    function wireEditEvents() {
        if (wireEditEvents.__wired) return;
        wireEditEvents.__wired = true;

        // 행추가
        document.addEventListener('parts-edit:add-row', () => {
            const wrap = document.querySelector('.line-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            // 인라인 편집 중이면 막기
            if (editCtxs.length > 0) {
            alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
            }
            addDraftRow();
        });

        // 수정(선택 1행 → 인라인 입력으로 전환)
        const lineWrap = document.querySelector('.line-parts-wrapper');
        const editBtn = lineWrap?.querySelector('#edit-btn-2');
        if (editBtn && !editBtn.__wired) {
            editBtn.__wired = true;
            editBtn.addEventListener('click', () => {
                if (!lineDT) return;

                // 이미 편집 중이면 막기
                if (editCtxs.length > 0) {
                    alert('이미 편집 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                const nodes = lineDT.rows('.selected').nodes().toArray();

                if (nodes.length === 0) {
                    alert('수정할 행을 선택해 주세요. (여러 행도 가능합니다)');
                    return;
                }

                // 선택된 각 행을 인풋으로 전환
                editCtxs = [];
                for (const rowNode of nodes) {
                    if (rowNode.classList.contains('draft-row')) {
                    alert('행추가 중인 드래프트는 "저장"으로 추가하세요. 수정은 저장된 행만 가능합니다.');
                    // 드래프트가 섞여있으면 전체 취소로 일관성 유지
                    editCtxs = [];
                    return;
                    }
                    const rowApi = lineDT.row(rowNode);
                    const before = { ...rowApi.data() };
                    const beforeKey = { Q_Code: before.Q_Code, EQ_Unit: before.EQ_Unit, P_N: before.P_N };
                    editCtxs.push({ rowNode, rowApi, before, beforeKey });
                    rowToInputs(rowNode);
                }

                

                // ✅ 수정 모드
                setButtonsForModeLine('editing');
            });
        }

        // 저장 (인라인 수정 → UPDATE, 드래프트 → INSERT)
        document.addEventListener('parts-edit:save', async () => {
            const wrap = document.querySelector('.line-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            try {
                // 1) 편집행 처리
                if (editCtxs.length > 0) {
                    const payloads = [];
                    for (const ctx of editCtxs) {
                    const after = collectInputs(ctx.rowNode);
                    const err = validateRow({ ...ctx.before, ...after });
                    if (err) {
                        alert(err);
                        document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: 'VALIDATION', message: err } }));
                        return;
                    }
                    payloads.push({ where: ctx.beforeKey, data: after });
                    }

                    // bulk 시도 → 실패하면 개별 업데이트 폴백
                    let bulkOk = false;
                    try {
                    const r = await fetch('/parts_line/bulk-update', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ rows: payloads })
                    });
                    bulkOk = r.ok;
                    if (!bulkOk) { try { throw new Error(await r.text()); } catch { throw new Error('bulk update failed'); } }
                    } catch {
                    // fallback: 순차 개별 업데이트
                    for (const p of payloads) {
                        const respU = await fetch('/parts_line/update', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(p)
                        });
                        if (!respU.ok) throw new Error(await respU.text());
                    }
                    }
                    editCtxs = [];
                }
                

                // 2) 드래프트 INSERT 처리
                const drafts = collectDraftRows();
                if (drafts.length) {
                    for (const d of drafts) {
                    const err = validateRow(d);
                    if (err) {
                        alert(err);
                        document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: 'VALIDATION', message: err } }));
                        return;
                    }
                    }
                    for (const d of drafts) {
                    const payload = {
                        EQ_Name: d.EQ_Name,
                        Parts_Name: d.Parts_Name,
                        Q_Code: d.Q_Code,
                        EQ_Unit: d.EQ_Unit,
                        P_N: d.P_N,
                        Count: d.Count ?? 0,
                        Stoker: d.Stoker ?? '',
                        Etc: d.Etc ?? ''
                    };
                    const resp = await fetch('/parts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    if (!resp.ok) throw new Error(await resp.text());
                    }
                    removeAllDraftRows();
                }

                alert('저장되었습니다.');
                await fetchItems();
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));
                // ✅ 저장 후 상태 반영(대개 idle)
                refreshButtonsByStateLine();
                
            } catch (e) {
                console.error(e);
                alert('Q-Code 값이 중복입니다.\n' + (e.message || e));
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: e?.message } }));

            }
        });

        // 삭제 
        document.addEventListener('parts-edit:delete-row', async () => {
            if (!lineDT) return;
            const selected = lineDT.rows('.selected').data().toArray();
            if (!selected.length) return alert('삭제할 행을 선택해 주세요.');

            // 편집 중이면 막기
            if (editCtxs.length > 0) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }

            const rows = selected.map((r) => ({
                Q_Code: r.Q_Code,
                EQ_Unit: r.EQ_Unit,
                P_N: r.P_N
            }));

            if (!confirm(`선택한 ${rows.length}건을 삭제할까요?`)) return;

            try {
                const resp = await fetch('/parts_line/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rows })
                });
                if (!resp.ok) throw new Error(await resp.text());
                await fetchItems();
                alert('삭제되었습니다.');
                refreshButtonsByStateLine();

            } catch (e) {
            console.error(e);
            alert('삭제 중 오류가 발생했습니다.\n' + (e.message || e));
            }
        });

        // 취소
        document.addEventListener('parts-edit:cancel', () => {
            const wrap = document.querySelector('.line-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            // 인라인 수정 복원
            if (editCtxs.length > 0) {
                for (const ctx of editCtxs) {
                    restoreRowFromData(ctx.rowNode, ctx.before);
                }
                editCtxs = [];
            }

            // 드래프트 제거
            removeAllDraftRows();
            refreshButtonsByStateLine();
        });
    }

    return { init, fetch: () => fetchItems(), searchWithFilters };
})();




window.PartsLine = PartsLine;