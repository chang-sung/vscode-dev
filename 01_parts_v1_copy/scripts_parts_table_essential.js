// ---------------------------------
// 파츠 구매 모듈 (parts_essential_list) — DataTables 버전 (전체 재고 컬럼 제거)
// ---------------------------------
const PartsEssential = (() => {
    let state = {
        items: [],
        isFetching: false,
        isAdmin: false,
        inited: false,
        container: null,
        lastFilters: {}
    };

    // DT 인스턴스
    let purchaseDT = null;

    // 인라인 수정 컨텍스트
    let editCtxs = [];

    // ✅ 행 선택 상태 저장용 (식별키: P_N)
    const selectedPNsPurchase = new Set();
    const __essentialAuth = { isPrivileged: false };

    // === 버튼 활성/비활성 유틸 ===
    // === 버튼 핸들/모드 관리 ===
    function getButtonsInEssentialWrap() {
        const wrap = document.querySelector('.essential-parts-wrapper');
        return {
            btnAdd:    wrap?.querySelector('#edit-add-btn'),
            btnDel:    wrap?.querySelector('#edit-delete-btn'),
            btnEdit:   wrap?.querySelector('#edit-btn'),
            btnCancel: wrap?.querySelector('#edit-cancel-btn'),
        };
    }

    function hasDraftRowNow() {
        const $tbody = $(state.container).find('.essential-table tbody');
        return $tbody.find('tr.draft-row').length > 0;
    }

    function isEditingNow() {
        return editCtxs.length > 0;
    }

    /**
     * mode: 'idle' | 'draft' | 'editing'
     *  - idle:    아무것도 아님 → [행추가/행삭제/수정] 가능, [취소] 비활성
     *  - draft:   드래프트 존재 → [행추가/행삭제/수정] 비활성, [취소]만 활성
     *  - editing: 수정 진행 중 → [행추가/행삭제/수정] 비활성, [취소]만 활성
     */
    function setButtonsForMode(mode) {
        const { btnAdd, btnDel, btnEdit, btnCancel } = getButtonsInEssentialWrap();
        if (!btnAdd || !btnDel || !btnEdit || !btnCancel) return;

        if (mode === 'draft') {
            btnAdd.disabled = true;
            btnDel.disabled = true;
            btnEdit.disabled = true;
            btnCancel.disabled = false;   // ✅ 취소는 항상 가능
        } else if (mode === 'editing') {
            btnAdd.disabled = true;
            btnDel.disabled = true;
            btnEdit.disabled = true;
            btnCancel.disabled = false;   // ✅ 취소는 항상 가능
        } else { // idle
            btnAdd.disabled = false;
            btnDel.disabled = false;
            btnEdit.disabled = false;
            btnCancel.disabled = false;    // ✅ 편집 모드 아닐 때만 꺼짐
        }
    }

    /** 현재 상황을 읽어서 버튼 모드 자동 결정 */
    function refreshButtonsByState() {
    if (hasDraftRowNow()) return setButtonsForMode('draft');
    if (isEditingNow())   return setButtonsForMode('editing');
    return setButtonsForMode('idle');
    }

    function setAdminMode(isPrivileged) {
        state.isAdmin = !!isPrivileged;
        __essentialAuth.isPrivileged = !!isPrivileged;

        renderTable(state.items);   // 모드 반영
        refreshButtonsByState();    // 버튼 상태 갱신
        enforceEssentialEditButtonVisibility(); // 파츠 편집 버튼 표시/숨김
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
            setAdminMode(false);               // 실패 시 기본 비권한
        }
    }

    // essential 탭의 '파츠 편집' 토글 버튼 표시/숨김
    function enforceEssentialEditButtonVisibility() {
        const wrap = document.querySelector('.essential-parts-wrapper');
        if (!wrap) return;
        const mainBtn = wrap.querySelector('#add-parts-btn');     // 파츠 편집 버튼
        const panel   = wrap.querySelector('.edit-btn-container'); // 내부 버튼 모음(열려있다면 닫기)

        if (!mainBtn) return;

        if (__essentialAuth.isPrivileged) {
            mainBtn.style.display = '';
        } else {
            mainBtn.style.display = 'none';
            if (panel) panel.style.display = 'none';
        }
    }

    function searchWithFilters(params) {
        // scripts_parts.js에서 전달: {EQ_Type, Parts_Name, EQ_Unit, Q_Code, P_N}
        state.lastFilters = {
        eqType: params?.EQ_Type ?? '',
        parts:  params?.Parts_Name ?? '',
        unit:   params?.EQ_Unit ?? '',
        qcode:  params?.Q_Code ?? '',
        pn:     params?.P_N ?? ''
        };
        selectedPNsPurchase.clear();
        renderTable(state.items); // 필터 적용해 재렌더
        refreshButtonsByState();
    }

    function buildUI(container) {
        container.innerHTML = `
        <div class="table-wrap">
            <table class="essential-table display compact nowrap" style="width:100%">
            <colgroup>
                <col style="width:80px">
                <col style="width:150px">
                <col style="width:150px">
                <col style="width:400px">
                <col style="width:400px">
                <col style="width:70px"> 
                <col style="width:70px"> 
                <col style="width:140px"> 
                <col style="width:180px"> 
            </colgroup>
            <thead>
                <tr>
                <th>구분</th>
                <th>파츠 분류</th>
                <th>Q-Code</th>
                <th>Unit</th>
                <th>업체P/N</th>
                <th>EMS 재고</th>
                <th>라인 재고</th>
                <th>구매현황</th>
                <th>비고</th>
                </tr>
            </thead>
            <tbody></tbody>
            </table>
        </div>
        `;
    }

    // 전역(상단) 필터만 적용
    function applyFiltersOnly(list) {
        return filterItemsByGlobalFilters(list, state.lastFilters);
    }


    // DT 데이터 구성 & 초기화/갱신
    function finalizePurchaseTable(rowsData) {
        const $table = $(state.container).find('.essential-table');

        // 파생 컬럼 포함 데이터셋 준비
        const dtData = (rowsData || []).map(it => {
        const ems  = Number(it.EMS재고 || 0);
        const line = Number(it.라인재고 || 0);

        // 구매 필요 조건: EMS 재고 == 0
        const needPurchase = (ems === 0);
        const purchaseStatus = needPurchase ? '구매 필요' : '';

        return {
            EQ_Name: it.EQ_Name ?? '',
            Parts_Name: it.Parts_Name ?? '',
            Q_Code: it.Q_Code ?? '',
            EQ_Unit: it.EQ_Unit ?? '',
            P_N: it.P_N ?? '',
            // 총재고(전체 재고) 제거
            EMS재고: ems,
            라인재고: line,
            // ↓ 정렬/표시용 필드
            need_flag: needPurchase ? 1 : 0,     // 정렬용 숫자
            Purchase_status: purchaseStatus,     // 표시용 텍스트
            Memo: it.Memo ?? '',
            _pn: it.P_N ?? ''
        };
        });

        // 이미 초기화되어 있으면 갱신
        if ($.fn.DataTable.isDataTable($table)) {
        purchaseDT.clear().rows.add(dtData).draw(false);
        return;
        }

        // 최초 초기화
        purchaseDT = $table.DataTable({
        data: dtData,
        columns: [
            { data: 'EQ_Name'    },
            { data: 'Parts_Name' },
            { data: 'Q_Code'     },
            { data: 'EQ_Unit'    },
            { data: 'P_N',
            createdCell: (td) => td.setAttribute('data-col','P_N')
            },
            // 총재고 컬럼 제거 후 인덱스 재정렬
            { data: 'EMS재고',   className: 'dt-right' },
            { data: '라인재고',  className: 'dt-right' },
            {
            data: 'Purchase_status',
            className: 'dt-center',
            render: (data, type, row) => {
                // 정렬/타입 계산 시에는 숫자 플래그 사용
                if (type === 'sort' || type === 'type') return row.need_flag;
                return data; // '구매 필요' 또는 ''
            }
            },
            { data: 'Memo',
            createdCell: (td) => td.setAttribute('data-col','Memo')
            }
        ],
        // DataTables 옵션 — 총재고 제거에 맞춰 기본 정렬 변경
        scrollY: '610px',
        scrollX: false,
        scrollCollapse: true,
        paging: false,
        searching: true,   // dom:'t'라 기본 검색창은 숨김(외부 검색/전역필터와 병행)
        fixedHeader: true,
        orderCellsTop: true,
        info: false,
        dom: 't',
        // 기본 정렬: EMS 재고 오름차순 (이전 총재고 asc 대체)
        order: [[5, 'asc']],
        createdRow: (row, data) => {
            // 인라인 편집/선택용 키 부여
            Array.from(row.cells).forEach(td => td.setAttribute('data-pn', data._pn));
            // (선택 복원은 사용하지 않음; 필요시 bindDTSelectable에서 restoreOnDraw: true로 전환)
        }
        });

        // ✅ 행 선택 토글(다중 선택, 재그림 복원 OFF)
        bindDTSelectable($table, {
            selectedSet: selectedPNsPurchase,
            getKey: (tr) => tr.querySelector('td')?.getAttribute('data-pn'),
            singleSelect: false,
            restoreOnDraw: false
        });
        wireOutsideDeselect($table, {
            selectedSet: selectedPNsPurchase,
            preserveSelectors: [
                'button', '.edit-group-btn',
                '#add-parts-btn', '#edit-add-btn', '#edit-delete-btn', '#edit-btn', '#edit-cancel-btn',
                '.dataTables_length', '.dataTables_filter', '.dataTables_paginate', '.dataTables_info',
                '.dtfh-floatingparent', '.dtfh-floatingheader'
            ]
        });
    }

    function renderTable(items) {
        if (!state.container) return;
        const filtered = applyFiltersOnly(items);
        finalizePurchaseTable(filtered);
    }

    async function fetchItems() {
        if (state.isFetching) return;
        state.isFetching = true;
        try {
        const r = await fetch('/parts_essential_list');
        state.items = await r.json();

        // 공통(상단 필터)에게 최신 데이터 전달 → 드롭다운 채움 폴백용
        document.dispatchEvent(new CustomEvent('parts:data-loaded', {
            detail: { which: 'essential', items: state.items }
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
        enforceEssentialEditButtonVisibility(); // 버튼 DOM이 만들어진 직후 1차 적용
        
        await checkClientAuthority();
        await fetchItems();

        wireEditEvents();           // ← 빠져있던 호출 추가
        refreshButtonsByState();

        state.inited = true;
    }

    /* ============================== 테이블 편집 (행추가, 행삭제, 수정, 취소)  ============================== */

    function inputHTML(name, val = '', type = 'text') {
        const v = String(val ?? '');
        return `
        <input class="dt-input" name="${escapeHtml(name)}"
            type="${escapeHtml(type)}" value="${escapeHtml(v)}"
            style="width:100%;box-sizing:border-box;">`;
    }

    // ---------- 행을 인풋으로 변환(수정) ----------
    function rowToInputs(rowNode) {
        const data  = purchaseDT.row(rowNode).data();
        const cells = rowNode.cells || rowNode.children;
        if (!cells || !data) return;

        // insert/update 허용 컬럼 기준으로만 인풋화
        cells[0].innerHTML = inputHTML('EQ_Name',      data.EQ_Name);
        cells[1].innerHTML = inputHTML('Parts_Name',   data.Parts_Name);
        cells[2].innerHTML = inputHTML('Q_Code',       data.Q_Code);
        cells[3].innerHTML = inputHTML('EQ_Unit',      data.EQ_Unit);
        cells[4].innerHTML = inputHTML('P_N',          data.P_N);

        // 파생 재고: 읽기전용
        cells[5].textContent = data.EMS재고 ?? 0;
        cells[6].textContent = data.라인재고 ?? 0;

        // 수정 허용
        cells[7].innerHTML = data.Purchase_status ?? '';   // 구매현황
        cells[8].innerHTML = inputHTML('Memo',             data.Memo);              // 비고

        rowNode.classList.add('editing-row');
        // ✅ Q-Code 입력 시 대문자 자동 변환
        $(rowNode).find('input.dt-input[name="Q_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function() { this.value = this.value.toUpperCase(); });
    }

    function collectInputs(rowNode) {
        const obj = {};
        $(rowNode).find('input.dt-input').each(function () {
            const n = this.name;
            let v = this.value ?? '';
            v = v.trim();
            if (n === 'Q_Code' || n === 'EQ_Name') v = v.toUpperCase();
            obj[n] = v;
        });
        return obj;
    }

    function restoreRowFromData(rowNode, data) {
        purchaseDT.row(rowNode).data({
            EQ_Name:          data.EQ_Name ?? '',
            Parts_Name:       data.Parts_Name ?? '',
            Q_Code:           data.Q_Code ?? '',
            EQ_Unit:          data.EQ_Unit ?? '',
            P_N:              data.P_N ?? '',
            EMS재고:          Number(data.EMS재고 ?? data.Count_Now ?? 0),
            라인재고:         Number(data.라인재고 ?? 0),
            Etc:              data.Etc ?? '',
            Purchase_status:  data.Purchase_status ?? '',
            Memo:             data.Memo ?? '',
            need_flag:        Number(data.need_flag ?? (Number(data.EMS재고 ?? data.Count_Now ?? 0) === 0 ? 1 : 0)),
            _pn:              data.P_N ?? '',
        });
        purchaseDT.draw(false);
    }

    function makeDraftInput(name, value = '', { type = 'text', placeholder = '' } = {}) {
        const v = String(value ?? '');
        return `
        <input class="dt-input"
            name="${escapeHtml(name)}"
            type="${escapeHtml(type)}"
            value="${escapeHtml(v)}"
            placeholder="${escapeHtml(placeholder)}"
            style="width:100%;box-sizing:border-box;">`;
    }

    function addDraftRow() {
        if (!purchaseDT) return;

        // 이미 드래프트가 있으면 추가 금지
        const $tbody = $(state.container).find('.essential-table tbody');
        const hasDraft = $tbody.find('tr.draft-row').length > 0;
        if (hasDraft) {
            alert('이미 추가 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
        }

        // 정렬용으로 EMS재고 = -1 (오름차순에서 항상 맨 위)
        const draftData = {
            EQ_Name: '', Parts_Name: '', Q_Code: '', EQ_Unit: '', P_N: '',
            EMS재고: -1,          // ← 정렬 값 (표시는 아래에서 0으로 바꿔줌)
            라인재고: 0,
            Purchase_status: '', Memo: '',
            need_flag: 1, _pn: '', _draft: true,
        };

        // 1) 행 추가 + draw
        const rowApi = purchaseDT.row.add(draftData);
        purchaseDT.draw(false);

        // 2) 노드 가져와 드래프트 마킹
        const node  = rowApi.node();
        node.classList.add('draft-row');
        node.classList.remove('selected');

        // 3) 셀을 인풋/표시로 교체 (EMS재고는 표시만 0으로)
        const cells = node.cells || node.children;
        cells[0].innerHTML = makeDraftInput('EQ_Name', '');
        cells[1].innerHTML = makeDraftInput('Parts_Name', '');
        cells[2].innerHTML = makeDraftInput('Q_Code', '');
        cells[3].innerHTML = makeDraftInput('EQ_Unit', '');
        cells[4].innerHTML = makeDraftInput('P_N', '');
        cells[5].textContent = '0';           // ← 보여줄 값
        cells[6].textContent = '0';
        cells[7].textContent = '';            // 구매현황은 입력 안 함
        cells[8].innerHTML = makeDraftInput('Memo', '');

        // 4) 혹시 현재 정렬 상태에 따라 중간에 들어갔다면, DOM상으로 최상단으로 이동
        //    (정렬이 다시 트리거되어도 EMS재고=-1 덕분에 항상 맨 위 유지)
        $tbody.prepend(node);
        // ✅ 드래프트 시작 → 버튼 잠그기
        setButtonsForMode('draft');

        // ✅ 드래프트행의 Q-Code 입력 시 대문자 자동 변환
        $(node).find('input.dt-input[name="Q_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function () { this.value = this.value.toUpperCase(); });
    }

    function collectDraftRows() {
        const rows = [];
        const $table = $(state.container).find('.essential-table');
        $table.find('tbody tr.draft-row').each(function () {
            const obj = {};
            $(this).find('input.dt-input').each(function () {
                const name = this.name;
                let val = this.value?.trim() ?? '';
                if (name === 'Q_Code' || name === 'EQ_Name') val = val.toUpperCase();
                obj[name] = val;
            });
            rows.push(obj);
        });
        return rows;
    }

    function removeAllDraftRows() {
        if (!purchaseDT) return;
        const $table = $(state.container).find('.essential-table');
        $table.find('tbody tr.draft-row').each(function () {
            purchaseDT.row(this).remove();
        });
        purchaseDT.draw(false);


        // ✅ 현재 상태에 맞춰 버튼 갱신
        refreshButtonsByState();

    }

    // ---------- 검증 ----------
    function validateRow(r, rowNode) {
        const required = ['EQ_Name', 'Parts_Name', 'Q_Code', 'EQ_Unit', 'P_N'];
        const missing = required.filter(k => !String(r[k] ?? '').trim());
        if (missing.length) {
            if (rowNode) {
            const el = rowNode.querySelector(`input[name="${missing[0]}"]`);
            if (el) el.focus();
            }
            return `필수 항목 누락: ${missing.join(', ')}`;
        }

        // ✅ Q-Code 형식 검증
        const q = String(r.Q_Code ?? '').trim().toUpperCase();
        const QCODE_REGEX = /^[A-Z0-9]{8}-[A-Z0-9]{6}$/;

        if (!QCODE_REGEX.test(q)) {
            if (rowNode) {
            const el = rowNode.querySelector('input[name="Q_Code"]');
            if (el) el.focus();
            }
            return 'Q-Code 형식이 올바르지 않습니다.';
        }

        return null;
    }

    // ---------- 이벤트 배선 ----------
    function wireEditEvents() {
        if (wireEditEvents.__wired) return;
        wireEditEvents.__wired = true;

        // (섹션 전용) 수정 버튼 직접 핸들 — 라인 모듈과 동일한 방식
        const essentialWrap = document.querySelector('.essential-parts-wrapper');
        const editBtn = essentialWrap?.querySelector('#edit-btn');
        if (editBtn && !editBtn.__wired) {
            editBtn.__wired = true;
            editBtn.addEventListener('click', () => {
                if (!purchaseDT) return;

                if (isEditingNow()) {
                    alert('이미 편집 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }
                const nodes = purchaseDT.rows('.selected').nodes().toArray();
                if (nodes.length === 0) {
                    alert('수정할 행을 1건 이상 선택해 주세요.');
                    return;
                }

                if (nodes.some(n => n.classList.contains('draft-row'))) {
                    alert('행추가 중인 드래프트가 포함되어 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                // ✅ 선택된 모든 행을 인풋으로 전환
                editCtxs = [];
                for (const rowNode of nodes) {
                    const rowApi = purchaseDT.row(rowNode);
                    const before = { ...rowApi.data() };
                    const beforeKey = { Q_Code: before.Q_Code, EQ_Unit: before.EQ_Unit, P_N: before.P_N };
                    editCtxs.push({ rowNode, rowApi, before, beforeKey });
                    rowToInputs(rowNode);
                }

                // ✅ 수정 모드로 버튼 잠금
                setButtonsForMode('editing');
            });
        }

        // 행추가
        document.addEventListener('parts-edit:add-row', () => {
            const wrap = document.querySelector('.essential-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;
            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }
            addDraftRow();
        });

        // 저장(드래프트 → INSERT, 수정행 → UPDATE)
        document.addEventListener('parts-edit:save', async () => {
            const wrap = document.querySelector('.essential-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            try {
                // 1) 수정 UPDATE
                if (isEditingNow()) {
                    // 수집
                    const updates = [];
                    for (const { rowNode, before, beforeKey } of editCtxs) {
                        const updated = collectInputs(rowNode);
                        const err = validateRow({ ...before, ...updated }, rowNode);
                        if (err) {
                        alert(err);
                        document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok:false, error:'VALIDATION', message: err } }));
                        return;
                        }

                        updates.push({
                        where: beforeKey,
                        data: {
                            EQ_Name:    updated.EQ_Name    ?? before.EQ_Name,
                            Parts_Name: updated.Parts_Name ?? before.Parts_Name,
                            Q_Code:     updated.Q_Code     ?? before.Q_Code,
                            EQ_Unit:    updated.EQ_Unit    ?? before.EQ_Unit,
                            P_N:        updated.P_N        ?? before.P_N,
                            Memo:       updated.Memo       ?? before.Memo,
                        },
                        _key: before._pn,    // 로컬 복원용
                        _before: before,     // 로컬 복원시 사용
                        _rowNode: rowNode
                        });
                    }

                    // 순차 저장
                    for (const u of updates) {
                        const resp = await fetch('/parts_essential_list/update', {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ where: u.where, data: u.data })
                        });
                        if (!resp.ok) {
                        const msg = await resp.text().catch(()=> '서버 오류');
                        throw new Error(msg);
                        }
                    }

                    // 로컬 테이블 반영 (전체 리프레시 없이)
                    for (const u of updates) { 
                        const after = {
                        ...u._before,
                        ...u.data
                        };
                        restoreRowFromData(u._rowNode, after);
                    }

                    // 종료
                    editCtxs = [];
                }

                // 2) 드래프트 INSERT
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
                            EQ_Name:    d.EQ_Name,
                            Parts_Name: d.Parts_Name,
                            Q_Code:     d.Q_Code,
                            EQ_Unit:    d.EQ_Unit,
                            P_N:        d.P_N,
                            Memo:       d.Memo ?? ''
                        };
                        const resp = await fetch('/parts_essential_list', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (!resp.ok) throw new Error(await resp.text());
                    }
                    removeAllDraftRows();
                }

                await fetchItems();
                alert('저장되었습니다.');
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));

                // ✅ 저장 완료 → 드래프트 모드 해제
                refreshButtonsByState();
            } catch (e) {
                console.error(e);
                alert('Q-Code 값이 중복입니다.\n' + (e.message || e));
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: e?.message } }));
            }
        });

        // 삭제(다중)
        document.addEventListener('parts-edit:delete-row', async () => {
            const wrap = document.querySelector('.essential-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            if (!purchaseDT) return;
            // 드래프트 선택 제거
            const $tbody = $(state.container).find('.essential-table tbody');
            const $draftSelected = $tbody.find('tr.draft-row.selected');
            if ($draftSelected.length) {
                // 드래프트는 로컬 제거만
                $draftSelected.each(function(){ purchaseDT.row(this).remove(); });
                purchaseDT.draw(false);
      
            }

            const selected = purchaseDT.rows('.selected').data().toArray();
            if (!selected.length) return alert('삭제할 행을 선택해 주세요.');

            // 수정 중이면 막기
            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }

            const rows = selected.map((r) => ({
                Q_Code:  r.Q_Code,
                EQ_Unit: r.EQ_Unit,
                P_N:     r.P_N
            }));

            if (!confirm(`선택한 ${rows.length}건을 삭제할까요?`)) return;

            try {
                const resp = await fetch('/parts_essential_list/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rows })
                });
                if (!resp.ok) throw new Error(await resp.text());
                await fetchItems();
                alert('삭제되었습니다.');
            } catch (e) {
                console.error(e);
                alert('삭제 중 오류가 발생했습니다.\n' + (e.message || e));
            }
        });

        // 취소(드래프트 제거/수정 복원)
        document.addEventListener('parts-edit:cancel', () => {
            const wrap = document.querySelector('.essential-parts-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            removeAllDraftRows();
            if (isEditingNow()) {
                for (const { rowNode, before } of editCtxs) {
                    restoreRowFromData(rowNode, before);
                }
                editCtxs = [];
            }

            // ✅ 취소 → 드래프트 모드 해제
            refreshButtonsByState();
        });
    }

    function __clearSelection() {
        try {
        selectedPNsPurchase.clear();
        const $t = $(state.container).find('.essential-table');
        if ($t.length) $t.find('tbody tr.selected').removeClass('selected');
        } catch(e){ console.warn('[PartsEssential.__clearSelection]', e); }
    }
    return { init, fetch: () => fetchItems(), searchWithFilters, __clearSelection };
})();


window.PartsEssential = PartsEssential;