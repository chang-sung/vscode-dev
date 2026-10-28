// ---------------------------------
// 파츠 교체/점검 이력 모듈 — DataTables 버전
// plm_eqp_repair_web_log 대응 버전
// ---------------------------------
const PartsHistory = (() => {
    let state = {
        items: [],
        isFetching: false,
        isAdmin: false,
        inited: false,
        container: null,
        lastFilters: {}
    };

    let historyDT = null;
    let editCtxs = [];

    // 선택 상태 저장용
    const selectedHistoryKeys = new Set();
    const __historyAuth = { isPrivileged: false };

    function nowDateTimeStr() {
        const d = new Date();
        const y = d.getFullYear();
        const m = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hh = String(d.getHours()).padStart(2, '0');
        const mm = String(d.getMinutes()).padStart(2, '0');
        const ss = String(d.getSeconds()).padStart(2, '0');
        return `${y}-${m}-${day} ${hh}:${mm}:${ss}`;
    }

    function makeRowKey(row) {
        return [
            row.WriteTime ?? '',
            row.Work_Time ?? '',
            row.EQ_Name ?? '',
            row.Unit ?? '',
            row.Parts_Code ?? '',
            row.Work_Name ?? ''
        ].join('||');
    }

    // =========================
    // 버튼 상태 유틸
    // =========================

    function getCurrentUserName() {
        return String(window.__currentUserName || '').trim();
    }

    function canEditOrDeleteRow(row) {
        const currentUser = getCurrentUserName();

        const writers = String(row?.Work_Name || '')
            .split(',')
            .map(v => v.trim())
            .filter(Boolean);

        return !!currentUser && writers.includes(currentUser);
    }

    function getUnauthorizedRows(rows) {
        return (rows || []).filter(row => !canEditOrDeleteRow(row));
    }

    function getButtonsInHistoryWrap() {
        const wrap = document.querySelector('.parts-history-wrapper');
        return {
            btnAdd:    wrap?.querySelector('#history-edit-add-btn'),
            btnDel:    wrap?.querySelector('#history-edit-delete-btn'),
            btnEdit:   wrap?.querySelector('#history-edit-btn'),
            btnCancel: wrap?.querySelector('#history-edit-cancel-btn'),
        };
    }

    function hasDraftRowNow() {
        const $tbody = $(state.container).find('.history-table tbody');
        return $tbody.find('tr.draft-row').length > 0;
    }

    function isEditingNow() {
        return editCtxs.length > 0;
    }

    function setButtonsForMode(mode) {
        const { btnAdd, btnDel, btnEdit, btnCancel } = getButtonsInHistoryWrap();
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

    function refreshButtonsByState() {
        if (hasDraftRowNow()) return setButtonsForMode('draft');
        if (isEditingNow()) return setButtonsForMode('editing');
        return setButtonsForMode('idle');
    }

    function setAdminMode(isPrivileged) {
        state.isAdmin = !!isPrivileged;
        __historyAuth.isPrivileged = !!isPrivileged;

        renderTable(state.items);
        refreshButtonsByState();
        enforceHistoryEditButtonVisibility();
    }

    async function checkClientAuthority() {
        let data = { authority: '사용자', userName: '' };

        try {
            const res = await fetch('/check-client-authority-parts');
            data = await res.json();

            const isPrivileged = (
            data.authority === '관리자' ||
            data.authority === '구매담당자' ||
            data.authority === '사용자'
            );
            setAdminMode(isPrivileged);
        } catch (e) {
            console.error('History authority check failed:', e);
            setAdminMode(false);
            }

            if (!document.getElementById('user-name-display')) {
            const userNameContainer = document.createElement('div');
            userNameContainer.id = 'user-name-display';

            const userNameBox = document.createElement('div');
            userNameBox.id = 'user-name-box';
            userNameBox.innerText = `${data.authority} 모드 ( ${data.userName} )`;

            userNameContainer.appendChild(userNameBox);
            document.body.prepend(userNameContainer);
        }
    }

    function enforceHistoryEditButtonVisibility() {
        const wrap = document.querySelector('.parts-history-wrapper');
        if (!wrap) return;

        const mainBtn = wrap.querySelector('#history-add-btn');
        const panel = wrap.querySelector('.edit-btn-container');

        if (!mainBtn) return;

        if (__historyAuth.isPrivileged) {
            mainBtn.style.display = '';
        } else {
            mainBtn.style.display = 'none';
            if (panel) panel.style.display = 'none';
        }
    }

    // =========================
    // 검색 필터 반영
    // =========================
    function searchWithFilters(params) {
        state.lastFilters = {
            eqGroup:    params?.EQ_Group ?? params?.eqGroup ?? '',
            eqType:     params?.EQ_Type ?? params?.eqType ?? '',
            unit:       params?.EQ_Unit ?? params?.unit ?? '',
            partsGroup: params?.Parts_Name ?? params?.parts ?? '',
            qcode:      params?.Q_Code ?? params?.qcode ?? '',
            pn:         params?.P_N ?? params?.pn ?? '',
            partsGrade: params?.Parts_Grade ?? '',
            serialNo:   params?.Serial_No ?? ''
        };

        selectedHistoryKeys.clear();

        populateHistoryFilterOptions(state.items);
        renderTable(state.items);
        refreshButtonsByState();
    }

    function populateHistoryFilterOptions(items) {
        const selEqGroup  = document.getElementById('history-eqp-group-select');
        const selEqType   = document.getElementById('history-eqp-type-select');
        const selUnit     = document.getElementById('history-unit-type-select');
        const selPartsDiv = document.getElementById('history-parts-multiselect');

        if (!selEqGroup || !selEqType || !selUnit || !selPartsDiv) return;

        const prevEqGroup  = selEqGroup.value || 'ALL';
        const prevEqType   = selEqType.value || 'ALL';
        const prevUnit     = selUnit.value || 'ALL';
        const safeItems = Array.isArray(items) ? items : [];

        const eqGroupList = Array.from(new Set(
            safeItems.map(it => String(it.EQ_Group || '').trim()).filter(Boolean)
        )).sort();

        selEqGroup.innerHTML =
            `<option value="ALL">설비그룹 전체</option>` +
            eqGroupList.map(v => `<option value="${v}">${v}</option>`).join('');

        selEqGroup.value = eqGroupList.includes(prevEqGroup) ? prevEqGroup : 'ALL';

        const selectedEqGroup = selEqGroup.value;

        const byGroupItems = (selectedEqGroup === 'ALL')
            ? safeItems
            : safeItems.filter(it => String(it.EQ_Group || '').trim() === selectedEqGroup);

        const eqList = Array.from(new Set(
            byGroupItems.map(it => String(it.EQ_Name || '').trim()).filter(Boolean)
        )).sort();

        selEqType.innerHTML =
            `<option value="ALL">설비 전체</option>` +
            eqList.map(v => `<option value="${v}">${v}</option>`).join('');

        selEqType.value = eqList.includes(prevEqType) ? prevEqType : 'ALL';

        const selectedEqType = selEqType.value;

        const byEqItems = byGroupItems.filter(it =>
            selectedEqType === 'ALL' || String(it.EQ_Name || '').trim() === selectedEqType
        );

        const unitList = Array.from(new Set(
            byEqItems.map(it => String(it.Unit || '').trim()).filter(Boolean)
        )).sort();

        selUnit.innerHTML =
            `<option value="ALL">Unit 전체</option>` +
            unitList.map(v => `<option value="${v}">${v}</option>`).join('');

        selUnit.value = unitList.includes(prevUnit) ? prevUnit : 'ALL';

        const selectedUnit = selUnit.value;

        const byUnitItems = byEqItems.filter(it =>
            selectedUnit === 'ALL' || String(it.Unit || '').trim() === selectedUnit
        );

        const partsList = Array.from(new Set(
            byUnitItems.map(it => String(it.Parts_Group || '').trim()).filter(Boolean)
        )).sort();

        buildHistoryPartsMultiSelect(partsList);
    }

    function getDefaultCheckedParts(partsList = []) {
        return partsList.filter(v => String(v).trim() !== 'PAD');
    }

    function buildHistoryPartsMultiSelect(partsList = []) {
        const multi = document.getElementById('history-parts-multiselect');
        if (!multi) return;

        const panel = multi.querySelector('.history-parts-panel');
        const label = multi.querySelector('.history-parts-label');
        const toggle = multi.querySelector('.history-parts-toggle');

        if (!panel || !label || !toggle) return;

        if (!multi.__wired) {
            multi.__wired = true;

            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                multi.classList.toggle('open');
            });

            panel.addEventListener('click', (e) => {
                e.stopPropagation();
            });

            document.addEventListener('click', () => {
                multi.classList.remove('open');
            });
        }

        const prevSelected = Array.isArray(state.lastFilters.partsGroups)
            ? state.lastFilters.partsGroups
            : [];

        const selectedList = prevSelected.length
            ? prevSelected.filter(v => partsList.includes(v))
            : getDefaultCheckedParts(partsList);

        state.lastFilters.partsGroups = selectedList;

        panel.innerHTML = `
            <label class="history-parts-row history-parts-all-row">
                <input type="checkbox" value="__ALL__">
                <span>전체 선택</span>
            </label>
            ${partsList.map(part => {
                const checked = selectedList.includes(part) ? 'checked' : '';
                return `
                    <label class="history-parts-row">
                        <input type="checkbox" value="${escapeHtml(part)}" ${checked}>
                        <span>${escapeHtml(part)}</span>
                    </label>
                `;
            }).join('')}
        `;

        const allChk = panel.querySelector('input[value="__ALL__"]');
        const partChks = Array.from(
            panel.querySelectorAll('input[type="checkbox"]:not([value="__ALL__"])')
        );

        const updateSelected = () => {
            const checkedValues = partChks
                .filter(chk => chk.checked)
                .map(chk => chk.value);

            state.lastFilters.partsGroups = checkedValues;
            state.lastFilters.partsGroup = '';

            if (allChk) {
                allChk.checked = checkedValues.length === partsList.length && partsList.length > 0;
                allChk.indeterminate = checkedValues.length > 0 && checkedValues.length < partsList.length;
            }

            if (checkedValues.length === 0) {
                label.textContent = '선택 없음';
            } else if (checkedValues.length === partsList.length) {
                label.textContent = '파츠 전체';
            } else if (checkedValues.length === 1) {
                label.textContent = checkedValues[0];
            } else {
                label.textContent = `파츠 ${checkedValues.length}개 선택`;
            }

            renderTable(state.items);
            refreshButtonsByState();
        };

        if (allChk) {
            allChk.addEventListener('change', () => {
                partChks.forEach(chk => {
                    chk.checked = allChk.checked;
                });
                updateSelected();
            });
        }

        partChks.forEach(chk => {
            chk.addEventListener('change', updateSelected);
        });

        updateSelected();
    }

    function rebuildEqOptionsBySelectedGroup() {
    populateHistoryFilterOptions(state.items);
    }

    function rebuildUnitOptionsByFilters() {
        populateHistoryFilterOptions(state.items);
    }

    function rebuildPartsOptionsByFilters() {
        populateHistoryFilterOptions(state.items);
    }

    function applyFiltersOnly(list) {
        const f = state.lastFilters || {};

        return (list || []).filter(it => {
            if (f.eqGroup && String(it.EQ_Group || '').trim() !== f.eqGroup) return false;
            if (f.eqType && String(it.EQ_Name || '').trim() !== f.eqType) return false;
            if (f.unit && String(it.Unit || '').trim() !== f.unit) return false;
            if (Array.isArray(f.partsGroups) && f.partsGroups.length > 0) {
            if (!f.partsGroups.includes(String(it.Parts_Group || '').trim())) return false;
            } else if (f.partsGroup && String(it.Parts_Group || '').trim() !== f.partsGroup) {
                return false;
            }

            if (f.qcode && !String(it.Parts_Code || '').toUpperCase().includes(f.qcode.toUpperCase())) return false;
            if (f.pn && !String(it.Parts_Name || '').toUpperCase().includes(f.pn.toUpperCase())) return false;
            if (f.partsGrade && !String(it.Parts_Grade || '').toUpperCase().includes(f.partsGrade.toUpperCase())) return false;
            if (f.serialNo && !String(it.Serial_Number || '').toUpperCase().includes(f.serialNo.toUpperCase())) return false;

            return true;
        });
    }

    // =========================
    // 모니터링 summary 카드 유틸
    // =========================

    function getHistorySummaryCounts(items) {
        let need = 0;
        let review = 0;

        for (const it of (items || [])) {
            const stateVal = String(it?.Eqp_State || '').trim();

            if (stateVal === '조치필요') need++;
            else if (stateVal === '평가중') review++;
        }

        return { need, review };
    }

    function renderHistorySummaryCards(sum) {
        const box = document.querySelector('.group3-container .summary-container');
        if (!box) return;

        const need = Number(sum?.need ?? 0);
        const review = Number(sum?.review ?? 0);

        box.innerHTML = `
            <div class="summary-cards">
                <div class="summary-card st-review" data-eqp-state="평가중">
                    <div class="label">평가중</div>
                    <div class="value">${review.toLocaleString()}</div>
                </div>
                <div class="summary-card st-need" data-eqp-state="조치필요">
                    <div class="label">조치필요</div>
                    <div class="value">${need.toLocaleString()}</div>
                </div>
                
            </div>
        `;
    }

    function refreshHistorySummaryCards() {
        const filtered = applyFiltersOnly(state.items);
        const sum = getHistorySummaryCounts(filtered);
        renderHistorySummaryCards(sum);
    }

    function wireHistorySummaryCards() {
        if (wireHistorySummaryCards.__wired) return;
        wireHistorySummaryCards.__wired = true;

        const host = document.querySelector('.group3-container .summary-container');
        if (!host) return;

        host.addEventListener('click', (e) => {
            const card = e.target?.closest?.('.summary-card');
            if (!card) return;

            const eqpState = card.dataset.eqpState;
            if (!eqpState) return;

            const alreadyActive = card.classList.contains('active');

            host.querySelectorAll('.summary-card').forEach(el => el.classList.remove('active'));

            if (alreadyActive) {
                renderTable(state.items); // 전체 복귀
                return;
            }

            card.classList.add('active');

            const filtered = applyFiltersOnly(state.items).filter(it =>
                String(it?.Eqp_State || '').trim() === eqpState
            );

            finalizeHistoryTable(filtered);
        });
    }

    // =========================
    // UI
    // =========================
    function buildUI(container) {
        container.innerHTML = `
        <div class="table-wrap">
            <table class="history-table display compact nowrap" style="width:100%">
                <colgroup>
                    <col style="width:150px">
                    <col style="width:50px">
                    <col style="width:100px">
                    <col style="width:50px">
                    <col style="width:120px">
                    <col style="width:100px">
                    <col style="width:120px">
                    <col style="width:160px">
                    <col style="width:100px">
                    <col style="width:90px">
                    <col style="width:90px">
                    <col style="width:90px">
                    <col style="width:340px">
                    <col style="width:100px">
                </colgroup>
                <thead>
                    <tr>
                        <th>작업시간</th>
                        <th>Site</th>
                        <th>설비그룹</th>
                        <th>설비</th>
                        <th>Unit</th>
                        <th>Sub Unit</th>
                        <th>파츠 분류</th>
                        <th>업체P/N</th>
                        <th>Q-Code</th>
                        <th>시리얼</th>
                        <th>파츠상태</th>
                        <th>점검결과</th>
                        <th>작업내용</th>
                        <th>작업자</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
        `;
    }

    // =========================
    // DataTable
    // =========================
    function finalizeHistoryTable(rowsData) {
        window.PartsHistoryCycle?.setRows(rowsData || []);
        const $table = $(state.container).find('.history-table');

        const dtData = (rowsData || []).map(it => ({
            WriteTime:      it.WriteTime ?? '',
            Work_Time:      it.Work_Time ?? '',
            Site:           it.Site ?? '',
            EQ_Group:       it.EQ_Group ?? '',
            EQ_Name:        it.EQ_Name ?? '',
            Unit:           it.Unit ?? '',
            Sub_Unit:       it.Sub_Unit ?? '',
            Parts_Group:    it.Parts_Group ?? '',
            Parts_Name:     it.Parts_Name ?? '',
            Parts_Code:     it.Parts_Code ?? '',
            Serial_Number:  it.Serial_Number ?? '',
            Parts_Grade:    it.Parts_Grade ?? '',
            Eqp_State:      it.Eqp_State ?? '',
            Comment:        it.Comment ?? '',
            Work_Name:      it.Work_Name ?? '',
            Work_IP:        it.Work_IP ?? '',
            _rowKey: makeRowKey(it)
        }));

        if ($.fn.DataTable.isDataTable($table)) {
            historyDT.clear().rows.add(dtData).draw(false);
            return;
        }

        historyDT = $table.DataTable({
            data: dtData,
            columns: [
                { data: 'Work_Time' },
                { data: 'Site' },
                { data: 'EQ_Group' },
                { data: 'EQ_Name' },
                { data: 'Unit' },
                { data: 'Sub_Unit' },
                { data: 'Parts_Group' },
                { data: 'Parts_Name' },
                { data: 'Parts_Code', createdCell: (td) => td.setAttribute('data-col', 'Parts_Code') },
                { data: 'Serial_Number', createdCell: (td) => td.setAttribute('data-col', 'Serial_Number') },
                { data: 'Parts_Grade' },
                { data: 'Eqp_State' },
                { data: 'Comment', createdCell: (td) => td.setAttribute('data-col', 'Comment') },
                { data: 'Work_Name' }
            ],
            scrollY: '575px',
            scrollX: true,
            scrollCollapse: true,
            paging: false,
            searching: true,
            fixedHeader: true,
            orderCellsTop: true,
            info: false,
            dom: 't',
            order: [[0, 'desc']],
            createdRow: (row, data) => {
                Array.from(row.cells).forEach(td => td.setAttribute('data-rowkey', data._rowKey));

                const state = String(data.Eqp_State || '').trim();

                if (state === '조치필요') {
                row.classList.add('row-need'); // 🔴 빨강
                } else if (state === '평가중') {
                row.classList.add('row-review'); // 🟠 주황
                }
            }
        });

        bindDTSelectable($table, {
            selectedSet: selectedHistoryKeys,
            getKey: (tr) => tr.querySelector('td')?.getAttribute('data-rowkey'),
            restoreOnDraw: false
        });

        wireOutsideDeselect($table, {
            selectedSet: selectedHistoryKeys,
            preserveSelectors: [
                'button', '.edit-group-btn',
                '#history-add-btn', '#history-edit-add-btn', '#history-edit-delete-btn',
                '#history-edit-btn', '#history-edit-cancel-btn',
                '.dataTables_length', '.dataTables_filter', '.dataTables_paginate',
                '.dataTables_info', '.dtfh-floatingparent', '.dtfh-floatingheader'
            ]
        });
    }

    function renderTable(items) {
        if (!state.container) return;
        const filtered = applyFiltersOnly(items);
        finalizeHistoryTable(filtered);

        const sum = getHistorySummaryCounts(filtered);
        renderHistorySummaryCards(sum);
    }

    // =========================
    // fetch
    // =========================
    async function fetchItems() {
        if (state.isFetching) return;
        state.isFetching = true;

        try {
            const f = state.lastFilters || {};
            const qs = new URLSearchParams();

            if (f.eqGroup) qs.set('eqGroup', f.eqGroup);
            if (f.eqType) qs.set('eqType', f.eqType);
            if (f.unit) qs.set('unit', f.unit);
            if (f.partsGroup) qs.set('partsGroup', f.partsGroup);
            if (f.qcode) qs.set('qcode', f.qcode);
            if (f.pn) qs.set('pn', f.pn);
            if (f.partsGrade) qs.set('partsGrade', f.partsGrade);
            if (f.serialNo) qs.set('serialNo', f.serialNo);

            const url = `/parts_replacement_history${qs.toString() ? `?${qs.toString()}` : ''}`;
            const r = await fetch(url);
            const json = await r.json();

            state.items = Array.isArray(json?.items) ? json.items : [];

            populateHistoryFilterOptions(state.items);

            document.dispatchEvent(new CustomEvent('parts:data-loaded', {
                detail: { which: 'history', items: state.items }
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
        enforceHistoryEditButtonVisibility();

        await checkClientAuthority();
        await fetchItems();

        wireEditEvents();
        refreshButtonsByState();
        wireHistorySummaryCards();

        state.inited = true;
    }

    // =========================
    // 행 입력 드롭다운
    // =========================
    const HISTORY_OPTIONS = {
        Site: ['A3', 'A4'],
        EQ_Group: ['PLM_S', 'PLM_T', 'ACV'],

        Unit_S: [
            'IN LFT',
            'OUT LFT',
            'LD PA TR',
            'LD TO TR',
            'PA PRE ALIGN',
            'ATT LD TR',
            'MAG TR',
            'FILM SPY TR',
            '2SHEET STG',
            'FILM LD TR',
            'PA DET',
            'FILM DET',
            'SET TBL1',
            'SET TBL2',
            'ATT STG1',
            'ATT STG2',
            'ATT ULD TR',
            'INSP STG',
            'ATT INSP',
            'ULD TO TR',
            'ULD PA TR'
        ],

        Unit_T: [
            'PA LOADER',
            'PA SHT',
            'PA TURNOVER',
            'PA TRANSFER',
            'PA LAMI SHT',
            'FILM LAMI SHT',
            'PA BUFFER',
            'LAMI TRANSFER',
            'LAMI SHT',
            'LAMI TURNOVER',
            'INSPECT TR',
            'PA 배출 SHT',
            '불량배출 TR',
            '양품배출 TR',
            'FILM LFT',
            'FILM LOADER',
            'FILM SHT',
            'FILM TRANSFER'
        ],

        Unit_ACV: [
            'CHAMBER',
            'Load C/V-1',
            'Load C/V-2',
            'Load C/V-3',
            'Load C/V-4',
            'Unload C/V-1',
            'Unload C/V-2',
            'Unload C/V-3',
            'Unload C/V-4'
        ],

        SubUnit1: ['미해당','A','B'],
        SubUnit2: ['미해당','X', 'Y', 'Z', 'TH', 'TT'],

        Parts_Group_S: [
            '라미 롤러',
            'PAD',
            '실린더',
            '로봇',
            '감속기',
            '모터',
            '볼스크류',
            '볼너트',
            'LM블럭',
            'LM레일',
            '베어링',
            'Auto Stage',
            '조명컨트롤러',
            '속도 임시변경',
            '기타'
        ],

        Parts_Group_T: [
            '라미 롤러',
            'PAD',
            '실린더',
            '로봇',
            '감속기',
            '모터',
            '볼스크류',
            '볼너트',
            'LM블럭',
            'LM레일',
            '베어링',
            '크로스롤러베어링',
            '조명컨트롤러',
            '속도 임시변경',
            '기타'
        ],

        Parts_Group_ACV: [
            '립씰',
            '패킹',
            '파워몰러',
            '기타'
        ],

        Parts_Grade: [
            'A급 교체',
            'B급 교체',
            '수리품',
            '모터 튜닝',
            '구리스 도포',
            '구매중'
        ],

        Eqp_State: ['완료', '조치필요', '평가중']
    };

    function getUnitOptionsByEqGroup(eqGroup = '') {
        const g = String(eqGroup || '').trim().toUpperCase();

        if (g === 'PLM_S') return HISTORY_OPTIONS.Unit_S || [];
        if (g === 'PLM_T') return HISTORY_OPTIONS.Unit_T || [];
        if (g === 'ACV')   return HISTORY_OPTIONS.Unit_ACV || [];

        return [];
    }

    function getPartsGroupOptionsByEqGroup(eqGroup = '') {
        const g = String(eqGroup || '').trim().toUpperCase();

        if (g === 'PLM_S') return HISTORY_OPTIONS.Parts_Group_S || [];
        if (g === 'PLM_T') return HISTORY_OPTIONS.Parts_Group_T || [];
        if (g === 'ACV')   return HISTORY_OPTIONS.Parts_Group_ACV || [];

        return [];
    }

    function buildEqNameOptions(eqGroup = '') {
        const g = String(eqGroup || '').trim().toUpperCase();
        let baseList = [];
        let suffix = '';

        if (g === 'PLM_S') {
            baseList = [
                ...Array.from({ length: 64 }, (_, i) => String(i + 35).padStart(2, '0')),
                'A0', 'A1', 'A2'
            ];
            suffix = 'G';
        } else if (g === 'PLM_T') {
            baseList = Array.from({ length: 30 }, (_, i) => String(i + 3).padStart(2, '0'));
            suffix = 'T';
        } else if (g === 'ACV') {
            baseList = Array.from({ length: 32 }, (_, i) => String(i + 1).padStart(2, '0'));
            suffix = 'G';
        } else {
            return [];
        }

        return baseList.map(v => `${v}${suffix}`);
    }

    function splitSubUnit(value = '') {
        const raw = String(value || '').trim();

        if (!raw) return { sub1: '', sub2: '' };

        if (raw.includes('-')) {
            const [sub1, sub2] = raw.split('-');
            return {
                sub1: sub1 || '',
                sub2: sub2 || ''
            };
        }

        return { sub1: raw || '', sub2: '' };
    }

    function isLamiRoller(partsGroup = '') {
        return String(partsGroup || '').trim() === '라미 롤러';
    }

    function toggleSerialInputByPartsGroup(rowNode, partsGroupValue = '') {
        const enable = isLamiRoller(partsGroupValue);

        const serialInput = rowNode.querySelector('input[name="Serial_Number"]');
        if (!serialInput) return;

        serialInput.disabled = !enable;
        serialInput.required = enable;

        if (!enable) {
            serialInput.value = '';
            serialInput.placeholder = '';
            serialInput.classList.remove('required-field');
        } else {
            serialInput.placeholder = '시리얼 입력';
            serialInput.classList.add('required-field');
        }
    }

    //----------------- parts_use_list lookup

    function getCurrentSubUnitTextFromRow(rowNode) {
        const sub1El = rowNode.querySelector('[name="Sub_Unit1"]');
        const sub2El = rowNode.querySelector('[name="Sub_Unit2"]');

        const sub1 = String(sub1El?.value || '').trim();
        const sub2 = String(sub2El?.value || '').trim();

        return buildSubUnit(sub1, sub2);
    }

    async function lookupPartsUseInfo({ unit, subUnit, partsGroup }) {
        const qs = new URLSearchParams({
            unit: String(unit || '').trim(),
            subUnit: String(subUnit || '').trim(),
            partsGroup: String(partsGroup || '').trim()
        });

        const resp = await fetch(`/parts_use_lookup?${qs.toString()}`);
        if (!resp.ok) {
            throw new Error(await resp.text().catch(() => 'lookup failed'));
        }

        const json = await resp.json();
        return json?.item || null;
    }

    async function autofillHistoryPartsInfo(rowNode) {
        if (!rowNode) return;

        const unitEl = rowNode.querySelector('[name="Unit"]');
        const partsGroupEl = rowNode.querySelector('[name="Parts_Group"]');
        const qCodeEl = rowNode.querySelector('[name="Parts_Code"]');
        const specEl = rowNode.querySelector('[name="Parts_Name"]'); // UI상 업체P/N 컬럼

        if (!unitEl || !partsGroupEl || !qCodeEl || !specEl) return;

        const unit = String(unitEl.value || '').trim();
        const partsGroup = String(partsGroupEl.value || '').trim();
        const subUnit = getCurrentSubUnitTextFromRow(rowNode);

        // 값 초기화 유틸
        const clearAutoFilledFields = () => {
            qCodeEl.value = '';
            specEl.value = '';

            qCodeEl.dispatchEvent(new Event('input', { bubbles: true }));
            specEl.dispatchEvent(new Event('input', { bubbles: true }));
        };

        // 필수 선택값 부족하면 기존 자동입력값 제거
        if (!unit || !partsGroup || !subUnit) {
            clearAutoFilledFields();
            return;
        }

        try {
            const item = await lookupPartsUseInfo({ unit, subUnit, partsGroup });

            // 조회 결과 없으면 기존 자동입력값 제거
            if (!item) {
                clearAutoFilledFields();
                return;
            }

            qCodeEl.value = String(item.Q_Code || '').trim().toUpperCase();
            specEl.value = String(item.Spec || '').trim();

            qCodeEl.dispatchEvent(new Event('input', { bubbles: true }));
            specEl.dispatchEvent(new Event('input', { bubbles: true }));
        } catch (err) {
            console.error('[autofillHistoryPartsInfo] failed:', err);

            // 조회 실패 시에도 이전 값 남지 않게 초기화하고 싶으면 유지
            clearAutoFilledFields();
        }
    }

    function bindAutoFillEvents(rowNode) {
        if (!rowNode) return;

        const unitEl = rowNode.querySelector('[name="Unit"]');
        const partsGroupEl = rowNode.querySelector('[name="Parts_Group"]');
        const sub1El = rowNode.querySelector('[name="Sub_Unit1"]');
        const sub2El = rowNode.querySelector('[name="Sub_Unit2"]');

        const handler = () => {
            autofillHistoryPartsInfo(rowNode);
        };

        unitEl?.addEventListener('change', handler);
        partsGroupEl?.addEventListener('change', handler);
        sub1El?.addEventListener('change', handler);
        sub2El?.addEventListener('change', handler);
    }


    // =========================
    // 편집 유틸
    // =========================

    function buildSubUnit(sub1, sub2) {
        const s1 = String(sub1 || '').trim();
        const s2 = String(sub2 || '').trim();

        if (s1 === '미해당' && s2) return s2;
        if (s1 && s2) return `${s1}-${s2}`;
        if (s1) return s1;
        return '';
    }

    function inputHTML(name, val = '', type = 'text') {
        const v = String(val ?? '');
        return `
        <input class="dt-input" name="${escapeHtml(name)}"
            type="${escapeHtml(type)}" value="${escapeHtml(v)}"
            style="width:100%;box-sizing:border-box;">`;
    }

    function selectHTML(name, options = [], selected = '', disabled = false, placeholder = '') {
        const current = String(selected ?? '').trim();
        return `
            <select
                class="dt-input dt-select"
                name="${escapeHtml(name)}"
                style="width:100%;box-sizing:border-box;"
                ${disabled ? 'disabled' : ''}
            >
                <option value="">${escapeHtml(placeholder)}</option>
                ${options.map(opt => {
                    const v = String(opt);
                    const sel = v === current ? 'selected' : '';
                    return `<option value="${escapeHtml(v)}" ${sel}>${escapeHtml(v)}</option>`;
                }).join('')}
            </select>
        `;
    }

    function subUnitHTML(subUnitValue = '') {
        const { sub1, sub2 } = splitSubUnit(subUnitValue);

        return `
            <div class="subunit-combo" style="display:flex;gap:4px;align-items:center;">
                <select class="dt-input dt-select" name="Sub_Unit1" style="width:48%;box-sizing:border-box;">
                    <option value=""></option>
                    ${HISTORY_OPTIONS.SubUnit1.map(opt => `
                        <option value="${escapeHtml(opt)}" ${opt === sub1 ? 'selected' : ''}>${escapeHtml(opt)}</option>
                    `).join('')}
                </select>
                <span style="flex:0 0 auto;">-</span>
                <select class="dt-input dt-select" name="Sub_Unit2" style="width:48%;box-sizing:border-box;">
                    <option value=""></option>
                    ${HISTORY_OPTIONS.SubUnit2.map(opt => `
                        <option value="${escapeHtml(opt)}" ${opt === sub2 ? 'selected' : ''}>${escapeHtml(opt)}</option>
                    `).join('')}
                </select>
            </div>
        `;
    }

    function rowToInputs(rowNode) {
        const data = historyDT.row(rowNode).data();
        const cells = rowNode.cells || rowNode.children;
        if (!cells || !data) return;

        const eqGroup = String(data.EQ_Group || '').trim();
        const eqNameOptions = buildEqNameOptions(eqGroup);
        const unitOptions = getUnitOptionsByEqGroup(eqGroup);
        const partsGroupOptions = getPartsGroupOptionsByEqGroup(eqGroup);
        const groupNotSelected = !eqGroup;

        cells[0].innerHTML  = inputHTML('Work_Time', data.Work_Time);
        cells[1].innerHTML  = selectHTML('Site', HISTORY_OPTIONS.Site, data.Site);
        cells[2].innerHTML  = selectHTML('EQ_Group', HISTORY_OPTIONS.EQ_Group, data.EQ_Group);
        cells[3].innerHTML  = selectHTML(
            'EQ_Name',
            eqNameOptions,
            data.EQ_Name,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[4].innerHTML  = selectHTML(
            'Unit',
            unitOptions,
            data.Unit,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[5].innerHTML  = subUnitHTML(data.Sub_Unit);
        cells[6].innerHTML  = selectHTML(
            'Parts_Group',
            partsGroupOptions,
            data.Parts_Group,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[7].innerHTML  = inputHTML('Parts_Name', data.Parts_Name);
        cells[8].innerHTML  = inputHTML('Parts_Code', data.Parts_Code);
        cells[9].innerHTML  = inputHTML('Serial_Number', data.Serial_Number);
        cells[10].innerHTML = selectHTML('Parts_Grade', HISTORY_OPTIONS.Parts_Grade, data.Parts_Grade);
        cells[11].innerHTML = selectHTML('Eqp_State', HISTORY_OPTIONS.Eqp_State, data.Eqp_State);
        cells[12].innerHTML = inputHTML('Comment', data.Comment);
        cells[13].innerHTML = inputHTML('Work_Name', data.Work_Name);

        rowNode.classList.add('editing-row');

        $(rowNode).find('input.dt-input[name="Parts_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function () { this.value = this.value.toUpperCase(); });

        $(rowNode).find('select[name="EQ_Group"]').on('change', function () {
            const selectedGroup = this.value;
            const $eqName = $(rowNode).find('select[name="EQ_Name"]');
            const $unit = $(rowNode).find('select[name="Unit"]');
            const $partsGroup = $(rowNode).find('select[name="Parts_Group"]');

            const eqOpts = buildEqNameOptions(selectedGroup);
            const unitOpts = getUnitOptionsByEqGroup(selectedGroup);
            const partsGroupOpts = getPartsGroupOptionsByEqGroup(selectedGroup);
            const disabled = !selectedGroup;

            $eqName.html(
                `<option value=""></option>` +
                eqOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $eqName.prop('disabled', disabled);
            $eqName.val('');

            $unit.html(
                `<option value=""></option>` +
                unitOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $unit.prop('disabled', disabled);
            $unit.val('');

            $partsGroup.html(
                `<option value=""></option>` +
                partsGroupOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $partsGroup.prop('disabled', disabled);
            $partsGroup.val('');

            // EQ 그룹 바뀌면 연관 자동입력 필드도 초기화
            $(rowNode).find('[name="Parts_Code"]').val('');
            $(rowNode).find('[name="Parts_Name"]').val('');
            $(rowNode).find('[name="Serial_Number"]').val('');
        });

        $(rowNode).find('select[name="Parts_Group"]').on('change', function () {
            toggleSerialInputByPartsGroup(rowNode, this.value);
        });

        // 초기 상태 반영
        toggleSerialInputByPartsGroup(rowNode, data.Parts_Group);

        // 자동조회 바인딩
        bindAutoFillEvents(rowNode);
    }

    function validateWorkNameFormat(value = '') {
        const text = String(value || '').trim();

        // 한글/영문/숫자 + 콤마만 허용
        // 스페이스 및 기타 특수문자 금지
        const regex = /^[가-힣a-zA-Z0-9,]+$/;

        return regex.test(text);
    }

    function normalizeWorkName(value = '') {
        return String(value || '')
            .trim()
            .replace(/\s*,\s*/g, ',')   // 콤마 앞뒤 공백 제거
            .replace(/\s+/g, ',')       // 공백 구분을 콤마로 변경
            .replace(/,+/g, ',')        // 콤마 중복 제거
            .replace(/^,|,$/g, '');     // 앞뒤 콤마 제거
    }

    function collectInputs(rowNode) {
        const obj = {};
        let sub1 = '';
        let sub2 = '';

        $(rowNode).find('input.dt-input, select.dt-input').each(function () {
            const n = this.name;
            let v = this.value ?? '';
            v = v.trim();

            if (n === 'Parts_Code' || n === 'EQ_Name') v = v.toUpperCase();
            if (n === 'Work_Name') v = normalizeWorkName(v);

            if (n === 'Sub_Unit1') {
                sub1 = v;
                return;
            }
            if (n === 'Sub_Unit2') {
                sub2 = v;
                return;
            }

            obj[n] = v;
        });

        obj.Sub_Unit = buildSubUnit(sub1, sub2);

        return obj;
    }

    function restoreRowFromData(rowNode, data) {
        historyDT.row(rowNode).data({
            WriteTime:      data.WriteTime ?? '',
            Work_Time:      data.Work_Time ?? '',
            Site:           data.Site ?? '',
            EQ_Group:       data.EQ_Group ?? '',
            EQ_Name:        data.EQ_Name ?? '',
            Unit:           data.Unit ?? '',
            Sub_Unit:       data.Sub_Unit ?? '',
            Parts_Group:    data.Parts_Group ?? '',
            Parts_Name:     data.Parts_Name ?? '',
            Parts_Code:     data.Parts_Code ?? '',
            Serial_Number:  data.Serial_Number ?? '',
            Parts_Grade:    data.Parts_Grade ?? '',
            Eqp_State:      data.Eqp_State ?? '',
            Comment:        data.Comment ?? '',
            Work_Name:      data.Work_Name ?? '',
            Work_IP:        data.Work_IP ?? '',
            _rowKey: makeRowKey(data)
        });
        historyDT.draw(false);
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
        if (!historyDT) return;

        const $tbody = $(state.container).find('.history-table tbody');
        const hasDraft = $tbody.find('tr.draft-row').length > 0;
        if (hasDraft) {
            alert('이미 추가 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
        }

        const now = nowDateTimeStr();

        const draftData = {
            WriteTime: now,
            Work_Time: now,
            Site: '',
            EQ_Group: '',
            EQ_Name: '',
            Unit: '',
            Sub_Unit: '',
            Parts_Group: '',
            Parts_Name: '',
            Parts_Code: '',
            Serial_Number: '',
            Parts_Grade: '',
            Eqp_State: '',
            Comment: '',
            Work_Name: getCurrentUserName() || '',
            Work_IP: '',
            _draft: true,
            _rowKey: `draft||${Date.now()}`
        };

        const rowApi = historyDT.row.add(draftData);
        historyDT.draw(false);

        const node = rowApi.node();
        node.classList.add('draft-row');
        node.classList.remove('selected');

        const cells = node.cells || node.children;
        const eqGroup = String(draftData.EQ_Group || '').trim();
        const eqNameOptions = buildEqNameOptions(eqGroup);
        const unitOptions = getUnitOptionsByEqGroup(eqGroup);
        const partsGroupOptions = getPartsGroupOptionsByEqGroup(eqGroup);
        const groupNotSelected = !eqGroup;

        cells[0].innerHTML  = makeDraftInput('Work_Time', draftData.Work_Time);
        cells[1].innerHTML  = selectHTML('Site', HISTORY_OPTIONS.Site, draftData.Site);
        cells[2].innerHTML  = selectHTML('EQ_Group', HISTORY_OPTIONS.EQ_Group, draftData.EQ_Group);
        cells[3].innerHTML  = selectHTML(
            'EQ_Name',
            eqNameOptions,
            draftData.EQ_Name,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[4].innerHTML  = selectHTML(
            'Unit',
            unitOptions,
            draftData.Unit,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[5].innerHTML  = subUnitHTML(draftData.Sub_Unit);
        cells[6].innerHTML  = selectHTML(
            'Parts_Group',
            partsGroupOptions,
            draftData.Parts_Group,
            groupNotSelected,
            groupNotSelected ? '' : ''
        );
        cells[7].innerHTML  = makeDraftInput('Parts_Name', draftData.Parts_Name);
        cells[8].innerHTML  = makeDraftInput('Parts_Code', draftData.Parts_Code);
        cells[9].innerHTML  = makeDraftInput('Serial_Number', draftData.Serial_Number);
        cells[10].innerHTML = selectHTML('Parts_Grade', HISTORY_OPTIONS.Parts_Grade, draftData.Parts_Grade);
        cells[11].innerHTML = selectHTML('Eqp_State', HISTORY_OPTIONS.Eqp_State, draftData.Eqp_State);
        cells[12].innerHTML = makeDraftInput('Comment', draftData.Comment);
        cells[13].innerHTML = makeDraftInput('Work_Name', draftData.Work_Name);

        $tbody.prepend(node);
        setButtonsForMode('draft');

        $(node).find('input.dt-input[name="Parts_Code"], input.dt-input[name="EQ_Name"]')
            .on('input', function () { this.value = this.value.toUpperCase(); });

        $(node).find('select[name="EQ_Group"]').on('change', function () {
            const selectedGroup = this.value;
            const $eqName = $(node).find('select[name="EQ_Name"]');
            const $unit = $(node).find('select[name="Unit"]');
            const $partsGroup = $(node).find('select[name="Parts_Group"]');

            const eqOpts = buildEqNameOptions(selectedGroup);
            const unitOpts = getUnitOptionsByEqGroup(selectedGroup);
            const partsGroupOpts = getPartsGroupOptionsByEqGroup(selectedGroup);
            const disabled = !selectedGroup;

            $eqName.html(
                `<option value=""></option>` +
                eqOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $eqName.prop('disabled', disabled);
            $eqName.val('');

            $unit.html(
                `<option value=""></option>` +
                unitOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $unit.prop('disabled', disabled);
            $unit.val('');

            $partsGroup.html(
                `<option value=""></option>` +
                partsGroupOpts.map(v => `<option value="${v}">${v}</option>`).join('')
            );
            $partsGroup.prop('disabled', disabled);
            $partsGroup.val('');

            $(node).find('[name="Parts_Code"]').val('');
            $(node).find('[name="Parts_Name"]').val('');
            $(node).find('[name="Serial_Number"]').val('');
        });

        $(node).find('select[name="Parts_Group"]').on('change', function () {
            toggleSerialInputByPartsGroup(node, this.value);
        });

        // 초기 상태 반영
        toggleSerialInputByPartsGroup(node, draftData.Parts_Group);

        // 자동조회 바인딩
        bindAutoFillEvents(node);
    }

    function collectDraftRows() {
        const rows = [];
        const $table = $(state.container).find('.history-table');

        $table.find('tbody tr.draft-row').each(function () {
            const obj = {
                WriteTime: nowDateTimeStr()
            };

            let sub1 = '';
            let sub2 = '';

            $(this).find('input.dt-input, select.dt-input').each(function () {
                const name = this.name;
                let val = this.value?.trim() ?? '';

                if (name === 'Parts_Code' || name === 'EQ_Name') val = val.toUpperCase();
                if (name === 'Work_Name') val = normalizeWorkName(val);

                if (name === 'Sub_Unit1') {
                    sub1 = val;
                    return;
                }
                if (name === 'Sub_Unit2') {
                    sub2 = val;
                    return;
                }

                obj[name] = val;
            });

            obj.Sub_Unit = buildSubUnit(sub1, sub2);

            rows.push(obj);
        });

        return rows;
    }

    function removeAllDraftRows() {
        if (!historyDT) return;
        const $table = $(state.container).find('.history-table');
        $table.find('tbody tr.draft-row').each(function () {
            historyDT.row(this).remove();
        });
        historyDT.draw(false);
        refreshButtonsByState();
    }

    const FIELD_LABELS = {
        Work_Time: '작업시간',
        Site: 'Site',
        EQ_Group: '설비그룹',
        EQ_Name: '설비',
        Unit: 'Unit',
        Sub_Unit: 'Sub Unit',
        Parts_Group: '파츠 분류',
        Parts_Name: '파츠명',
        Parts_Code: 'Q-Code',
        Parts_Grade: '파츠상태',
        Eqp_State: '점검결과',
        Serial_Number: '시리얼',
        Work_Name: '작업자'
    };

    function validateRow(r, rowNode) {
        const required = [
            'Work_Time',
            'Site',
            'EQ_Group',
            'EQ_Name',
            'Unit',
            'Sub_Unit',
            'Parts_Group',
            'Parts_Name',
            'Parts_Code',
            'Parts_Grade',
            'Eqp_State',
            'Work_Name'
        ];

        if (isLamiRoller(r.Parts_Group)) {
            required.push('Serial_Number');
        }

        const missing = required.filter(k => !String(r[k] ?? '').trim());

        // ✅ 작업자 이름 형식 검사
        if (!validateWorkNameFormat(r.Work_Name)) {

            if (rowNode) {
                const el = rowNode.querySelector('[name="Work_Name"]');
                if (el) el.focus();
            }

            return `작업자의 이름 구분은 콤마(,)로만 입력이 가능합니다.
        (예 : 김철민,변창성)`;
        }

        if (missing.length) {
            if (rowNode) {
                let el = rowNode.querySelector(`[name="${missing[0]}"]`);
                if (el) el.focus();
            }

            // ✅ 여기 핵심: 사용자용 이름으로 변환
            const readable = missing.map(k => FIELD_LABELS[k] || k);

            return `필수 항목 누락: ${readable.join(', ')}`;
        }

        return null;
    }

    // =========================
    // 이벤트
    // =========================
    function wireEditEvents() {
        if (wireEditEvents.__wired) return;
        wireEditEvents.__wired = true;

        const historyWrap = document.querySelector('.parts-history-wrapper');
        const editBtn = historyWrap?.querySelector('#history-edit-btn');

        if (editBtn && !editBtn.__wired) {
            editBtn.__wired = true;
            editBtn.addEventListener('click', () => {
                if (!historyDT) return;

                if (isEditingNow()) {
                    alert('이미 편집 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                const nodes = historyDT.rows('.selected').nodes().toArray();
                if (nodes.length === 0) {
                    alert('수정할 행을 1건 이상 선택해 주세요.');
                    return;
                }

                if (nodes.some(n => n.classList.contains('draft-row'))) {
                    alert('행추가 중인 드래프트가 포함되어 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                const selectedRows = historyDT.rows('.selected').data().toArray();
                const unauthorized = getUnauthorizedRows(selectedRows);

                if (unauthorized.length > 0) {
                    alert('작업자 본인의 이력만 수정할 수 있습니다.');
                    return;
                }

                editCtxs = [];
                for (const rowNode of nodes) {
                    const rowApi = historyDT.row(rowNode);
                    const before = { ...rowApi.data() };
                    const beforeKey = {
                        WriteTime: before.WriteTime,
                        Work_Time: before.Work_Time,
                        EQ_Name: before.EQ_Name,
                        Unit: before.Unit,
                        Parts_Code: before.Parts_Code,
                        Work_Name: before.Work_Name
                    };

                    editCtxs.push({ rowNode, rowApi, before, beforeKey });
                    rowToInputs(rowNode);
                }

                setButtonsForMode('editing');
            });
        }

        document.addEventListener('parts-edit:add-row', () => {
            const wrap = document.querySelector('.parts-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }
            addDraftRow();
        });

        document.addEventListener('parts-edit:save', async () => {
            const wrap = document.querySelector('.parts-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            try {
                // 1) UPDATE
                if (isEditingNow()) {
                    const updates = [];

                    for (const { rowNode, before, beforeKey } of editCtxs) {
                        const updated = collectInputs(rowNode);
                        const merged = { ...before, ...updated };

                        const err = validateRow(merged, rowNode);
                        if (err) {
                            alert(err);
                            document.dispatchEvent(new CustomEvent('parts-edit:result', {
                                detail: { ok: false, error: 'VALIDATION', message: err }
                            }));
                            return;
                        }

                        updates.push({
                            where: beforeKey,
                            data: {
                                Work_Time:      merged.Work_Time,
                                Site:           merged.Site,
                                EQ_Group:       merged.EQ_Group,
                                EQ_Name:        merged.EQ_Name,
                                Unit:           merged.Unit,
                                Sub_Unit:       merged.Sub_Unit,
                                Parts_Group:    merged.Parts_Group,
                                Parts_Name:     merged.Parts_Name,
                                Parts_Code:     merged.Parts_Code,
                                Serial_Number:  merged.Serial_Number,
                                Parts_Grade:    merged.Parts_Grade,
                                Eqp_State:      merged.Eqp_State,
                                Comment:        merged.Comment,
                                Work_Name:      merged.Work_Name
                            },
                            _before: before,
                            _rowNode: rowNode
                        });
                    }

                    for (const u of updates) {
                        const resp = await fetch('/parts_replacement_history/update', {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ where: u.where, data: u.data })
                        });
                        if (!resp.ok) {
                            const msg = await resp.text().catch(() => '서버 오류');
                            throw new Error(msg);
                        }
                    }

                    for (const u of updates) {
                        const after = { ...u._before, ...u.data };
                        restoreRowFromData(u._rowNode, after);
                    }

                    editCtxs = [];
                }

                // 2) INSERT
                const drafts = collectDraftRows();
                if (drafts.length) {
                    for (const d of drafts) {
                        const err = validateRow(d);
                        if (err) {
                            alert(err);
                            document.dispatchEvent(new CustomEvent('parts-edit:result', {
                                detail: { ok: false, error: 'VALIDATION', message: err }
                            }));
                            return;
                        }
                    }

                    for (const d of drafts) {
                        const resp = await fetch('/parts_replacement_history', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(d)
                        });
                        if (!resp.ok) throw new Error(await resp.text());
                    }

                    removeAllDraftRows();
                }

                await fetchItems();
                alert('저장되었습니다.');
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));
                refreshButtonsByState();

            } catch (e) {
                console.error(e);
                alert('저장 중 오류가 발생했습니다.\n' + (e.message || e));
                document.dispatchEvent(new CustomEvent('parts-edit:result', {
                    detail: { ok: false, error: e?.message }
                }));
            }
        });

        document.addEventListener('parts-edit:delete-row', async () => {
            const wrap = document.querySelector('.parts-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            if (!historyDT) return;

            const $tbody = $(state.container).find('.history-table tbody');
            const $draftSelected = $tbody.find('tr.draft-row.selected');
            if ($draftSelected.length) {
                $draftSelected.each(function () { historyDT.row(this).remove(); });
                historyDT.draw(false);
            }

            const selected = historyDT.rows('.selected').data().toArray();
            if (!selected.length) return alert('삭제할 행을 선택해 주세요.');

            const unauthorized = getUnauthorizedRows(selected);
            if (unauthorized.length > 0) {
                alert('작업자 본인의 이력만 삭제할 수 있습니다.');
                return;
            }

            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }

            const rows = selected.map(r => ({
                WriteTime: r.WriteTime,
                Work_Time: r.Work_Time,
                EQ_Name: r.EQ_Name,
                Unit: r.Unit,
                Parts_Code: r.Parts_Code,
                Work_Name: r.Work_Name
            }));

            if (!confirm(`선택한 ${rows.length}건을 삭제할까요?`)) return;

            try {
                const resp = await fetch('/parts_replacement_history/delete', {
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

        document.addEventListener('parts-edit:cancel', () => {
            const wrap = document.querySelector('.parts-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            removeAllDraftRows();

            if (isEditingNow()) {
                for (const { rowNode, before } of editCtxs) {
                    restoreRowFromData(rowNode, before);
                }
                editCtxs = [];
            }

            refreshButtonsByState();
        });
    }

    function clearSelection() {
        try {
            selectedHistoryKeys.clear();
            const $t = $(state.container).find('.history-table');
            if ($t.length) $t.find('tbody tr.selected').removeClass('selected');
        } catch (e) {
            console.warn('[PartsHistory.clearSelection]', e);
        }
    }

    return {
        init,
        fetch: () => fetchItems(),
        searchWithFilters,
        clearSelection,
        rebuildEqOptionsBySelectedGroup,
        rebuildUnitOptionsByFilters,
        rebuildPartsOptionsByFilters
    };
})();

window.PartsHistory = PartsHistory;

