// ---------------------------------
// 품질 이력 모듈 — DataTables 버전
// table: plc_log.plm_quality_history_web_log
// ---------------------------------
const QualityHistory = (() => {
    let state = {
        items: [],
        isFetching: false,
        isAdmin: false,
        inited: false,
        container: null,
        lastFilters: {}
    };

    let qualityDT = null;
    let editCtxs = [];

    const selectedQualityKeys = new Set();
    const __qualityAuth = { isPrivileged: false };

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

    function escapeHtml(v) {
        return String(v ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function makeRowKey(row) {
        return [
            row.WriteTime ?? '',
            row.WorkTime ?? '',
            row.EQ_Name ?? '',
            row.Lot_id ?? '',
            row.Type ?? '',
            row.Write_Name ?? ''
        ].join('||');
    }

    function getCurrentUserName() {
        return String(window.__currentUserName || '').trim();
    }

    function canEditOrDeleteRow(row) {
        const currentUser = getCurrentUserName();

        const writers = String(row?.Write_Name || '')
            .split(',')
            .map(v => v.trim())
            .filter(Boolean);

        const workers = String(row?.Work_Name || '')
            .split(',')
            .map(v => v.trim())
            .filter(Boolean);

        return !!currentUser && (writers.includes(currentUser) || workers.includes(currentUser));
    }

    function getUnauthorizedRows(rows) {
        return (rows || []).filter(row => !canEditOrDeleteRow(row));
    }

    function getButtonsInQualityWrap() {
        const wrap = document.querySelector('.quality-history-wrapper');
        return {
            btnAdd: wrap?.querySelector('#quality-history-edit-add-btn'),
            btnDel: wrap?.querySelector('#quality-history-edit-delete-btn'),
            btnEdit: wrap?.querySelector('#quality-history-edit-btn'),
            btnCancel: wrap?.querySelector('#quality-history-edit-cancel-btn')
        };
    }

    function hasDraftRowNow() {
        const $tbody = $(state.container).find('.quality-history-table tbody');
        return $tbody.find('tr.draft-row').length > 0;
    }

    function isEditingNow() {
        return editCtxs.length > 0;
    }

    function setButtonsForMode(mode) {
        const { btnAdd, btnDel, btnEdit, btnCancel } = getButtonsInQualityWrap();
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

    function enforceQualityEditButtonVisibility() {
        const wrap = document.querySelector('.quality-history-wrapper');
        if (!wrap) return;

        const mainBtn = wrap.querySelector('#quality-history-add-btn');
        const panel = wrap.querySelector('.edit-btn-container');

        if (!mainBtn) return;

        if (__qualityAuth.isPrivileged) {
            mainBtn.style.display = '';
        } else {
            mainBtn.style.display = 'none';
            if (panel) panel.style.display = 'none';
        }
    }

    function setAdminMode(isPrivileged) {
        state.isAdmin = !!isPrivileged;
        __qualityAuth.isPrivileged = !!isPrivileged;

        renderTable(state.items);
        refreshButtonsByState();
        enforceQualityEditButtonVisibility();
    }

    async function checkClientAuthority() {
        let data = { authority: '사용자', userName: '' };

        try {
            const res = await fetch('/check-client-authority-parts');
            data = await res.json();

            window.__currentUserName = data.userName || window.__currentUserName || '';

            const isPrivileged = (
                data.authority === '관리자' ||
                data.authority === '구매담당자' ||
                data.authority === '사용자'
            );

            setAdminMode(isPrivileged);
        } catch (e) {
            console.error('Quality authority check failed:', e);
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

    function searchWithFilters(params) {
        state.lastFilters = {
            eqType: params?.EQ_Type ?? params?.eqType ?? '',
            keyword: params?.allKeyword ?? params?.keyword ?? '',
            model: params?.Model ?? '',
            lotId: params?.Lot_id ?? '',
            type: params?.Type ?? '',
            category: params?.Category ?? ''
        };

        selectedQualityKeys.clear();
        renderTable(state.items);
        refreshButtonsByState();
    }

    function applyFiltersOnly(list) {
        const f = state.lastFilters || {};

        return (list || []).filter(it => {
            if (f.eqType && !String(it.EQ_Name || '').toUpperCase().includes(f.eqType.toUpperCase())) return false;
            if (f.model && !String(it.Model || '').toUpperCase().includes(f.model.toUpperCase())) return false;
            if (f.lotId && !String(it.Lot_id || '').toUpperCase().includes(f.lotId.toUpperCase())) return false;
            if (f.type && !String(it.Type || '').toUpperCase().includes(f.type.toUpperCase())) return false;
            if (f.category && !String(it.Category || '').toUpperCase().includes(f.category.toUpperCase())) return false;

            if (f.keyword) {
                const kw = String(f.keyword).toUpperCase();
                const targets = [
                    it.WriteTime,
                    it.WorkTime,
                    it.EQ_Name,
                    it.Model,
                    it.Lot_id,
                    it.Type,
                    it.Category,
                    it.Detail,
                    it.Comment,
                    it.Write_Name,
                    it.Feedback,
                    it.Work_Name,
                    it.Process_status
                ];

                const matched = targets.some(v => String(v || '').toUpperCase().includes(kw));
                if (!matched) return false;
            }

            return true;
        });
    }

    function getQualitySummaryCounts(items) {
        let pending = 0;
        let done = 0;

        for (const it of (items || [])) {
            const status = String(it?.Process_status || '처리 필요').trim();

            if (status === '완료') done++;
            else pending++;
        }

        return { pending, done };
    }

    function renderQualitySummaryCards(sum) {
        const box = document.querySelector('.group3-container .summary-container');
        if (!box) return;

        const pending = Number(sum?.pending ?? 0);
        const done = Number(sum?.done ?? 0);

        box.innerHTML = `
            <div class="summary-cards">
                <div class="summary-card st-req" data-quality-status="처리 필요">
                    <div class="label">처리 필요</div>
                    <div class="value">${pending.toLocaleString()}</div>
                </div>

                <div class="summary-card st-done" data-quality-status="완료">
                    <div class="label">완료</div>
                    <div class="value">${done.toLocaleString()}</div>
                </div>
            </div>
        `;
    }

    function wireQualitySummaryCards() {
        if (wireQualitySummaryCards.__wired) return;
        wireQualitySummaryCards.__wired = true;

        const host = document.querySelector('.group3-container .summary-container');
        if (!host) return;

        host.addEventListener('click', (e) => {
            const card = e.target?.closest?.('.summary-card');
            if (!card) return;

            const statusVal = card.dataset.qualityStatus;
            if (!statusVal) return;

            const activePage = localStorage.getItem('parts.activePage') || '';
            if (activePage !== 'quality_history') return;

            const alreadyActive = card.classList.contains('active');

            host.querySelectorAll('.summary-card').forEach(el => {
                el.classList.remove('active');
            });

            if (alreadyActive) {
                renderTable(state.items);
                return;
            }

            card.classList.add('active');

            const filtered = applyFiltersOnly(state.items).filter(it => {
                const status = String(it?.Process_status || '처리 필요').trim();
                return status === statusVal;
            });

            finalizeQualityTable(filtered);
        });
    }

    function buildUI(container) {
        container.innerHTML = `
            <div class="table-wrap">
                <table class="quality-history-table display compact nowrap" style="width:100%">
                    <colgroup>
                        <col style="width:90px">
                        <col style="width:150px">
                        <col style="width:50px">
                        <col style="width:50px">
                        <col style="width:160px">
                        <col style="width:70px">
                        <col style="width:70px">
                        <col style="width:380px">
                        <col style="width:200px">
                        <col style="width:80px">
                        <col style="width:280px">
                        <col style="width:80px">
                        <col style="width:80px">
                    </colgroup>
                    <thead>
                        <tr>
                            <th>등록일</th>
                            <th>발생일시</th>
                            <th>설비</th>
                            <th>모델</th>
                            <th>LOT ID</th>
                            <th>구분</th>
                            <th>대분류</th>
                            <th>현황</th>
                            <th>비고</th>
                            <th>작성자</th>
                            <th>처리내용</th>
                            <th>처리자</th>
                            <th>처리여부</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        `;
    }

    function normalizeRow(it) {
        return {
            WriteTime: it.WriteTime ?? '',
            WorkTime: it.WorkTime ?? '',
            EQ_Name: it.EQ_Name ?? '',
            Model: it.Model ?? '',
            Lot_id: it.Lot_id ?? '',
            Type: it.Type ?? '',
            Category: it.Category ?? '',
            Detail: it.Detail ?? '',
            Comment: it.Comment ?? '',
            Write_Name: it.Write_Name ?? '',
            Feedback: it.Feedback ?? '',
            Work_Name: it.Work_Name ?? '',
            Process_status: it.Process_status ?? '처리 필요',
            _rowKey: makeRowKey(it)
        };
    }

    function finalizeQualityTable(rowsData) {
        const $table = $(state.container).find('.quality-history-table');
        const dtData = (rowsData || []).map(normalizeRow);

        if ($.fn.DataTable.isDataTable($table)) {
            qualityDT.clear().rows.add(dtData).draw(false);
            return;
        }

        qualityDT = $table.DataTable({
            data: dtData,
            columns: [
                { 
                    data: 'WriteTime',
                    className: 'dt-center',
                    render: function(data) {
                        if (!data) return '';
                        return String(data).substring(0,10)
                    }
                },
                { 
                    data: 'WorkTime',
                    className: 'dt-center'
                },
                { 
                    data: 'EQ_Name',
                    className: 'dt-center'
                 },
                { 
                    data: 'Model',
                    className: 'dt-center'
                },
                { 
                    data: 'Lot_id',
                    className: 'dt-center'
                },
                { 
                    data: 'Type',
                    className: 'dt-center'
                },
                { 
                    data: 'Category',
                    className: 'dt-center'
                },
                {
                    data: 'Detail',
                    createdCell: function(td) {
                        td.style.textAlign = 'left';
                        td.style.paddingLeft = '10px';
                    }
                },
                {
                    data: 'Comment',
                    createdCell: function(td) {
                        td.style.textAlign = 'left';
                        td.style.paddingLeft = '10px';
                    }
                },
                { 
                    data: 'Write_Name',
                    className: 'dt-center'
                },
                {
                    data: 'Feedback',
                    createdCell: function(td) {
                        td.style.textAlign = 'left';
                        td.style.paddingLeft = '10px';
                    }
                },
                { 
                    data: 'Work_Name',
                    className: 'dt-center' 
                },
                {
                    data: 'Process_status',
                    className: 'dt-center',
                    render: function(data) {
                        return data || '처리 필요';
                    }
                }

            ],
            scrollY: '625px',
            scrollX: true,
            scrollCollapse: true,
            paging: false,
            searching: true,
            fixedHeader: true,
            orderCellsTop: true,
            info: false,
            dom: 't',
            order: [[1, 'desc'], [0, 'desc']],
            createdRow: (row, data) => {
                Array.from(row.cells).forEach(td =>
                    td.setAttribute('data-rowkey', data._rowKey)
                );

                const status = String(
                    data.Process_status || '처리 필요'
                ).trim();

                if (status === '완료') {
                    row.classList.add('row-done');
                } else {
                    row.classList.add('row-review');
                }
            }
        });

        bindDTSelectable($table, {
            selectedSet: selectedQualityKeys,
            getKey: (tr) => tr.querySelector('td')?.getAttribute('data-rowkey'),
            restoreOnDraw: false
        });

        wireOutsideDeselect($table, {
            selectedSet: selectedQualityKeys,
            preserveSelectors: [
                'button', '.edit-group-btn',
                '#quality-history-add-btn',
                '#quality-history-edit-add-btn',
                '#quality-history-edit-delete-btn',
                '#quality-history-edit-btn',
                '#quality-history-edit-cancel-btn',
                '.dataTables_length',
                '.dataTables_filter',
                '.dataTables_paginate',
                '.dataTables_info',
                '.dtfh-floatingparent',
                '.dtfh-floatingheader'
            ]
        });
    }

    function renderTable(items) {
        if (!state.container) return;

        const filtered = applyFiltersOnly(items);
        finalizeQualityTable(filtered);

        const sum = getQualitySummaryCounts(filtered);
        renderQualitySummaryCards(sum);
    }

    async function fetchItems() {
        if (state.isFetching) return;
        state.isFetching = true;

        try {
            const r = await fetch('/quality_history');
            const json = await r.json();

            state.items = Array.isArray(json?.items) ? json.items : [];

            document.dispatchEvent(new CustomEvent('parts:data-loaded', {
                detail: { which: 'quality_history', items: state.items }
            }));

            if (state.container) renderTable(state.items);
        } catch (e) {
            console.error('[QualityHistory.fetchItems] failed:', e);
            alert('품질 이력 조회 중 오류가 발생했습니다.');
        } finally {
            state.isFetching = false;
        }
    }

    async function init(container) {
        if (state.inited) return;

        state.container = container;
        buildUI(container);
        enforceQualityEditButtonVisibility();

        await checkClientAuthority();
        await fetchItems();

        wireEditEvents();
        refreshButtonsByState();
        wireQualitySummaryCards();

        state.inited = true;
    }

    const QUALITY_OPTIONS = {
        Type: ['PANEL', 'FILM', '기타'],
        Category: ['장변부 CURL', '장변부 역CURL', '단변부 CURL', '단변부 역CURL', 'TWIST CURL', 'ALIGN MARK이상', '협착', '이물', '기타'],
        Process_status: ['처리 필요', '완료']
    };

    function inputHTML(name, val = '', type = 'text') {
        const v = String(val ?? '');
        return `
            <input class="dt-input"
                name="${escapeHtml(name)}"
                type="${escapeHtml(type)}"
                value="${escapeHtml(v)}"
                style="width:100%;box-sizing:border-box;">
        `;
    }

    function selectHTML(name, options = [], selected = '') {
        const current = String(selected ?? '').trim();

        return `
            <select class="dt-input dt-select"
                name="${escapeHtml(name)}"
                style="width:100%;box-sizing:border-box;">
                <option value=""></option>
                ${options.map(opt => {
                    const v = String(opt);
                    const sel = v === current ? 'selected' : '';
                    return `<option value="${escapeHtml(v)}" ${sel}>${escapeHtml(v)}</option>`;
                }).join('')}
            </select>
        `;
    }

    function textareaHTML(name, val = '') {
        const v = String(val ?? '');
        return `
            <textarea class="dt-input"
                name="${escapeHtml(name)}"
                style="width:100%;box-sizing:border-box;">${escapeHtml(v)}</textarea>
        `;
    }

    function rowToInputs(rowNode) {
        const data = qualityDT.row(rowNode).data();
        const cells = rowNode.cells || rowNode.children;
        if (!cells || !data) return;

        cells[0].innerHTML = inputHTML('WriteTime', data.WriteTime);
        cells[1].innerHTML = inputHTML('WorkTime', data.WorkTime);
        cells[2].innerHTML = inputHTML('EQ_Name', data.EQ_Name);
        cells[3].innerHTML = inputHTML('Model', data.Model);
        cells[4].innerHTML = textareaHTML('Lot_id', data.Lot_id);
        cells[5].innerHTML = selectHTML('Type', QUALITY_OPTIONS.Type, data.Type);
        cells[6].innerHTML = selectHTML('Category', QUALITY_OPTIONS.Category, data.Category);
        cells[7].innerHTML = textareaHTML('Detail', data.Detail);
        cells[8].innerHTML = textareaHTML('Comment', data.Comment);
        cells[9].innerHTML = inputHTML('Write_Name', data.Write_Name);
        cells[10].innerHTML = textareaHTML('Feedback', data.Feedback);
        cells[11].innerHTML = inputHTML('Work_Name', data.Work_Name);
        cells[12].innerHTML = selectHTML('Process_status', QUALITY_OPTIONS.Process_status, data.Process_status || '처리 필요');

        rowNode.classList.add('editing-row');

        $(rowNode).find('input.dt-input[name="EQ_Name"]').on('input', function () {
            this.value = this.value.toUpperCase();
        });
    }

    function normalizeNameList(value = '') {
        return String(value || '')
            .trim()
            .replace(/\s*,\s*/g, ',')
            .replace(/\s+/g, ',')
            .replace(/,+/g, ',')
            .replace(/^,|,$/g, '');
    }

    function validateNameFormat(value = '') {
        const text = String(value || '').trim();
        if (!text) return true;

        const regex = /^[가-힣a-zA-Z0-9,]+$/;
        return regex.test(text);
    }

    function collectInputs(rowNode) {
        const obj = {};

        $(rowNode).find('input.dt-input, select.dt-input, textarea.dt-input').each(function () {
            const n = this.name;
            let v = this.value ?? '';
            v = v.trim();

            if (n === 'EQ_Name') v = v.toUpperCase();
            if (n === 'Write_Name' || n === 'Work_Name') v = normalizeNameList(v);

            obj[n] = v;
        });

        return obj;
    }

    function restoreRowFromData(rowNode, data) {
        qualityDT.row(rowNode).data(normalizeRow(data));
        qualityDT.draw(false);
    }

    function addDraftRow() {
        if (!qualityDT) return;

        const $tbody = $(state.container).find('.quality-history-table tbody');
        const hasDraft = $tbody.find('tr.draft-row').length > 0;

        if (hasDraft) {
            alert('이미 추가 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
            return;
        }

        const now = nowDateTimeStr();

        const draftData = {
            WriteTime: now,
            WorkTime: now,
            EQ_Name: '',
            Model: '',
            Lot_id: '',
            Type: '',
            Category: '',
            Detail: '',
            Comment: '',
            Write_Name: getCurrentUserName() || '',
            Feedback: '',
            Work_Name: '',
            Process_status: '처리 필요',
            _draft: true,
            _rowKey: `draft||${Date.now()}`
        };

        const rowApi = qualityDT.row.add(draftData);
        qualityDT.draw(false);

        const node = rowApi.node();
        node.classList.add('draft-row');
        node.classList.remove('selected');

        const cells = node.cells || node.children;

        cells[0].innerHTML = inputHTML('WriteTime', draftData.WriteTime);
        cells[1].innerHTML = inputHTML('WorkTime', draftData.WorkTime);
        cells[2].innerHTML = inputHTML('EQ_Name', draftData.EQ_Name);
        cells[3].innerHTML = inputHTML('Model', draftData.Model);
        cells[4].innerHTML = textareaHTML('Lot_id', draftData.Lot_id);
        cells[5].innerHTML = selectHTML('Type', QUALITY_OPTIONS.Type, draftData.Type);
        cells[6].innerHTML = selectHTML('Category', QUALITY_OPTIONS.Category, draftData.Category);
        cells[7].innerHTML = textareaHTML('Detail', draftData.Detail);
        cells[8].innerHTML = textareaHTML('Comment', draftData.Comment);
        cells[9].innerHTML = inputHTML('Write_Name', draftData.Write_Name);
        cells[10].innerHTML = textareaHTML('Feedback', draftData.Feedback);
        cells[11].innerHTML = inputHTML('Work_Name', draftData.Work_Name);
        cells[12].innerHTML = selectHTML('Process_status', QUALITY_OPTIONS.Process_status, draftData.Process_status);

        $tbody.prepend(node);
        setButtonsForMode('draft');

        $(node).find('input.dt-input[name="EQ_Name"]').on('input', function () {
            this.value = this.value.toUpperCase();
        });
    }

    function collectDraftRows() {
        const rows = [];
        const $table = $(state.container).find('.quality-history-table');

        $table.find('tbody tr.draft-row').each(function () {
            const obj = {};

            $(this).find('input.dt-input, select.dt-input, textarea.dt-input').each(function () {
                const name = this.name;
                let val = this.value?.trim() ?? '';

                if (name === 'EQ_Name') val = val.toUpperCase();
                if (name === 'Write_Name' || name === 'Work_Name') val = normalizeNameList(val);

                obj[name] = val;
            });

            if (!obj.WriteTime) obj.WriteTime = nowDateTimeStr();

            rows.push(obj);
        });

        return rows;
    }

    function removeAllDraftRows() {
        if (!qualityDT) return;

        const $table = $(state.container).find('.quality-history-table');

        $table.find('tbody tr.draft-row').each(function () {
            qualityDT.row(this).remove();
        });

        qualityDT.draw(false);
        refreshButtonsByState();
    }

    const FIELD_LABELS = {
        WriteTime: '등록일',
        WorkTime: '발생일시',
        EQ_Name: '설비',
        Model: '모델',
        Lot_id: 'LOT ID',
        Type: '구분',
        Category: '대분류',
        Detail: '현황',
        Comment: '비고',
        Write_Name: '작성자',
        Feedback: '처리내용',
        Work_Name: '처리자',
        Process_status: '처리여부'
    };

    function validateRow(r, rowNode) {
        const required = [
            'WorkTime',
            'EQ_Name',
            'Model',
            'Lot_id',
            'Type',
            'Category',
            'Detail',
            'Write_Name',
            'Process_status'
        ];

        const missing = required.filter(k => !String(r[k] ?? '').trim());

        if (!validateNameFormat(r.Write_Name)) {
            if (rowNode) {
                const el = rowNode.querySelector('[name="Write_Name"]');
                if (el) el.focus();
            }

            return `작성자의 이름 구분은 콤마(,)로만 입력이 가능합니다.
(예 : 김철민,변창성)`;
        }

        if (!validateNameFormat(r.Work_Name)) {
            if (rowNode) {
                const el = rowNode.querySelector('[name="Work_Name"]');
                if (el) el.focus();
            }

            return `처리자의 이름 구분은 콤마(,)로만 입력이 가능합니다.
(예 : 김철민,변창성)`;
        }

        if (missing.length) {
            if (rowNode) {
                const el = rowNode.querySelector(`[name="${missing[0]}"]`);
                if (el) el.focus();
            }

            const readable = missing.map(k => FIELD_LABELS[k] || k);
            return `필수 항목 누락: ${readable.join(', ')}`;
        }

        return null;
    }

    function makeUpdateWhere(row) {
        return {
            WriteTime: row.WriteTime,
            WorkTime: row.WorkTime,
            EQ_Name: row.EQ_Name,
            Lot_id: row.Lot_id,
            Type: row.Type,
            Write_Name: row.Write_Name
        };
    }

    function wireEditEvents() {
        if (wireEditEvents.__wired) return;
        wireEditEvents.__wired = true;

        const qualityWrap = document.querySelector('.quality-history-wrapper');
        const editBtn = qualityWrap?.querySelector('#quality-history-edit-btn');

        if (editBtn && !editBtn.__wired) {
            editBtn.__wired = true;

            editBtn.addEventListener('click', () => {
                if (!qualityDT) return;

                if (isEditingNow()) {
                    alert('이미 편집 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                const nodes = qualityDT.rows('.selected').nodes().toArray();

                if (nodes.length === 0) {
                    alert('수정할 행을 1건 이상 선택해 주세요.');
                    return;
                }

                if (nodes.some(n => n.classList.contains('draft-row'))) {
                    alert('행추가 중인 드래프트가 포함되어 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                    return;
                }

                const selectedRows = qualityDT.rows('.selected').data().toArray();
                const unauthorized = getUnauthorizedRows(selectedRows);

                if (unauthorized.length > 0) {
                    alert('작성자 또는 처리자 본인의 이력만 수정할 수 있습니다.');
                    return;
                }

                editCtxs = [];

                for (const rowNode of nodes) {
                    const rowApi = qualityDT.row(rowNode);
                    const before = { ...rowApi.data() };
                    const beforeKey = makeUpdateWhere(before);

                    editCtxs.push({ rowNode, rowApi, before, beforeKey });
                    rowToInputs(rowNode);
                }

                setButtonsForMode('editing');
            });
        }

        document.addEventListener('parts-edit:add-row', () => {
            const wrap = document.querySelector('.quality-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }

            addDraftRow();
        });

        document.addEventListener('parts-edit:save', async () => {
            const wrap = document.querySelector('.quality-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            try {
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
                                WriteTime: merged.WriteTime,
                                WorkTime: merged.WorkTime,
                                EQ_Name: merged.EQ_Name,
                                Model: merged.Model,
                                Lot_id: merged.Lot_id,
                                Type: merged.Type,
                                Category: merged.Category,
                                Detail: merged.Detail,
                                Comment: merged.Comment,
                                Write_Name: merged.Write_Name,
                                Feedback: merged.Feedback,
                                Work_Name: merged.Work_Name,
                                Process_status: merged.Process_status || '처리 필요'
                            },
                            _before: before,
                            _rowNode: rowNode
                        });
                    }

                    for (const u of updates) {
                        const resp = await fetch('/quality_history/update', {
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
                        const resp = await fetch('/quality_history', {
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
            const wrap = document.querySelector('.quality-history-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            if (!qualityDT) return;

            const $tbody = $(state.container).find('.quality-history-table tbody');
            const $draftSelected = $tbody.find('tr.draft-row.selected');

            if ($draftSelected.length) {
                $draftSelected.each(function () {
                    qualityDT.row(this).remove();
                });
                qualityDT.draw(false);
            }

            const selected = qualityDT.rows('.selected').data().toArray();

            if (!selected.length) {
                refreshButtonsByState();
                return alert('삭제할 행을 선택해 주세요.');
            }

            const unauthorized = getUnauthorizedRows(selected);

            if (unauthorized.length > 0) {
                alert('작성자 또는 처리자 본인의 이력만 삭제할 수 있습니다.');
                return;
            }

            if (isEditingNow()) {
                alert('현재 수정 중인 행이 있습니다. 저장 또는 취소 후 다시 시도하세요.');
                return;
            }

            const rows = selected.map(makeUpdateWhere);

            if (!confirm(`선택한 ${rows.length}건을 삭제할까요?`)) return;

            try {
                const resp = await fetch('/quality_history/delete', {
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
            const wrap = document.querySelector('.quality-history-wrapper');
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
            selectedQualityKeys.clear();

            const $t = $(state.container).find('.quality-history-table');

            if ($t.length) {
                $t.find('tbody tr.selected').removeClass('selected');
            }
        } catch (e) {
            console.warn('[QualityHistory.clearSelection]', e);
        }
    }

    return {
        init,
        fetch: () => fetchItems(),
        searchWithFilters,
        clearSelection
    };
})();

window.QualityHistory = QualityHistory;