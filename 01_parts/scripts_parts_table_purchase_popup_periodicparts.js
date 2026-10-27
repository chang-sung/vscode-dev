async function openPeriodicPartsWindow() {
    try {
        const res = await fetch('/parts/purchase_periodic_parts');
        if (!res.ok) throw new Error('서버 응답 오류');
        const items = await res.json();

        if (window.__periodicPartsWin && !window.__periodicPartsWin.closed) {
            window.__periodicPartsWin.focus();
            return;
        }

        const w = 1200, h = 450;
        const left = Math.max(0, (window.screenX || window.screenLeft || 0) + 80);
        const top  = Math.max(0, (window.screenY || window.screenTop || 0) + 80);

        const popup = window.open(
            '',
            'purchase_periodic_parts_popup',
            `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`
        );

        if (!popup) {
            alert('팝업이 차단되었습니다. 브라우저 팝업 허용 후 다시 시도해주세요.');
            return;
        }

        window.__periodicPartsWin = popup;

        const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
        );

        const rowsHtml = (items || []).map(it => {
            const price = (it.Price != null && it.Price !== '' && !isNaN(Number(it.Price)))
                ? Number(it.Price).toLocaleString()
                : '';

            return `
                <tr>
                    <td>${esc(it.Q_Code)}</td>
                    <td>${esc(it.Parts_Div)}</td>
                    <td>${esc(it.P_N)}</td>
                    <td style="text-align:right;">${esc(price)}</td>
                    <td style="text-align:right;">${esc(it.Avg_LeadTime)}</td>
                    <td>${esc(it.Memo)}</td>
                </tr>
            `;
        }).join('');

        popup.document.write(`
            <!DOCTYPE html>
            <html lang="ko">
            <head>
            <meta charset="utf-8" />
            <title>정기 구매 파츠 관리</title>
            <link rel="stylesheet" href="/css/100_utils/styles.css">
            <link rel="stylesheet" href="/css/01_parts/styles_parts.css">

            <script src="/js/100_utils/jquery.min.js"><\/script>
            <script src="/js/100_utils/dataTables.min.js"><\/script>
            <script src="/js/100_utils/dataTable.fixedHeader.min.js"><\/script>
            <script src="/js/100_utils/dataTable.fixedColumns.min.js"><\/script>
            <script src="/js/100_utils/Sortable.min.js"><\/script>

            <style>
            body {
                font-family: "맑은 고딕", system-ui, sans-serif;
                margin: 16px;
                background-color: #f7f7f7;
            }
            .periodic-popup-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
            }
            .periodic-popup-wrapper {
                background: #fff;
                box-shadow: 0 8px 20px rgba(0,0,0,.08);
                padding: 15px;
                border-radius: 5px;
            }
            .periodic-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                background: #fff;
                table-layout: fixed;
            }
            .periodic-table th,
            .periodic-table td {
                border: 1px solid #ddd;
                padding: 6px 8px;
                white-space: nowrap;
            }
            .periodic-table thead tr {
                background-color: #f0f4ff;
            }
            .periodic-table thead th {
                text-align: center !important;
            }
            .periodic-table tbody tr.selected {
                background-color:#87cefa !important;
                color:#fff;
            }
            .periodic-table tbody tr:hover {
                background-color:#f0f8ff;
                cursor:pointer;
            }
            .periodic-input {
                width: 100%;
                box-sizing: border-box;
                font-size: 12px;
            }
            .periodic-table tbody td:nth-child(4),
            .periodic-table tbody td:nth-child(5) {
                text-align: right !important;
            }
            .periodic-table tbody td:nth-child(6) {
                position: relative;
                padding-right: 24px;
            }
            .periodic-row-handle {
                position: absolute;
                right: 6px;
                top: 50%;
                transform: translateY(-50%);
                width: 16px;
                height: 16px;
                cursor: grab;
                opacity: .9;
            }
            .periodic-row-handle:hover {
                cursor: grabbing;
                opacity: 1;
            }
            .periodic-row-handle svg {
                pointer-events: none;
            }
            </style>
            </head>

            <body>
            <div class="periodic-popup-title">정기 구매 파츠 관리</div>

            <div class="periodic-popup-wrapper">
                <div class="edit-btn-wrapper" id="periodic-edit-tools" style="display:none;">
                    <button id="periodic-main-btn" class="edit-group-btn">파츠 편집</button>

                    <div class="edit-btn-container" style="display:none">
                        <button id="periodic-add-btn" class="edit-group-btn">행추가</button>
                        <button id="periodic-delete-btn" class="edit-group-btn">행삭제</button>
                        <button id="periodic-edit-btn" class="edit-group-btn">수정</button>
                        <button id="periodic-cancel-btn" class="edit-group-btn">취소</button>
                    </div>
                </div>

                <table class="periodic-table display compact">
                    <thead>
                        <tr>
                            <th style="width:100px;">코드</th>
                            <th style="width:200px;">파츠분류</th>
                            <th style="width:200px;">업체P/N</th>
                            <th style="width:100px;">단가</th>
                            <th style="width:100px;">평균입고일</th>
                            <th style="width:300px;">비고</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rowsHtml || '<tr><td colspan="6" style="text-align:center;">등록된 주기교체 파츠가 없습니다.</td></tr>'}
                    </tbody>
                </table>
            </div>

            <script>
            (function () {
                const jq = window.jQuery || window.$;
                if (!jq) return;

                let periodicDT = null;
                let editMode = false;
                let backupRows = [];
                let periodicSortable = null;
                let $tbody = null;

                const HANDLE_SVG =
                    '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
                        '<path d="M7 10l5-5 5 5M7 14l5 5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" />' +
                    '</svg>';

                function enablePeriodicDnD() {
                    if (!window.Sortable || periodicSortable || !$tbody || !$tbody.length) return;

                    const tbodyEl = $tbody[0];

                    jq(tbodyEl).find('tr').each(function () {
                        const tds = this.querySelectorAll('td');
                        if (!tds || tds.length !== 6) return;

                        const td = tds[5];
                        if (td.querySelector('.periodic-row-handle')) return;

                        const text = td.textContent.trim();
                        td.innerHTML =
                            '<span class="memo-text">' + text + '</span>' +
                            '<span class="periodic-row-handle" title="드래그해서 순서 변경">' +
                                HANDLE_SVG +
                            '</span>';
                    });

                    periodicSortable = Sortable.create(tbodyEl, {
                        handle: '.periodic-row-handle',
                        animation: 150
                    });
                }

                function disablePeriodicDnD() {
                    if (periodicSortable) {
                        periodicSortable.destroy();
                        periodicSortable = null;
                    }

                    if (!$tbody || !$tbody.length) return;

                    $tbody.find('tr').each(function () {
                        const tds = this.querySelectorAll('td');
                        if (!tds || tds.length !== 6) return;

                        const memoSpan = tds[5].querySelector('.memo-text');
                        if (memoSpan) {
                            tds[5].textContent = memoSpan.textContent.trim();
                        }
                    });
                }

                jq(function () {
                    const $table = jq('.periodic-table');
                    if (!$table.length) return;

                    $tbody = $table.find('tbody');

                    const $btnMain   = jq('#periodic-main-btn');
                    const $btnAdd    = jq('#periodic-add-btn');
                    const $btnDel    = jq('#periodic-delete-btn');
                    const $btnEdit   = jq('#periodic-edit-btn');
                    const $btnCancel = jq('#periodic-cancel-btn');
                    const $editBox   = jq('.edit-btn-container');
                    const $editWrapper = $btnMain.closest('.edit-btn-wrapper');

                    if (jq.fn.DataTable) {
                        periodicDT = $table.DataTable({
                            paging: false,
                            searching: false,
                            info: false,
                            ordering: true,
                            scrollX: true,
                            scrollY: '260px',
                            scrollCollapse: true,
                            autoWidth: false,

                            columnDefs: [
                                { targets: 0, width: '100px' },
                                { targets: 1, width: '200px' },
                                { targets: 2, width: '200px' },
                                { targets: 3, width: '100px' },
                                { targets: 4, width: '100px' },
                                { targets: 5, width: '300px' },
                            ]
                        });
                    }

                    let lastSelectedIndex = null;

                    $tbody.off('click.periodicRowSelect').on('click.periodicRowSelect', 'tr', function (e) {
                        if (jq(e.target).is('input,textarea,select,button,a')) return;

                        const $rows = $tbody.find('tr');
                        const $row = jq(this);
                        const idx = $rows.index($row);

                        const isCtrl = !!(e.ctrlKey || e.metaKey);
                        const isShift = !!e.shiftKey;
                        const wasSelected = $row.hasClass('selected');

                        if (isShift && window.getSelection) {
                            const sel = window.getSelection();
                            if (sel && sel.removeAllRanges) sel.removeAllRanges();
                        }

                        if (isShift && lastSelectedIndex != null) {
                            const start = Math.min(lastSelectedIndex, idx);
                            const end = Math.max(lastSelectedIndex, idx);

                            if (!isCtrl) $rows.removeClass('selected');

                            for (let i = start; i <= end; i++) {
                                $rows.eq(i).addClass('selected');
                            }

                            lastSelectedIndex = idx;
                            return;
                        }

                        if (isCtrl) {
                            $row.toggleClass('selected');
                            lastSelectedIndex = idx;
                            return;
                        }

                        const selectedCount = $rows.filter('.selected').length;

                        if (selectedCount === 1 && wasSelected) {
                            $row.removeClass('selected');
                            lastSelectedIndex = null;
                            return;
                        }

                        $rows.removeClass('selected');
                        $row.addClass('selected');
                        lastSelectedIndex = idx;
                    });

                    (async () => {
                        let canEdit = false;

                        try {
                            const res = await fetch('/check-client-authority-parts');
                            if (res.ok) {
                                const data = await res.json();
                                const role = (data.authority || '').trim();

                                if (role === '관리자' || role === '구매담당자') {
                                    canEdit = true;
                                }
                            }
                        } catch (e) {
                            console.warn('[Periodic popup] authority check failed:', e);
                        }

                        if (!canEdit) return;

                        $editWrapper.css('display', 'flex');

                        $btnMain.on('click', function () {
                            if (!editMode) {
                                editMode = true;
                                $editBox.css('display', 'flex');
                                $btnMain.text('저장');

                                backupRows = [];
                                $tbody.find('tr').each(function () {
                                    const tds = this.querySelectorAll('td');
                                    if (!tds || tds.length !== 6) return;

                                    backupRows.push([
                                        tds[0].textContent.trim(),
                                        tds[1].textContent.trim(),
                                        tds[2].textContent.trim(),
                                        tds[3].textContent.trim().replace(/,/g, ''),
                                        tds[4].textContent.trim(),
                                        tds[5].textContent.trim()
                                    ]);
                                });

                                enablePeriodicDnD();
                                return;
                            }

                            const rowsToSave = [];

                            $tbody.find('tr').each(function () {
                                const tds = this.querySelectorAll('td');
                                if (!tds || tds.length !== 6) return;

                                const vals = [];

                                for (let i = 0; i < 6; i++) {
                                    const td = tds[i];
                                    const input = td.querySelector('input.periodic-input');
                                    let v;

                                    if (input) {
                                        v = input.value;
                                    } else if (i === 5) {
                                        const span = td.querySelector('.memo-text');
                                        v = span ? span.textContent.trim() : td.textContent.trim();
                                    } else {
                                        v = td.textContent.trim();
                                    }

                                    if (i === 3) v = v.replace(/,/g, '');
                                    vals.push(v);
                                }

                                rowsToSave.push({
                                    Q_Code: vals[0],
                                    Parts_Div: vals[1],
                                    P_N: vals[2],
                                    Price: vals[3],
                                    Avg_LeadTime: vals[4],
                                    Memo: vals[5]
                                });
                            });

                            fetch('/parts/purchase_periodic_parts/save', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ rows: rowsToSave })
                            })
                            .then(res => {
                                if (!res.ok) throw new Error('서버 응답 오류');
                                return res.json();
                            })
                            .then(() => {
                                $tbody.find('tr').each(function () {
                                    const tds = this.querySelectorAll('td');
                                    if (!tds || tds.length !== 6) return;

                                    for (let i = 0; i < 6; i++) {
                                        const td = tds[i];
                                        const input = td.querySelector('input.periodic-input');
                                        if (!input) continue;

                                        let v = input.value;

                                        if (i === 3 && v) {
                                            const num = Number(v.replace(/,/g, ''));
                                            td.innerHTML = '<span style="float:right;">' +
                                                (isNaN(num) ? '' : num.toLocaleString()) +
                                                '</span>';
                                        } else {
                                            td.textContent = v;
                                        }
                                    }
                                });

                                disablePeriodicDnD();

                                editMode = false;
                                $editBox.hide();
                                $btnMain.text('파츠 편집');
                                $tbody.find('tr').removeClass('selected');

                                alert('주기교체 파츠 리스트가 저장되었습니다.');
                            })
                            .catch(err => {
                                console.error('[Periodic popup] save error:', err);
                                alert('저장 중 오류가 발생했습니다.');
                            });
                        });

                        $btnEdit.on('click', function () {
                            if (!editMode) return;

                            const $selected = $tbody.find('tr.selected');
                            if (!$selected.length) {
                                alert('수정할 행을 선택해 주세요.');
                                return;
                            }

                            $selected.each(function () {
                                const tds = this.querySelectorAll('td');
                                if (!tds || tds.length !== 6) return;

                                for (let i = 0; i < 6; i++) {
                                    const td = tds[i];
                                    if (td.querySelector('input.periodic-input')) continue;

                                    let text = td.textContent.trim();
                                    if (i === 3) text = text.replace(/,/g, '');

                                    const type = (i === 3 || i === 4) ? 'number' : 'text';

                                    td.innerHTML =
                                        '<input class="periodic-input" type="' + type + '" value="' + text + '">';
                                }
                            });
                        });

                        $btnAdd.on('click', function () {
                            if (!editMode) return;

                            const tr = document.createElement('tr');

                            for (let i = 0; i < 6; i++) {
                                const td = document.createElement('td');
                                const type = (i === 3 || i === 4) ? 'number' : 'text';
                                td.innerHTML = '<input class="periodic-input" type="' + type + '">';
                                tr.appendChild(td);
                            }

                            if (periodicDT) {
                                periodicDT.row.add(tr).draw(false);
                            } else {
                                $tbody.append(tr);
                            }

                        });

                        $btnDel.on('click', function () {
                            if (!editMode) return;

                            const $selected = $tbody.find('tr.selected');

                            if (!$selected.length) {
                                alert('삭제할 행을 선택해 주세요.');
                                return;
                            }

                            if (!window.confirm($selected.length + '건을 삭제할까요?')) return;

                            if (periodicDT) {
                                $selected.each(function () {
                                    periodicDT.row(this).remove();
                                });
                                periodicDT.draw(false);
                            } else {
                                $selected.remove();
                            }
                        });

                        $btnCancel.on('click', function () {
                            if (!editMode) return;
                            if (!window.confirm('편집 내용을 모두 취소할까요?')) return;

                            disablePeriodicDnD();

                            if (periodicDT) {
                                periodicDT.clear();

                                backupRows.forEach(function (r) {
                                    periodicDT.row.add([
                                        r[0],
                                        r[1],
                                        r[2],
                                        r[3] ? Number(r[3]).toLocaleString() : '',
                                        r[4],
                                        r[5]
                                    ]);
                                });

                                periodicDT.draw(false);
                            } else {
                                $tbody.empty();

                                backupRows.forEach(function (r) {
                                    const tr = document.createElement('tr');

                                    tr.innerHTML =
                                        '<td>' + r[0] + '</td>' +
                                        '<td>' + r[1] + '</td>' +
                                        '<td>' + r[2] + '</td>' +
                                        '<td style="text-align:right;">' + (r[3] ? Number(r[3]).toLocaleString() : '') + '</td>' +
                                        '<td style="text-align:right;">' + r[4] + '</td>' +
                                        '<td>' + r[5] + '</td>';

                                    $tbody.append(tr);
                                });
                            }

                            editMode = false;
                            $editBox.hide();
                            $btnMain.text('파츠 편집');
                            $tbody.find('tr').removeClass('selected');
                        });

                    })();
                });
            })();
            <\/script>
            </body>
            </html>
        `);

        popup.document.close();
        popup.focus();

    } catch (e) {
        console.error('[openPeriodicPartsWindow] error:', e);
        alert('주기교체 파츠 리스트를 불러오는 중 오류가 발생했습니다.');
    }
}

$(function () {
    $('#periodic-parts-inform-btn').on('click', function () {
        openPeriodicPartsWindow();
    });
});
