async function openNaPartsWindow() {
    try {
        // 1) NA 파츠 리스트 조회
        const res = await fetch('/parts/purchase_na_parts');
        if (!res.ok) throw new Error('서버 응답 오류');
        const items = await res.json();

        // 2) 팝업 재사용 처리
        if (window.__naPartsWin && !window.__naPartsWin.closed) {
            window.__naPartsWin.focus();
            return;
        }

        const w = 1180, h = 600;
        const left = Math.max(0, (window.screenX || window.screenLeft || 0) + 80);
        const top  = Math.max(0, (window.screenY || window.screenTop || 0) + 80);

        const features = `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`;
        const popup = window.open('', 'purchase_na_parts_popup', features);

        if (!popup) {
            alert('팝업이 차단되었습니다. 브라우저 팝업 허용 후 다시 시도해주세요.');
            return;
        }

        window.__naPartsWin = popup;

        // 3) 테이블 행 HTML 생성
        const rowsHtml = (items || []).map(it => {
            const eqName      = it.EQ_Name        ?? '';
            const naQCode     = it.NA_Q_Code      ?? '';
            const naPartsName = it.NA_Parts_Name  ?? '';
            const naPn        = it.NA_P_N         ?? '';
            const naPriceRaw  = it.NA_Price;
            const qCode       = it.Q_Code         ?? '';
            const parts       = it.Parts_Name     ?? '';
            const pn          = it.P_N            ?? '';
            const priceRaw    = it.Price;
            const memo        = it.Memo           ?? '';

            const naPrice = (naPriceRaw != null && naPriceRaw !== '' && !isNaN(Number(naPriceRaw)))
                ? Number(naPriceRaw).toLocaleString()
                : '';
            const price   = (priceRaw != null && priceRaw !== '' && !isNaN(Number(priceRaw)))
                ? Number(priceRaw).toLocaleString()
                : '';

            const esc = (s) => String(s).replace(/[&<>"']/g, c =>
                ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
            ));

            return `
                <tr>
                    <td>${esc(eqName)}</td>
                    <td>${esc(naQCode)}</td>
                    <td>${esc(naPartsName)}</td>
                    <td>${esc(naPn)}</td>
                    <td style="text-align:right;">${esc(naPrice)}</td>
                    <td>${esc(qCode)}</td>
                    <td>${esc(parts)}</td>
                    <td>${esc(pn)}</td>
                    <td style="text-align:right;">${esc(price)}</td>
                    <td>${esc(memo)}</td>
                </tr>
            `;
        }).join('');

        // 4) 팝업 HTML 작성 (★ script 닫는 태그는 전부 <\/script> 로!)
        popup.document.write(`
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8" />
    <title>구매 불가 파츠 관리</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="stylesheet" href="/css/100_utils/styles.css">
    <link rel="stylesheet" href="/css/01_parts/styles_parts.css">

    <!-- JS 라이브러리 -->
    <script src="/js/100_utils/jquery.min.js"><\/script>
    <script src="/js/100_utils/dataTables.min.js"><\/script>
    <script src="/js/100_utils/dataTable.fixedHeader.min.js"><\/script>
    <script src="/js/100_utils/dataTable.fixedColumns.min.js"><\/script>
    <script src="/js/100_utils/Sortable.min.js"><\/script>

    <style>
        body {
            font-family: "맑은 고딕", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 16px;
            background-color: #f7f7f7;
        }
        .na-popup-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .na-popup-wrapper {
            background: #fff;
            box-shadow: 0 8px 20px rgba(0,0,0,.08);
            padding: 15px;
            border-radius: 5px;
        }

        .na-table thead th {
            text-align: center !important;
        }

        table.na-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: #fff;
        }
        table.na-table thead tr {
            background-color: #f0f4ff;
        }
        table.na-table th,
        table.na-table td {
            border: 1px solid #ddd;
            padding: 6px 8px;
            text-align: left;
            white-space: nowrap;
        }
        table.na-table th {
            font-weight: 600;
        }

        
        table.na-table tbody tr.selected {
            background-color:#87cefa !important; 
            color:#fff; 
        }

        table.na-table tbody tr:hover {
            background-color:#f0f8ff; 
            cursor:pointer; 
        }

        .na-input {
            width: 100%;
            /* 필요하면 padding만 살짝 줘도 됨 */
            box-sizing: border-box;
            margin-left: 0;
            margin-right: 0;
            font-size: 12px;
        }

        .na-table thead th:nth-child(1),
        .na-table tbody td:nth-child(1) {
            text-align: center !important;
        }
            
        table.na-table tbody tr:nth-child(even) {
            background-color: #fafafa;
        }

        /* 단가 컬럼: 5번째(NA_Price), 9번째(Price) 항상 오른쪽 정렬 */
        .na-table tbody td:nth-child(5),
        .na-table tbody td:nth-child(9) {
            text-align: right !important;
        }

        .na-table tbody td:nth-child(10) {
            position: relative;  
            padding-right: 24px;  /* 핸들 공간 확보 */
        }

        /* 드래그 핸들 (사유 컬럼) */
        .na-row-handle {
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            width: 16px;
            height: 16px;
            cursor: grab;
            opacity: .9;
        }

        .na-row-handle:hover {
            cursor: grabbing;
            opacity: 1;
        }
        .na-row-handle svg {
            pointer-events: none; /* 드래그시 SVG 안쪽에서 이벤트 막힘 방지 */
        }

    </style>

</head>
<body class="parts-popup">
    <div class="na-popup-title">구매 불가 파츠 관리</div>
    <div class="na-popup-wrapper">
        <div class="edit-btn-wrapper" id="na-edit-tools" style="display:none;">
            <button id="add-parts-btn" class="edit-group-btn">파츠 편집</button>
            
            <div class="edit-btn-container" style="display:none">
                <button id="edit-add-btn-2" class="edit-group-btn">행추가</button>
                <button id="edit-delete-btn-2" class="edit-group-btn">행삭제</button>
                <button id="edit-btn-2" class="edit-group-btn">수정</button>
                <button id="edit-cancel-btn-2" class="edit-group-btn">취소</button>
            </div>
        </div>

        <table class="na-table display compact">
            <thead>
                <!-- JS에서 동적으로 2단 헤더 생성 -->
            </thead>
            <tbody>
                ${rowsHtml || '<tr><td colspan="10" style="text-align:center;">등록된 구매 제한 품목이 없습니다.</td></tr>'}
            </tbody>
        </table>
    </div>

    <script>
        // ===========================
        // NA 구매제한 팝업용 편집 버튼 바인딩
        // ===========================
        (function() {
            const jq = window.jQuery || window.$;
            if (!jq) {
                console.warn('[NA popup] jQuery not found. DataTables 편집 기능 비활성화');
                return;
            }

            let naDT = null;
            let editMode = false;
            let backupRows = [];
            let naSortable = null; // ✅ NA 팝업용 DnD 인스턴스
            let $tbody = null;     // ✅ 바깥 스코프에 선언

            const HANDLE_SVG =
                '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
                    '<path d="M7 10l5-5 5 5M7 14l5 5 5-5"' +
                        ' fill="none"' +
                        ' stroke="currentColor"' +
                        ' stroke-width="2"' +
                        ' stroke-linecap="round" />' +
                '</svg>';

            // ✅ 사유 컬럼(마지막 컬럼)에 드래그 핸들 붙이고 Sortable 활성화
            function enableNaDnD() {
                if (!window.Sortable) {
                    console.warn('[NA popup] SortableJS not loaded');
                    return;
                }
                if (naSortable) return;
                if (!$tbody || !$tbody.length) return;   // ✅ 방어 코드

                const tbodyEl = $tbody[0];

                jq(tbodyEl).find('tr').each(function () {
                    const tds = this.querySelectorAll('td');
                    if (!tds || !tds.length) return;
                    const td = tds[tds.length - 1]; // 마지막 컬럼 = 사유

                    if (!td) return;
                    if (td.querySelector('.na-row-handle')) return;

                    const text = td.textContent.trim();

                    td.innerHTML =
                        '<span class="memo-text">' + text + '</span>' +
                        '<span class="na-row-handle" title="드래그해서 순서 변경">' +
                            HANDLE_SVG +
                        '</span>';
                });

                naSortable = Sortable.create(tbodyEl, {
                    handle: '.na-row-handle',
                    animation: 150
                });
            }

            // ✅ DnD 해제 + 핸들 제거
            function disableNaDnD() {
                if (naSortable) {
                    naSortable.destroy();
                    naSortable = null;
                }
                if (!$tbody || !$tbody.length) return;

                $tbody.find('tr').each(function () {
                    const tds = this.querySelectorAll('td');
                    if (!tds || !tds.length) return;
                    const td = tds[tds.length - 1];
                    if (!td) return;

                    const memoSpan = td.querySelector('.memo-text');
                    if (memoSpan) {
                        const text = memoSpan.textContent.trim();
                        td.textContent = text;
                    }
                });
            }

            jq(function () {
                const $table    = jq('.na-table');
                if (!$table.length) return;

                const $thead    = $table.find('thead');
                $tbody = $table.find('tbody');

                const $btnMain  = jq('#add-parts-btn');
                const $btnAdd   = jq('#edit-add-btn-2');
                const $btnDel   = jq('#edit-delete-btn-2');
                const $btnEdit  = jq('#edit-btn-2');
                const $btnCancel= jq('#edit-cancel-btn-2');
                const $editBox  = jq('.edit-btn-container');
                const $editWrapper = $btnMain.closest('.edit-btn-wrapper'); // 버튼 감싸는 래퍼

                if (!$btnMain.length) return;

                // ==========================
                // 2단 헤더 구성용 컬럼 정의
                // ==========================
                const cols = [
                    // 🔹 단일 헤더 (rowspan=2)
                    { key: 'EQ_Name', title: '구분', width: 50 },

                    // 🔹 구매 불가품 그룹
                    { key: 'NA_Q_Code',     title: '코드',       width: 50, group: '구매 불가품' },
                    { key: 'NA_Parts_Name', title: '파츠 분류', width: 100, group: '구매 불가품' },
                    { key: 'NA_P_N',        title: '업체P/N',   width: 100, group: '구매 불가품' },
                    { key: 'NA_Price',      title: '단가',      width: 50,  group: '구매 불가품' },

                    // 🔹 대체품 그룹
                    { key: 'Q_Code',     title: '코드',       width: 50, group: '대체품' },
                    { key: 'Parts_Name', title: '파츠 분류', width: 100, group: '대체품' },
                    { key: 'P_N',        title: '업체P/N',   width: 100, group: '대체품' },
                    { key: 'Price',      title: '단가',      width: 50,  group: '대체품' },

                    // 🔹 단일 헤더 (rowspan=2)
                    { key: 'Memo', title: '사유', width: 200 }
                ];

                const grouped   = cols.filter(c => c.group);
                const ungrouped = cols.filter(c => !c.group);

                // 왼쪽 단일 컬럼 (구분)
                const leftSingles  = ungrouped.filter(c => c.key !== 'Memo');
                // 오른쪽 단일 컬럼 (사유)
                const rightSingles = ungrouped.filter(c => c.key === 'Memo');

                const groupMap = {};
                grouped.forEach(c => {
                    if (!groupMap[c.group]) groupMap[c.group] = [];
                    groupMap[c.group].push(c);
                });

                // ==========================
                // 2단 <thead> HTML 생성
                // ==========================
                let topRow = '<tr>';
                let bottomRow = '<tr>';

                // 1) 왼쪽 단일 컬럼
                leftSingles.forEach(function(c) {
                    topRow += '<th rowspan="2" style="width:' + c.width + 'px;">' + c.title + '</th>';
                });

                // 2) 그룹 헤더
                Object.keys(groupMap).forEach(groupName => {
                    const colsInGroup = groupMap[groupName];

                    // 그룹별 색상 지정
                    let bgColor = '';
                    if (groupName === '구매 불가품') bgColor = '#ffe5e5'; // 연한 빨간색
                    if (groupName === '대체품')     bgColor = '#e6ffe6'; // 연한 초록색

                    topRow += '<th colspan="' + colsInGroup.length + '" ' +
                            'style="text-align:center; background-color:' + bgColor + ';">' +
                            groupName +
                            '</th>';
                });

                // 3) 오른쪽 단일 컬럼 (사유)
                rightSingles.forEach(function(c) {
                    topRow += '<th rowspan="2" style="width:' + c.width + 'px;">' + c.title + '</th>';
                });

                topRow += '</tr>';

                // 4) 하단 행: 그룹 내부 컬럼
                Object.keys(groupMap).forEach(groupName => {
                    const colsInGroup = groupMap[groupName];

                    let bgColor = '';
                    if (groupName === '구매 불가품') bgColor = '#ffffff';
                    if (groupName === '대체품')     bgColor = '#ffffff';

                    colsInGroup.forEach(c => {
                        bottomRow += '<th style="width:' + c.width + 'px; text-align:center; background-color:' +
                                    bgColor + ';">' + c.title + '</th>';
                    });
                });

                bottomRow += '</tr>';
                $thead.html(topRow + bottomRow);

                // 1) DataTables 초기화 (권한과 무관하게 조회는 모두 가능)
                if (jq.fn.DataTable) {
                    naDT = $table.DataTable({
                        paging:   false,
                        searching:false,
                        info:     false,
                        ordering: true,
                        scrollX:  true,
                        scrollY:  '243px',   // ✅ 원하는 높이로 조정
                        scrollCollapse: true,
                        autoWidth: false     // 컬럼 폭 계속 넓어지는 것 방지용 (권장)
                    });
                }

                // 2) 행 선택 로직: 단일 / Ctrl / Shift 구간 선택 (권한과 무관)
                let lastSelectedIndex = null;

                $tbody
                    .off('click.naRowSelect')
                    .on('click.naRowSelect', 'tr', function (e) {
                        // 셀 안의 input/버튼 클릭 시는 행 선택 무시
                        if (jq(e.target).is('input,textarea,select,button,a')) return;

                        const $rows = $tbody.find('tr');
                        const $row  = jq(this);
                        const idx   = $rows.index($row);

                        const isCtrl  = !!(e.ctrlKey || e.metaKey);
                        const isShift = !!e.shiftKey;

                        // Shift 드래그 시 텍스트 선택 생기는 것 제거
                        if (isShift && window.getSelection) {
                            const sel = window.getSelection();
                            if (sel && sel.removeAllRanges) sel.removeAllRanges();
                        }

                        const wasSelected = $row.hasClass('selected');

                        // ---- Shift: 구간 선택 ----
                        if (isShift && lastSelectedIndex != null) {
                            const start = Math.min(lastSelectedIndex, idx);
                            const end   = Math.max(lastSelectedIndex, idx);

                            if (!isCtrl) {
                                $rows.removeClass('selected');
                            }

                            for (let i = start; i <= end; i++) {
                                $rows.eq(i).addClass('selected');
                            }

                            // 구간 마지막 클릭한 행을 anchor 로 업데이트
                            lastSelectedIndex = idx;
                            return;
                        }

                        // ---- Ctrl/⌘: 개별 토글 ----
                        if (isCtrl) {
                            if (wasSelected) {
                                $row.removeClass('selected');
                            } else {
                                $row.addClass('selected');
                            }
                            lastSelectedIndex = idx;
                            return;
                        }

                        // ---- 기본: 단일 선택 ----
                        const selectedCount = $rows.filter('.selected').length;

                        // 하나만 선택된 상태에서 다시 그 행 클릭 → 전체 해제
                        if (selectedCount === 1 && wasSelected) {
                            $row.removeClass('selected');
                            lastSelectedIndex = null;
                            return;
                        }

                        // 그 외에는 항상 단일 선택
                        $rows.removeClass('selected');
                        $row.addClass('selected');
                        lastSelectedIndex = idx;
                    });

                // 3) 권한 체크 후, 편집 버튼/기능 제어
                (async () => {
                    let canEdit = false;

                    try {
                        const res = await fetch('/check-client-authority-parts');
                        if (res.ok) {
                            const data = await res.json();
                            const role = (data.authority || '').trim(); // '사용자' | '관리자' | '구매담당자'
                            if (role === '관리자' || role === '구매담당자') {
                                canEdit = true;
                            }
                        }
                    } catch (e) {
                        console.warn('[NA popup] authority check failed:', e);
                    }

                    // 관리자 또는 구매담당자 → 편집 버튼 보이기
                    if (canEdit) {
                        if ($editWrapper.length) {
                            $editWrapper.css('display', 'flex');   // 또는 block
                        } else {
                            $btnMain.show();
                        }
                    } else {
                        // 일반 사용자: 기본이 display:none 이라서 아무것도 안 함
                        return;
                    }

                    // ==========================
                    // 편집 기능 바인딩
                    // ==========================

                    // 1) 메인 버튼: 파츠 편집 ↔ 저장
                    $btnMain.on('click', function () {
                        // 편집 모드 아니면 → 패널 열고 백업만
                        if (!editMode) {
                            editMode = true;
                            $editBox.css('display', 'flex');
                            $btnMain.text('저장');

                            backupRows = [];
                            $tbody.find('tr').each(function () {
                                const tds = this.querySelectorAll('td');
                                if (!tds || tds.length !== 10) return;
                                backupRows.push([
                                    tds[0].textContent.trim(),                          // EQ_Name
                                    tds[1].textContent.trim(),                          // NA_Q_Code
                                    tds[2].textContent.trim(),                          // NA_Parts_Name
                                    tds[3].textContent.trim(),                          // NA_P_N
                                    tds[4].textContent.trim().replace(/,/g, ''),        // NA_Price
                                    tds[5].textContent.trim(),                          // Q_Code
                                    tds[6].textContent.trim(),                          // Parts_Name
                                    tds[7].textContent.trim(),                          // P_N
                                    tds[8].textContent.trim().replace(/,/g, ''),        // Price
                                    tds[9].textContent.trim()                           // Memo
                                ]);
                            });

                            // ✅ 편집 모드 진입 시 행 순서 변경(드래그) 활성화
                            enableNaDnD();
                            return;
                        }

                        // 이미 편집 모드 → 저장
                        const rowsToSave = [];
                        $tbody.find('tr').each(function () {
                            const tds = this.querySelectorAll('td');
                            if (!tds || tds.length !== 10) return;

                            const vals = [];
                            for (let i = 0; i < 10; i++) {
                                const td = tds[i];
                                const input = td.querySelector('input.na-input');
                                let v;

                                if (input) {
                                    v = input.value;
                                } else {
                                    // ✅ Memo(사유) 컬럼은 memo-text만 읽기
                                    if (i === 9) {
                                        const span = td.querySelector('.memo-text');
                                        v = span ? span.textContent.trim() : td.textContent.trim();
                                    } else {
                                        v = td.textContent.trim();
                                    }
                                }

                                // 숫자 컬럼 처리: i == 4(NA_Price), i == 8(Price)
                                if (i === 4 || i === 8) {
                                    v = v.replace(/,/g, '');
                                }
                                vals.push(v);
                            }

                            rowsToSave.push({
                                EQ_Name:       vals[0],
                                NA_Q_Code:     vals[1],
                                NA_Parts_Name: vals[2],
                                NA_P_N:        vals[3],
                                NA_Price:      vals[4],
                                Q_Code:        vals[5],
                                Parts_Name:    vals[6],
                                P_N:           vals[7],
                                Price:         vals[8],
                                Memo:          vals[9]
                            });
                        });

                        // 👉 서버로 저장 요청
                        fetch('/parts/purchase_na_parts/save', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ rows: rowsToSave })
                        })
                        .then(res => {
                            if (!res.ok) throw new Error('서버 응답 오류');
                            return res.json();
                        })
                        .then(data => {

                            // 저장 성공 시, 화면의 인풋을 다시 텍스트로 돌리기
                            $tbody.find('tr').each(function () {
                                const tds = this.querySelectorAll('td');
                                if (!tds || tds.length !== 10) return;

                                for (let i = 0; i < 10; i++) {
                                    const td = tds[i];
                                    const input = td.querySelector('input.na-input');
                                    if (!input) continue;

                                    let v = input.value;

                                    // 숫자 컬럼: i 4(NA_Price), 8(Price) 는 천단위 콤마
                                    if ((i === 4 || i === 8) && v) {
                                        const num = Number(v.replace(/,/g, ''));
                                        td.innerHTML =
                                            '<span style="float:right;">' +
                                            (isNaN(num) ? '' : num.toLocaleString()) +
                                            '</span>';
                                    } else {
                                        td.textContent = v;
                                    }
                                }
                            });

                            editMode = false;
                            $editBox.hide();
                            $btnMain.text('파츠 편집');
                            $tbody.find('tr').removeClass('selected');

                            // ✅ 저장 후 행 순서 변경 기능 해제 + 핸들 제거
                            disableNaDnD();

                            alert('구매 제한 리스트가 저장되었습니다.');
                        })
                        .catch(err => {
                            console.error('[NA popup] save error:', err);
                            alert('저장 중 오류가 발생했습니다. 다시 시도해 주세요.');
                        });

                        // (방어적) 인풋 → 텍스트 복원
                        $tbody.find('tr').each(function () {
                            const tds = this.querySelectorAll('td');
                            if (!tds || tds.length !== 10) return;

                            for (let i = 0; i < 10; i++) {
                                const td = tds[i];
                                const input = td.querySelector('input.na-input');
                                if (!input) continue;

                                let v = input.value;
                                if ((i === 4 || i === 8) && v) {
                                    const num = Number(v.replace(/,/g, ''));
                                    td.innerHTML =
                                        '<span style="float:right;">' +
                                        (isNaN(num) ? '' : num.toLocaleString()) +
                                        '</span>';
                                } else {
                                    td.textContent = v;
                                }
                            }
                        });

                        // ✅ 편집 취소 시에도 행 순서 변경 기능 해제
                        disableNaDnD();

                        editMode = false;
                        $editBox.hide();
                        $btnMain.text('파츠 편집');
                        $tbody.find('tr').removeClass('selected');
                    });

                    // 2) 수정: 선택 행만 인풋 전환
                    $btnEdit.on('click', function () {
                        if (!editMode) return;

                        const $selected = $tbody.find('tr.selected');
                        if (!$selected.length) {
                            alert('수정할 행을 선택해 주세요.');
                            return;
                        }

                        $selected.each(function () {
                            const tds = this.querySelectorAll('td');
                            if (!tds || tds.length !== 10) return;

                            for (let i = 0; i < 10; i++) {
                                const td = tds[i];
                                if (td.querySelector('input.na-input')) continue;

                                let text = td.textContent.trim();
                                if (i === 4 || i === 8) text = text.replace(/,/g, '');

                                const type = (i === 4 || i === 8 ? 'number' : 'text');
                                td.innerHTML =
                                    '<input class="na-input" type="' + type + '" value="' + text + '">';
                            }
                        });
                    });

                    // 3) 행추가: 새 행 인풋 상태로 추가
                    $btnAdd.on('click', function () {
                        if (!editMode) return;

                        const tr = document.createElement('tr');
                        for (let i = 0; i < 10; i++) {
                            const td = document.createElement('td');
                            const type = (i === 4 || i === 8 ? 'number' : 'text');
                            td.innerHTML =
                                '<input class="na-input" type="' + type + '">';
                            tr.appendChild(td);
                        }

                        if (naDT) {
                            naDT.row.add(tr).draw(false);
                        } else {
                            $tbody.append(tr);
                        }
                    });

                    // 4) 행삭제
                    $btnDel.on('click', function () {
                        if (!editMode) return;

                        const $selected = $tbody.find('tr.selected');
                        if (!$selected.length) {
                            alert('삭제할 행을 선택해 주세요.');
                            return;
                        }
                        if (!window.confirm($selected.length + '건을 삭제할까요?')) return;

                        if (naDT) {
                            $selected.each(function () {
                                naDT.row(this).remove();
                            });
                            naDT.draw(false);
                        } else {
                            $selected.remove();
                        }
                    });

                    // 5) 취소: 백업 데이터로 전체 원복
                    $btnCancel.on('click', function () {
                        if (!editMode) return;
                        if (!window.confirm('편집 내용을 모두 취소할까요?')) return;

                        if (naDT) {
                            naDT.clear();
                            backupRows.forEach(function (r) {
                                naDT.row.add([
                                    r[0],
                                    r[1],
                                    r[2],
                                    r[3],
                                    r[4] ? Number(r[4]).toLocaleString() : '',
                                    r[5],
                                    r[6],
                                    r[7],
                                    r[8] ? Number(r[8]).toLocaleString() : '',
                                    r[9]
                                ]);
                            });
                            naDT.draw(false);
                        } else {
                            $tbody.empty();
                            backupRows.forEach(function (r) {
                                const tr = document.createElement('tr');
                                tr.innerHTML =
                                    '<td>' + r[0] + '</td>' +
                                    '<td>' + r[1] + '</td>' +
                                    '<td>' + r[2] + '</td>' +
                                    '<td>' + r[3] + '</td>' +
                                    '<td style="text-align:right;">' +
                                        (r[4] ? Number(r[4]).toLocaleString() : '') +
                                    '</td>' +
                                    '<td>' + r[5] + '</td>' +
                                    '<td>' + r[6] + '</td>' +
                                    '<td>' + r[7] + '</td>' +
                                    '<td style="text-align:right;">' +
                                        (r[8] ? Number(r[8]).toLocaleString() : '') +
                                    '</td>' +
                                    '<td>' + r[9] + '</td>';
                                $tbody.append(tr);
                            });
                        }

                        editMode = false;
                        $editBox.hide();
                        $btnMain.text('파츠 편집');
                        $tbody.find('tr').removeClass('selected');
                    });

                })(); // 권한 체크 IIFE 끝
            });
        })();
    <\/script>



</body>
</html>
        `);

        popup.document.close();
        popup.focus();

    } catch (e) {
        console.error('[openNaPartsWindow] error:', e);
        alert('구매 제한 리스트를 불러오는 중 오류가 발생했습니다.');
    }
}

// 메인 화면의 "구매 제한" 버튼에 이벤트 연결
$(function () {
    $('#na-parts-inform-btn').on('click', function () {
        openNaPartsWindow();
    });
});
