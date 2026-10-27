// ---------------------------------
// 파츠 구매 모듈 (parts_purchase_list) — DataTables 버전
// ---------------------------------
const PartsBuy = (() => {
    let state = {
        items: [],
        isFetching: false,
        isAdmin: false,
        inited: false,
        container: null,
        lastFilters: {}, // (EQ_Type/Parts/Unit/Q_Code/P_N)
        dateRange: { start: null, end: null },
        pendingOrderNos: null   // ✅ 드래그로 바뀐 No 순서를 임시 저장
    };

    let buyDT = null;                   // DataTables 인스턴스
    const selectedKeys = new Set();     // 선택 상태(복합키)
    let buyEditCtxs = null; // [{ rowNode, rowApi, before, beforeKey }, ...]
    let __buyDelBusy = false;
    let __purchaseSortable = null; //  DnD 인스턴스 보관

    // ============================================ 행 드래그앤 드롭
    function getDT() { return buyDT; }

    // DnD 활성화
    function enableDnD() {
        if (!window.Sortable) return console.warn('SortableJS not loaded');
        if (!buyDT || __purchaseSortable) return;
        const tbody = buyDT.table().body();
        if (!tbody) return;

        __purchaseSortable = Sortable.create(tbody, {
            handle: '.row-handle',
            animation: 150,
            onEnd: () => {
            const orderedNos = Array.from(tbody.querySelectorAll('tr'))
                .map(tr => buyDT.row(tr).data()?.No)
                .filter(n => Number.isFinite(Number(n)));
            state.pendingOrderNos = orderedNos;
            applyDomOrderToDataTable();
            }
        });

        // ✅ DnD 켜짐 → 핸들 렌더링 반영
        if (buyDT) buyDT.rows().invalidate('data').draw(false);
    }

    function disableDnD() {
        if (__purchaseSortable) {
            __purchaseSortable.destroy();
            __purchaseSortable = null;
        }
        // ✅ DnD 꺼짐 → 핸들 숨김 반영
        if (buyDT) buyDT.rows().invalidate('data').draw(false);
    }

    function readCurrentDomOrderNos() {
        if (!buyDT) return [];
        const tbody = buyDT.table().body();
        return Array.from(tbody.querySelectorAll('tr'))
            .map(tr => buyDT.row(tr).data())
            .filter(Boolean)
            .map(r => r?.No)
            .filter(n => Number.isFinite(Number(n)));
    }

    function applyDomOrderToDataTable() {
        if (!buyDT) return;
        const nos = readCurrentDomOrderNos();
        if (!nos.length) return;

        const all = buyDT.rows().data().toArray();
        const byNo = new Map(all.map(r => [String(r.No), r]));
        const ordered = [];
        for (const n of nos) {
            const r = byNo.get(String(n));
            if (r) {
            ordered.push(r);
            byNo.delete(String(n));
            }
        }
        // 필터 등으로 빠진 나머지(번호 없는 행 포함)는 뒤에 유지
        const rest = Array.from(byNo.values());
        buyDT.clear().rows.add(ordered.concat(rest)).draw(false);
    }

    // =================================================================

    async function copySelectedQCodeAndReason() {
        if (!buyDT) return;
        const rows = buyDT.rows('.selected').data().toArray();
        if (!rows.length) { alert('복사할 행을 선택하세요.'); return; }

        const lines = rows.map(r => {
            const q = (r.Q_Code ?? '').toString().replace(/\r?\n/g, ' ');
            const p = (r.P_N ?? '').toString().replace(/\r?\n/g, ' ');
            const c = (r.Req_Comment ?? '').toString().replace(/\r?\n/g, ' ');
            return `${q}\t${p}\t${c}`;
        });
        const text = lines.join('\n');

        // 1) 우선 'copy' 이벤트를 트리거해서 안전하게 넣는 경로
        let succeeded = false;
        const onCopy = (e) => {
            try {
                e.preventDefault(); // 기본 선택 복사 막고
                e.clipboardData.setData('text/plain', text); // 우리가 만든 텍스트 주입
                succeeded = true;
            } catch (_) { /* no-op */ }
        };
        document.addEventListener('copy', onCopy, { once: true });

        try {
            // execCommand('copy')가 'copy' 이벤트를 발생시킵니다.
            const ok = document.execCommand && document.execCommand('copy');
            if (ok && succeeded) {
                toastCopy(`복사됨: ${rows.length}건 (Q-Code + 요청사유)`);
                return;
            }
        } catch (_) {
            /* fallthrough */
        }

        // 2) 실패 시, 안전 컨텍스트면 Clipboard API로 재시도
        try {
            await navigator.clipboard.writeText(text);
            toastCopy(`복사됨: ${rows.length}건 (Q-Code + 요청사유)`);
            return;
        } catch (_) {
            // 3) 마지막 폴백: 숨은 textarea 선택 복사
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            ta.setSelectionRange(0, ta.value.length);
            try {
                const ok2 = document.execCommand && document.execCommand('copy');
                if (ok2) {
                    toastCopy(`복사됨: ${rows.length}건 (Q-Code + 요청사유)`);
                    return;
                }
            } catch (_) { /* no-op */ }
            finally {
                document.body.removeChild(ta);
            }
        }
    }


    const PURCHASE_STATUS_FLOW = [
        "구매 요청",
        "구매 대기",
        "Drop",
        "구매 중",
        "입고 완료",
        ""
    ];

    // HTML 이스케이프 (툴팁/내용 XSS 방지)
    function esc(s) {
        return String(s ?? '').replace(/[&<>"']/g, c =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    // 말줄임 span 생성 (title에 전체 텍스트)
    function ellipsize(val) {
        const txt = esc(val ?? '');
        return `<span class="dt-ellipsis" title="${txt}">${txt}</span>`;
    }

    // 프론트 표시 키 -> DB 실제 컬럼 키 매핑 (저장 전 공통 적용)
    function mapFrontendToDB(upd) {
        const out = { ...upd };
        // PO_* → Order_*
        if ('PO_Qty'  in out) { out.Order_Qty  = out.PO_Qty;  delete out.PO_Qty; }
        if ('PO_No'   in out) { out.Order_Num  = out.PO_No;   delete out.PO_No;  }
        if ('PO_Date' in out) { out.Order_Date = out.PO_Date; delete out.PO_Date; }

        // 표시용 → DB 실컬럼
        if ('EQ_Name'    in out) { out.EQP_Div   = out.EQ_Name;    delete out.EQ_Name; }
        if ('Parts_Name' in out) { out.Parts_Div = out.Parts_Name; delete out.Parts_Name; }
        if ('EQ_Unit'    in out) { out.Unit      = out.EQ_Unit;    delete out.EQ_Unit; }
        if ('Purchase_status' in out) { out.Purchase_Status = out.Purchase_status; delete out.Purchase_status; }

        return out;
    }

    function canDeleteRow(row) {
        // 1) 구매담당자: 전체 삭제 가능
        if (state.isPurchaser) return true;

        // 2) 관리자/사용자: 본인 글 + (구매 요청 | 구매 대기)만 삭제 가능
        const status = (row.Purchase_status || row.Purchase_Status || '').trim();
        const currentUser = (state.userName || '').trim();
        const requester   = (row.Requester || '').trim();
        const isOwn = currentUser && requester && currentUser === requester;

        if (!isOwn) return false;
        return (status === '구매 요청' || status === '구매 대기');
    }

    // ✅ 구매담당자: 수정 금지 컬럼
    const PURCHASER_EXCLUDED_COLS = new Set([
        'Request_Date',   // 요청일
        'EMS재고',        // 재고(EMS)
        '라인재고',       // 재고(라인)
        //'Requester',     // 요청자
        'Avg_LeadTime',   // 평균 입고일
        'No', '_key'      // 내부 키
    ]);

    // ✅ 컬럼 타입 규칙
    function getColEditSpec(col) {
        if (col === 'Req_Qty' || col === 'PO_Qty') return { type: 'number', min: 0, step: 1 };
        if (col === 'PO_Date' || col === 'Arrival_Date') return { type: 'date' };
        if (col === 'Purchase_Catg') return { type: 'select', options: ['EMS','MRO','SAP','STG','재계약','채번중',''] };
        if (col === 'Site') return { type: 'select', options: ['P1DD', 'P1DJ', 'MRO'] };
        if (col === 'EQ_Name') return { type: 'select', options: ['A3 PLM'] };
        if (col === 'Purchase_status') {
            const opts = PURCHASE_STATUS_FLOW.filter(Boolean); // ["구매 요청","구매 대기","Drop","구매 완료","입고 완료"]
            return { type: 'select', options: opts };
        }

        return { type: 'text', maxlength: 200 };
    }

    // 권한에 따라 편집 가능한 컬럼 세트 계산
    function getEditableColumnsForRow(rowData) {
        const currentUser = (state.userName || '').trim();
        const requester   = (rowData?.Requester || '').trim();
        const status      = (rowData?.Purchase_status || rowData?.Purchase_Status || '').trim();

        // ✅ 구매담당자: 표시되는 대부분의 컬럼을 허용하되, 금지 목록은 제외
        if (state.isPurchaser) {
            let cols = Object.keys(rowData).filter(c => !PURCHASER_EXCLUDED_COLS.has(c));
            // ✅ 상태가 '구매 요청'일 때는 구매구분 수정 금지
            if (status === '구매 요청') {
                cols = cols.filter(c => c !== 'Purchase_Catg');
            }
            return cols.map(c => [c, getColEditSpec(c)]);
        }

        // [그 외 기존 로직 유지] 일반 사용자: 본인 + '구매 요청' 상태만 제한적 편집
        if (status === '구매 요청' && currentUser && requester && currentUser === requester) {
            return [
                ['EQ_Name',     getColEditSpec('EQ_Name')],
                ['Q_Code',      { type:'text',   maxlength:100 }],
                ['Parts_Name',  { type:'text',   maxlength:100 }],
                ['EQ_Unit',     { type:'text',   maxlength:100 }],
                ['P_N',         { type:'text',   maxlength:150 }],
                ['Req_Qty',     { type:'number', min:0, step:1 }],
                ['Req_Comment', { type:'text',   maxlength:120 }],
                ['Site',        getColEditSpec('Site')]
            ];
        }

        // 관리자( state.isManager )는 기존처럼 처리되고, 나머지는 편집 불가
        return [];
    }

    // updates 객체에서 권한상 허용된 컬럼만 필터
    function filterUpdatesByPermission(rowData, updates) {
        const allowed = new Set(getEditableColumnsForRow(rowData).map(([c]) => c));
        const filtered = {};
        for (const k of Object.keys(updates)) {
        if (allowed.has(k)) filtered[k] = updates[k];
        }

        // ✅ 이중 안전장치: 구매요청 상태라면 구매구분 값 제거
        const status = (rowData?.Purchase_status || rowData?.Purchase_Status || '').trim();
        if (status === '구매 요청' && 'Purchase_Catg' in filtered) {
            delete filtered.Purchase_Catg;
        }

        return filtered;
    }

    function rowToInputsPurchase(rowNode, editableMap) {
        const data  = buyDT.row(rowNode).data();
        const cells = rowNode.cells || rowNode.children;
        if (!cells || !data) return;

        // 컬럼 dataSrc → 인덱스 캐시
        const indexByCol = {};
        buyDT.columns().every(function () {
        indexByCol[this.dataSrc()] = this.index();
        });

        for (const [col, opt] of editableMap) {
        const idx = indexByCol[col];
        if (idx == null) continue;

        const v = data[col] ?? '';
        const status = (data.Purchase_status || data.Purchase_Status || '').trim();

        // ✅ 구매요청 상태에서는 구매구분을 입력 컨트롤로 만들지 않고 텍스트만
        if (col === 'Purchase_Catg' && status === '구매 요청') {
            cells[idx].innerHTML = `<span class="p-readonly">${esc(String(v)) || ''}</span>`;
            cells[idx].setAttribute('data-col', col);
            continue;
        }

        if (opt.type === 'number') {
            cells[idx].innerHTML =
            `<input type="number" class="p-edit" value="${esc(String(v))}" step="${opt.step ?? 1}" min="${opt.min ?? 0}">`;
        } else if (opt.type === 'date') {
            cells[idx].innerHTML =
            `<input type="text" class="p-edit p-date" value="${esc(String(v))}" placeholder="YYYY-MM-DD">`;
        } else if (opt.type === 'select') {
            const opts = (opt.options || []).map(o => {
                const sel = String(o) === String(v) ? ' selected' : '';
                return `<option value="${esc(o)}"${sel}>${esc(o)}</option>`;
            }).join('');
            cells[idx].innerHTML = `<select class="p-edit">${opts}</select>`;
        } else { // text
            const max = opt.maxlength ? ` maxlength="${opt.maxlength}"` : '';
            cells[idx].innerHTML =
            `<input type="text" class="p-edit" value="${esc(String(v))}"${max}>`;
        }
        cells[idx].setAttribute('data-col', col);
        }

        // 날짜 인풋에 flatpickr 붙이기 (표시는 YYYY-MM-DD만)
        if (window.flatpickr) {
        $(rowNode).find('.p-date').each(function () {
            flatpickr(this, {
                dateFormat: 'Y-m-d',
                allowInput: true,
                locale: flatpickr.l10ns?.ko || 'ko'
            });
        });
        }

        // ✅ 추가: Q_Code 입력 시 자동 대문자 변환
        // data-col은 <td>에 있으므로 td[data-col="Q_Code"] 아래 input을 잡아야 함
        $(rowNode)
        .find('td[data-col="Q_Code"] input.p-edit')
        .on('input', function () { this.value = this.value.toUpperCase(); });

        rowNode.classList.add('editing-row');
    }

    // 숫자 문자열을 느슨하게 파싱(콤마/공백/통화기호 제거)
    function parseNumberLoose(v) {
        if (v == null) return null;
        if (typeof v === 'number') return Number.isFinite(v) ? v : null;
        const s = String(v)
            .replace(/[,\s₩원]/g, '') // 천단위 콤마, 공백, 통화기호 제거
            .replace(/[^0-9.+\-eE]/g, ''); // 숫자/부호/소수점/지수 외 제거
        if (s === '' || s === '.' || s === '+' || s === '-' ) return null;
        const n = Number(s);
        return Number.isFinite(n) ? n : null;
    }

    function restoreRowFromDataPurchase(rowNode, data) {
        // DataTables 데이터로 원복 후 redraw
        buyDT.row(rowNode).data({ ...data });
        buyDT.draw(false);
    }

    function setPurchaseActionButtonsEnabled(enabled) {
        const wrap   = document.querySelector('.parts-purchase-wrapper');
        if (!wrap) return;
        const btnEdit = wrap.querySelector('#edit-btn-3');
        const btnDel  = wrap.querySelector('#edit-delete-btn-3');
        if (btnEdit) btnEdit.disabled = !enabled;
        if (btnDel)  btnDel.disabled  = !enabled;
    }

    /* ---------------------------
        권한 체크 (상단 표시 포함)
    --------------------------- */
    function setAdminMode(isAdmin) {
        state.isAdmin = isAdmin;
        renderTable(state.items); // 모드 반영
    }

    async function checkClientAuthority() {
        try {
            const res = await fetch('/check-client-authority-parts');
            const data = await res.json();
            state.role = data.authority; // '사용자' | '관리자' | '구매담당자'
            state.userName = data?.userName || 'Unknown';

            state.isPurchaser   = (state.role === '구매담당자'); // 구매담당자
            state.isManager     = (state.role === '관리자');     // 관리자
            state.isPrivileged  = state.isPurchaser || state.isManager; // 관리자/구매담당자
           
            setAdminMode(state.isPrivileged);

            // 상단 버튼 표시 제어
            const btnApprove = document.getElementById('parts-purchase-btn-approve');
            const btnReject  = document.getElementById('parts-purchase-btn-reject');
            if (btnApprove) btnApprove.style.display = state.isPrivileged ? '' : 'none';
            if (btnReject)  btnReject.style.display  = state.isPrivileged ? '' : 'none';

            // ✅ 결재 정보 버튼: "구매담당자"만 보이게
            const btnManagerInfo = document.getElementById('purchase-manager-inform-btn');
            if (btnManagerInfo) {
                btnManagerInfo.style.display = state.isPurchaser ? '' : 'none';
            }

            // ✅ 정기 구매 파츠 버튼: "구매담당자"만 보이게
            const btnPeriodicParts = document.getElementById('periodic-parts-inform-btn');
            if (btnPeriodicParts) {
                btnPeriodicParts.style.display = state.isPurchaser ? '' : 'none';
            }

            // ✅ 컨테이너도 권한에 따라 토글
            const box2 = document.querySelector('.edit-btn-container-2');
            if (box2) box2.style.display = state.isPrivileged ? 'flex' : 'none';

            if (!document.getElementById('user-name-display')) {
                const userNameContainer = document.createElement('div');
                userNameContainer.id = 'user-name-display';
                const userNameBox = document.createElement('div');
                userNameBox.id = 'user-name-box';
                userNameBox.innerText = `${data.authority} 모드 ( ${data.userName} )`;
                userNameContainer.appendChild(userNameBox);
                document.body.prepend(userNameContainer);
            }
            // 권한 클래스 토글 (구매담당자면 role-purchaser 클래스 on)
            document.body.classList.toggle('role-purchaser', state.isPurchaser);
        } catch (e) {
            console.error('Authority check failed:', e);
        }
    }

    // 구매 대기 선택된 행 키/필드 추출 유틸
    function getSelectedRows() {
        if (!buyDT || selectedKeys.size === 0) return [];
        const all = buyDT.rows().data().toArray();
        const need = new Set(selectedKeys);
        return all
        .filter(r => need.has(r._key))
        .map(r => ({
            No:         r.No ?? null,
            Q_Code:     r.Q_Code ?? null,
            Unit:       r.EQ_Unit ?? null,
            P_N:        r.P_N ?? null,
            EQP_Div:    r.EQ_Name ?? null,     // 화면 필드(EQ_Name) = 서버 컬럼 EQP_Div
            Parts_Div:  r.Parts_Name ?? null,  // 화면 필드(Parts_Name) = 서버 컬럼 Parts_Div
            Requester:  r.Requester ?? null,
            Site:       r.Site ?? null,  
            Purchase_Status: r.Purchase_status ?? null, // 참고: 현재 상태(클라 검증용)
        }));
    }

    /* ---------------------------
        상단 전역 필터 연동
    --------------------------- */
    function searchWithFilters(params) {
        state.lastFilters = {
            eqType: params?.EQ_Type ?? '',
            parts:  params?.Parts_Name ?? '',
            unit:   params?.EQ_Unit ?? '',
            qcode:  params?.Q_Code ?? '',
            pn:     params?.P_N ?? '',
            requester: params?.Requester ?? '',
            Purchase_Status: Array.isArray(params?.Purchase_Status)
                ? params.Purchase_Status.slice()
                : (params?.Purchase_Status ?? ''),
            Purchase_Catg: params?.Purchase_Catg ?? '',
            allKeyword: params?.allKeyword ?? '',
            dupOnly: !!params?.dupOnly
        };

        selectedKeys.clear();
        renderTable(state.items);
    }

    /* ---------------------------
        UI 스켈레톤
    --------------------------- */
    function buildUI(container) {
        const privileged = !!state.isAdmin; // 관리자/구매담당자 = true

        // 공통(행 데이터 순서와 일치)
        const base = [
            { key: 'Request_Date', title: '요청일', width: 90 },
            { key: 'EQ_Name',      title: '구분',   width: 70  },
            { key: 'Site',      title: 'SITE',   width: 70  },
            { key: 'Q_Code',       title: 'Q-Code', width: 140 },
            { key: 'Parts_Name',   title: '파츠 분류', width: 140 },
            { key: 'P_N',          title: '업체P/N', width: 240 },

            // ↓↓↓ 2단 그룹 ① 재고(EMS/라인)
            { key: 'EMS재고',      title: 'EMS',   width: 40,  group: '재고' },
            { key: '라인재고',     title: '라인',  width: 40,  group: '재고' },

            { key: 'Requester',    title: '요청자', width: 70 },
            { key: 'Req_Comment',  title: '요청사유', width: 180 },
            { key: 'Purchase_status', title: '구매현황', width: 80 },

            // ↓↓↓ 2단 그룹 ② 구매 수량(요청/구매)
            { key: 'Req_Qty',      title: '요청',  width: 40,  group: '구매 수량' },
            { key: 'PO_Qty',       title: '구매',  width: 40,  group: '구매 수량' },

            { key: 'Unit_Price',  title: '단가', width: 80,  group: '가격' },
            { key: 'Total_Price', title: '금액', width: 80,  group: '가격' },

            { key: 'Purchase_Catg', title: '구매<br>구분', width: 60 },

            { key: 'Manager_Comment', title: '구매담당자 코멘트', width: 140 },
            { key: 'Avg_LeadTime',    title: '평균<br>입고일',       width: 60 },
        ];

        // 권한 전용
        const adminOnly = [
            { key: 'Req_Num',      title: '의뢰번호', width: 120, adminOnly: true },
            { key: 'ERP_PR',       title: 'ERP PR',   width: 110, adminOnly: true },
            { key: 'Q_Code_2',     title: '부품',     width: 140, adminOnly: true },
            { key: 'PO_No',        title: '발주번호', width: 100, adminOnly: true },
            { key: 'PO_Date',      title: '발주일', width: 120, adminOnly: true },
            { key: 'Arrival_Date', title: '입고일',   width: 120, adminOnly: true },
            { key: 'PO_Reject',    title: 'PO Reject', width: 100, adminOnly: true },
        ];

        const cols = privileged ? [...base, ...adminOnly] : base;

        // ── 헤더 2단 구성 정보 만들기 ─────────────────────────
        const groups = {};   // {groupName: [colIndex,...]}
        const row1 = [];     // [{type:'single'|'group', text, span}]
        const row2 = [];     // [{text, admin}...]

        for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (c.group) {
            if (!groups[c.group]) groups[c.group] = [];
            groups[c.group].push(i);
        }
        }

        let i = 0;
        while (i < cols.length) {
        const col = cols[i];
        if (col.group) {
            const g = groups[col.group];
            if (g[0] === i) {
            row1.push({ type: 'group', text: col.group, span: g.length });
            g.forEach(idx => {
                const cc = cols[idx];
                row2.push({ text: cc.title, admin: !!cc.adminOnly });
            });
            }
            i++;
        } else {
            row1.push({ type: 'single', text: col.title, admin: !!col.adminOnly });
            i++;
        }
        }

        const colgroupHTML = cols.map(c => `<col style="width:${c.width}px">`).join('');
        const row1HTML = row1.map(cell => {
        if (cell.type === 'group') {
            return `<th class="th-group" colspan="${cell.span}">${cell.text}</th>`;
        }
        return `<th class="${cell.admin ? 'admin-col' : ''}" rowspan="2">${cell.text}</th>`;
        }).join('');
        const row2HTML = row2.map(cell => `<th class="${cell.admin ? 'admin-col' : ''}">${cell.text}</th>`).join('');

        container.innerHTML = `
        <div class="table-wrap">
            <table class="purchase-mgmt-table display compact nowrap" style="width:100%">
            <colgroup>${colgroupHTML}</colgroup>
            <thead>
                <tr class="top-group">${row1HTML}</tr>
                <tr class="sub-head">${row2HTML}</tr>
            </thead>
            <tbody></tbody>
            </table>
        </div>
        `;
    }

    /* ---------------------------
        전역(상단) 필터만 적용
    --------------------------- */
    function applyFilters(list) {
        // DB 원본 → 글로벌 필터 스키마로 정규화
        const normalized = (list || []).map(it => ({
            ...it,
            EQ_Name:    it.EQP_Div ?? '',
            Parts_Name: it.Parts_Div ??  '',
            Q_Code:     it.Q_Code ?? '',
            P_N:        it.P_N ?? '',
            Requester:  it.Requester ?? '',
            Site:       it.Site ?? '',
            Purchase_status: it.Purchase_Status ?? it.Purchase_status ?? '',
            Purchase_Catg:   it.Purchase_Catg ?? ''
        }));

        // 1) 기존 전역 필터 적용
        let out = filterItemsByGlobalFilters(normalized, state.lastFilters);

        // 2) ✅ 중복 구매만 보기 적용 (현재 날짜 범위로 로딩된 state.items 기준)
        if (state.lastFilters?.dupOnly) {
            const TARGET = new Set(['구매 요청', '구매 대기', '구매 중']);
            const dupSet = getDuplicateQCodeSet(list); // list = state.items(현재 날짜 범위)

            out = out.filter(it => {
                const q = String(it?.Q_Code ?? '').trim().toUpperCase();
                const st = String(it?.Purchase_Status ?? it?.Purchase_status ?? '').trim();
                return dupSet.has(q) && TARGET.has(st);
            });
        }

        return out;
    }

    /* ---------------------------
        DataTables 초기화/갱신
    --------------------------- */
    function finalizeBuyTable(rowsData) {
        const $table = $(state.container).find('.purchase-mgmt-table');

        // 1) 표에 쓸 데이터 매핑
        const dtData = (rowsData || []).map(it => {
        const ems  = Number(it.EMS재고 ?? it.Count_Now ?? 0);
        const line = Number(it.라인재고 ?? 0);

        function tryFormatDate(val) {
            if (!val) return '';
            const d = new Date(val);
            if (isNaN(d)) return val;
            d.setHours(d.getHours() + 9); // KST 보정: UTC 기준 +9시간 추가 후 YYYY-MM-DD (UTC로 잘림)
            return d.toISOString().slice(0, 10); // YYYY-MM-DD
        }

        return {
            // 공통
            No:               it.No ?? null,
            Request_Date:     tryFormatDate(it.Req_Date),
            EQ_Name:          it.EQP_Div ?? '',
            Site:             it.Site ?? '',  
            Q_Code:           it.Q_Code ?? '',
            Parts_Name:       it.Parts_Div ?? '',
            P_N:              it.P_N ?? '',
            EMS재고:          ems,
            라인재고:         line,
            Requester:        it.Requester ?? '',
            Req_Comment:      it.Req_Comment ?? '',
            Purchase_status:  it.Purchase_Status ?? '',
            Req_Qty:          Number(it.Req_Qty ?? 0),
            PO_Qty:           Number(it.Order_Qty ?? 0),
            Unit_Price:       it.Unit_Price  == null ? null : Number(it.Unit_Price),
            Total_Price:      it.Total_Price == null ? null : Number(it.Total_Price),

            Manager_Comment:  it.Manager_Comment ?? '',
            Avg_LeadTime:     it.Avg_LeadTime ?? '신규품',

            // 관리자 전용
            Purchase_Catg:    it.Purchase_Catg ?? '',
            Req_Num:          it.Req_Num ?? '',
            ERP_PR:           it.ERP_PR ?? '',
            Q_Code_2:         it.Q_Code_2 ?? '',
            PO_No:            it.Order_Num ?? '',
            PO_Date:          tryFormatDate(it.Order_Date),
            Arrival_Date:     tryFormatDate(it.Arrival_Date),
            PO_Reject:        it.PO_Reject ?? '',

            // 선택/편집용 키
            _key: (it.No != null ? String(it.No) : `${it.Site ?? ''}|${it.Q_Code ?? ''}|${it.Unit ?? ''}|${it.P_N ?? it.Spec ?? ''}|${it.Req_Num ?? ''}`)
        };
        });

        // 2) 권한별 컬럼 정의
        const baseDefs = [
            { data:'Request_Date' },
            { data:'EQ_Name' },
            { data:'Site' },  
            { data:'Q_Code' },
            { data:'Parts_Name',   render: (d,t) => t === 'display' ? ellipsize(d) : d },
            { data:'P_N',          render: (d,t) => t === 'display' ? ellipsize(d) : d, className:'dt-left' },
            { data:'EMS재고'  },
            { data:'라인재고' },
            { data:'Requester' },
            { data:'Req_Comment',  render: (d,t) => t === 'display' ? ellipsize(d) : d, className:'dt-left'  },
            { data:'Purchase_status' },
            { data:'Req_Qty' },
            { data:'PO_Qty' },
            { data:'Unit_Price',  className:'dt-right',
                render:(d,t)=> t==='display' ? (d==null?'':d.toLocaleString()) : d },
            { data:'Total_Price', className:'dt-right',
                render:(d,t)=> t==='display' ? (d==null?'':d.toLocaleString()) : d },
            { data:'Purchase_Catg' },
            { data:'Manager_Comment', render: (d,t) => t === 'display' ? ellipsize(d) : d, className:'dt-left'  },
            { data:'Avg_LeadTime' },
        ];

        const adminDefs = [
            { data:'Req_Num' },
            { data:'ERP_PR' },
            { data:'Q_Code_2' },
            { data:'PO_No' },
            { data:'PO_Date' },
            { data:'Arrival_Date' },
            { data:'PO_Reject' },
        ];

        const cols = state.isAdmin ? [...baseDefs, ...adminDefs] : baseDefs;

        // ✅ Purchase_Catg 컬럼 index 찾기
        const purchaseCatgIdx = cols.findIndex(c => c.data === 'Purchase_Catg');
        const catgCellDef = (purchaseCatgIdx >= 0) ? [{
        targets: purchaseCatgIdx,
        createdCell: (td, cellData) => {
                td.classList.remove('catg-EMS','catg-SAP','catg-MRO','catg-STG');
                if (!cellData) return;

                const key = String(cellData).trim();

                if (key === 'EMS') {
                    td.classList.add('catg-EMS');
                }
                else if (key === 'SAP') {
                    td.classList.add('catg-SAP');
                }
                else if (key === 'MRO') {
                    td.classList.add('catg-MRO');
                }
                else if (key === 'STG') {
                    td.classList.add('catg-STG');
                }
                else if (key === '재계약') {
                    td.classList.add('catg-CONTRACT');
                }
                else if (key === '채번중') {
                    td.classList.add('catg-NUMBERING');
                }
            }
        }] : [];

        const poRejectIdx = cols.findIndex(c => c.data === 'PO_Reject');
        const poRejectCellDef = (poRejectIdx >= 0) ? [{
            targets: poRejectIdx,
            className: 'po-reject-cell',
            render: (d, t, row) => {
                if (t !== 'display') return d;
                const showHandle = state.isPrivileged && !!__purchaseSortable; // 편집(DnD) ON일 때만

                const HANDLE_SVG = `
                    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" class="row-handle in-po-reject">
                    <path d="M7 10l5-5 5 5M7 14l5 5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                    </svg>
                `.trim();
                const handle = showHandle ? `${HANDLE_SVG}` : '';
                const txt = esc(d ?? '');
                return `<span class="po-reject-text">${txt}</span>${handle}`;
            }
        }] : [];

        // 3) 초기화 또는 갱신
        if ($.fn.DataTable.isDataTable($table)) {
        buyDT.clear().rows.add(dtData).draw(false);
        buyDT.columns.adjust();
        buyDT.fixedHeader?.adjust?.();
        buyDT.fixedColumns().relayout?.();
        } else {
            buyDT = $table.DataTable({
                data: dtData,
                columns: cols,
                columnDefs: [...catgCellDef, ...poRejectCellDef],
                scrollY: '540px',
                scrollX: true,
                scrollCollapse: true,
                paging: false,
                searching: true,
                fixedHeader: true,
                orderCellsTop: true,
                info: false,
                dom: 't',
                ordering: true, // 0=Request_Date, 10=Purchase_status, 6=EMS재고
                order: [], // 초기순서는 서버 결과(p.No ASC) 그대로 유지
                createdRow: (row, data) => {
                const key = data._key;
                row.setAttribute('data-key', key);
                Array.from(row.cells).forEach(td => td.setAttribute('data-key', key));
                },
                rowCallback: (row, data) => {
                    const status = (data.Purchase_status || '').trim();
                    row.classList.remove('st-req', 'st-wait', 'st-drop', 'st-done', 'st-arrived');
                    switch (status) {
                        case '구매 요청': row.classList.add('st-req'); break;
                        case '구매 대기': row.classList.add('st-wait'); break;
                        case 'Drop':      row.classList.add('st-drop'); break;
                        case '구매 중': row.classList.add('st-done'); break;
                        case '입고 완료': row.classList.add('st-arrived'); break;
                        default: /* no-op */
                    }
                }
            });

            // 4) 행 선택 바인딩
            bindDTSelectable($table, {
                selectedSet: selectedKeys,
                getKey: (tr) => tr.getAttribute('data-key'),
                restoreOnDraw: false
            });

            // ✅ 바깥 클릭 시 전체 선택 해제: 공용 유틸 재사용
            wireOutsideDeselect($table, {
                selectedSet: selectedKeys,
                preserveSelectors: [
                'button',
                '.edit-group-btn',
                '#parts-purchase-btn',
                '#parts-purchase-btn-approve',
                '#parts-purchase-btn-reject',
                '#purchase-date-range',
                '.flatpickr-calendar',     // 달력 클릭 시 선택 유지
                '.flatpickr-day',          // 달력 내부
                '.search-btn'
                ]
            });

            // ✅ Q-Code 입력 대문자 자동 변환(델리게이션, 1회만 설치)
            if (!state.__upperWired) {
                state.__upperWired = true;
                $(state.container).on('input', 'td[data-col="Q_Code"] input.p-edit', function () {
                    this.value = this.value.toUpperCase();
                });
            }

        }
    }

    function renderTable(items) {
        if (!state.container) return;
        const filtered = applyFilters(items);
        finalizeBuyTable(filtered);
    }

    /* ---------------------------
        데이터 로딩
    --------------------------- */
    async function fetchItems() {
        if (state.isFetching) return;
        state.isFetching = true;
        try {
            // 날짜 범위(없으면 3개월 기본)
            const getDefault3MonthsRange = () => {
            const today = new Date();
            const y = today.getFullYear();
            const m = today.getMonth();
            const first = new Date(y, m - 2, 1);
            const pad = (n) => String(n).padStart(2, '0');
            return {
                start: `${first.getFullYear()}-${pad(first.getMonth()+1)}-01`,
                end:   `${y}-${pad(m+1)}-${pad(today.getDate())}`
            };
            };
            const range = (state.dateRange?.start && state.dateRange?.end)
                ? state.dateRange
                : getDefault3MonthsRange();
            const qs = new URLSearchParams({ start: range.start, end: range.end }).toString();
            const r = await fetch(`/parts_purchase_list?${qs}`);
 
            state.items = await r.json();

            document.dispatchEvent(new CustomEvent('parts:data-loaded', {
            detail: { which: 'purchase_buy', items: state.items }
            }));

            if (state.container) renderTable(state.items);

            // 🔒 표가 리프레시되었으니, 이전 편집 컨텍스트/플래그는 반드시 초기화
            buyEditCtxs = null;
            purchaseEditOn = false;
            selectedKeys.clear();
            state.pendingOrderNos = null; // ✅ 드래그 보류 순서 초기화
        } finally {
            state.isFetching = false;
        }
    }

    async function fetchItemsBySearch(searchType, keyword) {
        if (state.isFetching) return;
        state.isFetching = true;

        try {
            const qs = new URLSearchParams({
                searchType: searchType || '',
                keyword: keyword || ''
            }).toString();

            const r = await fetch(`/parts_purchase_list?${qs}`);
            state.items = await r.json();

            document.dispatchEvent(new CustomEvent('parts:data-loaded', {
                detail: { which: 'purchase_buy', items: state.items }
            }));

            if (state.container) renderTable(state.items);

            // 🔒 검색 결과로 표가 리프레시되었으니 편집/선택 상태 초기화
            buyEditCtxs = null;
            purchaseEditOn = false;
            selectedKeys.clear();
            state.pendingOrderNos = null;
        } finally {
            state.isFetching = false;
        }
    }

    /* ---------------------------
        초기화
    --------------------------- */
    async function init(container) {
        if (state.inited) return;
        state.container = container;

        await checkClientAuthority();
        buildUI(container);
        wireDateRange(container);
       
        await fetchItems();
        await fetchPurchaseSummaryAllTime(); // 요약 카드 갱신
        wireStatusButtons();   // 구매 승인, 반려
        wirePurchaseEditEvents();  // 편집 토글/행추가/삭제/저장
        wireCopyHotkey();   //  추가
        wireStatusShortcutIcons();
        wireSummaryCards();
        wireDupOnlyCheckbox(); // 추가
       

        state.inited = true;
    }

    /*----- 팝업 -----*/
   
    function wireStatusButtons() {
        const btnApprove = document.getElementById('parts-purchase-btn-approve');
        const btnReject  = document.getElementById('parts-purchase-btn-reject');

        const call = async (action) => {
            if (!state.isPrivileged) {
            return alert('권한이 없습니다. (관리자/구매담당자만 가능)');
            }

            const rows = getSelectedRows();
            if (rows.length === 0) {
            return alert('행을 선택해주세요.');
            }

            // ✅ 승인자 이름 확보 (없으면 authority 한번 더 조회)
            if (action === 'approve' && (!state.userName || state.userName === 'Unknown')) {
            try { await checkClientAuthority(); } catch (_) {}
            }

            try {
            const payload = { action, rows };

            // ✅ approve일 때만 approver 추가 전송
            if (action === 'approve') {
                payload.approver = state.userName || 'Unknown';
            }

            const res = await fetch('/parts_purchase/status-chg', {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error(await res.text().catch(() => '서버 오류'));
            const data = await res.json();

            await fetchItems();

            // ✅ 서버가 {updated, logged} 같이 주면 메시지도 더 정확히 가능
            if (action === 'approve') {
                alert(`${data.updated ?? 0}건 구매 대기로 변경했습니다.`);
            } else {
                alert(`${data.updated ?? 0}건 Drop으로 변경했습니다.`);
            }

            selectedKeys.clear();
            } catch (e) {
            console.error(e);
            alert('상태 변경 중 오류가 발생했습니다.');
            }
        };

        if (btnApprove && !btnApprove.__wired) {
            btnApprove.__wired = true;
            btnApprove.addEventListener('click', () => call('approve'));
        }
        if (btnReject && !btnReject.__wired) {
            btnReject.__wired = true;
            btnReject.addEventListener('click', () => call('reject'));
        }
    }

    function wirePurchaseEditEvents() {
        if (wirePurchaseEditEvents.__wired) return;
        wirePurchaseEditEvents.__wired = true;

        const wrap = document.querySelector('.parts-purchase-wrapper');
        if (!wrap) return;

        const btnToggle = wrap.querySelector('#add-parts-btn');      // 파츠 편집(패널 토글용)
        const box       = wrap.querySelector('.edit-btn-container'); // 내부 버튼 그룹
        const btnEdit   = wrap.querySelector('#edit-btn-3');         // 수정(→저장 토글)
        const btnCancel = wrap.querySelector('#edit-cancel-btn-3');  // 취소
        const btnDel    = wrap.querySelector('#edit-delete-btn-3');  // 삭제

        if (!btnToggle || !box || !btnEdit || !btnCancel || !btnDel) return;

        // 패널 show/hide + 메인 버튼 = 저장
        btnToggle.onclick = async () => {
            const isOpen = box.style.display !== 'none';

            if (!isOpen) {
                // 패널 열기: 메인 버튼은 '저장' 역할
                box.style.display = '';
                btnToggle.textContent = '저장';
                if (state.isPrivileged) enableDnD();
                return;
            }

        };

        // 수정 ↔ 저장 (단일 행 인라인)
        btnEdit.onclick = async () => {
            if (!buyDT) return;

            // 아직 편집 모드가 아니라면 → 단일 행 선택 체크 후 인풋 전환
            // 아직 편집 모드가 아니라면 → "선택된 모든 행"을 편집 인풋으로 전환
            if (!buyEditCtxs) {
                const nodes = buyDT.rows('.selected').nodes().toArray();
                if (nodes.length === 0) {
                alert('수정할 행을 선택해 주세요.');
                return;
                }

                const edited = [];
                let skipped_status = 0;
                let skipped_perm   = 0;
                let skipped_draft  = 0;

                for (const rowNode of nodes) {
                const rowApi = buyDT.row(rowNode);
                const before = { ...rowApi.data() };

                // 구매담당자 외에는 '구매 요청' 상태만 편집 허용
                if (!state.isPurchaser) {
                    const status = (before.Purchase_status || before.Purchase_Status || '').trim();
                    if (status !== '구매 요청') { skipped_status++; continue; }
                }
                // 권한 체크
                const editableMap = getEditableColumnsForRow(before);
                if (!editableMap.length) { skipped_perm++; continue; }
                // 드래프트 편집 금지
                if (rowNode.classList.contains('__draft')) { skipped_draft++; continue; }

                const beforeKey = {
                    Q_Code:    before.Q_Code,
                    Unit:      before.EQ_Unit,
                    P_N:       before.P_N,
                    EQP_Div:   before.EQ_Name,
                    Parts_Div: before.Parts_Name,
                    Requester: before.Requester
                };

                // 이 행을 인풋으로 전환
                rowToInputsPurchase(rowNode, editableMap);
                edited.push({ rowNode, rowApi, before, beforeKey });
                }

                if (edited.length === 0) {
                let msg = '수정 가능한 행이 없습니다.';
                if (skipped_status) msg += `\n- 상태 제한으로 제외: ${skipped_status}건`;
                if (skipped_perm)   msg += `\n- 권한 부족으로 제외: ${skipped_perm}건`;
                if (skipped_draft)  msg += `\n- 드래프트로 제외: ${skipped_draft}건`;
                alert(msg);
                return;
                }

                // ✅ 다중행 편집 컨텍스트 시작
                buyEditCtxs = edited;
                purchaseEditOn = true;               // 저장은 메인 버튼으로
                setPurchaseActionButtonsEnabled(false);

                if (skipped_status || skipped_perm || skipped_draft) {
                let msg = `선택 ${nodes.length}건 중 ${edited.length}건만 편집 모드로 전환했습니다.`;
                if (skipped_status) msg += `\n- 상태 제한 제외: ${skipped_status}건`;
                if (skipped_perm)   msg += `\n- 권한 부족 제외: ${skipped_perm}건`;
                if (skipped_draft)  msg += `\n- 드래프트 제외: ${skipped_draft}건`;
                alert(msg);
                }
                return;
            }

            // 이미 편집 모드라면 → 전체 저장(메인 저장 흐름 재사용)
            try {
                await onPurchaseSaveMinimal();
            } catch (e) {
                console.error(e);
                alert('저장 중 오류: ' + (e.message || e));
            }    
        };

        // 취소
        btnCancel.onclick = () => {

            if (buyEditCtxs && buyEditCtxs.length) {
                for (const ctx of buyEditCtxs) {
                    restoreRowFromDataPurchase(ctx.rowNode, ctx.before);
                }
                buyEditCtxs = null;
                purchaseEditOn = false;
            }
           
            box.style.display = 'none';
            btnToggle.textContent = '파츠 편집';
            setPurchaseActionButtonsEnabled(true);
            disableDnD();
        };

        // 삭제
        btnDel.onclick = async (e) => {
            // ✅ 다른 리스너로 전파 차단 (중복 알림 방지)
            if (e && typeof e.stopImmediatePropagation === 'function') {
                e.stopImmediatePropagation();
                e.preventDefault();
            }

            // ✅ 연속 클릭/이중 바인딩 대비 가드
            if (__buyDelBusy) return;
            __buyDelBusy = true;
            try {
                if (!buyDT) return;

                const nodes = buyDT.rows('.selected').nodes().toArray();
                if (nodes.length === 0) {
                alert('삭제할 행을 선택해 주세요.');
                return;
                }

                const rowsData = nodes.map(n => buyDT.row(n).data());
                const deletables = rowsData.filter(r => canDeleteRow(r));

                if (deletables.length === 0) {
                const hasNonPending = rowsData.some(r => {
                    const st = (r.Purchase_status || r.Purchase_Status || '').trim();
                    return st !== '구매 요청' && st !== '구매 대기';
                });

                if (hasNonPending) {
                    alert('구매 상태가 "구매 요청 or 구매 대기"일 때만 삭제가 가능합니다.');
                } else {
                    alert('해당 파츠 요청자만 삭제할 수 있습니다.');
                }
                return;
                }

                if (deletables.length !== rowsData.length) {
                if (!confirm(`선택 ${rowsData.length}건 중 ${deletables.length}건만 삭제 권한이 있습니다. 가능한 건만 삭제할까요?`)) return;
                } else {
                if (!confirm(`선택한 ${deletables.length}건을 삭제할까요?`)) return;
                }

                const payloadRows = deletables.map(r => {
                if (r.No != null) return { No: r.No };
                return {
                    where: {
                        Q_Code:    r.Q_Code ?? null,
                        Unit:      r.EQ_Unit ?? null,
                        P_N:       r.P_N ?? null,
                        EQP_Div:   r.EQ_Name ?? null,
                        Parts_Div: r.Parts_Name ?? null,
                        Requester: r.Requester ?? null,
                        Site:      r.Site ?? null,
                    }
                };
                });

                const res = await fetch('/parts_purchase/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rows: payloadRows, force: !!state.isPurchaser })
                });
                if (!res.ok) throw new Error(await res.text());

                alert('삭제되었습니다.');
                selectedKeys.clear();
                await PartsBuy.fetch();
                await fetchPurchaseSummaryAllTime(); // 요약 카드 갱신
            } catch (e2) {
                console.error(e2);
                alert('삭제 중 오류가 발생했습니다.\n' + (e2.message || e2));
            } finally {
                __buyDelBusy = false;
            }
        };

        document.addEventListener('parts-edit:save', async () => {
            const wrap = document.querySelector('.parts-purchase-wrapper');
            if (!wrap || wrap.getAttribute('aria-hidden') === 'true') return;

            try {
            await onPurchaseSaveMinimal(); // 내부에서 ok:true/false에 맞춰 result 쏨
            } catch (e) {
            console.error(e);
            // 혹시 내부에서 못 보냈다면 실패 이벤트라도 보냄 (버튼 '저장중…' 복구용)
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false, error: e?.message } }));
            }
        });
    }

    /** 편집모드 전환: 지정 컬럼만 input/textarea 로 바꾸고, 해제 시 텍스트 복원 */
    function setPurchaseEditMode(on) {
        purchaseEditOn = on;
        if (!buyDT) return;

        const editableCols = new Set([
            'Purchase_Catg', 'Req_Qty', 'Req_Comment', 'Manager_Comment',
            'Order_Qty', 'Order_Num', 'Order_Date', 'Arrival_Date',
            'ERP_PR', 'PO_Reject', 'Purchase_status'
        ]);

        $(state.container).find('.purchase-mgmt-table tbody tr').each(function () {
        const rowData = buyDT.row(this).data();
        if (!rowData) return;

        Array.from(this.cells).forEach((td, idx) => {
            const col = buyDT.column(idx).dataSrc();
            if (!col || !editableCols.has(col)) return;

            if (on) {
            td.setAttribute('data-col', col);
            td.innerHTML = editorForPurchase(col, rowData[col]);
            } else {
            const v = rowData[col] ?? '';
            td.textContent = (typeof v === 'number') ? String(v) : v;
            }
        });
        });
    }

    function editorForPurchase(col, val) {
        const v = String(val ?? '');
        const status = (rowData?.Purchase_status || rowData?.Purchase_Status || '').trim?.() || '';

        if (/Comment$/.test(col)) return `<textarea class="p-edit form-control" rows="1">${esc(v)}</textarea>`;
        if (/Date$/.test(col))    return `<input type="text" class="p-edit p-date form-control" value="${esc(v)}" placeholder="YYYY-MM-DD">`;
        if (col === 'Req_Qty' || col === 'PO_Qty') return `<input type="number" class="p-edit form-control" step="1" value="${esc(v)}">`;
        if (col === 'Purchase_Catg') {
            if (status === '구매 요청') return `<span class="p-readonly">${esc(v)}</span>`;
            const opts = ['EMS','MRO','SAP','재계약','채번중'].map(o => {
                const sel = o === v ? ' selected' : '';
                return `<option value="${esc(o)}"${sel}>${esc(o)}</option>`;
            }).join('');
            return `<select class="p-edit form-control">${opts}</select>`;
        }

        if (col === 'Purchase_status') {
            const opts = PURCHASE_STATUS_FLOW
            .filter(Boolean)
            .map(o => `<option value="${esc(o)}"${o===v?' selected':''}>${esc(o)}</option>`)
            .join('');
            return `<select class="p-edit form-control">${opts}</select>`;
        }

        if (col === 'Unit_Price') {
            // 콤마 입력 허용 → 저장 시 parseNumberLoose로 정규화
            return `<input type="text" class="p-edit form-control" value="${esc(v)}" >`;
        }

        return `<input type="text" class="p-edit form-control" value="${esc(v)}">`;
    }

    async function onPurchaseSaveMinimal() {
        if (!buyDT) return;

        const editedCount = $(state.container)
            .find('.purchase-mgmt-table tbody .p-edit').length;

        const hasReorder = Array.isArray(state.pendingOrderNos) && state.pendingOrderNos.length > 0;

        // 저장 버튼 핸들
        const saveBtn = document.querySelector('.parts-purchase-wrapper #edit-save-btn-3');
        const box = document.querySelector('.parts-purchase-wrapper .edit-btn-container');
        const mainBtn = document.querySelector('.parts-purchase-wrapper #add-parts-btn');

        if (editedCount === 0 && !hasReorder) {
            // ✅ 성공처럼 종료 + UI 원복
            setPurchaseEditMode(false);
            if (box) box.style.display = 'none';
            if (mainBtn) mainBtn.textContent = '파츠 편집';
            if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = '저장';   // ← "저장중..." 복원
            }
            setPurchaseActionButtonsEnabled(true);
            alert('저장되었습니다.');
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));
            buyEditCtxs = null; // ← 다중 컨텍스트 정리
            return;
        }

        try {
            // 행별로 변경 수집
            const byKey = new Map(); // key → { rowData, updates{} }

            $(state.container).find('.purchase-mgmt-table tbody tr').each(function () {
                const rowData = buyDT.row(this).data();
                if (!rowData) return;

                $(this).find('.p-edit').each(function () {
                const td  = this.closest('td');
                const col = td?.getAttribute('data-col');
                if (!col) return;

                let v = this.value ?? '';
                // 숫자 컬럼 통일 처리
                const numCols = new Set(['Req_Qty','PO_Qty','Unit_Price','Total_Price']);
                    if (numCols.has(col)) {
                    v = parseNumberLoose(v);
                } else {
                    v = String(v).trim();
                    if (col === 'Q_Code') v = v.toUpperCase();
                }


                // ✅ 달라진 경우에만 updates에 포함
                // 비교도 동일한 규칙으로: 숫자는 숫자끼리 비교, 그 외는 문자열 비교
                let before = rowData[col];
                if (numCols.has(col)) {
                    before = parseNumberLoose(before);
                    const bothNull = (v == null && before == null);
                    const changed = bothNull ? false : (v !== before);
                    if (!changed) return;
                } else {
                    before = (before == null ? '' : String(before).trim());
                    const changed = String(v) !== String(before);
                    if (!changed) return;
                }

                const key = rowData._key;
                if (!byKey.has(key)) byKey.set(key, { rowData, updates: {} });
                byKey.get(key).updates[col] = v;

                });
            });

            const payloads = [];
            for (const {rowData, updates} of byKey.values()) {
                // 1) 권한 필터
                let filtered = filterUpdatesByPermission(rowData, updates);
                if (!Object.keys(filtered).length) continue;

                // 2) 프론트 → DB 컬럼 매핑
                filtered = mapFrontendToDB(filtered);
                if (!Object.keys(filtered).length) continue;

                const p = { updates: filtered };
                if (rowData.No != null) {
                p.No = rowData.No;
                } else {
                p.where = {
                    Q_Code:    rowData.Q_Code ?? null,
                    Unit:      rowData.EQ_Unit ?? null,
                    P_N:       rowData.P_N ?? null,
                    EQP_Div:   rowData.EQ_Name ?? null,
                    Parts_Div: rowData.Parts_Name ?? null,
                    Requester: rowData.Requester ?? null,
                    Site:      rowData.Site ?? null,
                };
                }
                payloads.push(p);
            }

            // 셀 변경이 없더라도, 순서 변경이 있으면 저장 진행
            const hasCellChanges = payloads.length > 0;

            if (!hasCellChanges && !hasReorder) {
                alert('변경된 내용이 없습니다.');
                setPurchaseActionButtonsEnabled(true);
                document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok:false } }));
                return;
            }

            if (hasCellChanges) {
                const res = await fetch('/parts_purchase/bulk-update', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({ rows: payloads })
                });
                if (!res.ok) throw new Error(await res.text());
            }
            // ✅ 순서 확정: pendingOrderNos가 있으면 서버에 한번에 확정
            if (hasReorder) {
                const resp = await fetch('/parts_purchase/reorder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ orderedNos: state.pendingOrderNos })
            });
                const json = await resp.json().catch(()=> ({}));
                if (!resp.ok || json.ok === false) {
                  throw new Error(json?.error || 'reorder failed');
                }
            }

            // 편집 종료 + 새로고침
            setPurchaseEditMode(false);
            const box = document.querySelector('.parts-purchase-wrapper .edit-btn-container');
            const mainBtn = document.querySelector('.parts-purchase-wrapper #add-parts-btn');
            if (box) box.style.display = 'none';
            if (mainBtn) mainBtn.textContent = '파츠 편집';
            await PartsBuy.fetch();
            alert('저장되었습니다.');
            setPurchaseActionButtonsEnabled(true);
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: true } }));
            buyEditCtxs = null; // ← 저장 성공 후 정리
            disableDnD(); // ✅ 저장 후 DnD OFF
            state.pendingOrderNos = null; // ✅ 확정 후 클리어

        } catch (e) {
            console.error(e);
            alert('저장 중 오류: ' + (e.message || e));
            setPurchaseActionButtonsEnabled(true);
            document.dispatchEvent(new CustomEvent('parts-edit:result', { detail: { ok: false } }));
        } finally {
            // 혹시라도 남았을 로딩 UI 정리
            if (saveBtn) {
            saveBtn.disabled = false;
            saveBtn.textContent = '저장';
            }
            setPurchaseActionButtonsEnabled(true);
        }
    }

    // ✅ 날짜 범위 헬퍼: 3개월 기본/표시/달력/조회 버튼까지 한 번에
    function wireDateRange() {
        const input = document.getElementById('purchase-date-range');
        const btn   = document.getElementById('purchase-date-init-btn');
        if (!input) return;

        const getDefault3MonthsRange = () => {
            const today = new Date();
            const y = today.getFullYear();
            const m = today.getMonth();
            const first = new Date(y, m - 2, 1);                  // 두 달 전 1일
            const pad = (n) => String(n).padStart(2, '0');
            return {
            start: `${first.getFullYear()}-${pad(first.getMonth()+1)}-01`,
            end:   `${y}-${pad(m+1)}-${pad(today.getDate())}`
            };
        };

        const show = ({start, end}) => { input.value = `${start} ~ ${end}`; };

        // 초기 기본 범위(미설정 시)
        if (!state.dateRange?.start || !state.dateRange?.end) {
            state.dateRange = getDefault3MonthsRange();
        }
        show(state.dateRange);

        // ⬇️ flatpickr 인스턴스 붙이기 (+ 인스턴스 보관)
        if (window.flatpickr) {
            const fp = flatpickr(input, {
            mode: 'range',
            dateFormat: 'Y-m-d',
            rangeSeparator: ' ~ ',
            allowInput: false,
            clickOpens: true,
            defaultDate: [state.dateRange.start, state.dateRange.end],
            locale: flatpickr.l10ns?.ko || 'ko',
            // 날짜가 닫힐 때(선택 완료 후)에만 처리 → 2개 모두 선택한 경우 자동 조회
            onClose: (selectedDates) => {
                if (selectedDates && selectedDates.length === 2) {
                const pad = (n) => String(n).padStart(2, '0');
                const s = selectedDates[0], e = selectedDates[1];
                const start = `${s.getFullYear()}-${pad(s.getMonth()+1)}-${pad(s.getDate())}`;
                const end   = `${e.getFullYear()}-${pad(e.getMonth()+1)}-${pad(e.getDate())}`;
                state.dateRange = { start, end };
                show(state.dateRange);
                // ✅ 자동 조회
                fetchItems();
                }
            }
            });
            // 인스턴스 보관(초기화 버튼에서 사용)
            input.__fp = fp;
        }

        // “날짜 초기화” 버튼 → 기본 3개월 범위로 복원 + 바로 조회
        if (btn && !btn.__wired) {
            btn.__wired = true;
            btn.addEventListener('click', () => {
            state.dateRange = getDefault3MonthsRange();
            show(state.dateRange);

            // 캘린더에도 반영
            const toDate = (s) => new Date(`${s}T00:00:00`);
            if (input.__fp) {
                // setDate(date, triggerChange, format) — triggerChange=true로 이벤트 발생
                input.__fp.setDate([toDate(state.dateRange.start), toDate(state.dateRange.end)], true);
            }
            // ✅ 즉시 조회
            fetchItems();
            });
        }
    }

    function wireCopyHotkey() {
        if (wireCopyHotkey.__wired) return;
        wireCopyHotkey.__wired = true;

        document.addEventListener('keydown', (e) => {
            // 입력 UI에 포커스면 무시
            const ae = document.activeElement;
            const tag = (ae?.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || ae?.isContentEditable) return;

            // Ctrl/Cmd + C
            const isCopy = (e.key === 'c' || e.key === 'C') && (e.ctrlKey || e.metaKey);
            if (!isCopy) return;

            // 우리 테이블이 존재 & 선택행이 있을 때만 동작
            const hasSel = buyDT && buyDT.rows('.selected').data().toArray().length > 0;
            if (!hasSel) return;

            // 다른 복사 이벤트와 충돌 방지
            e.preventDefault();
            e.stopPropagation();

            copySelectedQCodeAndReason();
        }, true);
    }

    function wireStatusShortcutIcons() {
        if (wireStatusShortcutIcons.__wired) return;
        wireStatusShortcutIcons.__wired = true;

        // 상태 매핑
        const MAP = {
            'p_req.png'     : '구매 요청',
            'p_wait.png'    : '구매 대기',
            'p_done.png'    : '구매 중',
            'p_arrived.png' : '입고 완료'
        };

        // 컨테이너
        const box = document.querySelector('.edit-btn-container-4');
        if (!box) return;

        // 클릭 처리 (위임)
        box.addEventListener('click', (e) => {
            const img = e.target && e.target.tagName === 'IMG' ? e.target : null;
            if (!img) return;

            // 1) status 판별: data-status 우선, 없으면 파일명으로
            let status = img.dataset?.status;
            if (!status) {
            const src = img.getAttribute('src') || '';
            const file = src.split('/').pop(); // p_req.png 같은 파일명
            status = MAP[file];
            }
            if (!status) return; // 매핑 안 되면 무시

            // 2) 토글 동작: 이미 그 상태로만 필터링 중이면 '전체'로 초기화
            const cur = (PartsBuy?.getStatusFlow ? null : null); // 단순 보조 (없어도 됨)
            // 현재 lastFilters를 알 수 없으니, dropdown/hidden 값으로 체크
            const hidden = document.getElementById('purchase-status-values');
            const nowVals = (hidden?.value || '').split(',').filter(Boolean);
            const isSame = (nowVals.length === 1 && nowVals[0] === status);

            const nextStatuses = isSame ? [] : [status]; // 같은 걸 눌렀으면 전체로(해제), 아니면 해당 상태만

            // 3) 상태 멀티셀렉트 UI 갱신(있으면)
            if (typeof window.setCheckedStatusesToDropdown === 'function') {
            window.setCheckedStatusesToDropdown(nextStatuses);
            } else if (hidden) {
            hidden.value = nextStatuses.join(',');
            const label = document.querySelector('#purchase-status-multiselect .msel-label');
            if (label) {
                if (nextStatuses.length === 0) label.textContent = '상태 전체';
                else label.textContent = nextStatuses[0] + (nextStatuses.length > 1 ? ` 외 ${nextStatuses.length - 1}` : '');
            }
            }

            // 4) 실제 필터 적용 (다른 필터 유지하고 상태만 덮어쓰기)
            PartsBuy.searchWithFilters({ Purchase_Status: nextStatuses });

            // 5) 액티브 표시(선택된 아이콘 강조)
            box.querySelectorAll('img').forEach(el => el.classList.remove('active'));
            if (nextStatuses.length === 1) img.classList.add('active');
        });
    }

    function renderPurchaseSummaryCards(sum) {
        const box = document.querySelector('.group3-container .summary-container');
        if (!box) return;

        const req   = Number(sum?.req   ?? 0);
        const wait  = Number(sum?.wait  ?? 0);
        const doing = Number(sum?.doing ?? 0);

        const dup = countDuplicateQCodes(state.items);

        box.innerHTML = `
            <div class="summary-cards">
                <div class="summary-card st-req"  data-status="구매 요청">
                    <div class="label">구매 요청</div>
                    <div class="value">${req.toLocaleString()}</div>
                </div>
                <div class="summary-card st-wait" data-status="구매 대기">
                    <div class="label">구매 대기</div>
                    <div class="value">${wait.toLocaleString()}</div>
                </div>
                <div class="summary-card st-done" data-status="구매 중">
                    <div class="label">구매 중</div>
                    <div class="value">${doing.toLocaleString()}</div>
                </div>

                <div class="summary-card st-dup" data-dup="1">
                    <div class="label">중복 구매품</div>
                    <div class="value">${dup.toLocaleString()}</div>
                </div>
            </div>
        `;
    }

    async function fetchPurchaseSummaryAllTime() {
        try {
            // ✅ 날짜필터 없이 "전체" 집계만 내려주는 API
            const res = await fetch('/parts_purchase/summary');
            if (!res.ok) throw new Error(await res.text().catch(()=> 'summary fetch failed'));
            const json = await res.json();

            // 예: { req: 1, wait: 2, doing: 2 }
            renderPurchaseSummaryCards(json);
        } catch (e) {
            console.error(e);
            renderPurchaseSummaryCards({ req: 0, wait: 0, doing: 0 });
        }
    }

    function countDuplicateQCodes(items) {
        const TARGET = new Set(['구매 요청', '구매 대기', '구매 중']);

        const cntByQ = new Map();

        for (const it of (items || [])) {

            const st = String(it?.Purchase_Status ?? it?.Purchase_status ?? '').trim();
            if (!TARGET.has(st)) continue;

            const q = String(it?.Q_Code ?? '').trim().toUpperCase();
            if (!q) continue;

            cntByQ.set(q, (cntByQ.get(q) || 0) + 1);
        }

        // ✅ "중복 Q-Code의 종류 수" (count >= 2 인 Q_Code 개수)
        let dupKinds = 0;
        for (const c of cntByQ.values()) {
            if (c >= 2) dupKinds++;
        }
        return dupKinds;
    }

    function getDuplicateQCodeSet(items) {
        const TARGET = new Set(['구매 요청', '구매 대기', '구매 중']);
        const cntByQ = new Map();

        for (const it of (items || [])) {
            const st = String(it?.Purchase_Status ?? it?.Purchase_status ?? '').trim();
            if (!TARGET.has(st)) continue;

            const q = String(it?.Q_Code ?? '').trim().toUpperCase();
            if (!q) continue;

            cntByQ.set(q, (cntByQ.get(q) || 0) + 1);
        }

        const dupSet = new Set();
        for (const [q, c] of cntByQ.entries()) {
            if (c >= 2) dupSet.add(q);
        }
        return dupSet;
    }

    function wireDupOnlyCheckbox() {
        const cb = document.getElementById('purchase-dup-only');
        if (!cb || cb.__wired) return;
        cb.__wired = true;

        cb.addEventListener('change', () => {
            PartsBuy.searchWithFilters({
            ...state.lastFilters,
            dupOnly: cb.checked
            });
        });
    }

    function wireSummaryCards() {
        if (wireSummaryCards.__wired) return;
        wireSummaryCards.__wired = true;

        const host = document.querySelector('.group3-container .summary-container');
        if (!host) return;

        host.addEventListener('click', (e) => {
            const card = e.target?.closest?.('.summary-card');
            if (!card) return;

            const last   = PartsBuy.getLastFilters ? PartsBuy.getLastFilters() : (state.lastFilters || {});
            const hidden = document.getElementById('purchase-status-values');
            const label  = document.querySelector('#purchase-status-multiselect .msel-label');

            // ✅ 1) 중복 카드 클릭
            if (card.dataset.dup === '1') {
            const nextDup = !last.dupOnly;

            // 상태 전체로 초기화
            if (typeof window.setCheckedStatusesToDropdown === 'function') {
                window.setCheckedStatusesToDropdown([]); // 상태 전체
            } else {
                if (hidden) hidden.value = '';
                if (label)  label.textContent = '상태 전체';
            }

            PartsBuy.searchWithFilters({
                ...last,
                Purchase_Status: [],
                dupOnly: nextDup
            });

            // ✅ active 표시: 중복이 켜지면 dup 카드만 active
            host.querySelectorAll('.summary-card').forEach(el => el.classList.remove('active'));
            if (nextDup) card.classList.add('active');

            return;
            }

            // ✅ 2) 상태 카드 클릭 (구매 요청/대기/중 등)
            const status = card.dataset.status;
            if (!status) return;

            const nowVals = (hidden?.value || '').split(',').filter(Boolean);
            const isSameOne = (nowVals.length === 1 && nowVals[0] === status);

            // 같은 카드 재클릭이면 전체
            const nextStatuses = isSameOne ? [] : [status];

            // ✅ 상태 카드 누르면 중복 필터는 무조건 OFF
            const nextDupOnly = false;

            // UI 동기화
            if (typeof window.setCheckedStatusesToDropdown === 'function') {
                window.setCheckedStatusesToDropdown(nextStatuses);
            } else {
                if (hidden) hidden.value = nextStatuses.join(',');
                if (label)  label.textContent = nextStatuses.length ? nextStatuses[0] : '상태 전체';
            }

            PartsBuy.searchWithFilters({
            ...last,
            Purchase_Status: nextStatuses,
            dupOnly: nextDupOnly
            });

            // active 표시
            host.querySelectorAll('.summary-card').forEach(el => el.classList.remove('active'));
            if (nextStatuses.length === 1) card.classList.add('active');
        });
    }

    return {
        init,
        fetch: async () => {
            await fetchItems();
            await fetchPurchaseSummaryAllTime();
        },

        fetchBySearch: async (searchType, keyword) => {
            const kw = String(keyword || '').trim();

            // 검색어 없으면 기존 날짜 기준 조회로 복귀
            if (!kw) {
                await fetchItems();
                await fetchPurchaseSummaryAllTime();
                return;
            }

            await fetchItemsBySearch(searchType, kw);
            await fetchPurchaseSummaryAllTime();
        },

        searchWithFilters,
        getStatusFlow: () => PURCHASE_STATUS_FLOW.filter(Boolean),
        isPrivileged: () => !!state.isPrivileged,
        getDT,
        enableDnD,
        disableDnD,
        getLastFilters: () => ({ ...state.lastFilters })
    };


})();

window.PartsBuy = PartsBuy;