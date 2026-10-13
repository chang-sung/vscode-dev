async function openMotorHistoryPopup() {
    const popup = window.open("", "_blank", "width=1400,height=900");
    if (!popup) {
        alert("팝업이 차단되었습니다. 팝업 허용 후 다시 시도해주세요.");
        return;
    }

    popup.document.write(`
        <html>
        <head>
            <meta charset="UTF-8">
            <title>교체 이력 상세</title>
            <link rel="stylesheet" href="/css/07_mon_load/styles_motor_load_popup.css">
            <script src="/js/100_utils/jquery.min.js"></script>
            <script src="/js/100_utils/dataTables.min.js"></script>
            <script src="/js/100_utils/dataTable.fixedHeader.min.js"></script>
            <script src="/js/100_utils/dataTable.fixedColumns.min.js"></script>
            <script src="/js/100_utils/xlsx.full.min.js"></script>
        </head>
        <body>
            <div id="popup-loading-overlay">
                <div class="popup-loading-spinner"></div>
            </div>

            <h2 class="title-with-range">
                <span>교체 이력 상세</span>
                <span class="date-range" id="dateRangeText">전체 데이터 조회</span>
            </h2>

            <div class="top-controls">

                <div class="controls2-container">
                    <span class="controls-title">필터 선택</span>

                    <div class="filter-inline-container">
                        <div class="field">
                            <label for="siteFilter">Site</label>
                            <select id="siteFilter" class="popup-form-control">
                                <option value="ALL">전체</option>
                            </select>
                        </div>

                        <div class="field">
                            <label for="eqpFilter">설비</label>
                            <select id="eqpFilter" class="popup-form-control">
                                <option value="ALL">전체</option>
                            </select>
                        </div>

                        <div class="field">
                            <label for="eqStateFilter">점검결과</label>
                            <select id="eqStateFilter" class="popup-form-control">
                                <option value="ALL">전체</option>
                            </select>
                        </div>

                        <div class="field field-search-type">
                            <label for="keywordType">검색</label>
                            <select id="keywordType" class="popup-form-control">
                                <option value="all">전체</option>
                                <option value="Parts_Code">Q-Code</option>
                                <option value="Parts_Name">업체P/N</option>
                                <option value="Unit">Unit</option>
                                <option value="Work_Name">작업자</option>
                            </select>
                        </div>

                        <div class="field field-search-input">
                            <input type="text" id="keywordInput" class="popup-input-control keyword-wide" placeholder="검색어 입력">
                        </div>

                        <div class="field field-search-btn">
                            <label>&nbsp;</label>
                            <button id="keywordSearchBtn" class="popup-btn search-btn-inline">조회</button>
                        </div>
                        
                    </div>
                </div>

                <div class="controls3-container">
                    <span class="controls-title">엑셀 저장</span>
                    <button class="popup-btn excel-btn" onclick="downloadExcel()">엑셀 저장</button>
                </div>
            </div>

            <div class="total-count-wrapper">
                <span id="totalCount">총 0건</span>
            </div>

            <div class="table-scroll">
                <table id="motorHistoryTable">
                    <colgroup>
                        <col style="width:120px"> <!-- 작업시간 -->
                        <col style="width:50px">  <!-- Site -->
                        <col style="width:70px">  <!-- 설비그룹 -->
                        <col style="width:70px">  <!-- 설비 -->
                        <col style="width:120px"> <!-- Unit -->
                        <col style="width:120px"> <!-- Sub Unit -->
                        <col style="width:100px"> <!-- 파츠 분류 -->
                        <col style="width:150px"> <!-- 업체P/N -->
                        <col style="width:110px"> <!-- Q-Code -->
                        <col style="width:90px">  <!-- 파츠상태 -->
                        <col style="width:90px">  <!-- 점검결과 -->
                        <col style="width:200px"> <!-- 작업내용 -->
                        <col style="width:100px"> <!-- 작업자 -->
                    </colgroup>

                    <thead>
                        <tr>
                            <th class="sortable">작업시간</th>
                            <th class="sortable">Site</th>
                            <th class="sortable">설비그룹</th>
                            <th class="sortable">설비</th>
                            <th class="sortable">Unit</th>
                            <th class="sortable">Sub Unit</th>
                            <th class="sortable">파츠 분류</th>
                            <th class="sortable">업체P/N</th>
                            <th class="sortable">Q-Code</th>
                            <th class="sortable">파츠상태</th>
                            <th class="sortable">점검결과</th>
                            <th class="sortable">작업내용</th>
                            <th class="sortable">작업자</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <script>
                let originalRows = [];
                let filteredRows = [];
                let sortState = { key: 'Work_Time', asc: false };

                const colKeys = [
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
                    'Comment',
                    'Work_Name'
                ];

                function showLoading() {
                    document.getElementById("popup-loading-overlay").style.display = "flex";
                }

                function hideLoading() {
                    document.getElementById("popup-loading-overlay").style.display = "none";
                }

                function compareValues(a, b, key, asc) {
                    if (key === 'Work_Time') {
                        const da = new Date(a || 0);
                        const db = new Date(b || 0);
                        return asc ? da - db : db - da;
                    }

                    const sa = (a ?? '').toString();
                    const sb = (b ?? '').toString();
                    return asc ? sa.localeCompare(sb, 'ko') : sb.localeCompare(sa, 'ko');
                }

                function normalize(v) {
                    return (v ?? '').toString().trim();
                }

                function buildFilterOptions(rows) {
                    const eqpSet = Array.from(new Set(
                        rows.map(r => normalize(r.EQ_Name)).filter(Boolean)
                    )).sort((a, b) => a.localeCompare(b, 'ko'));

                    const siteSet = Array.from(new Set(
                        rows.map(r => normalize(r.Site)).filter(Boolean)
                    )).sort((a, b) => a.localeCompare(b, 'ko'));

                    const stateSet = Array.from(new Set(
                        rows.map(r => normalize(r.Eqp_State)).filter(Boolean)
                    )).sort((a, b) => a.localeCompare(b, 'ko'));

                    const eqpFilter = document.getElementById('eqpFilter');
                    const siteFilter = document.getElementById('siteFilter');
                    const eqStateFilter = document.getElementById('eqStateFilter');

                    eqpFilter.innerHTML =
                        '<option value="ALL">전체</option>' +
                        eqpSet.map(v => \`<option value="\${v}">\${v}</option>\`).join('');

                    siteFilter.innerHTML =
                        '<option value="ALL">전체</option>' +
                        siteSet.map(v => \`<option value="\${v}">\${v}</option>\`).join('');

                    eqStateFilter.innerHTML =
                        '<option value="ALL">전체</option>' +
                        stateSet.map(v => \`<option value="\${v}">\${v}</option>\`).join('');
                }

                function applyFilter() {
                    showLoading();

                    setTimeout(() => {
                        const eqpVal = document.getElementById('eqpFilter').value;
                        const siteVal = document.getElementById('siteFilter').value;
                        const eqStateVal = document.getElementById('eqStateFilter').value;
                        const keywordType = document.getElementById('keywordType').value;
                        const keywordInput = document.getElementById('keywordInput').value.trim().toLowerCase();

                        filteredRows = originalRows.filter(row => {
                            const matchEqp =
                                eqpVal === 'ALL' ||
                                normalize(row.EQ_Name) === eqpVal;

                            const matchSite =
                                siteVal === 'ALL' ||
                                normalize(row.Site) === siteVal;

                            const matchEqState =
                                eqStateVal === 'ALL' ||
                                normalize(row.Eqp_State) === eqStateVal;

                            let matchKeyword = true;
                            if (keywordInput) {
                                if (keywordType === 'all') {
                                    const values = [
                                        row.Work_Time, row.Site, row.EQ_Group, row.EQ_Name,
                                        row.Unit, row.Sub_Unit, row.Parts_Group, row.Parts_Name,
                                        row.Parts_Code, row.Parts_Grade,
                                        row.Eqp_State, row.Comment, row.Work_Name
                                    ].map(v => (v ?? '').toString().toLowerCase());

                                    matchKeyword = values.some(v => v.includes(keywordInput));
                                } else {
                                    const target = (row[keywordType] ?? '').toString().toLowerCase();
                                    matchKeyword = target.includes(keywordInput);
                                }
                            }

                            return matchEqp && matchSite && matchEqState && matchKeyword;
                        });

                        render(filteredRows);
                        hideLoading();
                    }, 10);
                }

                function render(rows) {
                    const enriched = [...rows].sort((a, b) => compareValues(
                        a[sortState.key], b[sortState.key], sortState.key, sortState.asc
                    ));

                    const tbody = document.querySelector('#motorHistoryTable tbody');
                    tbody.innerHTML = enriched.map(row => {
                        let bgColor = 'transparent';

                        if ((row.Eqp_State || '').trim() === '조치필요') {
                            bgColor = '#ffe3e3';
                        } else if ((row.Eqp_State || '').trim() === '평가중') {
                            bgColor = '#fff4d6';
                        }

                        return \`
                            <tr style="background-color:\${bgColor};">
                                <td>\${row.Work_Time || ''}</td>
                                <td>\${row.Site || ''}</td>
                                <td>\${row.EQ_Group || ''}</td>
                                <td>\${row.EQ_Name || ''}</td>
                                <td>\${row.Unit || ''}</td>
                                <td>\${row.Sub_Unit || ''}</td>
                                <td>\${row.Parts_Group || ''}</td>
                                <td>\${row.Parts_Name || ''}</td>
                                <td>\${row.Parts_Code || ''}</td>
                                <td>\${row.Parts_Grade || ''}</td>
                                <td>\${row.Eqp_State || ''}</td>
                                <td>\${row.Comment || ''}</td>
                                <td>\${row.Work_Name || ''}</td>
                            </tr>
                        \`;
                    }).join('');

                    document.getElementById('totalCount').innerText = \`총 \${enriched.length}건\`;
                }

                async function loadRows() {
                    showLoading();
                    try {
                        // ✅ 팝업 최초 로딩은 항상 전체 데이터 조회
                        const res = await fetch('/parts_replacement_history');
                        const json = await res.json();

                        if (!res.ok || !json.ok) {
                            throw new Error(json?.message || '교체 이력 조회 실패');
                        }

                        const allowedPartsGroups = ['모터', '볼스크류', '볼너트', '감속기', 'LM블럭', '로봇'];

                        originalRows = (Array.isArray(json.items) ? json.items : []).filter(row => {
                            const group = (row.Parts_Group || '').toString().trim();
                            return allowedPartsGroups.includes(group);
                        });

                        filteredRows = [...originalRows];

                        buildFilterOptions(originalRows);
                        render(filteredRows);
                    } catch (err) {
                        console.error('교체 이력 데이터 로딩 실패:', err);
                        originalRows = [];
                        filteredRows = [];
                        render([]);
                        alert('교체 이력을 불러오지 못했습니다.');
                    } finally {
                        hideLoading();
                    }
                }
                    
                function downloadExcel() {
                    const rows = filteredRows.map(row => ({
                        작업시간: row.Work_Time || '',
                        Site: row.Site || '',
                        설비그룹: row.EQ_Group || '',
                        설비: row.EQ_Name || '',
                        Unit: row.Unit || '',
                        SubUnit: row.Sub_Unit || '',
                        파츠분류: row.Parts_Group || '',
                        업체PN: row.Parts_Name || '',
                        QCode: row.Parts_Code || '',
                        파츠상태: row.Parts_Grade || '',
                        점검결과: row.Eqp_State || '',
                        작업내용: row.Comment || '',
                        작업자: row.Work_Name || ''
                    }));

                    const ws = XLSX.utils.json_to_sheet(rows);
                    const wb = XLSX.utils.book_new();
                    XLSX.utils.book_append_sheet(wb, ws, '교체이력');
                    XLSX.writeFile(wb, 'motor_change_history.xlsx');
                }

                document.getElementById('siteFilter').addEventListener('change', applyFilter);
                document.getElementById('eqStateFilter').addEventListener('change', applyFilter);
                document.getElementById('eqpFilter').addEventListener('change', applyFilter);
                document.getElementById('keywordSearchBtn').addEventListener('click', applyFilter);
                document.getElementById('keywordInput').addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') applyFilter();
                });

                document.querySelectorAll('#motorHistoryTable thead th.sortable').forEach((th, idx) => {
                    th.addEventListener('click', () => {
                        const key = colKeys[idx];
                        if (!key) return;

                        if (sortState.key === key) {
                            sortState.asc = !sortState.asc;
                        } else {
                            sortState.key = key;
                            sortState.asc = true;
                        }

                        document.querySelectorAll('#motorHistoryTable thead th.sortable')
                            .forEach(el => el.classList.remove('asc', 'desc'));

                        th.classList.add(sortState.asc ? 'asc' : 'desc');
                        render(filteredRows);
                    });
                });

                const defaultSortTh = document.querySelectorAll('#motorHistoryTable thead th.sortable')[0];
                if (defaultSortTh) defaultSortTh.classList.add('desc');

                (async function init() {
                    await loadRows();
                })();
            <\/script>
        </body>
        </html>
    `);

    popup.document.close();
}