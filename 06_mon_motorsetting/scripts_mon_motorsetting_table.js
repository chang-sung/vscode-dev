const selectedMotorRows = new Set(); // 모터 테이블 행 선택 상태

function getSpecByType(motorNum, pos, type, field, bound) {
    const key = `${motorNum}_${pos}`;
    return motorSpecGlobal?.[key]?.[type]?.[`${field}_${bound}`];
}

/* ---------------------------------------
   SPEC 헬퍼: EQ별 Type(Phone/Foldable) 반영
   - motorSpecGlobal[key][type][FIELD_bound] 구조용
--------------------------------------- */
function getSpecByEqType(eqNameAny, motorNum, pos, field, bound) {
  const type = window.getEqSpecType ? window.getEqSpecType(eqNameAny) : "phone";
  const key = `${motorNum}_${pos}`;
  return motorSpecGlobal?.[key]?.[type]?.[`${field}_${bound}`];
}

// === pos 유효성 헬퍼: motor 정의에 있는 pos만 허용 ===
function getAllowedPosSet(motorNum) {
    const info = window.motor?.[motorNum];
    if (!info || !info.pos) return new Set();
    // 키 타입 혼재 가능 → 문자열로 통일
    return new Set(Object.keys(info.pos).map(String));
}

function countAbnormalHighlightMotorsFromTbody(tbody) {
    let totalImportant = 0;
    let abnormalImportant = 0;

    const rows = tbody.querySelectorAll("tr");

    rows.forEach(row => {
        const displayEqName = row.querySelector(".eq-name")?.textContent;
        if (!displayEqName) return;

        // 화면표시명 -> 원본 EQ로 복구 (Foldable 판정/데이터 접근용)
        const rawEqName = window.EqNameUtils?.reverse ? EqNameUtils.reverse(displayEqName) : displayEqName;

        const motors = motorDataGlobal?.[rawEqName];
        if (!motors) return;

        for (const motorNum in motors) {
        const motorInfo = window.motor?.[motorNum];
        if (!motorInfo) continue;

        const motorValues = motors[motorNum];
        if (!motorValues) continue;

        // ✅ 헤더에 존재하는 pos 화이트리스트
        const allowedPos = getAllowedPosSet(motorNum);

        // ===== ACC (MODEL) pos=0 =====
        const model = motorValues?.[0]?.MODEL;
        const modelUpper = getSpecByEqType(rawEqName, motorNum, 0, "MODEL", "upper");
        const modelLower = getSpecByEqType(rawEqName, motorNum, 0, "MODEL", "lower");

        if (model !== undefined && modelUpper !== undefined && modelLower !== undefined && modelUpper !== modelLower) {
            totalImportant++;
            const val = parseInt(model);
            if (!Number.isNaN(val) && (val > modelUpper || val < modelLower)) {
            abnormalImportant++;
            }
        }

        // ===== DEC (FIX) pos=0 =====
        const fix = motorValues?.[0]?.FIX;
        const fixUpper = getSpecByEqType(rawEqName, motorNum, 0, "FIX", "upper");
        const fixLower = getSpecByEqType(rawEqName, motorNum, 0, "FIX", "lower");

        if (fix !== undefined && fixUpper !== undefined && fixLower !== undefined && fixUpper !== fixLower) {
            totalImportant++;
            const val = parseInt(fix);
            if (!Number.isNaN(val) && (val > fixUpper || val < fixLower)) {
            abnormalImportant++;
            }
        }

        // ===== SPEED (pos>=1, 헤더 pos만) =====
        for (const pos in motorValues) {
            if (pos === "0") continue;
            if (!allowedPos.has(String(pos))) continue;

            const speed = motorValues?.[pos]?.SPEED;
            const speedUpper = getSpecByEqType(rawEqName, motorNum, pos, "SPEED", "upper");
            const speedLower = getSpecByEqType(rawEqName, motorNum, pos, "SPEED", "lower");

            if (speed !== undefined && speedUpper !== undefined && speedLower !== undefined && speedUpper !== speedLower) {
            totalImportant++;

            const adjusted = (motorInfo.group === "TH")
                ? Math.floor(speed / 1000)
                : Math.floor(speed / 100);

            const val = parseInt(adjusted);
            if (!Number.isNaN(val) && (val > speedUpper || val < speedLower)) {
                abnormalImportant++;
            }
            }
        }
        }
    });

    return { totalImportant, abnormalImportant };
}


function countAllOutliersFromTbody(tbody) {
    let totalOutliers = 0;

    const rows = tbody.querySelectorAll("tr");

    rows.forEach(row => {
        const displayEqName = row.querySelector(".eq-name")?.textContent;
        if (!displayEqName) return;

        const rawEqName = window.EqNameUtils?.reverse ? EqNameUtils.reverse(displayEqName) : displayEqName;

        const motors = motorDataGlobal?.[rawEqName];
        if (!motors) return;

        for (const motorNum in motors) {
        const motorInfo = window.motor?.[motorNum];
        if (!motorInfo) continue;

        const motorValues = motors[motorNum];
        if (!motorValues) continue;

        const allowedPos = getAllowedPosSet(motorNum);

        // ===== ACC (MODEL) pos=0 =====
        const acc = motorValues?.[0]?.MODEL;
        const accUpper = getSpecByEqType(rawEqName, motorNum, 0, "MODEL", "upper");
        const accLower = getSpecByEqType(rawEqName, motorNum, 0, "MODEL", "lower");

        if (acc !== undefined && accUpper !== undefined && accLower !== undefined) {
            const val = parseInt(acc);
            if (!Number.isNaN(val) && (val > accUpper || val < accLower)) {
            totalOutliers++;
            }
        }

        // ===== DEC (FIX) pos=0 =====
        const dec = motorValues?.[0]?.FIX;
        const decUpper = getSpecByEqType(rawEqName, motorNum, 0, "FIX", "upper");
        const decLower = getSpecByEqType(rawEqName, motorNum, 0, "FIX", "lower");

        if (dec !== undefined && decUpper !== undefined && decLower !== undefined) {
            const val = parseInt(dec);
            if (!Number.isNaN(val) && (val > decUpper || val < decLower)) {
            totalOutliers++;
            }
        }

        // ===== SPEED (pos>=1, 헤더 pos만) =====
        for (const pos in motorValues) {
            if (pos === "0") continue;
            if (!allowedPos.has(String(pos))) continue;

            const speed = motorValues?.[pos]?.SPEED;
            const speedUpper = getSpecByEqType(rawEqName, motorNum, pos, "SPEED", "upper");
            const speedLower = getSpecByEqType(rawEqName, motorNum, pos, "SPEED", "lower");

            if (speed !== undefined && speedUpper !== undefined && speedLower !== undefined) {
            const adjusted = (motorInfo.group === "TH")
                ? Math.floor(speed / 1000)
                : Math.floor(speed / 100);

            const val = parseInt(adjusted);
            if (!Number.isNaN(val) && (val > speedUpper || val < speedLower)) {
                totalOutliers++;
            }
            }
        }
        }
    });

    return totalOutliers;
}

function renderMotorGroupTable(groupName, eqFilterList = null, visibleMotors = null) {
    const motorTableContainer = document.getElementById("motor-table-container");
    motorTableContainer.innerHTML = "";

    const motors = Object.entries(motor)
        .filter(([_, m]) => groupName === "ALL_GROUPS" || m.group === groupName)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));

    const table = document.createElement("table");
    table.classList.add("motor-table", "hover-row");
    table.id = "motorTable";

    const thead = createMotorTableHeader(motors, visibleMotors);
    const tbody = createMotorTableBody(motors, eqFilterList, visibleMotors);

    table.appendChild(thead);
    table.appendChild(tbody);

    finalizeMotorTable(table, tbody);
}

function createMotorTableHeader(motors, visibleMotors) {
    const thead = document.createElement("thead");

    // 모터 없을 때: 기본 헤더만
    if (
        motors.length === 0 ||
        (visibleMotors && motors.every(([motorNum]) => !visibleMotors.has(motorNum)))
    ) {
        const headerRow = document.createElement("tr");
        headerRow.innerHTML = `
        <th class="type-col">Type</th>
        <th class="eq-name">EQ_Name</th>
        <th class="write-time">WriteTime</th>
        <th class="remark-header">비고</th>
        `;
        thead.appendChild(headerRow);
        return thead;
    }

    /* ---------------------------
        SPEC 헬퍼 (Type: Phone/Foldable)
    --------------------------- */
    function getSpecValue(motorNum, pos, type, field, bound) {
        const key = `${motorNum}_${pos}`;
        return motorSpecGlobal?.[key]?.[type]?.[`${field}_${bound}`];
    }

    function specCellRangeHTML(motorNum, pos, type, field) {
        const lower = getSpecValue(motorNum, pos, type, field, "lower");
        const upper = getSpecValue(motorNum, pos, type, field, "upper");

        return `
            <td class="spec-cell spec-editable spec-${type.toLowerCase()} no-sort"
                data-type="${type}"
                data-motor="${motorNum}"
                data-pos="${pos}"
                data-field="${field}">
            <div class="spec-display">
                <span class="spec-lower" style="color:#0077cc;">${lower ?? "-"}</span>
                <span style="color:#999;"> / </span>
                <span class="spec-upper" style="color:#cc4444;">${upper ?? "-"}</span>
            </div>
            </td>
        `;
    }
    
    // 1줄째: 모터 그루핑 헤더
    const motorHeaderRow = document.createElement("tr");

    // ✅ 총 헤더 행 수: 4
    motorHeaderRow.innerHTML = `
        <th class="type-col" rowspan="4">Type</th>
        <th class="eq-name" rowspan="4">EQ_Name</th>
        <th class="write-time" rowspan="2">WriteTime</th>
    `;

    motors.forEach(([motorNum, m]) => {
        if (visibleMotors && !visibleMotors.has(motorNum)) return;

        const posCount = Object.keys(m.pos).length + 2;
        motorHeaderRow.innerHTML += `
        <th colspan="${posCount}">SM${motorNum}_${m.name}</th>
        `;
    });

    motorHeaderRow.innerHTML += `<th class="remark-header" rowspan="2">비고</th>`;
    thead.appendChild(motorHeaderRow);

    /* ---------------------------
        2줄째: ACC / DEC / Position 라벨
    --------------------------- */
    const posHeaderRow = document.createElement("tr");

    motors.forEach(([motorNum, m]) => {
        if (visibleMotors && !visibleMotors.has(motorNum)) return;

        posHeaderRow.innerHTML += `<th>ACC</th><th>DEC</th>`;

        Object.keys(m.pos)
        .sort((a, b) => Number(a) - Number(b))
        .forEach((pos) => {
            posHeaderRow.innerHTML += `<th>${m.pos[pos].name}</th>`;
        });
    });

    thead.appendChild(posHeaderRow);

    /* ---------------------------
        3~4줄째: SPEC (P/F) - 한 셀에 하한/상한
        P: Phone, F: Foldable
    --------------------------- */
    const specRowP = document.createElement("tr");
    const specRowF = document.createElement("tr");
    specRowP.classList.add("spec-p-row");
    specRowF.classList.add("spec-f-row");

    // ✅ P행 맨 앞에만 라벨 셀(rowspan=2) - thead이므로 th 사용
    specRowP.innerHTML = `
        <th class="spec-title-cell write-time" rowspan="2">
        <div class="spec-cell-inner">
            <div class="spec-label-left">SPEC</div>
            <div class="spec-label-right">
            <div style="font-weight:700;">Phone</div>
            <div style="font-weight:700; margin-top:6px;">Foldable</div>
            </div>
        </div>
        </th>
    `;
    specRowF.innerHTML = ``; // rowspan 때문에 F행 첫칸은 없음

    motors.forEach(([motorNum, m]) => {
        if (visibleMotors && !visibleMotors.has(motorNum)) return;

        // ACC/DEC pos=0
        specRowP.innerHTML += specCellRangeHTML(motorNum, 0, "phone", "MODEL");
        specRowP.innerHTML += specCellRangeHTML(motorNum, 0, "phone", "FIX");

        specRowF.innerHTML += specCellRangeHTML(motorNum, 0, "foldable", "MODEL");
        specRowF.innerHTML += specCellRangeHTML(motorNum, 0, "foldable", "FIX");

        // SPEED pos>=1
        Object.keys(m.pos)
        .sort((a, b) => Number(a) - Number(b))
        .forEach((pos) => {
            specRowP.innerHTML += specCellRangeHTML(motorNum, pos, "phone", "SPEED");
            specRowF.innerHTML += specCellRangeHTML(motorNum, pos, "foldable", "SPEED");
        });
    });

    // remark 셀은 P행에만, 2행 덮기
    specRowP.innerHTML += `<th class="remark-spec-cell" rowspan="2"></th>`;

    thead.appendChild(specRowP);
    thead.appendChild(specRowF);

    return thead;
}


function createMotorTableBody(motors, eqFilterList, visibleMotors) {
    const tbody = document.createElement("tbody");

    for (const eqName in motorDataGlobal) {
        if (eqFilterList && !eqFilterList.includes(eqName)) continue;

        const row = document.createElement("tr");
        row.setAttribute("data-key", eqName);

        const displayEqName = EqNameUtils.transform(eqName);
        const motorsMap = motorDataGlobal[eqName];
        const writeTime = motorsMap?.[1]?.[0]?.WriteTime?.split(".")[0] ?? "";

        // ✅ Type 표시 (foldable이면 Foldable)
        const type = window.getEqSpecType ? window.getEqSpecType(eqName) : "phone";
        const typeText = {phone: "Phone", foldable: "Foldable", watch: "Watch"}[type] ?? type;

        // ✅ Type 컬럼 추가
        row.innerHTML = `
        <td class="type-col">${typeText}</td>
        <td class="eq-name">${displayEqName}</td>
        <td class="write-time">${writeTime}</td>
        `;

        motors.forEach(([motorNum, m]) => {
        if (visibleMotors && !visibleMotors.has(motorNum)) return;
        row.append(...createMotorCells(eqName, displayEqName, motorNum, m, motorsMap?.[motorNum]));
        });

        row.innerHTML += `<td class="remark-cell"></td>`;
        tbody.appendChild(row);
    }

    return tbody;
}

function createMotorCells(rawEqName, displayEqName, motorNum, m, values) {
    const tds = [];

    // ACC (MODEL) pos=0
    tds.push(createMotorCell(rawEqName, displayEqName, motorNum, "ACC", 0, values?.[0]?.MODEL, "MODEL"));

    // DEC (FIX) pos=0
    tds.push(createMotorCell(rawEqName, displayEqName, motorNum, "DEC", 0, values?.[0]?.FIX, "FIX"));

    // SPEED pos>=1
    Object.keys(m.pos).sort((a, b) => Number(a) - Number(b)).forEach(pos => {
        const raw = values?.[pos]?.SPEED ?? "-";
        const val = raw === "-" ? "-" : (m.group === "TH" ? Math.floor(raw / 1000) : Math.floor(raw / 100));
        tds.push(createMotorCell(rawEqName, displayEqName, motorNum, pos, pos, val, "SPEED"));
    });

    return tds;
}

function createMotorCell(rawEqName, displayEqName, motorNum, posLabel, posNum, value, field) {
    const td = document.createElement("td");
    td.textContent = value ?? "-";
    td.dataset.motor = motorNum;
    td.dataset.pos = posLabel;

    // ✅ 코멘트 키는 기존 로직 유지(둘 다 맵에 있음)
    const commentKey = `${displayEqName}_${motorNum}_${posNum}`;
    if (commentMap[commentKey]) {
        const icon = document.createElement("span");
        icon.textContent = "📝";
        icon.title = commentMap[commentKey];
        icon.style.marginLeft = "4px";
        icon.style.cursor = "pointer";
        td.appendChild(icon);
    }

    // ✅ EQ별 SPEC Type 결정
    const type = window.getEqSpecType ? window.getEqSpecType(rawEqName) : "phone";

    const upper = getSpecByType(motorNum, posNum, type, field, "upper");
    const lower = getSpecByType(motorNum, posNum, type, field, "lower");

    if (upper !== undefined && lower !== undefined) {
        const isImportant = upper !== lower;

        if (isImportant) {
        td.style.backgroundColor = "#C0C0C0"; // 중요 위치 회색
        }

        if (value !== "-") {
        const val = parseInt(value);
        if (val > upper || val < lower) {
            if (isImportant) td.classList.add("cell-abnormal-important");
            else td.classList.add("cell-abnormal-general");
        }
        }
    }

    return td;
}

// DataTables는 스크롤 헤더를 별도 테이블로 복제하므로 wrapper에 offset을 공유한다.
function syncMotorStickyColumns(table, wrapper) {
    const row = table.tBodies[0]?.querySelector("tr");
    const typeCell = row?.querySelector(".type-col") || table.querySelector("th.type-col");
    const eqCell = row?.querySelector(".eq-name") || table.querySelector("th.eq-name");
    const typeWidth = typeCell?.getBoundingClientRect().width || 0;
    const eqWidth = eqCell?.getBoundingClientRect().width || 0;
    if (!typeWidth || !eqWidth) return;

    wrapper.style.setProperty("--motor-type-width", typeWidth + "px");
    wrapper.style.setProperty("--motor-eq-width", eqWidth + "px");
}

function finalizeMotorTable(table, tbody) {
    const container = document.getElementById("motor-table-container");
    container.innerHTML = "";

    const wrapper = document.createElement("div");
    wrapper.className = "motor-table-wrapper";
    wrapper.style.overflowX = "auto";
    wrapper.style.whiteSpace = "nowrap";
    wrapper.appendChild(table);
    container.appendChild(wrapper);

    // ✅ DataTable 적용 (안전하게 초기화)
    try {
        if ($.fn.DataTable.isDataTable("#motorTable")) {
            $("#motorTable").DataTable().destroy();
        }

        $("#motorTable").DataTable({
            scrollY: '610px',
            scrollX: true,
            scrollCollapse: true,
            paging: false,
            searching: false,
            fixedHeader: true,
            columnDefs: [
                { orderable: false, targets: 'no-sort' } // ✅ 추가
            ]
        });
    } catch (e) {
        console.error("❌ DataTable 초기화 오류:", e);
    }

    const syncSticky = () => syncMotorStickyColumns(table, wrapper);
    $(table).on("draw.dt.motorSticky column-sizing.dt.motorSticky", syncSticky);
    $(window).off("resize.motorSticky").on("resize.motorSticky", syncSticky);
    syncSticky();
    requestAnimationFrame(syncSticky);

    // ✅ 이상치/정상 비율 렌더링
    renderEqStatus(tbody);

    // ✅ 공용 선택 바인딩(Shift 구간, Ctrl/⌘ 토글, draw 후 복원)
    const $table = $('#motorTable');
    bindDTSelectable($table, {
        selectedSet: selectedMotorRows,
        getKey: (tr) => tr.getAttribute('data-key'),
        restoreOnDraw: true,       // 🔁 필터/정렬/재그림 후 선택 복원
        onChange: () => {}         // 필요시 선택 변경 후크
    });

    // ✅ 테이블 바깥 클릭 시 전체 해제(버튼/필터 UI는 제외)
    wireOutsideDeselect($table, {
        selectedSet: selectedMotorRows,
        preserveSelectors: ['button', '.status-btn', '.dt-filter', '.dataTables_filter']
    });


}

function renderEqStatus(tbodyElement) {
    const eqRows = tbodyElement.querySelectorAll("tr");
    const totalCount = eqRows.length;
    let abnormalCount = 0;

    eqRows.forEach(row => {
        const isAbnormalRow = Array.from(row.cells).some(td =>
        td.classList.contains("cell-abnormal-important") ||
        td.classList.contains("cell-abnormal-general")
        );
        if (isAbnormalRow) abnormalCount++;
    });

    const normalRate = totalCount === 0 ? 0 : Math.round(((totalCount - abnormalCount) / totalCount) * 100);
    const { totalImportant, abnormalImportant } = countAbnormalHighlightMotorsFromTbody(tbodyElement);
    const totalOutliers = countAllOutliersFromTbody(tbodyElement);
    const generalOutliers = totalOutliers - abnormalImportant;

    const statusHTML = `
        <div class="summary-cards motor-summary-cards">
            <div class="summary-card">
              <div class="label">전체설비</div>
              <div class="value">${totalCount}</div>
            </div>

            <div class="summary-card status-abnormal">
              <div class="label">이상설비</div>
              <div class="value">${abnormalCount}</div>
            </div>

            <div class="summary-card status-tttm">
              <div class="label">TTTM</div>
              <div class="value">${normalRate}%</div>
            </div>


             <div class="summary-card st-important">
              <div class="label">중요위치</div>
              <div class="value">${Number(totalImportant).toLocaleString()}</div>
            </div>

            <!-- ✅ 버튼 제거: 카드 자체가 클릭 대상 -->
            <div class="summary-card st-ab-imp summary-click" data-action="important" role="button" tabindex="0">
              <div class="label">이상치(중요)</div>
              <div class="value">${Number(abnormalImportant).toLocaleString()}</div>
            </div>

            <div class="summary-card st-ab-gen summary-click" data-action="general" role="button" tabindex="0">
              <div class="label">이상치(일반)</div>
              <div class="value">${Number(generalOutliers).toLocaleString()}</div>
            </div>
        </div>
    `;
    
    document.querySelector(".spec_mon_group").innerHTML = statusHTML;
}


function bindSpecCellSelection() {
    document.addEventListener("click", (e) => {
        const td = e.target.closest("td.spec-editable");
        if (!td) return;

        // ✅ DataTables 정렬/클릭 충돌 방지(헤더영역)
        e.preventDefault();
        e.stopPropagation();

        // ✅ 편집 중이면 선택 변경 막기
        if (isEditingSpec) return;

        const additive = !!(e.ctrlKey || e.metaKey);

        const { motor, pos, type, field } = td.dataset;
        const key = `${motor}_${pos}_${type}_${field}`;

        // ✅ Ctrl/⌘ 없이 클릭: 단일 선택/해제
        if (!additive) {
        // 이미 선택된 셀을 다시 클릭하면 → 전체 해제
        if (selectedSpecCells.has(key)) {
            clearSelectedSpecCells();
            return;
        }

        // 다른 셀 클릭이면 → 기존 해제 후 1개 선택
        clearSelectedSpecCells();
        selectedSpecCells.add(key);
        td.classList.add("spec-selected");
        return;
        }

        // ✅ Ctrl/⌘ 클릭: 다중 토글
        if (selectedSpecCells.has(key)) {
        selectedSpecCells.delete(key);
        td.classList.remove("spec-selected");
        } else {
        selectedSpecCells.add(key);
        td.classList.add("spec-selected");
        }
    }, true); // ✅ capture로 먼저 먹기
    }


document.addEventListener("DOMContentLoaded", bindSpecCellSelection);

function bindSummaryCardActions() {
    const host = document.querySelector(".spec_mon_group");
    if (!host) return;

    // ✅ 카드 클릭 (이벤트 위임)
    host.addEventListener("click", (e) => {
        const card = e.target.closest(".summary-card.summary-click");
        if (!card) return;

        const action = card.dataset.action;
        if (action === "abnormal-important") focusFirstOutlierCell("cell-abnormal-important");
        if (action === "abnormal-general") focusFirstOutlierCell("cell-abnormal-general");
    });

    // ✅ 키보드 접근성(Enter/Space)
    host.addEventListener("keydown", (e) => {
        const card = e.target.closest(".summary-card.summary-click");
        if (!card) return;
        if (e.key !== "Enter" && e.key !== " ") return;
        e.preventDefault();
        card.click();
    });
}

// ✅ 첫 이상치 셀로 스크롤 + 강조
function focusFirstOutlierCell(className) {
    const cell = document.querySelector(`#motorTable tbody td.${className}`);
    if (!cell) return;

    // 기존 강조 제거
    document.querySelectorAll("#motorTable td.scroll-focus").forEach(td => td.classList.remove("scroll-focus"));

    // 스크롤 대상(가로) - DataTables 스크롤 바디
    const scrollBody = document.querySelector(".dataTables_scrollBody");
    if (scrollBody) {
        // 가로 스크롤: 셀의 left 위치로 이동
        const left = cell.offsetLeft;
        scrollBody.scrollTo({ left: Math.max(0, left - 120), behavior: "smooth" });
    }

    // 세로 스크롤: 셀이 보이게
    cell.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });

    // 강조
    cell.classList.add("scroll-focus");
}

document.addEventListener("DOMContentLoaded", bindSummaryCardActions);