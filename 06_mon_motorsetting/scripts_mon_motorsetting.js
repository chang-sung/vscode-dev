// 전역 변수 정의
let motorDataGlobal = {};
let motorSpecGlobal = {};
let commentMap = {};
let eqTypeMap = {};

let currentEqFilterGroup = "all";
let currentEqTypeFilter = 'all'; // all | phone | foldable
let currentGroupName = "ALL_GROUPS";

let showOnlyAbnormal = false;
let showOnlyOutlier = false;
let isEditingSpec = false;
let currentUserInfo = null;

let currentIndexImportant = 0;
let currentIndexGeneral = 0;

let lastAppliedEqGroup = null; // 마지막으로 적용된 조(Shift) 기억

// 권한 체크 및 사용자 정보 표시
async function checkClientAuthority() {
    try {
        const response = await fetch('/check-client-authority');
        const data = await response.json();

        window.currentUserInfo = data;   // ✅ 전역 저장

        setAdminMode(data.isAdmin);

        const userNameContainer = document.createElement('div');
        userNameContainer.id = 'user-name-display';

        const userNameBox = document.createElement('div');
        userNameBox.id = 'user-name-box';
        userNameBox.innerText = `${data.authority} 모드 ( ${data.userName} )`;

        userNameContainer.appendChild(userNameBox);
        document.body.prepend(userNameContainer);
    } catch (error) {
        console.error('Error checking client authority:', error);
    }
}

function setAdminMode(isAdmin) {
    // 기존 버튼 (예: 공지사항 작성 버튼)
    const writeButton = document.getElementById('notice-write-button');
    if (writeButton) writeButton.style.display = isAdmin ? 'inline-block' : 'none';

    // ✅ SPEC 변경 버튼도 관리자일 때만 표시
    const specEditBtn = document.getElementById('toggle-spec-edit-btn');
    if (specEditBtn) specEditBtn.style.display = isAdmin ? 'inline-block' : 'none';
}


function getBaseEqList() {
    return currentEqFilterGroup === "all"
        ? [].concat(...Object.values(groups))
        : (groups[currentEqFilterGroup] || []);
}

// ✅ 설비 선택 드롭다운 채우기
function rebuildEqpSelect(eqList) {
    const select = document.getElementById('eqp-select');
    if (!select) return;

    const prev = select.value || 'all';
    select.innerHTML = '<option value="all">EQP 전체</option>';

    // EqNameUtils.transform 이 없을 때 대비
    const safeTransform = (eq) =>
        (window.EqNameUtils && typeof EqNameUtils.transform === 'function')
            ? EqNameUtils.transform(eq)
            : eq;

    const list = Array.isArray(eqList) ? eqList : [];
    const sorted = [...list].sort((a, b) =>
        safeTransform(a).localeCompare(safeTransform(b), 'ko')
    );

    const frag = document.createDocumentFragment();
    for (const eq of sorted) {
        const opt = document.createElement('option');
        opt.value = eq;                 // 원본 키 유지
        opt.textContent = safeTransform(eq); // 표기명
        frag.appendChild(opt);
    }
    select.appendChild(frag);

    // 이전 선택 복구 (목록에 있으면 유지)
    select.value = sorted.includes(prev) ? prev : 'all';
}

function initShiftFromQuery() {
    const sp = new URLSearchParams(location.search);
    const shift = (sp.get('shift') || '').toLowerCase();

    // 유효한 값만 허용
    const valid = ['a', 'b', 'c', 'd', 'all'];
    if (!valid.includes(shift)) return;

    const select = document.getElementById('eq-group-select');
    if (!select) return;

    // 드롭다운 선택 적용
    select.value = shift;

    // 내부 상태 갱신
    currentEqFilterGroup = shift;
    lastAppliedEqGroup = null; // eqp-select 초기화 유도

    // 데이터 로딩 후 필터 적용
    const applyWhenReady = () => {
        if (Object.keys(motorDataGlobal || {}).length > 0) {
        applyCombinedFilters();
        } else {
        setTimeout(applyWhenReady, 200); // 데이터 준비될 때까지 반복
        }
    };
    applyWhenReady();
}

// 초기 실행
document.addEventListener("DOMContentLoaded", () => {
    checkClientAuthority();
    loadAllMotorData();

    // ✅ 항상 표시
    document.querySelector(".unit-dropdown-wrapper").style.display = "inline-block";
    document.getElementById("important-checkbox").closest("label").style.display = "inline-block";

    // ✅ 페이지 로드시 '전체보기' 버튼 기본 active 처리
    const allBtn = document.querySelector('.button_group .bay-group-btn.all-groups-btn');
    if (allBtn) {
        allBtn.classList.add('active');
    }

    initShiftFromQuery();
});

// SPEC 편집 버튼
document.getElementById("toggle-spec-edit-btn").addEventListener("click", toggleSpecEditMode);

// 그룹 버튼 클릭
document.querySelectorAll(".bay-group-btn").forEach(btn => {
    btn.addEventListener("click", function () {
        showLoading(); // ✅ 로딩 오버레이 시작

        const group = this.getAttribute("data-group");
        currentGroupName = group;

        document.querySelectorAll(".bay-group-btn").forEach(b => b.classList.remove("active"));
        this.classList.add("active");

        // ✅ 체크박스 초기화
        document.getElementById("abnormal-checkbox").checked = false;
        document.getElementById("outlier-checkbox").checked = false;
        document.getElementById("important-checkbox").checked = false;

        // ✅ 상태 변수도 초기화
        showOnlyAbnormal = false;
        showOnlyOutlier = false;


        setTimeout(() => {
            resetSpecEditButton();

            const eqListBase = currentEqFilterGroup === "all"
                ? [].concat(...Object.values(groups))
                : groups[currentEqFilterGroup] || [];

            const filteredList = showOnlyAbnormal
                ? eqListBase.filter(eq => {
                    const motors = motorDataGlobal[eq];
                    return motors && eqpAbnormal(eq, motors);
                })
                : eqListBase;

            renderMotorGroupTable(group, filteredList);
            applyCombinedFilters();

            const dropdownWrapper = document.querySelector(".unit-dropdown-wrapper");
            const importantCheckbox = document.getElementById("important-checkbox");

            /* 필터링 옵션 축 서치 드롭다운 박스 숨기기
            if (group === "ALL_GROUPS") {
                dropdownWrapper.style.display = "inline-block";
                importantCheckbox.closest("label").style.display = "inline-block";
            } else {
                dropdownWrapper.style.display = "none";
                importantCheckbox.closest("label").style.display = "none";
            }
            */

            hideLoading(); // ✅ 로딩 오버레이 종료
        }, 50); // 약간의 지연을 주어 로딩 UI가 먼저 렌더링되게 함
    });
});


function getOutlierMotorsFromEqList(eqList) {
  const outlierMotors = new Set();

  eqList.forEach(eq => {
    const motors = motorDataGlobal[eq];
    if (!motors) return;

    const abnormalMotors = motorAbnormal(eq, motors); // ✅ eqName 전달
    abnormalMotors.forEach(mNum => outlierMotors.add(mNum));
  });

  return outlierMotors;
}

function getImportantMotors() {
    const importantMotors = new Set();

    for (const key in motorSpecGlobal) {
        const entry = motorSpecGlobal[key];
        if (!entry) continue;

        const [motorNum, pos] = key.split("_");
        const posNum = String(pos);

        // ✅ phone/foldable 둘 다 검사 (둘 중 하나라도 upper/lower 다르면 중요축)
        const typesToCheck = ["phone", "foldable", "watch"];

        for (const type of typesToCheck) {
        const spec = entry?.[type];
        if (!spec) continue;

        if (posNum === "0") {
            // MODEL
            const mu = spec.MODEL_upper, ml = spec.MODEL_lower;
            if (mu !== undefined && ml !== undefined && mu !== ml) {
            importantMotors.add(String(motorNum));
            break;
            }
            // FIX
            const fu = spec.FIX_upper, fl = spec.FIX_lower;
            if (fu !== undefined && fl !== undefined && fu !== fl) {
            importantMotors.add(String(motorNum));
            break;
            }
        } else {
            // SPEED
            const su = spec.SPEED_upper, sl = spec.SPEED_lower;
            if (su !== undefined && sl !== undefined && su !== sl) {
            importantMotors.add(String(motorNum));
            break;
            }
        }
        }
    }

    return importantMotors;
}
/* ---------------------------------------
   특정 EQ가 이상인지 판단 (행 필터용)
   - currentGroupName(ALL_GROUPS/그룹) 반영
   - pos 화이트리스트(헤더에 있는 pos만) 반영
   - EQ별 spec type(Phone/Foldable) 반영
--------------------------------------- */
function eqpAbnormal(eqName, motors) {
    if (!motors) return false;

    for (const motorNum in motors) {
        const motorInfo = window.motor?.[motorNum];
        if (!motorInfo) continue;

        // 그룹 필터 반영
        if (currentGroupName !== "ALL_GROUPS" && motorInfo.group !== currentGroupName) continue;

        const values = motors[motorNum];
        if (!values) continue;

        const allowedPos = getAllowedPosSet(motorNum);

        // ---- ACC (MODEL) pos=0 ----
        const model = values?.[0]?.MODEL;
        const modelUpper = getSpecByEqType(eqName, motorNum, 0, "MODEL", "upper");
        const modelLower = getSpecByEqType(eqName, motorNum, 0, "MODEL", "lower");

        if (model !== undefined && modelUpper !== undefined && modelLower !== undefined) {
        const val = parseInt(model);
        if (!Number.isNaN(val) && (val > modelUpper || val < modelLower)) return true;
        }

        // ---- DEC (FIX) pos=0 ----
        const fix = values?.[0]?.FIX;
        const fixUpper = getSpecByEqType(eqName, motorNum, 0, "FIX", "upper");
        const fixLower = getSpecByEqType(eqName, motorNum, 0, "FIX", "lower");

        if (fix !== undefined && fixUpper !== undefined && fixLower !== undefined) {
        const val = parseInt(fix);
        if (!Number.isNaN(val) && (val > fixUpper || val < fixLower)) return true;
        }

        // ---- SPEED pos>=1 (헤더 pos만) ----
        for (const pos in values) {
        if (pos === "0") continue;
        if (!allowedPos.has(String(pos))) continue;

        const speed = values?.[pos]?.SPEED;
        const speedUpper = getSpecByEqType(eqName, motorNum, pos, "SPEED", "upper");
        const speedLower = getSpecByEqType(eqName, motorNum, pos, "SPEED", "lower");

        if (speed !== undefined && speedUpper !== undefined && speedLower !== undefined) {
            const adjusted =
            motorInfo.group === "TH" ? Math.floor(speed / 1000) : Math.floor(speed / 100);

            const val = parseInt(adjusted);
            if (!Number.isNaN(val) && (val > speedUpper || val < speedLower)) return true;
        }
        }
    }

    return false;
}

/* ---------------------------------------
   motors(설비 1대의 전체 모터맵)에서
   이상 모터 번호 Set 반환 (이상치 컬럼/버튼용)
   - currentGroupName은 여기서 보통 "전체" 기준이지만,
     필요하면 동일하게 걸 수 있음(아래 주석 참고)
--------------------------------------- */
function motorAbnormal(eqName, motors) {
    const abnormalMotors = new Set();
    if (!motors) return abnormalMotors;

    for (const motorNum in motors) {
        const motorInfo = window.motor?.[motorNum];
        if (!motorInfo) continue;

        // ✅ 필요하면 그룹 필터까지 같이 적용하고 싶을 때만 사용
        // if (currentGroupName !== "ALL_GROUPS" && motorInfo.group !== currentGroupName) continue;

        const values = motors[motorNum];
        if (!values) continue;

        const allowedPos = getAllowedPosSet(motorNum);

        // ---- ACC (MODEL) pos=0 ----
        const model = values?.[0]?.MODEL;
        const modelUpper = getSpecByEqType(eqName, motorNum, 0, "MODEL", "upper");
        const modelLower = getSpecByEqType(eqName, motorNum, 0, "MODEL", "lower");

        if (model !== undefined && modelUpper !== undefined && modelLower !== undefined) {
        const val = parseInt(model);
        if (!Number.isNaN(val) && (val > modelUpper || val < modelLower)) {
            abnormalMotors.add(String(motorNum));
            continue;
        }
        }

        // ---- DEC (FIX) pos=0 ----
        const fix = values?.[0]?.FIX;
        const fixUpper = getSpecByEqType(eqName, motorNum, 0, "FIX", "upper");
        const fixLower = getSpecByEqType(eqName, motorNum, 0, "FIX", "lower");

        if (fix !== undefined && fixUpper !== undefined && fixLower !== undefined) {
        const val = parseInt(fix);
        if (!Number.isNaN(val) && (val > fixUpper || val < fixLower)) {
            abnormalMotors.add(String(motorNum));
            continue;
        }
        }

        // ---- SPEED pos>=1 ----
        for (const pos in values) {
        if (pos === "0") continue;
        if (!allowedPos.has(String(pos))) continue;

        const speed = values?.[pos]?.SPEED;
        const speedUpper = getSpecByEqType(eqName, motorNum, pos, "SPEED", "upper");
        const speedLower = getSpecByEqType(eqName, motorNum, pos, "SPEED", "lower");

        if (speed !== undefined && speedUpper !== undefined && speedLower !== undefined) {
            const adjusted =
            motorInfo.group === "TH" ? Math.floor(speed / 1000) : Math.floor(speed / 100);

            const val = parseInt(adjusted);
            if (!Number.isNaN(val) && (val > speedUpper || val < speedLower)) {
            abnormalMotors.add(String(motorNum));
            break;
            }
        }
        }
    }

    return abnormalMotors;
}

// EQ 그룹(Shift조) 선택 변경 핸들러
document.getElementById("eq-group-select").addEventListener("change", function () {
    currentEqFilterGroup = this.value;   // 현재 조 갱신
    applyCombinedFilters();              // 스피너는 내부에서 처리
});

document.getElementById('eqp-select').addEventListener('change', () => {
    applyCombinedFilters();
});

document.getElementById("abnormal-checkbox").addEventListener("change", function () {
    showOnlyAbnormal = this.checked;
    applyCombinedFilters();
});

document.getElementById("outlier-checkbox").addEventListener("change", function () {
    showOnlyOutlier = this.checked;
    applyCombinedFilters();
});

document.getElementById("important-checkbox").addEventListener("change", function () {
    applyCombinedFilters();
});

// 설비 구분(phone / foldable) 선택 변경
document.getElementById("type-select")?.addEventListener("change", function () {
    currentEqTypeFilter = this.value;  // all | phone | foldable
    applyCombinedFilters();
});

function applyCombinedFilters() {
    showLoading();   // ✅ 스피너 시작

    setTimeout(() => {
        // ⬇️ 조(Shift) 바뀐 경우에만 설비 드롭다운 초기화
        if (lastAppliedEqGroup !== currentEqFilterGroup) {
            const eqpSelect = document.getElementById('eqp-select');
            if (eqpSelect) eqpSelect.value = 'all';
            lastAppliedEqGroup = currentEqFilterGroup;
        }

        // 1) Shift(조) 기준 베이스 EQ 리스트
        let filteredEqList = getBaseEqList();

        // 2) "이상설비만 보기" 적용
        if (showOnlyAbnormal) {
            filteredEqList = filteredEqList.filter(eq => {
                const motors = motorDataGlobal[eq];
                return motors && eqpAbnormal(eq, motors);
            });
        }

        // 3) 설비 선택 드롭다운 갱신
        rebuildEqpSelect(filteredEqList);

        // 4) eqp-select 선택값 반영
        const selectedEq = document.getElementById('eqp-select')?.value || 'all';
        if (selectedEq !== 'all') {
            if (!filteredEqList.includes(selectedEq)) {
                document.getElementById('eqp-select').value = 'all';
            } else {
                filteredEqList = [selectedEq];
            }
        }

        // 5) 컬럼 가시성(중요축/이상치)
        let visibleMotors = null;
        if (document.getElementById("important-checkbox").checked) {
            visibleMotors = getImportantMotors();
        }
        if (showOnlyOutlier) {
            const outlierMotors = getOutlierMotorsFromEqList(filteredEqList);
            visibleMotors = visibleMotors
                ? new Set([...visibleMotors].filter(m => outlierMotors.has(m)))
                : outlierMotors;
        }

        if (currentEqTypeFilter !== 'all') {
            filteredEqList = filteredEqList.filter(eq =>
                getEqSpecType(eq) === currentEqTypeFilter
            );
        }

        // 6) 테이블 렌더
        renderMotorGroupTable(currentGroupName, filteredEqList, visibleMotors);

        hideLoading();   // ✅ 렌더 끝나면 스피너 종료
    }, 10);
}

// 이상치(중요/일반) : 버튼이든 카드든 동일 동작
document.addEventListener("click", function (e) {
    // 1) 기존 버튼(혹시 남아있으면)
    if (e.target.classList.contains("btn-abnormal-important")) {
        scrollToNextHighlightedCell("important");
        return;
    }
    if (e.target.classList.contains("btn-abnormal-general")) {
        scrollToNextHighlightedCell("general");
        return;
    }

    // 2) ✅ 카드 클릭
    const card = e.target.closest(".summary-card.summary-click");
    if (!card) return;

    const type = card.dataset.action; // "important" | "general"
    if (type === "important" || type === "general") {
        scrollToNextHighlightedCell(type);
    }
});

// ✅ 공통 스크롤 함수
function scrollToFirstHighlightedCell(type) {
    const selector = type === "important"
        ? ".cell-abnormal-important"
        : ".cell-abnormal-general";

    const cells = Array.from(document.querySelectorAll(selector));
    if (cells.length === 0) return;

    const index = type === "important" ? currentIndexImportant : currentIndexGeneral;

    const cell = cells[index % cells.length];
    cell.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });

    // 강조 효과 주기
    cell.classList.add("scroll-focus");
    setTimeout(() => {
        cell.classList.remove("scroll-focus");
    }, 1000);

    if (type === "important") {
        currentIndexImportant++;
    } else {
        currentIndexGeneral++;
    }
}

function scrollToNextHighlightedCell(type) {
    const tableWrapper = document.querySelector(".dataTables_scrollBody");

    const selector = type === "important"
        ? "#motorTable td.cell-abnormal-important"
        : "#motorTable td.cell-abnormal-general";

    const cells = Array.from(document.querySelectorAll(selector));

    if (cells.length === 0) return;

    let index;
    if (type === "important") {
        index = currentIndexImportant;
    } else {
        index = currentIndexGeneral;
    }

    const cell = cells[index % cells.length];
    const left = cell.offsetLeft;
    const top = cell.offsetTop;

    tableWrapper.scrollTo({
        top: top - 100,
        left: left - 100,
        behavior: "smooth"
    });

    cell.classList.add("scroll-focus");
    setTimeout(() => cell.classList.remove("scroll-focus"), 1500);

    if (type === "important") {
        currentIndexImportant = (index + 1) % cells.length;
    } else {
        currentIndexGeneral = (index + 1) % cells.length;
    }
}

function getAbnormalCells(type) {
    const selector = type === "important"
        ? ".cell-abnormal-important"
        : ".cell-abnormal-general";

    return Array.from(document.querySelectorAll(selector));
}

/* ---------------------------
   공용: DataTables 행 선택 토글 바인딩
--------------------------- */
function bindDTSelectable($table, {
    selectedSet,
    getKey,
    restoreOnDraw = false,
    onChange
    } = {}) {
    if (!$table || typeof getKey !== 'function') return;

    const tableEl = $table[0];
    const $tbody  = $table.find('tbody');

    // 🔹 (추가) Shift 클릭 시 텍스트 블럭 생성 방지
    $tbody
        .off('mousedown.__rowselect')
        .on('mousedown.__rowselect', 'tr', function (e) {
        if (e.shiftKey && !$(e.target).is('input,textarea,select,button,a')) {
            e.preventDefault();              // 텍스트 드래그 방지
        }
        });

    // (기존) anchor 저장
    const getLastAnchor = () => (tableEl.__lastAnchorIndex ?? null);
    const setLastAnchor = (idx) => { tableEl.__lastAnchorIndex = (typeof idx === 'number' ? idx : null); };

    const indexOfRow = (tr) => $tbody.find('tr').index(tr);
    const rowsBetween = (a, b) => {
        const $rows = $tbody.find('tr');
        const i1 = Math.min(a, b), i2 = Math.max(a, b);
        const arr = [];
        for (let i = i1; i <= i2; i++) arr.push($rows[i]);
        return arr;
    };

    $tbody
        .off('click.__rowselect')
        .on('click.__rowselect', 'tr', function (e) {
        if ($(e.target).is('input,textarea,select,button,a')) return;

        const key = getKey(this);
        if (!key) return;

        const isCtrl  = !!(e.ctrlKey || e.metaKey);
        const isShift = !!e.shiftKey;

        // 🔹 (추가) 혹시 생긴 브라우저 텍스트 선택 즉시 해제
        if (isShift && window.getSelection) {
            const sel = window.getSelection();
            if (sel && sel.removeAllRanges) sel.removeAllRanges();
        }

        const $rows = $tbody.find('tr');
        const wasSelected = this.classList.contains('selected');
        const currIdx = indexOfRow(this);
        let changed = false;

        // ---- Shift 구간 선택 ----
        if (isShift) {
            const last = getLastAnchor();

            if (last == null) {
            if (!isCtrl) {
                $rows.removeClass('selected');
                if (selectedSet) selectedSet.clear();
            }
            if (!wasSelected || isCtrl) {
                this.classList.add('selected');
                if (selectedSet) selectedSet.add(key);
                changed = true;
            }
            setLastAnchor(currIdx);
            if (onChange && changed) onChange(selectedSet);
            return;
            }

            const range = rowsBetween(last, currIdx);
            if (!isCtrl) {
            $rows.removeClass('selected');
            if (selectedSet) selectedSet.clear();
            }
            for (const tr of range) {
            const k = getKey(tr);
            if (!k) continue;
            if (!tr.classList.contains('selected')) {
                tr.classList.add('selected');
                if (selectedSet) selectedSet.add(k);
                changed = true;
            }
            }
            setLastAnchor(currIdx);
            if (onChange && changed) onChange(selectedSet);
            return;
        }

        // ---- Ctrl/⌘ 토글 ----
        if (isCtrl) {
            if (wasSelected) {
            this.classList.remove('selected');
            if (selectedSet) selectedSet.delete(key);
            } else {
            this.classList.add('selected');
            if (selectedSet) selectedSet.add(key);
            }
            setLastAnchor(currIdx);
            if (onChange) onChange(selectedSet);
            return;
        }

        // ---- 기본(단일) ----
        const selectedCount = $rows.filter('.selected').length;
        if (selectedCount === 1 && wasSelected) {
            this.classList.remove('selected');
            if (selectedSet) selectedSet.delete(key);
            setLastAnchor(null);
            if (onChange) onChange(selectedSet);
            return;
        }

        $rows.removeClass('selected');
        if (selectedSet) selectedSet.clear();
        this.classList.add('selected');
        if (selectedSet) selectedSet.add(key);
        setLastAnchor(currIdx);
        if (onChange) onChange(selectedSet);
        });

    // redraw 복원
    if (restoreOnDraw && selectedSet) {
        $table.off('draw.__rowselect').on('draw.__rowselect', function () {
        $table.find('tbody tr').each(function () {
            const k = getKey(this);
            if (k && selectedSet.has(k)) this.classList.add('selected');
            else this.classList.remove('selected');
        });
        });
    } else {
        $table.off('draw.__rowselect');
    }
}

function wireOutsideDeselect($table, {
        selectedSet,
        preserveSelectors = ['button']  // 버튼 클릭 시 유지
} = {}) {
    if (!$table || !$table.length) return;
    const tableEl = $table[0];
    if (tableEl.__outsideDeselectWired) return; // 중복 방지
    tableEl.__outsideDeselectWired = true;

    const handler = (e) => {
        // 1) 테이블 내부 클릭이면 무시
        if (tableEl.contains(e.target)) return;

        // 2) 화이트리스트(버튼 등) 클릭이면 유지
        for (const sel of preserveSelectors) {
        if (sel && e.target.closest?.(sel)) return;
        }

        // 3) 선택 해제
        const $rows = $($table).find('tbody tr.selected');
        if ($rows.length === 0) return;
        $rows.removeClass('selected');
        if (selectedSet?.clear) selectedSet.clear();
    };

    // document 전역 바인딩(한 번만)
    document.addEventListener('mousedown', handler);
    // 필요시 정리용 핸들 저장
    tableEl.__outsideDeselectCleanup = () => {
        document.removeEventListener('mousedown', handler);
        tableEl.__outsideDeselectWired = false;
    };
}