let activeSpecInputs = [];

// 저장하기 전 스크롤 위치 저장
const scrollContainer = document.querySelector('.dataTables_scrollBody');
const savedScrollLeft = scrollContainer?.scrollLeft || 0;
const savedScrollTop = scrollContainer?.scrollTop || 0;
// ✅ 선택된 SPEC 셀(편집 대상) 관리
const selectedSpecCells = new Set(); // key = motor_pos_type_field (bound는 셀 내부에 upper/lower 둘 다 있으니 제외)

function clearSelectedSpecCells() {
  selectedSpecCells.clear();
  document
    .querySelectorAll("td.spec-editable.spec-selected")
    .forEach(td => td.classList.remove("spec-selected"));
}

/* -------------------- SPEC 편집 모드 버튼 초기화 -------------------- */
function resetSpecEditButton() {
  const btn = document.getElementById("toggle-spec-edit-btn");
  isEditingSpec = false;
  if (btn) btn.innerText = "SPEC 변경";

  document.querySelectorAll(".spec-editable").forEach(td => {
    const { motor, pos, field, type } = td.dataset;
    const key = `${motor}_${pos}`;

    const lower = motorSpecGlobal?.[key]?.[type]?.[`${field}_lower`];
    const upper = motorSpecGlobal?.[key]?.[type]?.[`${field}_upper`];

    td.innerHTML = `
      <div class="spec-display">
        <span class="spec-lower" style="color:#0077cc;">${lower ?? "-"}</span>
        <span style="color:#999;"> / </span>
        <span class="spec-upper" style="color:#cc4444;">${upper ?? "-"}</span>
      </div>
    `;
  });
}

/* -------------------- SPEC 편집 모드 토글 -------------------- */
async function toggleSpecEditMode() {
    const btn = document.getElementById('toggle-spec-edit-btn');
    const scrollContainer = document.querySelector('.dataTables_scrollBody');
    const savedScroll = {
        left: scrollContainer?.scrollLeft || 0,
        top: scrollContainer?.scrollTop || 0
    };

    // ===== 편집 모드 진입 =====
    if (!isEditingSpec) {

        isEditingSpec = true;
        btn.innerText = '저장';
        showLoading();

        setTimeout(() => {
            resetDataTable(); // DataTable 해제
            const editableCount = document.querySelectorAll('.spec-editable').length;

            if (editableCount === 0) {
                alert('편집 대상 SPEC 셀(.spec-editable)이 없습니다. 헤더 생성부에 .spec-editable과 data-* 속성을 추가하세요.');
                isEditingSpec = false;
                btn.innerText = 'SPEC 변경';
                hideLoading();
                return;
            }

            const ok = createSpecInputs();  // 선택된 셀만
            if (!ok) {
                isEditingSpec = false;
                btn.innerText = "SPEC 변경";
                hideLoading();
                return;
            }

                restoreScroll(savedScroll); // 스크롤 복구
                hideLoading();
   
        }, 50);

    // ===== 저장 수행 =====
    } else {
        showLoading();

        /* ---- 상/하한 유효성 검사 ---- */
        const inputs = activeSpecInputs;
        const pairMap = new Map(); // key: motor_pos_type_field → { upper, lower }

        for (const input of inputs) {
            const { motor, pos, field, bound, type } = input.dataset; // ✅ type 추가
            const key = `${motor}_${pos}_${type}_${field}`;

            const obj = pairMap.get(key) || {};
            const num = parseFloat(input.value);
            if (!Number.isNaN(num)) obj[bound] = num;
            pairMap.set(key, obj);
        }

        let invalid = false;
        for (const [key, obj] of pairMap) {
            const { upper, lower } = obj;
            if (Number.isFinite(upper) && Number.isFinite(lower) && upper < lower) {
                const [motor, pos, type, field] = key.split("_");

                const label =
                pos === "0"
                    ? (field === "MODEL" ? "ACC" : field === "FIX" ? "DEC" : field)
                    : field;

                alert(`⚠️ [${type}] SM${motor}, Position ${pos}, ${label}:\n상한값은 하한값보다 작을 수 없습니다.`);
                invalid = true;
                break;
            }
        }

        if (invalid) {
            hideLoading();
            return;
        }


        /* ---- 변경분 수집 및 저장 ---- */
        const updates = collectSpecUpdates();

        try {
            await Promise.all(updates);

            // 저장 완료 직후 먼저 선택 클리어
            clearSelectedSpecCells();
            activeSpecInputs = [];

            await refreshMotorSpecGlobal();

            // 보기 모드 복귀
            isEditingSpec = false;
            btn.innerText = "SPEC 변경";

            // 테이블 재렌더
            renderMotorGroupTable(currentGroupName);
            clearSelectedSpecCells(); // 새 테이블 생겼으니 선택도 리셋

            // 필터 초기화 + 재적용
            resetFilterUIAndState();
            applyCombinedFilters();

            // 스크롤 복구
            setTimeout(() => restoreScroll(savedScroll), 100);
        } catch (err) {
            alert("SPEC 저장 중 문제가 발생했습니다. 다시 시도해주세요.");
        } finally {
            hideLoading();
        }
    }
}

/* -------------------- DataTable 유틸 -------------------- */
function resetDataTable() {
    if ($.fn.DataTable.isDataTable("#motorTable")) {
        $("#motorTable").DataTable().destroy();
    }
}

function initDataTable(callback) {
    $("#motorTable").DataTable({
        scrollY: '635px',
        scrollX: true,
        scrollCollapse: true,
        paging: false,
        searching: false,
        fixedHeader: !isEditingSpec,
        initComplete: callback
    });
}

/* -------------------- 입력칸 생성 (셀당 1개) -------------------- */
function createSpecInputs() {
  // ✅ 선택된 셀이 없으면 경고
  if (selectedSpecCells.size === 0) {
    alert("변경할 SPEC 셀을 먼저 클릭해서 선택해주세요.");
    return false;
  }

  activeSpecInputs = []; // ✅ 추가

  document.querySelectorAll("#motorTable thead td.spec-editable").forEach(td => {
    const { motor, pos, field, type } = td.dataset;
    const cellKey = `${motor}_${pos}_${type}_${field}`;

    // ✅ 선택된 셀만 편집
    if (!selectedSpecCells.has(cellKey)) return;

    // 이미 input 있으면 패스
    if (td.querySelector("input")) return;

    const key = `${motor}_${pos}`;
    const curLower = motorSpecGlobal?.[key]?.[type]?.[`${field}_lower`];
    const curUpper = motorSpecGlobal?.[key]?.[type]?.[`${field}_upper`];

    td.innerHTML = "";

    const wrap = document.createElement("div");
    wrap.className = "spec-edit-wrap";

    const inputLower = document.createElement("input");
    inputLower.type = "number";
    inputLower.className = "spec-input spec-input-lower";
    inputLower.value = (curLower ?? "").toString();
    inputLower.placeholder = "하한";
    inputLower.style.color = "#0077cc";
    inputLower.dataset.motor = motor;
    inputLower.dataset.pos = pos;
    inputLower.dataset.field = field;
    inputLower.dataset.type = type;
    inputLower.dataset.bound = "lower";

    const slash = document.createElement("span");
    slash.textContent = " / ";
    slash.style.color = "#999";
    slash.style.margin = "0 6px";

    const inputUpper = document.createElement("input");
    inputUpper.type = "number";
    inputUpper.className = "spec-input spec-input-upper";
    inputUpper.value = (curUpper ?? "").toString();
    inputUpper.placeholder = "상한";
    inputUpper.style.color = "#cc4444";
    inputUpper.dataset.motor = motor;
    inputUpper.dataset.pos = pos;
    inputUpper.dataset.field = field;
    inputUpper.dataset.type = type;
    inputUpper.dataset.bound = "upper";

    activeSpecInputs.push(inputLower, inputUpper);

    wrap.appendChild(inputLower);
    wrap.appendChild(slash);
    wrap.appendChild(inputUpper);
    td.appendChild(wrap);
  });

  return true;
}

/* -------------------- 변경분 수집 및 저장 -------------------- */
function collectSpecUpdates() {
  const updates = [];
  const updatedKeySet = new Set();

  const inputs = activeSpecInputs;

  inputs.forEach(input => {
    const { motor, pos, field, bound, type } = input.dataset;
    const key = `${motor}_${pos}`;
    const cellKey = `${motor}_${pos}_${type}_${field}`;
    const uniqueKey = `${key}_${type}_${field}_${bound}`;

    if (!selectedSpecCells.has(cellKey)) return;

    const newValue = input.value.trim();
    const oldValue = motorSpecGlobal?.[key]?.[type]?.[`${field}_${bound}`]?.toString() ?? "";

    const isEmpty = newValue === "";
    const isSame = newValue === oldValue;
    const isDuplicate = updatedKeySet.has(uniqueKey);
    const willUpdate = !isEmpty && !isSame && !isDuplicate;

    if (isEmpty || isSame || isDuplicate) return;

    updatedKeySet.add(uniqueKey);

    updates.push(fetch("/api/spec/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        Motor_Num: motor,
        POS_Num: pos,
        field,
        division: bound,
        type,
        value: newValue,
        userName: currentUserInfo?.userName || "unknown"
      })
    }));
  });

  return updates;
}

/* -------------------- SPEC 글로벌 값 새로고침 -------------------- */
async function refreshMotorSpecGlobal() {
    const specRes = await fetch('/api/spec');
    motorSpecGlobal = await specRes.json();
}

/* -------------------- 스크롤 복원 -------------------- */
function restoreScroll({ left, top }) {
    const container = document.querySelector('.dataTables_scrollBody');
    container?.scrollTo({ left, top, behavior: 'instant' });
}

/* -------------------- 필터 초기화 -------------------- */
function resetFilterUIAndState() {
    currentEqFilterGroup = "all";
    showOnlyAbnormal = false;
    showOnlyOutlier = false;
    currentIndexImportant = 0;
    currentIndexGeneral = 0;

    const eqGroupSel = document.getElementById("eq-group-select");
    const eqpSel = document.getElementById("eqp-select");
    if (eqGroupSel) eqGroupSel.value = "all";
    if (eqpSel) eqpSel.value = "all";

    const unitSelected = document.querySelector(".dropdown-selected");
    const unitMenu = document.getElementById("unitDropdownMenu");
    if (unitSelected) unitSelected.textContent = "설비 Unit 선택";
    if (unitMenu) unitMenu.style.display = "none";

    const abnormalCb = document.getElementById("abnormal-checkbox");
    const outlierCb = document.getElementById("outlier-checkbox");
    const importantCb = document.getElementById("important-checkbox");
    if (abnormalCb) abnormalCb.checked = false;
    if (outlierCb) outlierCb.checked = false;
    if (importantCb) importantCb.checked = false;
}

/* -------------------- ESC 취소 기능 -------------------- */
document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && isEditingSpec) {
        e.preventDefault();
        isEditingSpec = false;

        clearSelectedSpecCells();

        const btn = document.getElementById("toggle-spec-edit-btn");
        if (btn) btn.innerText = "SPEC 변경";

        applyCombinedFilters(); // render 직접 호출보다 이게 더 좋아
    }
});