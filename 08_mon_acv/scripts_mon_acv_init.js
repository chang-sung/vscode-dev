const eqplist = [
    "ACV#4G", "ACV#5G", "ACV#6G", "ACV#7G", "ACV#8G", "ACV#9G", "ACV#10G", "ACV#12G",
    "ACV#13G", "ACV#14G", "ACV#15G", "ACV#16G", "ACV#17G", "ACV#18G", "ACV#19G", "ACV#20G",
    "ACV#21G", "ACV#22G", "ACV#23G", "ACV#24G", "ACV#25G", "ACV#26G", "ACV#27G", "ACV#28G",
    "ACV#29G", "ACV#30G", "ACV#31G", "ACV#32G"
];

const zonelist = {
    zone_2: ["ACV#04G", "ACV#12G", "ACV#13G", "ACV#14G"],
    zone_3: ["ACV#05G", "ACV#06G", "ACV#07G", "ACV#08G", "ACV#09G", "ACV#10G"],
    zone_4: ["ACV#15G", "ACV#16G", "ACV#17G", "ACV#18G", "ACV#19G", "ACV#20G"],
    zone_6: ["ACV#21G", "ACV#22G", "ACV#23G", "ACV#24G", "ACV#25G"],
    zone_8: ["ACV#26G", "ACV#27G", "ACV#28G", "ACV#29G", "ACV#30G", "ACV#31G", "ACV#32G"]
};

const shiftGroups = {
    a: ["ACV#10G", "ACV#13G", "ACV#18G", "ACV#19G", "ACV#20G", "ACV#28G"],
    b: ["ACV#6G", "ACV#7G", "ACV#12G", "ACV#26G", "ACV#27G", "ACV#30G", "ACV#31G"],
    c: ["ACV#5G", "ACV#21G", "ACV#22G", "ACV#23G", "ACV#24G", "ACV#25G", "ACV#29G"],
    d: ["ACV#4G", "ACV#8G", "ACV#9G", "ACV#14G", "ACV#15G", "ACV#16G", "ACV#17G", "ACV#32G"]
};

// 전역 저장
window.eqplist = eqplist;
window.zonelist = zonelist;
window.shiftGroups = shiftGroups;

// ====== flatpickr 프로그램적 변경 가드 ======
let __isProgrammaticDateSet = false;
function __beginProgDateChange() { __isProgrammaticDateSet = true; }
function __endProgDateChange() { setTimeout(() => { __isProgrammaticDateSet = false; }, 0); }

// ====== 날짜 유틸 ======

// 특정 일자(day: 'YYYY-MM-DD')로 date-range를 설정하고 change 이벤트 트리거
function setDateRangeToDay(dayStr) {
    const input = document.getElementById("date-range");
    if (!input || !dayStr) return;

    if (window.__acvDatePicker?.setDate) {
        __beginProgDateChange();
        try {
            // true → change 이벤트 발생(가드로 무시)
            window.__acvDatePicker.setDate([dayStr, dayStr], true);
        } finally {
            __endProgDateChange();
        }
    } else {
        __beginProgDateChange();
        try {
            input.value = `${dayStr} ~ ${dayStr}`;
            input.dispatchEvent(new Event("change"));
        } finally {
            __endProgDateChange();
        }
    }
}

// date-range를 비움 (silent: true면 change 이벤트 미발생)
function clearDateRange(silent = true) {
    const input = document.getElementById("date-range");
    if (!input) return;

    if (window.__acvDatePicker?.clear) {
        __beginProgDateChange();               // ✅ 가드 시작
        try {
            window.__acvDatePicker.clear();    // 내부 onChange 발생 → 가드로 무시
        } finally {
            __endProgDateChange();             // ✅ 가드 해제
        }
    }
    input.value = "";

    if (!silent) {
        __beginProgDateChange();
        try {
            input.dispatchEvent(new Event("change"));
        } finally {
            __endProgDateChange();
        }
    }
}

// ====== 드롭다운 구성 ======

// Shift → 설비 리스트
function getEqListByShift(shiftKey = "all") {
    if (!window.eqplist) window.eqplist = [];
    if (!window.shiftGroups) window.shiftGroups = {};
    if (!shiftKey || shiftKey === "all") return [...window.eqplist];
    const group = window.shiftGroups[String(shiftKey).toLowerCase()];
    return Array.isArray(group) ? [...group] : [];
}

// 설비 드롭다운 채우기 (#eq-group-select) — shift 기반
function populateEqGroupSelectByShift(shiftKey = "all") {
    const select = document.getElementById("eq-group-select");
    if (!select) return;

    const eqList = getEqListByShift(shiftKey)
        .sort((a, b) => String(a).localeCompare(String(b), "en", { numeric: true }));

    select.innerHTML = `<option value="all">설비 선택</option>`;
    eqList.forEach(eq => {
        const opt = document.createElement("option");
        opt.value = eq;
        opt.textContent = eq;
        select.appendChild(opt);
    });
}

// 레시피 드롭다운 채우기(백엔드에서 전체 목록 받아와 채움)
async function populateRecipeSelect() {
    try {
        const response = await fetch("/get-acv-recipes");
        const recipes = await response.json();

        const recipeSelect = document.getElementById("recipe-select");
        if (!recipeSelect) return;

        recipeSelect.innerHTML = `<option value="all">Recipe 전체</option>`;
        recipes.forEach(recipe => {
            const opt = document.createElement("option");
            opt.value = recipe;
            opt.textContent = recipe;
            recipeSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("❌ Recipe 목록 로드 실패:", err);
    }
}


// Shift/EQ를 반영해 recipe 목록 구성
function populateRecipeSelectByEq() {
    const sel = document.getElementById("recipe-select");
    if (!sel) return;

    const prev = sel.value || "all";
    const eq   = document.getElementById("eq-group-select")?.value || "all";
    const shiftKey = document.getElementById("shift-select")?.value || "all";

    // 1) 데이터 소스 확보
    // - getFilteredMonData가 존재하면: shift는 반영하되(recipe는 무시), eq는 이후 수동 필터
    // - 없으면: 빈 배열(백엔드에서 /get-acv-recipes?eq= 호출하는 방식으로 확장 가능)
    let base = [];
    if (typeof getFilteredMonData === "function") {
        // shift 필터는 유지, recipe는 무시, eq는 일단 무시 후 아래에서 수동 필터
        base = getFilteredMonData({ ignoreRecipe: true, ignoreEq: true, shift: shiftKey }) || [];
    }

    // 2) 설비 선택값 반영
    if (eq !== "all") {
        base = base.filter(r => (r.EQ_Name || r.eq || "").trim() === eq);
    }

    // 3) 고유 레시피 목록 생성
    const unique = [...new Set(
        base.map(r => (r.recipe ?? "").trim()).filter(v => v)
    )].sort((a, b) => a.localeCompare(b, "en", { numeric: true }));

    // 4) 드롭다운 채우기
    sel.innerHTML = `<option value="all">Recipe 전체</option>`;
    if (unique.length > 0) {
        unique.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r;
            opt.textContent = r;
            sel.appendChild(opt);
        });
    } else {
        // 해당 설비에 해당하는 레시피가 없을 때 UX용 안내 옵션(Optional)
        const opt = document.createElement("option");
        opt.value = "__none__";
        opt.textContent = "해당 설비에 레시피 없음";
        opt.disabled = true;
        sel.appendChild(opt);
    }

    // 5) 기존 선택 유지 (없으면 all)
    sel.value = unique.includes(prev) ? prev : "all";
}

// Shift+Recipe 반영해 EQ 목록 구성 — 외부의 getFilteredMonData가 있는 경우에 사용
function populateEqSelectByRecipe() {
    const sel = document.getElementById("eq-group-select");
    if (!sel) return;

    const prev = sel.value || "all";
    const base = (typeof getFilteredMonData === "function")
        ? getFilteredMonData({ ignoreEq: true })
        : [];

    const unique = [...new Set(base.map(r => r.EQ_Name))]
        .sort((a, b) => String(a).localeCompare(String(b), "en", { numeric: true }));

    sel.innerHTML = `<option value="all">설비 전체</option>`;
    unique.forEach(eq => {
        const opt = document.createElement("option");
        opt.value = eq;
        opt.textContent = eq;
        sel.appendChild(opt);
    });

    sel.value = unique.includes(prev) ? prev : "all";
}

// (레거시) 전체 설비로 채우기
function populateEqGroupSelect() {
    const select = document.getElementById("eq-group-select");
    if (!select) return;
    select.innerHTML = `<option value="all">설비 전체</option>`;
    window.eqplist.forEach(eqName => {
        const option = document.createElement("option");
        option.value = eqName;
        option.textContent = eqName;
        select.appendChild(option);
    });
}

// ====== 로딩 오버레이 ======
function showLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = "flex";
    document.querySelectorAll("button, select, input[type='checkbox']").forEach(el => { el.disabled = true; });
}
function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.style.display = "none";
    document.querySelectorAll("button, select, input[type='checkbox']").forEach(el => { el.disabled = false; });
}

// ====== 안내 문구 ======
function showAnalysisText(message, targetSelector) {
    const target = document.querySelector(targetSelector);
    if (!target) return;
    const existing = target.querySelector(".analysis-text");
    if (existing) existing.remove();
    const msgEl = document.createElement("div");
    msgEl.className = "analysis-text";
    msgEl.innerText = message;
    target.appendChild(msgEl);
}
function clearAnalysisText(targetSelector) {
    const target = document.querySelector(targetSelector);
    if (!target) return;
    const existing = target.querySelector(".analysis-text");
    if (existing) existing.remove();
}

// ====== 사이클 상태머신 & 단일 진입점 ======

// 현재 탭/선택값으로 상태 계산
function getCycleState() {
    const isCycleMode = document.querySelector(".acv-chart-wrapper").style.display !== "none";
    if (!isCycleMode) return { mode: "not-cycle" };

    const eq = document.getElementById("eq-group-select")?.value || "all";
    const recipe = document.getElementById("recipe-select")?.value || "all";
    const dateV = document.getElementById("date-range")?.value?.trim() || "";

    if (eq === "all")     return { mode: "need-eq", msg: "설비를 선택하세요." };
    if (recipe === "all") return { mode: "need-recipe", msg: "레시피를 선택하세요." };
    if (!dateV)           return { mode: "eq+recipe-no-date", msg: "최근 일자를 자동으로 찾는 중…" };

    return { mode: "ready", msg: null, eq, recipe };
}

// 사이클 UI 공통 초기화
function resetCycleUI({ clearChart = true, clearSelector = true, msg = null } = {}) {
    try { acvChartInstance?.dispose?.(); } catch (_) {}
    if (clearChart) acvChartInstance = null;

    const cc = document.getElementById("cycle-counter");
    if (cc) cc.textContent = "";

    if (clearSelector) {
        const sel = document.getElementById("cycle-selector");
        if (sel) sel.innerHTML = `<option value="">Track In 시간 선택</option>`;
    }

    if (msg) showAnalysisText(msg, "#acv-chart-container");
    else clearAnalysisText("#acv-chart-container");

    // 상세 인포 테이블 초기화
    if (typeof updateInfoTable === "function") {
        updateInfoTable({ cv1: "-", cv2: "-", cv3: "-", cv4: "-" }, []);
    }
}

// 설비+레시피 최신 일자 조회 → 성공 시 setDateRangeToDay(day)
async function maybeLoadLatestForEqRecipe() {
    const eq = document.getElementById("eq-group-select")?.value;
    const recipe = document.getElementById("recipe-select")?.value;

    if (!eq || eq === "all" || !recipe || recipe === "all") return false;

    try {
        showLoading();
        const resp = await fetch(`/get-latest-eq-recipe-day?eq=${encodeURIComponent(eq)}&recipe=${encodeURIComponent(recipe)}`);
        const { day } = await resp.json();
        hideLoading();

        if (day) {
            setDateRangeToDay(day); // 내부에서 change 발생 → fetchAndRenderCycles 실행(가드로 안전)
            await Promise.resolve(); // 이벤트 큐 한 틱 양보
            await handleCycleSelectionsChange({ from: "auto-latest-day", requireDate: true });
            return true;
        } else {
            clearDateRange(true);
            resetCycleUI({ msg: "선택한 설비-레시피의 데이터가 없습니다." });
            return false;
        }
    } catch (e) {
        hideLoading();
        console.error("❌ latest eq+recipe 조회 실패:", e);
        return false;
    }
}

// 상태에 따른 단일 진입점
async function handleCycleSelectionsChange({ from = "unknown", requireDate = false } = {}) {
    const s = getCycleState();
    if (s.mode === "not-cycle") return;

    if (s.mode === "need-eq") {
        clearDateRange(true);
        resetCycleUI({ msg: "설비를 선택하세요." });
        return;
    }
    if (s.mode === "need-recipe") {
        clearDateRange(true);
        resetCycleUI({ msg: "레시피를 선택하세요." });
        return;
    }
    if (s.mode === "eq+recipe-no-date") {
        clearDateRange(true);
        resetCycleUI({ msg: "최근 일자를 자동으로 찾는 중…" });
        const ok = await maybeLoadLatestForEqRecipe();
        if (!ok) {
            resetCycleUI({ msg: "선택한 설비-레시피의 데이터가 없습니다." });
        }
        return;
    }
    if (s.mode === "ready") {
        clearAnalysisText("#acv-chart-container");
        if (requireDate) {
            const hasDate = !!(document.getElementById("date-range")?.value?.trim());
            if (!hasDate) {
                resetCycleUI({ msg: "날짜를 선택하거나, 레시피 변경으로 최근일자를 불러오세요." });
                return;
            }
        }
        if (typeof fetchAndRenderCycles === "function") {
            fetchAndRenderCycles(s.eq, { requireDate: true });
        }
        return;
    }

    resetCycleUI({ msg: "설비와 레시피를 선택하세요." });
}

// ====== EQ 선택 이벤트 ======
function setupEqGroupChangeHandler() {
    const select = document.getElementById("eq-group-select");
    if (!select) return;

    select.addEventListener("change", async function () {
        const isCycleMode = document.querySelector(".acv-chart-wrapper").style.display !== "none";
        if (isCycleMode) {
            handleCycleSelectionsChange({ from: "eq-change" });
            return;
        }

        // Efficiency 탭
        const isEffiMode = document.querySelector(".acv-effi-wrapper").style.display !== "none";
        if (isEffiMode && typeof loadEfficiencyAnalysis === "function") {
            const shiftKey = document.getElementById("shift-select")?.value || "all";
            loadEfficiencyAnalysis(this.value, shiftKey);
        }
    });
}

// ====== 뷰 전환 ======
function showWrapper(type) {
    const cycleWrapper = document.querySelector(".acv-chart-wrapper");
    const effiWrapper = document.querySelector(".acv-effi-wrapper");
    const monWrapper = document.querySelector(".acv-mon-wrapper");

    const show = (el, on) => { if (el) el.style.display = on ? "block" : "none"; };

    if (type === "cycle") {
        show(cycleWrapper, true);
        show(effiWrapper, false);
        show(monWrapper, false);
        setActiveButton("btn-cycle");
    } else if (type === "efficiency") {
        show(cycleWrapper, false);
        show(effiWrapper, true);
        show(monWrapper, false);
        setActiveButton("btn-efficiency");
    } else if (type === "monitor") {
        show(cycleWrapper, false);
        show(effiWrapper, false);
        show(monWrapper, true);
        setActiveButton("btn-monitor");
    }
}

function setActiveButton(activeId) {
    document.querySelectorAll(".bay-group-btn").forEach(btn => {
        btn.classList.toggle("active", btn.id === activeId);
    });
}

// Shift 유지 여부에 따른 필터 초기화
function resetFilters(keepShift = true) {
    const eqSelect = document.getElementById("eq-group-select");
    if (eqSelect) eqSelect.value = "all";

    const recipeSelect = document.getElementById("recipe-select");
    if (recipeSelect) recipeSelect.value = "all";

    if (!keepShift) {
        const shiftSelect = document.getElementById("shift-select");
        if (shiftSelect) {
            shiftSelect.value = "all";
            populateEqGroupSelectByShift("all");
        }
    }

    clearDateRange(true);

    if (typeof acvChartInstance !== "undefined" && acvChartInstance) acvChartInstance.clear?.();
    const cycleCounter = document.getElementById("cycle-counter");
    if (cycleCounter) cycleCounter.textContent = "";

    if (typeof updateInfoTable === "function") {
        updateInfoTable({ cv1: "-", cv2: "-", cv3: "-", cv4: "-" }, []);
    }
}

// ====== 초기화 ======
document.addEventListener("DOMContentLoaded", () => {
    // 날짜 선택기
    window.__acvDatePicker = flatpickr("#date-range", {
        mode: "range",
        dateFormat: "Y-m-d",
        locale: "ko"
    });

    // 초기 표/버튼 상태
    if (typeof updateInfoTable === "function") {
        updateInfoTable({ cv1: "-", cv2: "-", cv3: "-", cv4: "-" }, []);
    }
    document.getElementById("btn-monitor")?.classList.add("active");

    // Shift=all 기준으로 설비 목록 채우기
    populateEqGroupSelectByShift("all");

    // EQ 선택 이벤트
    setupEqGroupChangeHandler();

    // Recipe 선택 옵션 로드
    populateRecipeSelect();

    // 모드 전환 버튼
    setupToggleViewButtons();

    // 저장 버튼 연결
    if (typeof setupMonExcelExport === "function") {
        setupMonExcelExport();
    }

    // trendChart 필터 UI/이벤트
    if (typeof renderTrendChartFilters === "function") {
        renderTrendChartFilters();
    }

    // 처음 로딩 시: 모니터 뷰 + 요약표
    showWrapper("monitor");
    setTimeout(() => { if (typeof openSummaryPopup === "function") openSummaryPopup(); }, 10);

    // ====== 이벤트 바인딩 ======

    // 날짜 초기화 버튼
    document.getElementById("date-init-btn")?.addEventListener("click", () => {
        document.getElementById("date-range").value = "";
        const isCycleMode = document.querySelector(".acv-chart-wrapper").style.display !== "none";
        const isEffiMode = document.querySelector(".acv-effi-wrapper").style.display !== "none";

        if (isCycleMode) {
            clearDateRange(true);
            handleCycleSelectionsChange({ from: "date-init" });
        }

        if (isEffiMode && typeof loadEfficiencyAnalysis === "function") {
            weeklyChart?.dispose?.();
            dailyChart?.dispose?.();
            eqpChart?.dispose?.();
            weeklyChart = dailyChart = eqpChart = null;
            const shiftKey = document.getElementById("shift-select")?.value || "all";
            loadEfficiencyAnalysis("all", shiftKey);
        }
    });

    // 날짜 변경
    document.getElementById("date-range")?.addEventListener("change", () => {
        if (__isProgrammaticDateSet) return;  // ✅ 프로그램적 변경일 때는 무시
        handleCycleSelectionsChange({ from: "date-change", requireDate: true });
    });

    // Shift 변경
    document.getElementById("shift-select")?.addEventListener("change", () => {
        const shiftKey = document.getElementById("shift-select").value || "all";

        // 설비 드롭다운 갱신 및 초기화
        populateEqGroupSelectByShift(shiftKey);
        const eqSelect = document.getElementById("eq-group-select");
        if (eqSelect) eqSelect.value = "all";

        const isCycleMode = document.querySelector(".acv-chart-wrapper").style.display !== "none";
        const isEffiMode = document.querySelector(".acv-effi-wrapper").style.display !== "none";

        if (isCycleMode) {
            handleCycleSelectionsChange({ from: "shift-change" });
        }

        if (isEffiMode && typeof loadEfficiencyAnalysis === "function") {
            loadEfficiencyAnalysis("all", shiftKey);
        }
    });

    // 레시피 변경
    document.getElementById("recipe-select")?.addEventListener("change", async () => {
        const isCycleMode = document.querySelector(".acv-chart-wrapper").style.display !== "none";
        if (!isCycleMode) return;

        clearDateRange(true);
        handleCycleSelectionsChange({ from: "recipe-change" });
    });

    // 상단 버튼 active 토글
    document.querySelectorAll(".bay-group-btn").forEach(btn => {
        btn.addEventListener("click", function () {
            document.querySelectorAll(".bay-group-btn").forEach(b => b.classList.remove("active"));
            this.classList.add("active");
        });
    });
});

// ====== 탭 전환 버튼 로직 ======
function setupToggleViewButtons() {
    const btnCycle = document.getElementById("btn-cycle");
    const btnEfficiency = document.getElementById("btn-efficiency");
    const btnMonitor = document.getElementById("btn-monitor");

    btnCycle?.addEventListener("click", () => {
        resetFilters(false); // Shift 유지
        showWrapper("cycle");
        handleCycleSelectionsChange({ from: "tab-cycle" });
    });

    btnEfficiency?.addEventListener("click", () => {
        resetFilters(false); // Shift 유지
        showWrapper("efficiency");
        setTimeout(() => {
            const eqName = document.getElementById("eq-group-select").value || "all";
            const shiftKey = document.getElementById("shift-select")?.value || "all";
            if (typeof loadEfficiencyAnalysis === "function") {
                loadEfficiencyAnalysis(eqName, shiftKey);
            }
        }, 100);
    });

    btnMonitor?.addEventListener("click", () => {
        // 1) 필터 완전 초기화(Shift/설비/Recipe/날짜 모두)
        resetFilters(false);

        // 2) 드롭다운 다시 채우기
        populateEqGroupSelectByShift("all");
        const recipeSel = document.getElementById("recipe-select");
        if (recipeSel) recipeSel.value = "all";
        populateRecipeSelect();

        // 3) 차트/분석 영역 비우기
        try { hideChartContainer?.(); } catch (_) { /* noop */ }
        try { clearChart?.(); } catch (_) { /* noop */ }
        clearAnalysisText?.("#acv-chart-container");

        const cc = document.getElementById("cycle-counter");
        if (cc) cc.textContent = "";

        if (typeof updateInfoTable === "function") {
            updateInfoTable({ cv1: "-", cv2: "-", cv3: "-", cv4: "-" }, []);
        }

        showWrapper("monitor");
        if (typeof openSummaryPopup === "function") openSummaryPopup();
    });
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