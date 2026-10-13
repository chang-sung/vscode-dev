/* ===================== 전역 상태 ===================== */

let selectedStartDate = null;
let selectedEndDate = null;
let eqListReady = Promise.resolve();

let monitoringChartsContainer;
let axisTypeDropdown;
let axisDropdown;
let currentEqp;

const MONITORING_PLACEHOLDER_TARGET = "#monitoring-chart-container";


/* ===================== 공용 헬퍼 ===================== */

function showLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'flex';
}

function hideLoading() {
    const el = document.getElementById('loading-overlay');
    if (el) el.style.display = 'none';
}

function showAnalysisText(message, targetSelector) {
    const target = document.querySelector(targetSelector);
    if (!target) return;

    const existing = target.querySelector(".analysis-text");
    if (existing) existing.remove();

    const msgEl = document.createElement("div");
    msgEl.className = "analysis-text";
    msgEl.innerText = message;
    target.appendChild(msgEl);
    target.classList.add("with-border");
}

function selectEqpOption(eqpInput) {
    const $eq = $('#eqp-dropdown');
    if (!$eq.length || !eqpInput) return false;

    const s = String(eqpInput).trim().toUpperCase();

    $eq.val(s);
    if ($eq.val() === s) return true;

    if (/^\d+G$/.test(s)) {
        const full = `PLM#${s}`;
        $eq.val(full);
        if ($eq.val() === full) return true;
    }

    const shortKey = (typeof toShortEqpLabel === 'function') ? toShortEqpLabel(s) : s;
    const opt = $eq.find(`option[data-short="${shortKey}"]`)[0];
    if (opt) {
        $eq.val(opt.value);
        return true;
    }

    return false;
}

function updateMainPlaceholder() {
    const hasEqp = !!currentEqp;
    const typeVal = axisTypeDropdown?.val();
    const hasType = !!typeVal && typeVal !== "Select Type";

    const targetSelector = MONITORING_PLACEHOLDER_TARGET;
    const activeContainer = monitoringChartsContainer;
    const target = document.querySelector(targetSelector);

    if (!target) return false;

    const existing = target.querySelector(".analysis-text");
    if (existing) existing.remove();

    if (!activeContainer?.length) return false;

    if (!hasEqp) {
        if (typeof removeCharts === 'function') removeCharts(activeContainer);
        $('#chart-loading').hide();
        showAnalysisText("설비를 선택하세요.", targetSelector);
        return false;
    }

    if (!hasType) {
        if (typeof removeCharts === 'function') removeCharts(activeContainer);
        $('#chart-loading').hide();
        showAnalysisText("타입을 선택하세요.", targetSelector);
        return false;
    }

    return true;
}

function getAllSmList() {
    const ids = new Set();
    [...ballScrewGroups, ...autoStageGroups].forEach(group => {
        group.indices.forEach(n => ids.add(`SM${n}`));
    });

    return Array.from(ids).sort((a, b) => parseInt(a.slice(2), 10) - parseInt(b.slice(2), 10));
}

function getSmListByType(type) {
    const groups =
        type === 'Ball Screw' ? ballScrewGroups :
        type === 'Auto Stage' ? autoStageGroups :
        [...ballScrewGroups, ...autoStageGroups];

    const ids = new Set();
    groups.forEach(group => {
        group.indices.forEach(n => ids.add(`SM${n}`));
    });

    return Array.from(ids).sort((a, b) => parseInt(a.slice(2), 10) - parseInt(b.slice(2), 10));
}

function setAxisDropdown(list, selected = 'ALL') {
    if (!axisDropdown?.length) return;

    axisDropdown.empty();
    axisDropdown.append('<option value="ALL">ALL</option>');

    list.forEach(sm => {
        axisDropdown.append(
            `<option value="${sm}" ${sm === selected ? 'selected' : ''}>${sm}</option>`
        );
    });
}

function buildDateArgs() {
    const s = selectedStartDate ? `${selectedStartDate} 00:00:00` : null;
    const e = selectedEndDate ? `${selectedEndDate} 23:59:59` : null;
    return { s, e };
}

function rebuildEqpDropdownByShift(shiftKey = 'all') {
    const sel = document.querySelector('#eqp-dropdown');
    if (!sel) return;

    const prevValue = sel.value;

    let sourceList = [];
    if (shiftKey === 'all' || !window.shiftGroups) {
        sourceList = Array.isArray(window.__eqMasterList) ? [...window.__eqMasterList] : [];
    } else {
        sourceList = [...(window.shiftGroups[shiftKey] || [])];
    }

    const safeShort = (eqp) => {
        try {
            return (typeof toShortEqpLabel === 'function') ? toShortEqpLabel(eqp) : (eqp || '');
        } catch {
            return String(eqp || '');
        }
    };

    sourceList.sort((a, b) => safeShort(a).localeCompare(safeShort(b), 'en', { numeric: true }));

    sel.innerHTML = '';
    sel.insertAdjacentHTML('beforeend', `<option value="">설비 선택</option>`);

    for (const full of sourceList) {
        const short = safeShort(full);
        sel.insertAdjacentHTML(
            'beforeend',
            `<option value="${full}" title="${full}" data-short="${short}">${short}</option>`
        );
    }

    if (prevValue && sourceList.includes(prevValue)) {
        sel.value = prevValue;
    } else {
        sel.value = '';
    }
}

function updateMonitoringLayout() {
    const row = document.getElementById('monitoring-content-row');
    const eqp = document.getElementById('eqp-dropdown')?.value;

    if (!row) return;

    if (eqp) row.classList.add('has-eqp');
    else row.classList.remove('has-eqp');
}

/* ===================== 문서 준비 ===================== */

$(document).ready(function () {
    // 1. 로딩 스피너 먼저 표시
    showLoading();

    // 2. DOM 참조 먼저 잡기
    monitoringChartsContainer = $('#monitoring-chart-container');
    axisDropdown = $('.axis-dropdown');
    axisTypeDropdown = $('.axis-type-dropdown');

    // 3. 브라우저가 스피너를 먼저 그릴 시간을 준 뒤 초기화 시작
    requestAnimationFrame(async () => {

        const toShort = (eqp) => {
            if (typeof toShortEqpLabel === 'function') return toShortEqpLabel(eqp);
            if (!eqp) return '';
            const trimmed = String(eqp).trim();
            return trimmed.includes('#') ? trimmed.split('#').pop().trim() : trimmed;
        };

        const allSmMaster = getAllSmList();
        setAxisDropdown(allSmMaster);

        eqListReady = new Promise((resolve) => {
            $.get('/get-eqname-list', (data) => {
                if (Array.isArray(data) && data.length) {
                    data.sort((a, b) => toShort(a).localeCompare(toShort(b), 'en', { numeric: true }));
                    window.__eqMasterList = [...data];

                    data.forEach(eqp => {
                        const shortLabel = toShort(eqp);
                        $('.eqp-dropdown').append(
                            `<option value="${eqp}" title="${eqp}" data-short="${shortLabel}">${shortLabel}</option>`
                        );
                    });
                } else {
                    console.error("No EQP data received or data is empty");
                }

                resolve();
            }).fail(() => {
                resolve();
            });
        });

        /* ===== 이벤트 바인딩 ===== */

        $('#eqp-dropdown').on('change', function () {
            currentEqp = $(this).val() || null;

            // ✅ 설비 변경 시 axis 초기화
            const selectedAxisType = $('#axis-type-dropdown').val() || 'Ball Screw';
            const smList = getSmListByType(selectedAxisType);
            setAxisDropdown(smList, 'ALL');

            updateMonitoringLayout();
            renderMainTableAllEqps(selectedAxisType);
        });

        axisTypeDropdown.on('change', function () {
            const selectedAxisType = $(this).val() || 'Ball Screw';
            renderMainTableAllEqps(selectedAxisType);
        });

        axisDropdown.on('change', function () {
            renderMonitoringCharts();
        });

        const datePicker = flatpickr("#date-range", {
            mode: "range",
            dateFormat: "Y-m-d",
            locale: "ko",
            onClose(selectedDates) {
                if (selectedDates.length === 2) {
                    const toKSTDate = (d) =>
                        new Date(d.getTime() - (d.getTimezoneOffset() * 60000))
                            .toISOString()
                            .slice(0, 10);

                    selectedStartDate = toKSTDate(selectedDates[0]);
                    selectedEndDate = toKSTDate(selectedDates[1]);
                    $("#date-range").val(`${selectedStartDate} ~ ${selectedEndDate}`);

                    renderMonitoringCharts();
                }
            }
        });

        $("#date-init-btn").on("click", function () {
            if (datePicker) datePicker.clear();

            selectedStartDate = null;
            selectedEndDate = null;
            $("#date-range").val("").attr("placeholder", "날짜를 선택하세요");

            const eqp = currentEqp;
            const type = axisTypeDropdown?.val();

            if (eqp && type && type !== "Select Type") {
                renderMainTableAllEqps(type || 'Ball Screw');
            } else {
                if (typeof removeCharts === 'function') removeCharts(monitoringChartsContainer);
                updateMainPlaceholder();
            }
        });

        $('#eq-group-select').on('change', function () {
            const shiftKey = ($(this).val() || 'all').toLowerCase();

            // ✅ 설비 목록 재구성
            rebuildEqpDropdownByShift(shiftKey);

            // ✅ 선택 설비 초기화
            currentEqp = null;
            $('#eqp-dropdown').val('');

            // ✅ 축 타입 기준으로 axis 목록 초기화
            const type = $('#axis-type-dropdown').val() || 'Ball Screw';
            const smList = getSmListByType(type);
            setAxisDropdown(smList, 'ALL');

            // ✅ 차트 제거
            if (typeof removeCharts === 'function' && monitoringChartsContainer?.length) {
                removeCharts(monitoringChartsContainer);
            }

            updateMonitoringLayout();
            renderMainTableAllEqps(type);
        });

        // 교체 이력 버튼 클릭 이벤트
        $('#change-history-btn').on('click', async function () {
            if (typeof openMotorHistoryPopup === 'function') {
                await openMotorHistoryPopup();
            } else {
                console.error('openMotorHistoryPopup 함수가 없습니다.');
            }
        });

        $('#ref-setting-btn').on('click', async function () {
            const eqp = currentEqp;
            const axis = axisDropdown?.val();

            if (!eqp) {
                alert('설비를 먼저 선택하세요.');
                return;
            }

            if (!axis || axis === 'ALL') {
                alert('축(SM)을 선택하세요. (ALL은 지원하지 않습니다)');
                return;
            }

            showLoading();

            try {
                const resp = await fetch('/api/motor-refs/recompute-selected', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        eqp,
                        smKeys: [axis],
                        limitN: 3,
                        forceRecompute: true
                    })
                });

                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.detail || err.error || 'REF 재계산 실패');
                }

                const result = await resp.json();
                console.log('[REF][recompute-selected last10] result:', result);

                if (result?.detail?.[axis] === 'empty') {
                    alert(`최근 10개 데이터가 부족하여 ${axis}의 REF를 재설정하지 못했습니다.`);
                } else {
                    alert(`최근 10개 데이터 기준으로 ${axis}의 REF가 재설정되었습니다.`);
                }

                renderMonitoringCharts();
            } catch (e) {
                console.error('[REF recompute selected last10] error:', e);
                alert('REF 재설정 중 오류가 발생했습니다.');
            } finally {
                hideLoading();
            }
        });

        $('#abnormal-eqp-filter-checkbox').on('change', function () {
            const selectedAxisType = $('#axis-type-dropdown').val() || 'Ball Screw';
            renderMainTableAllEqps(selectedAxisType);
        });

        updateMainPlaceholder();
        await showStatusView();

        /* ===================== 딥링크 초기화 ===================== */
        const params = new URLSearchParams(location.search);
        const eqpQ = params.get('eqp');
        const typeQ = params.get('type');
        const axisQ = params.get('axis');

        if (eqpQ && typeQ) {
            await eqListReady;

            const normType = ((t) => {
                const s = (t || '').toLowerCase().replace(/[-_]/g, ' ');
                if (s.includes('ball')) return 'Ball Screw';
                if (s.includes('auto')) return 'Auto Stage';
                return t || '';
            })(typeQ);

            $('#axis-type-dropdown').val(normType);

            selectEqpOption(eqpQ);

            const $eq = $('#eqp-dropdown');
            currentEqp = $eq.val();

            updateMonitoringLayout();

            const list = getSmListByType(normType);
            setAxisDropdown(list, list.includes(axisQ) ? axisQ : 'ALL');

            await renderMainTableAllEqps(normType || 'Ball Screw');

            if (axisQ) {
                $('#axis-dropdown').val(list.includes(axisQ) ? axisQ : 'ALL');
                renderMonitoringCharts();
            }
        }

        // 4. 모든 초기화 끝난 뒤 로딩 숨기기
        requestAnimationFrame(() => {
            hideLoading();
        });
    });
});


/* ===================== 뷰 전환 ===================== */

function setViewButtonsActive(targetKey) {
    $('.button_group .bay-group-btn').removeClass('active');
    $(`.button_group .bay-group-btn[data-group="${targetKey}"]`).addClass('active');
}

function resetFilterOptions({ typeValue = 'Ball Screw', axisValue = 'ALL', clearDate = true } = {}) {
    currentEqp = undefined;

    if (clearDate) {
        selectedStartDate = null;
        selectedEndDate = null;
        $('#date-range').val('').attr('placeholder', '날짜를 선택하세요');
    }

    const $eq = $('.eqp-dropdown');
    if ($eq.length) $eq.prop('selectedIndex', 0);

    const $type = $('#axis-type-dropdown');
    if ($type.length) {
        if (typeValue) $type.val(typeValue);
        else $type.prop('selectedIndex', 0);
    }

    const $axis = $('.axis-dropdown');
    if ($axis.length) {
        $axis.empty().append('<option value="ALL">ALL</option>').val(axisValue || 'ALL');
    }

    if (typeof removeCharts === 'function') {
        removeCharts(monitoringChartsContainer);
    }

    if (typeof updateMainPlaceholder === 'function') updateMainPlaceholder();

    $('#eq-group-select').val('all');
    updateMonitoringLayout();
}

async function showStatusView() {
    resetFilterOptions({ typeValue: 'Ball Screw', axisValue: 'ALL', clearDate: true });

    $('#monitoring-tab-section').show();
    $('.main-table-wrapper').stop(true, true).show();
    $('#monitoring-chart-wrapper').stop(true, true).show();

    setViewButtonsActive('axis-status-btn');

    const $typeSel = $('#axis-type-dropdown');
    if ($typeSel.length) $typeSel.val('Ball Screw');

    $('#eq-group-select').val('all');

    await eqListReady;
    rebuildEqpDropdownByShift('all');

    await renderMainTableAllEqps($typeSel.val() || 'Ball Screw');
}

// 버튼 바인딩
$(document).on('click', '.bay-group-btn', async function () {
    const key = $(this).data('group');
    if (key === 'axis-status-btn') {
        await showStatusView();
    }
});


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
    const $tbody = $table.find('tbody');

    $tbody
        .off('mousedown.__rowselect')
        .on('mousedown.__rowselect', 'tr', function (e) {
            if (e.shiftKey && !$(e.target).is('input,textarea,select,button,a')) {
                e.preventDefault();
            }
        });

    const getLastAnchor = () => (tableEl.__lastAnchorIndex ?? null);
    const setLastAnchor = (idx) => {
        tableEl.__lastAnchorIndex = (typeof idx === 'number' ? idx : null);
    };

    const indexOfRow = (tr) => $tbody.find('tr').index(tr);
    const rowsBetween = (a, b) => {
        const $rows = $tbody.find('tr');
        const i1 = Math.min(a, b);
        const i2 = Math.max(a, b);
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

            const isCtrl = !!(e.ctrlKey || e.metaKey);
            const isShift = !!e.shiftKey;

            if (isShift && window.getSelection) {
                const sel = window.getSelection();
                if (sel && sel.removeAllRanges) sel.removeAllRanges();
            }

            const $rows = $tbody.find('tr');
            const wasSelected = this.classList.contains('selected');
            const currIdx = indexOfRow(this);
            let changed = false;

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
    preserveSelectors = ['button']
} = {}) {
    if (!$table || !$table.length) return;

    const tableEl = $table[0];
    if (tableEl.__outsideDeselectWired) return;
    tableEl.__outsideDeselectWired = true;

    const handler = (e) => {
        if (tableEl.contains(e.target)) return;

        for (const sel of preserveSelectors) {
            if (sel && e.target.closest?.(sel)) return;
        }

        const $rows = $($table).find('tbody tr.selected');
        if ($rows.length === 0) return;

        $rows.removeClass('selected');
        if (selectedSet?.clear) selectedSet.clear();
    };

    document.addEventListener('mousedown', handler);

    tableEl.__outsideDeselectCleanup = () => {
        document.removeEventListener('mousedown', handler);
        tableEl.__outsideDeselectWired = false;
    };
}