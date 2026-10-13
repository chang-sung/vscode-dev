let __globalBoardRows = [];
const selectedLoadRows = new Set();
let mainTableRenderVersion = 0;

/* ===================== 공통 유틸 ===================== */

function naturalSortEqp(a, b) {
    return String(a).localeCompare(String(b), 'en', { numeric: true });
}

function getSmListForType(type) {
    const groups =
        type === 'Ball Screw' ? (window.ballScrewGroups || []) :
        type === 'Auto Stage' ? (window.autoStageGroups || []) :
        [];

    const set = new Set();
    groups.forEach(g => (g.indices || []).forEach(n => set.add(`SM${n}`)));

    return Array.from(set).sort((a, b) => {
        return parseInt(a.slice(2), 10) - parseInt(b.slice(2), 10);
    });
}

function getAllEqList() {
    const fromDropdown = Array.from(document.querySelectorAll('.eqp-dropdown option'))
        .map(o => o.textContent?.trim())
        .filter(t => t && t !== '설비 선택');

    if (fromDropdown.length) {
        return Array.from(new Set(fromDropdown)).sort(naturalSortEqp);
    }

    const merged = [];
    if (window.bayGroups) {
        Object.values(window.bayGroups).forEach(arr => merged.push(...arr));
    }

    return Array.from(new Set(merged)).sort(naturalSortEqp);
}

function getSelectedEqShortFromDropdown() {
    const sel = document.querySelector('#eqp-dropdown');
    if (!sel) return null;

    const opt = sel.options[sel.selectedIndex];
    if (!opt || !opt.value) return null;

    const shortFromData = opt.getAttribute('data-short');
    if (shortFromData) return shortFromData.trim();

    const v = String(opt.value).trim();
    return v.includes('#') ? v.split('#').pop().trim() : v;
}

function getEqListByShift(shiftKey = 'all') {
    const all = getAllEqList();
    if (!window.shiftGroups || !shiftKey || shiftKey === 'all') return all;

    const toShort = (eqp) => {
        if (typeof toShortEqpLabel === 'function') return toShortEqpLabel(eqp);
        if (!eqp) return '';
        const trimmed = String(eqp).trim();
        return trimmed.includes('#') ? trimmed.split('#').pop().trim() : trimmed;
    };

    const raw = window.shiftGroups[shiftKey] || [];
    const allow = new Set(raw.map(toShort));
    return all.filter(eq => allow.has(eq));
}

function getAxisTypeForSm(sm) {
    const n = parseInt(String(sm).replace(/^SM/i, ''), 10);
    if (!Number.isFinite(n)) return '';

    const inGroup = (groups) => {
        if (!Array.isArray(groups)) return false;
        return groups.some(g => Array.isArray(g.indices) && g.indices.includes(n));
    };

    if (inGroup(window.ballScrewGroups)) return 'Ball Screw';
    if (inGroup(window.autoStageGroups)) return 'Auto Stage';
    return '';
}

/* ===================== 요약 ===================== */

function normalizeLampColor(color) {
    const c = String(color || '').trim().toLowerCase();
    if (c === 'red') return 'red';
    if (c === 'orange') return 'orange';
    return 'green';
}

function buildBoardMap(rows) {
    const map = {};

    (rows || []).forEach(row => {
        const eq = String(row.EQ_Name || '').trim();
        const type = String(row.Type || '').trim();
        const axis = String(row.Axis || '').trim().toUpperCase();

        if (!eq || !axis) return;

        if (!map[eq]) map[eq] = {};
        if (!map[eq][type]) map[eq][type] = {};

        map[eq][type][axis] = {
            lampColor: normalizeLampColor(row.Lamp_color),
            count: Number(row.Outlier_count || 0)
        };
    });

    return map;
}

function renderMonitoringSummaryCards(items) {
    const box = document.querySelector('.group3-container .summary-container');
    if (!box) return;

    if (!items || !items.length) {
        box.innerHTML = `<div class="summary-empty">이상 모니터링 항목이 없습니다.</div>`;
        return;
    }

    box.innerHTML = `
        <div class="summary-cards">
            ${items.map(item => `
                <div class="summary-card st-${item.lampColor}"
                    data-eqp="${item.eqp}"
                    data-type="${item.type}"
                    data-axis="${item.axis}">

                    <div class="value">${item.eqp}</div>
                    <div class="value value-sm">${item.type}</div>
                    <div class="value">${item.axis}</div>
                </div>
            `).join('')}
        </div>
    `;

    box.querySelectorAll('.summary-card').forEach(card => {
      card.addEventListener('click', async () => {
          const eqp = card.dataset.eqp || '';
          const type = card.dataset.type || '';
          const axis = card.dataset.axis || '';

          const eqpDropdown = document.getElementById('eqp-dropdown');
          const axisTypeDropdown = document.getElementById('axis-type-dropdown');

          let selectedEqpValue = '';

          if (eqpDropdown) {
              const targetOpt = Array.from(eqpDropdown.options).find(opt => {
                  const shortText = (opt.textContent || '').trim();
                  const shortData = (opt.getAttribute('data-short') || '').trim();
                  return shortText === eqp || shortData === eqp;
              });

              if (targetOpt) {
                  eqpDropdown.value = targetOpt.value;
                  selectedEqpValue = targetOpt.value;
              }
          }

          if (axisTypeDropdown && type) {
              axisTypeDropdown.value = type;
          }

          // ✅ 중요: summary 카드 클릭은 dropdown change 이벤트를 자동 발생시키지 않으므로
          // currentEqp를 직접 갱신해줘야 함
          if (selectedEqpValue) {
              currentEqp = selectedEqpValue;
          }

          if (typeof updateMonitoringLayout === 'function') {
              updateMonitoringLayout();
          }

          // 이전 차트 응답이 새 카드의 선택을 덮어쓰지 않도록 즉시 무효화한다.
          removeCharts(monitoringChartsContainer);

          // 차트 조회 전에 축을 확정해 테이블 렌더에서 한 번만 조회한다.
          setAxisDropdown(getSmListForType(type || 'Ball Screw'), axis || 'ALL');

          if (typeof renderMainTableAllEqps === 'function') {
              await renderMainTableAllEqps(type || 'Ball Screw');
          }
      });
  });
}

function renderBoardSummary(boardRows, {
    shiftKey = 'all',
    selectedType = '',
    abnormalOnly = false
} = {}) {
    __globalBoardRows = Array.isArray(boardRows) ? boardRows : [];

    const allowedEqSet = new Set(getEqListByShift(shiftKey));
    const abnormalEqSet = abnormalOnly ? getAbnormalEqSet(boardRows, selectedType) : null;

    const items = __globalBoardRows
        .filter(row => {
            const eq = String(row.EQ_Name || '').trim();
            const type = String(row.Type || '').trim();
            const lamp = normalizeLampColor(row.Lamp_color);

            if (!eq) return false;
            if (!allowedEqSet.has(eq)) return false;     // ✅ shift 필터
            if (selectedType && type !== selectedType) return false; // ✅ 타입 필터
            if (abnormalEqSet && !abnormalEqSet.has(eq)) return false; // ✅ 이상설비만 보기와 동기화
            if (lamp === 'green') return false;

            return true;
        })
        .map(row => ({
            eqp: String(row.EQ_Name || '').trim(),
            type: String(row.Type || '').trim(),
            axis: String(row.Axis || '').trim().toUpperCase(),
            lampColor: normalizeLampColor(row.Lamp_color),
            count: Number(row.Outlier_count || 0)
        }))
        .sort((a, b) => {
            const eqCmp = naturalSortEqp(a.eqp, b.eqp);
            if (eqCmp !== 0) return eqCmp;
            return a.axis.localeCompare(b.axis, 'en', { numeric: true });
        });

    renderMonitoringSummaryCards(items);
}

/* ===================== 테이블 빌드 ===================== */

function buildColGroup(eqList) {
    const cols = ['<col class="col-axis">'];
    for (let i = 0; i < eqList.length; i++) {
        cols.push('<col class="col-eqp">');
    }
    return `<colgroup>${cols.join('')}</colgroup>`;
}

function buildHeader(eqList) {
    return `
        <tr>
            <th class="axis-header">축</th>
            ${eqList.map(eq => `<th class="eqp-header" data-eqp="${eq}">${eq}</th>`).join('')}
        </tr>
    `;
}

function buildBodyAxisRows(eqList, smList) {
    return smList.map(sm => {
        const fullLabel = (window.smLabels?.[sm] || sm).trim();
        const cells = eqList
            .map(eq => `<td data-eqp="${eq}" data-sm="${sm}"></td>`)
            .join('');

        return `
            <tr data-key="${sm}">
                <th class="axis-label-cell">${fullLabel}</th>
                ${cells}
            </tr>
        `;
    }).join('');
}

/* ===================== 데이터 ===================== */

async function fetchBoardSummaryRows() {
    const resp = await fetch('/api/motor-board/summary');
    const json = await resp.json();

    if (!json.ok) {
        throw new Error('motor-board-summary api failed');
    }

    return json.data || [];
}

function fillLampCells($table, boardMap, selectedType) {
    $table.find('tbody td[data-eqp][data-sm]').each(function () {
        const $td = $(this);
        const eq = String($td.data('eqp') || '').trim();
        const sm = String($td.data('sm') || '').trim().toUpperCase();

        const item = boardMap[eq]?.[selectedType]?.[sm];
        const colorClass = item?.lampColor || 'green';
        const cnt = Number(item?.count || 0);

        const label = cnt > 99 ? '99+' : String(cnt);
        const html = colorClass === 'green'
            ? `<span class="lamp ${colorClass}" title="0"></span>`
            : `<span class="lamp ${colorClass} with-text" title="${cnt}">${label}</span>`;

        $td.attr('data-count', cnt).html(html);
    });
}


function getAbnormalEqSet(boardRows, selectedType) {
    const abnormalSet = new Set();

    (boardRows || []).forEach(row => {
        const eq = String(row.EQ_Name || '').trim();
        const type = String(row.Type || '').trim();
        const lamp = normalizeLampColor(row.Lamp_color);

        if (!eq) return;
        if (selectedType && type !== selectedType) return;

        if (lamp === 'orange' || lamp === 'red') {
            abnormalSet.add(eq);
        }
    });

    return abnormalSet;
}

/* ===================== DataTable ===================== */

function destroyMainTableDT($table) {
    if ($.fn.DataTable.isDataTable($table)) {
        $table.DataTable().destroy();
    }
}

function initMainTableDT($table) {
    return $table.DataTable({
        paging: false,
        searching: false,
        info: false,
        ordering: false,
        autoWidth: false,
        deferRender: true,
        dom: 't'
    });
}

/* ===================== 메인 렌더 ===================== */

async function renderMainTableAllEqps(selectedType = 'Ball Screw') {
    const renderVersion = ++mainTableRenderVersion;
    const $table = $('#main-table');

    try {
        const boardRows = await fetchBoardSummaryRows();
        if (renderVersion !== mainTableRenderVersion) return;
        const boardMap = buildBoardMap(boardRows);

        const shiftKey = ($('#eq-group-select').val() || 'all').toLowerCase();
        const selectedEqShort = getSelectedEqShortFromDropdown();

        let eqList = getEqListByShift(shiftKey);

        if (selectedEqShort) {
            eqList = eqList.filter(eq => eq === selectedEqShort);
        }

        const abnormalOnly = $('#abnormal-eqp-filter-checkbox').is(':checked');

        if (abnormalOnly) {
            const abnormalEqSet = getAbnormalEqSet(boardRows, selectedType);
            eqList = eqList.filter(eq => abnormalEqSet.has(eq));
        }

        const smList = getSmListForType(selectedType);

        const prevAxis = $('.axis-dropdown').val() || 'ALL';
        setAxisDropdown(smList, smList.includes(prevAxis) ? prevAxis : 'ALL');

        destroyMainTableDT($table);

        const html = [
            buildColGroup(eqList),
            `<thead>${buildHeader(eqList)}</thead>`,
            `<tbody>${buildBodyAxisRows(eqList, smList)}</tbody>`
        ].join('');

        $table.html(html);

        // ✅ summary도 현재 shift/type/abnormal 조건 반영
        renderBoardSummary(boardRows, {
            shiftKey,
            selectedType,
            abnormalOnly
        });

        fillLampCells($table, boardMap, selectedType);

        initMainTableDT($table);

        bindDTSelectable($table, {
            selectedSet: selectedLoadRows,
            getKey: (tr) => tr.getAttribute('data-key'),
            restoreOnDraw: true,
            onChange: () => {}
        });

        wireOutsideDeselect($table, {
            selectedSet: selectedLoadRows,
            preserveSelectors: [
                'button',
                '.status-btn',
                '.axis-dropdown',
                '#eq-group-select',
                '#eqp-dropdown'
            ]
        });

        renderMonitoringCharts();
    } catch (e) {
        console.error('renderMainTableAllEqps failed:', e);
    }
}

/* ===================== 테이블 아래 차트 ===================== */

function renderMonitoringCharts() {
    if (!monitoringChartsContainer?.length) return;

    if (!currentEqp) {
        removeCharts(monitoringChartsContainer);
        return;
    }

    const selectedAxis = $('#axis-dropdown').val() || 'ALL';
    const { s, e } = buildDateArgs();
    const selectedAxisType = $('#axis-type-dropdown').val();

    createMotorCharts(
        currentEqp,
        selectedAxis,
        selectedAxisType,
        s,
        e,
        monitoringChartsContainer
    );
}

window.renderMonitoringCharts = renderMonitoringCharts;

async function refreshSummaryUI() {
    try {
        const boardRows = await fetchBoardSummaryRows();
        const boardMap = buildBoardMap(boardRows);

        renderBoardSummary(boardRows);

        const $table = $('#main-table');
        const selectedType = $('#axis-type-dropdown').val() || 'Ball Screw';

        if ($table.length) {
            fillLampCells($table, boardMap, selectedType);
        }
    } catch (e) {
        console.error('refreshSummaryUI failed:', e);
    }
}