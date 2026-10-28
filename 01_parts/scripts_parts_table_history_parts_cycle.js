// 설비 수리 이력의 조회 결과를 받아 교체 간격을 계산하고 별도 표에 표시한다.
window.PartsHistoryCycle = (() => {
    const DAY_MS = 24 * 60 * 60 * 1000;
    const REPLACEMENT_GRADES = new Set(['A급 교체', 'B급 교체']);
    let sourceRows = [];
    let isOpen = false;
    let button;
    let panel;
    let stack;

    const clean = value => String(value ?? '').trim();

    // DB의 timezone 없는 작업시간을 동일한 기준으로 해석한다.
    // Date.parse의 브라우저별 차이와 DST에 따른 하루 길이 차이를 피한다.
    function parseWorkTime(value) {
        const match = /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$/.exec(clean(value));
        if (!match) return null;
        const [, year, month, day, hour = '00', minute = '00', second = '00'] = match;
        const time = Date.UTC(+year, +month - 1, +day, +hour, +minute, +second);
        const date = new Date(time);
        if (date.getUTCFullYear() !== +year || date.getUTCMonth() !== +month - 1 ||
            date.getUTCDate() !== +day || date.getUTCHours() !== +hour ||
            date.getUTCMinutes() !== +minute || date.getUTCSeconds() !== +second) return null;
        return time;
    }

    function calculate(rows) {
        const groups = new Map();
        let invalidDateCount = 0;

        for (const row of rows || []) {
            if (clean(row.Eqp_State) !== '완료' || !REPLACEMENT_GRADES.has(clean(row.Parts_Grade))) continue;
            const time = parseWorkTime(row.Work_Time);
            if (time === null) {
                invalidDateCount++;
                continue;
            }

            const eqName = clean(row.EQ_Name);
            const subUnit = clean(row.Sub_Unit);
            const partsGroup = clean(row.Parts_Group);
            // Unit, 등급, P/N은 그룹 키에 넣지 않는다. 요청된 세 컬럼만으로 묶는다.
            const key = JSON.stringify([eqName, subUnit, partsGroup]);
            if (!groups.has(key)) groups.set(key, {
                eqName, subUnit, partsGroup, times: new Set(), eqGroups: new Set(), units: new Set()
            });
            const group = groups.get(key);
            group.times.add(time);
            if (clean(row.EQ_Group)) group.eqGroups.add(clean(row.EQ_Group));
            if (clean(row.Unit)) group.units.add(clean(row.Unit));
        }

        const items = Array.from(groups.values(), group => {
            const times = Array.from(group.times).sort((a, b) => a - b);
            const count = times.length;
            const intervalCount = count - 1;
            const lastTime = times[count - 1];
            const previousTime = intervalCount ? times[count - 2] : null;
            return {
                eqName: group.eqName, subUnit: group.subUnit, partsGroup: group.partsGroup,
                eqGroup: Array.from(group.eqGroups).sort().join(', '),
                unit: Array.from(group.units).sort().join(', '),
                count, intervalCount, previousTime, lastTime,
                recentDays: intervalCount ? (lastTime - previousTime) / DAY_MS : null,
                averageDays: intervalCount ? (lastTime - times[0]) / DAY_MS / intervalCount : null
            };
        });
        items.sort((a, b) => a.eqName.localeCompare(b.eqName, 'ko', { numeric: true }) ||
            a.subUnit.localeCompare(b.subUnit, 'ko', { numeric: true }) ||
            a.partsGroup.localeCompare(b.partsGroup, 'ko'));
        return { items, invalidDateCount };
    }

    function formatTime(time) {
        return time === null ? '—' : new Date(time).toISOString().slice(0, 19).replace('T', ' ');
    }

    function render() {
        if (!panel || !isOpen) return;
        const { items, invalidDateCount } = calculate(sourceRows);
        const tbody = panel.querySelector('tbody');
        const fragment = document.createDocumentFragment();

        for (const item of items) {
            const tr = document.createElement('tr');
            const values = [item.eqGroup || '—', item.eqName || '—', item.unit || '—',
                item.subUnit || '—', item.partsGroup || '—',
                item.count, item.intervalCount, formatTime(item.previousTime), formatTime(item.lastTime),
                item.recentDays === null ? '산출 불가' : item.recentDays.toFixed(2),
                item.averageDays === null ? '산출 불가' : item.averageDays.toFixed(2)];
            for (const value of values) {
                const td = document.createElement('td');
                td.textContent = value;
                tr.appendChild(td);
            }
            fragment.appendChild(tr);
        }
        if (!items.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 11;
            td.textContent = '조건에 맞는 완료된 A급·B급 교체 이력이 없습니다.';
            tr.appendChild(td);
            fragment.appendChild(tr);
        }
        tbody.replaceChildren(fragment);
        const insufficient = items.filter(item => !item.intervalCount).length;
        panel.querySelector('#history-cycle-status').textContent =
            `${items.length.toLocaleString()}개 항목 · 교체 이력 1건인 ${insufficient.toLocaleString()}개 항목은 주기 산출 불가` +
            (invalidDateCount ? ` · 작업시간 오류 ${invalidDateCount.toLocaleString()}건 제외` : '');
    }

    function setRows(rows) {
        sourceRows = Array.isArray(rows) ? rows.slice() : [];
        render();
    }

    function adjustHistoryLayout() {
        const $ = window.jQuery;
        const table = document.querySelector('.parts-history-container .history-table');
        if (!$ || !table || !$.fn.DataTable.isDataTable(table)) return;
        const dt = $(table).DataTable();
        dt.columns.adjust();
        if (dt.fixedHeader?.adjust) dt.fixedHeader.adjust();
    }

    function init() {
        button = document.getElementById('history-cycle-btn');
        panel = document.getElementById('history-cycle-panel');
        stack = document.querySelector('.parts-history-wrapper .history-tables-stack');
        if (!button || !panel || !stack) return;

        button.addEventListener('click', () => {
            isOpen = !isOpen;
            button.classList.toggle('active', isOpen);
            button.setAttribute('aria-pressed', String(isOpen));
            button.setAttribute('aria-expanded', String(isOpen));
            panel.hidden = !isOpen;
            stack.classList.toggle('cycle-open', isOpen);
            if (!isOpen) stack.scrollTop = 0;
            render();
            requestAnimationFrame(adjustHistoryLayout);
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();

    return { setRows, calculate };
})();
