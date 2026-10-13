let ec = null;
let resizeHandler = null;
let fetchCtrl = null;      // AbortController
let lastReqId = 0;         // 요청 가드용 id
let trendFiltersBound = false;

// 날짜 범위 파싱 
function getDateRange() {
    const v = document.getElementById('date-range')?.value || '';
    if (!v.includes('~')) return { start: null, end: null };
    const [s, eRaw] = v.split('~').map(t => t.trim());
    const e = eRaw || s;
    return { start: `${s} 00:00:00`, end: `${e} 23:59:59` };
}

// 차트 초기화
function ensureChart() {
    const $trendChart = $('#acv-breakdown-container');
    if ($trendChart.length === 0) return null;   // 컨테이너 없음
    const el = $trendChart[0];                   // DOM 노드

    if (!ec) {
        ec = echarts.init(el);
    }

    // 리사이즈 핸들러 갱신
    if (resizeHandler) {
        $(window).off('resize', resizeHandler);
    }
    resizeHandler = () => { if (ec) ec.resize(); };
    $(window).on('resize', resizeHandler);

    return ec;
}

// 차트/리스너 완전 정리
function clearChart() {
    const $trendChart = $('#acv-breakdown-container');
    if ($trendChart.length === 0) return;

    if (ec) {
        try { ec.dispose(); } catch (_) {}
        ec = null;
    }
    if (resizeHandler) {
        $(window).off('resize', resizeHandler);
        resizeHandler = null;
    }
    $trendChart.empty(); // 컨테이너 비우기
}

// 메시지 표시
function showMessage(html) {
    const $trendChart = $('#acv-breakdown-container');
    if ($trendChart.length) {
        $trendChart.html(`
            <div style="
                display: flex;
                justify-content: center;   /* 가로 가운데 */
                align-items: center;       /* 세로 가운데 */
                height: 100%;              /* 부모 높이에 맞춤 */
            ">
                ${html}
            </div>
        `);
    }
}

// recipe를 쿼리로 함께 전달
async function fetchBreakdown(eq, start, end, recipe) {
    // 이전 요청 취소
    if (fetchCtrl) {
        try { fetchCtrl.abort(); } catch(e) {}
    }
    fetchCtrl = new AbortController();

    const params = new URLSearchParams();
    params.set('eq', eq || 'all');
    if (start && end) { params.set('start', start); params.set('end', end); }

    const r = (recipe ?? (document.getElementById('recipe-select')?.value ?? 'all'));
    if (r && r !== 'all') params.set('recipe', r);

    const res = await fetch(`/get-acv-trend-chart?${params.toString()}`, {
        signal: fetchCtrl.signal
    });
    if (!res.ok) throw new Error('get-acv-trend-chart 실패');
    return res.json();
}

// 값 변환
function toNumOrNull(v) {
    if (v === null || v === undefined) return null;
    if (v === '\\N' || v === '-' || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}
const _num = (v) => {
    const n = toNumOrNull(v);
    return n == null ? null : n;
};

// 메트릭/단위/축 매핑 
const METRICS = [
    'micro_press_count',
    'micro_press_time',
    'leak_interval_time',
    'press_time',
    'exhaust_time',
    'packing_on_time',
    'packing_off_time',
];

const UNITS = {
    micro_press_count: 'count',
    micro_press_time: 's',
    leak_interval_time: 's',
    press_time: 's',
    exhaust_time: 's',
    packing_on_time: 's',
    packing_off_time: 's',
};

// 보조축(오른쪽)으로 빼야 하는 키
const SECONDARY_AXIS_KEYS = new Set(['micro_press_count']);

//  일자별 집계 규칙 
const AGG_RULES = {
    micro_press_count: 'max',
    micro_press_time: 'max',
    leak_interval_time: 'min',
    press_time: 'max',
    exhaust_time: 'max',
    packing_on_time: 'max',
    packing_off_time: 'max',
};

function _ymd(ts) {
    const d = new Date(ts);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

// rows: [{ ts: datetime/string, metrics... }]
function aggregateDailyRows(rows, xKey = 'ts') {
    const daily = new Map(); // 'YYYY-MM-DD' -> { ts(ms at 00:00:00), metric... }

    for (const r of rows) {
        const ts = r[xKey] ? new Date(r[xKey]).getTime() : null;
        if (!ts) continue;

        const key = _ymd(ts);
        if (!daily.has(key)) {
            const dt = new Date(key + 'T00:00:00');
            daily.set(key, { ts: dt.getTime() });
        }
        const bucket = daily.get(key);

        for (const k of METRICS) {
            const v = _num(r[k]);
            if (v == null) continue;
            if (!(k in bucket) || bucket[k] == null) {
                bucket[k] = v;
            } else {
                const rule = AGG_RULES[k];
                if (rule === 'max') bucket[k] = Math.max(bucket[k], v);
                else if (rule === 'min') bucket[k] = Math.min(bucket[k], v);
            }
        }
    }

    return Array.from(daily.values()).sort((a, b) => a.ts - b.ts);
}

//  시리즈 구성 
function buildDailySeries(rows, metricKeys) {
    return metricKeys.map((key) => {
        const data = rows.map((r) => [r.ts, r[key] ?? null]);
        return {
            name: key,
            type: 'line',
            showSymbol: true,      // ✅ 항상 동그라미 표시
            symbolSize: 6,         // ✅ 동그라미 크기 (기본 4 → 6으로 살짝 키움)
            connectNulls: true,
            yAxisIndex: SECONDARY_AXIS_KEYS.has(key) ? 1 : 0,
            data,
        };
    });
}

// 보기 좋은 라벨
function prettyName(key) {
    const map = {
        micro_press_count: 'Micro-press 횟수',
        micro_press_time: 'Micro-press 시간',
        leak_interval_time: 'Leak 간격',
        press_time: 'Press',
        exhaust_time: 'Exhaust',
        packing_on_time: 'Packing On',
        packing_off_time: 'Packing Off',
    };
    return map[key] || key;
}

//  옵션 
function optionForSingleEq(eq, dailyRows) {
    const series = buildDailySeries(dailyRows, METRICS);

    // ✅ recipe-select의 현재 값 가져오기
    const recipeVal = document.getElementById('recipe-select')?.value || 'all';
    let countMax = 10; // 기본값(예: 5bar 계열)

    // ✅ recipe 시작 문자가 6 또는 8이면 max를 30으로
    if (typeof recipeVal === 'string' && (recipeVal.startsWith('6') || recipeVal.startsWith('8'))) {
        countMax = 30;
    }

    // ✅ 초기 노출 시리즈(나머지는 숨김)
    const defaultSelected = {
        micro_press_count: true,
        leak_interval_time: true,
        micro_press_time: false,
        press_time: false,
        exhaust_time: false,
        packing_on_time: false,
        packing_off_time: false
    };

    return {
        title: { text: `고장 차트(일자별) - ${eq}`, left: 'center' },
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                if (!params?.length) return '';
                const ts = params[0].value?.[0];
                const d = ts ? new Date(ts) : null;
                const dateStr = d
                    ? `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
                    : '';
                const lines = [`<div>${dateStr}</div>`];
                params.forEach((p) => {
                    const key = p.seriesName;
                    const unit = UNITS[key] || '';
                    const val = p.value?.[1];
                    const valStr = (val == null) ? '-' : String(val);
                    lines.push(`${prettyName(key)}: <b>${valStr}</b>${unit ? ' ' + unit : ''}`);
                });
                return lines.join('<br/>');
            },
        },
        legend: {
            top: 28,
            type: 'scroll',
            formatter: (name) => prettyName(name),
            selected: defaultSelected, // ✅ 초기 표시 제어
        },
        grid: { left: 56, right: 64, top: 64, bottom: 56 },
        xAxis: { type: 'time', boundaryGap: false },
        yAxis: [
            // 주축(Time)
            { type: 'value', name: 'Time (s)', min: 0, max: 500, axisLine: { show: true }, splitLine: { show: true } },
            // 보조축(Count) — recipe 조건에 따라 max 변경
            { type: 'value', name: 'Count', min: 0, max: countMax, axisLine: { show: true }, splitLine: { show: false } },
        ],
        dataZoom: [{ type: 'inside' }, { type: 'slider' }],
        series,
    };
}

//  렌더(직접 호출) 
async function renderTrendChart() {
    const eq = document.getElementById('eq-group-select')?.value || 'all';
    const recipe = document.getElementById('recipe-select')?.value || 'all';
    const { start, end } = getDateRange();

    if (eq === 'all' || recipe === 'all') {
        hideChartContainer();
        clearChart();
        return;
    }

    // ✅ 이 렌더의 고유 id
    const reqId = ++lastReqId;

    showChartContainer();

    let rows = [];
    try {
        rows = await fetchBreakdown(eq, start, end, recipe);
    } catch (e) {
        // abort 시도 중이면 조용히 무시
        if (e?.name !== 'AbortError') {
            console.error(e);
            showMessage('데이터 조회 실패');
        }
        return;
    }

    // ✅ 늦게 도착한 응답이면 무시
    if (reqId !== lastReqId) return;

    if (!rows || rows.length === 0) {
        clearChart();
        showMessage('데이터가 없습니다.');
        return;
    }

    rows.forEach((r) => { if (!r.ts && r.track_in) r.ts = r.track_in; });
    const dailyRows = aggregateDailyRows(rows, 'ts');

    const chart = ensureChart();
    if (!chart) return;
    chart.setOption(optionForSingleEq(eq, dailyRows), true);
}

//  필터 변경 시 재렌더 
function showChartContainer() {
    $('#acv-breakdown-container').show();
}

function hideChartContainer() {
    $('#acv-breakdown-container').hide();
}

function renderTrendChartFilters() {
    if (trendFiltersBound) return;  // ← 중복 방지
    trendFiltersBound = true;

    const eqSel = document.getElementById('eq-group-select');
    const recipeSel = document.getElementById('recipe-select');

    const safeRenderOrHide = () => {
        const eq = eqSel?.value || 'all';
        const recipe = recipeSel?.value || 'all';
        if (eq === 'all' || recipe === 'all') {
            hideChartContainer();
            clearChart();
            return;
        }
        renderTrendChart();
    };

    // 설비 변경
    eqSel?.addEventListener('change', safeRenderOrHide);

    // ✅ 레시피 변경: 설비를 'all'로 초기화하고 change 이벤트까지 발생
    recipeSel?.addEventListener('change', () => {
        // 비행 중 요청 취소 + 즉시 숨김
        if (fetchCtrl) { try { fetchCtrl.abort(); } catch(e) {} }
        hideChartContainer();
        clearChart();

        if (eqSel) {
            eqSel.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });

    document.getElementById('date-range')?.addEventListener('change', safeRenderOrHide);
    document.getElementById('date-init-btn')?.addEventListener('click', safeRenderOrHide);
}
