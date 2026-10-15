let weeklyChart = null;
let dailyChart = null;
let eqpChart = null;

// ✅ 효율 분석 전체 로딩
function loadEfficiencyAnalysis(selectedEq = "all") {
    resetEfficiencyDetailTable();
    showLoading();  // ✅ 로딩 시작

    const weeklyChartEl = document.getElementById("effi-weekly-chart");
    const dailyChartEl = document.getElementById("effi-daily-chart");
    const eqpChartEl = document.getElementById("effi-eqp-chart");

    const dateRange = document.getElementById("date-range")?.value;

    let url = `/get-first-cv-data`;
    const params = [];

    if (selectedEq && selectedEq !== 'all') {
        params.push(`eq=${encodeURIComponent(selectedEq)}`);
    }

    if (dateRange) {
        // flatpickr 한국어 구분자와 관계없이 선택한 날짜를 사용
        const picker = window.__acvDatePicker;
        const dates = picker?.selectedDates?.length
            ? picker.selectedDates.map(date => picker.formatDate(date, "Y-m-d"))
            : dateRange.match(/\d{4}-\d{2}-\d{2}/g) || [];
        const [start, endRaw] = dates;
        const end = endRaw || start;
        params.push(`start=${start} 00:00:00`);
        params.push(`end=${end} 23:59:59`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }

    clearAnalysisText("#effi-middle-container");

    fetch(url)
        .then(res => res.json())
        .then(async (data) => {
            hideLoading();  // ✅ 성공 시 로딩 종료

            if (!data || data.length === 0) {
                weeklyChart?.dispose();
                dailyChart?.dispose();
                eqpChart?.dispose();
                weeklyChart = dailyChart = eqpChart = null;
                showAnalysisText("데이터가 없습니다.", "#effi-middle-container");
                return;
            }

            renderWeeklyEfficiencyChart(data, weeklyChartEl);
            renderDailyEfficiencyChart(data, dailyChartEl);
            await renderEqpEfficiencyChart(data, eqpChartEl);
        })
        .catch(err => {
            hideLoading();  // ✅ 실패 시에도 로딩 종료
            console.error("❌ 설비 효율 데이터 조회 실패:", err);
            showAnalysisText("데이터를 불러오는 데 실패했습니다.", "#effi-middle-container");
        });
}

// ✅ 주차별 효율 차트
function renderWeeklyEfficiencyChart(data, chartEl) {
    const weeklyMap = {};

    data.forEach(row => {
        const weekKey = row.Week;
        if (!weeklyMap[weekKey]) {
            weeklyMap[weekKey] = { total: 0, count: 0 };
        }
        weeklyMap[weekKey].total += row.effi;
        weeklyMap[weekKey].count++;
    });

    const sortedWeeks = Object.keys(weeklyMap).sort();
    const recent5 = sortedWeeks.slice(-5);
    const values = recent5.map(week => {
        const { total, count } = weeklyMap[week];
        return parseFloat((total / count).toFixed(1));
    });

    weeklyChart?.dispose();
    weeklyChart = echarts.init(chartEl);

    weeklyChart.setOption({
        title: { text: '주차별 평균 효율 (%)', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: recent5 },
        yAxis: { type: 'value', min: 0, max: 100 },
        series: [{
            name: '효율',
            type: 'bar',
            data: values,
            label: { show: true, position: 'top' }
        }]
    });

    // ✅ 오른쪽 상세 표 연결
    weeklyChart.on('click', function (params) {
        const selectedWeek = params.name;
        const filtered = data.filter(row => row.Week === selectedWeek);
        showEfficiencyDetailTable(filtered);
    });

    setTimeout(() => weeklyChart.resize(), 100);
}

// ✅ 일자별 효율 차트
function renderDailyEfficiencyChart(data, chartEl) {
    const efficiencyByDate = {};

    data.forEach(row => {
        const dateKey = formatDateOnly(row.writeTime);
        if (!efficiencyByDate[dateKey]) {
            efficiencyByDate[dateKey] = { total: 0, count: 0 };
        }
        efficiencyByDate[dateKey].total += row.effi;
        efficiencyByDate[dateKey].count++;
    });

    const sortedDates = Object.keys(efficiencyByDate).sort();
    const recent15 = sortedDates.slice(-15);
    const values = recent15.map(date => {
        const { total, count } = efficiencyByDate[date];
        return parseFloat((total / count).toFixed(1));
    });

    dailyChart?.dispose();
    dailyChart = echarts.init(chartEl);

    dailyChart.setOption({
        title: { text: '일자별 평균 효율 (%)', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: recent15 },
        yAxis: { type: 'value', min: 0, max: 100 },
        series: [{
            name: '효율',
            type: 'bar',
            data: values,
            label: { show: true, position: 'top' }
        }]
    });

    dailyChart.on('click', function (params) {
        const selectedDate = params.name;
        const filtered = data.filter(row => formatDateOnly(row.writeTime) === selectedDate);
        showEfficiencyDetailTable(filtered);
    });

    setTimeout(() => dailyChart.resize(), 100);
}

// ✅ 설비별 평균 효율 차트
async function renderEqpEfficiencyChart(data, chartEl) {
    // 같은 날을 시작/종료일로 선택한 경우도 사용자 날짜 필터로 인정
    const userFiltered = !!document.getElementById("date-range")?.value?.trim();

    // (1) 날짜 자동 보정: 최근 7일
    let filteredData = data;
    if (!userFiltered) {
        const today = new Date();
        const sevenDaysAgo = new Date(today);
        sevenDaysAgo.setDate(today.getDate() - 6);
        sevenDaysAgo.setHours(0, 0, 0, 0);
        filteredData = data.filter(row => {
        const writeTime = new Date(row.writeTime);
        return writeTime >= sevenDaysAgo && writeTime <= today;
        });
    }

    // (2) 최신 레시피 맵 가져오기: [{eq, recipe}]
    let latestRecipeMap = {};
    try {
        const resp = await fetch('/get-latest-recipe-map');
        const arr = await resp.json();
        // 빠른 조회를 위해 맵으로 변환
        latestRecipeMap = (arr || []).reduce((acc, { eq, recipe }) => {
        acc[eq] = (recipe && recipe.trim()) || '미정';
        return acc;
        }, {});
    } catch (e) {
        console.warn('⚠️ 최신 레시피 맵 로드 실패 - 레시피 라벨 없이 진행:', e);
    }

    // (3) 평균 효율 집계
    const efficiencyByEq = {};
    filteredData.forEach(row => {
        const eq = row.eq;
        if (!efficiencyByEq[eq]) efficiencyByEq[eq] = { total: 0, count: 0 };
        efficiencyByEq[eq].total += row.effi;
        efficiencyByEq[eq].count++;
    });

    const zonelist = window.zonelist;
    const zoneColors = {
        zone_2: '#5470C6',
        zone_3: '#91CC75',
        zone_4: '#EE6666',
        zone_6: '#73C0DE',
        zone_8: '#FAC858'
    };

    const yLabels = [];
    const seriesData = [];

    Object.entries(zonelist).forEach(([zone, eqList]) => {
        eqList.forEach(eq => {
        const info = efficiencyByEq[eq];
        const avg = info ? Math.round(info.total / info.count) : 0;
        const latestRecipe = latestRecipeMap[eq] ?? '미정';

        yLabels.push(zone.toUpperCase());
        seriesData.push({
            value: avg,
            name: eq,             // 원본 설비명
            zone: zone,
            latestRecipe,         // ✅ 최신 레시피 보관
            itemStyle: {
            color: info ? (zoneColors[zone] || '#ccc') : '#ddd'
            }
        });
        });
    });

    // (4) 유효 데이터 체크
    const hasMeaningfulData = seriesData.some(d => d.value > 0);
    if (!hasMeaningfulData) {
        eqpChart?.dispose();
        eqpChart = null;
        showAnalysisText(userFiltered ? "선택한 기간에 데이터가 없습니다." : "최근 7일간 데이터가 없습니다.", "#effi-middle-container");
        return;
    }

    clearAnalysisText("#effi-middle-container");

    if (!chartEl) return;
    eqpChart?.dispose();
    eqpChart = echarts.init(chartEl);

    eqpChart.setOption({
        title: { text: '설비 평균 효율 (%)', left: 'center' },
        graphic: userFiltered ? [] : [
        { type: 'text', top: 30, right: 30, style: { text: '로딩 데이터: 최근 7일', fill: '#888', fontSize: 12 } }
        ],
        tooltip: {
        trigger: 'item',
        formatter: param => {
            const d = param.data;
            return [
            `ZONE: ${d.zone.toUpperCase()}`,
            `설비: ${d.name}`,
            `RECIPE(현재): ${d.latestRecipe || '미정'}`,
            `효율: ${d.value}%`
            ].join('<br/>');
        }
        },
        grid: { left: 65, right: 10, top: 50, bottom: 60 },
        xAxis: { type: 'value', min: 0, max: 100 },
        yAxis: { type: 'category', inverse: true, data: yLabels, axisLabel: { fontSize: 12 } },
        series: [
        // 1) 막대 안쪽 왼쪽 라벨 → "EQ (RECIPE)"
        {
            name: '효율 (이름+레시피)',
            type: 'bar',
            data: seriesData.map(d => ({ ...d })),
            label: {
                show: true,
                position: 'insideLeft',
                formatter: param => {
                    const d = param.data;
                    const label = `${d.name} - ${d.latestRecipe || '미정'}`;
                    return d.value === 0 ? `{black|${label}}` : `{white|${label}}`;
                },
                rich: {
                    black: { color: '#000', fontSize: 13 },
                    white: { color: '#fff', fontSize: 13 }
                }
            }
        },
        // 2) 막대 오른쪽 값 라벨
        {
            name: '효율 (값)',
            type: 'bar',
            data: seriesData.map(d => ({ ...d })),
            label: {
            show: true,
            position: 'right',
            formatter: p => p.data.value === 0 ? '' : `${p.data.value}%`,
            fontSize: 12,
            color: '#000'
            },
            itemStyle: { color: 'transparent' },
            barGap: '-100%'
        }
        ]
    });

    // 오른쪽 상세 표 클릭 연결
    eqpChart.on('click', function (params) {
        const selectedEq = params.data.name;
        const filtered = filteredData.filter(row => row.eq === selectedEq);
        showEfficiencyDetailTable(filtered);
    });

    setTimeout(() => eqpChart.resize(), 100);
}


// ✅ 날짜 형식 유틸
function formatDateTime(isoStr) {
    const d = new Date(isoStr);
    const mm = (d.getMonth() + 1).toString().padStart(2, '0');
    const dd = d.getDate().toString().padStart(2, '0');
    const hh = d.getHours().toString().padStart(2, '0');
    const mi = d.getMinutes().toString().padStart(2, '0');
    const ss = d.getSeconds().toString().padStart(2, '0');
    return `${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function formatDateOnly(isoStr) {
    const d = new Date(isoStr);
    const mm = (d.getMonth() + 1).toString().padStart(2, '0');
    const dd = d.getDate().toString().padStart(2, '0');
    return `${mm}-${dd}`;
}