// scripts_mon_acv_chart.js

let acvChartInstance = null;
let cycleChunks = [];
let cvLabelChunks = [];
let currentCycleIndex = 0;
let recipeData = null;   // 마지막(최신) 레시피(하위 호환)
let recipeList = [];     // 사이클별 레시피 목록(인덱스=사이클 인덱스)

function clearCycleChartWithMsg(msg = null) {
    console.log('[cycle] clearCycleChartWithMsg called. msg =', msg);

    try {
        if (typeof acvChartInstance !== 'undefined' && acvChartInstance) {
        console.log('[cycle] disposing echarts instance…');
        acvChartInstance.dispose();
        }
    } catch (e) {
        console.warn('[cycle] dispose failed:', e);
    }
    acvChartInstance = null;

    // 상태 초기화
    cycleChunks = [];
    cvLabelChunks = [];
    recipeData = null;
    recipeList = [];
    currentCycleIndex = 0;

    // UI 초기화
    const counter = document.getElementById('cycle-counter');
    if (counter) counter.textContent = '';
    const sel = document.getElementById('cycle-selector');
    if (sel) sel.innerHTML = `<option value="">Track In 시간 선택</option>`;

    // 컨테이너 정리 + 안내문
    const container = document.getElementById('acv-chart-container');
    if (container) {
        container.innerHTML = '';
    }
    if (msg) {
        showAnalysisText(msg, '#acv-chart-container');
    } else {
        clearAnalysisText?.('#acv-chart-container');
    }
}

function formatYmd(d) {
    if (!(d instanceof Date) || isNaN(d)) return null;
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

// flatpickr에서 선택된 범위를 YYYY-MM-DD 문자열로 반환
function getSelectedRangeYmd() {
    const input = document.getElementById("date-range");
    if (window.__acvDatePicker && Array.isArray(window.__acvDatePicker.selectedDates)) {
        const [s, e] = window.__acvDatePicker.selectedDates;
        if (s || e) {
            const start = formatYmd(s || e);
            const end   = formatYmd(e || s);
            return [start, end];
        }
    }
    // 폴백: input 문자열 파싱 (구분자 둘 다 대응)
    const raw = input?.value || '';
    let parts = raw.split(' ~ ');
    if (parts.length < 2) parts = raw.split(' to ');
    const start = (parts[0] || '').trim();
    const end   = (parts[1] || start).trim();
    return [start || null, end || null];
}

function fetchAndRenderCycles(eqName) {
    showLoading();

    const container = document.getElementById("acv-chart-container");
    if (!container) return;

    // 구분자에 의존하지 않고 flatpickr API로 읽음
    let [start, end] = getSelectedRangeYmd();

    // 날짜가 없으면 차트 그리지 않고 안내문만 출력
    if (!start || !end) {
        hideLoading();
        acvChartInstance?.dispose?.();
        acvChartInstance = null;
        document.getElementById("cycle-counter").textContent = '';
        showAnalysisText("날짜를 선택하거나, 데이터가 있는 레시피를 선택하세요.", "#acv-chart-container");
        return;
    }

    clearAnalysisText("#acv-chart-container");

    const recipeSel = document.getElementById('recipe-select');
    const recipeVal = recipeSel ? recipeSel.value : 'all';

    if (!recipeVal || recipeVal === 'all') {
        return;
    }

    let url = `/get-acv-data?eq=${encodeURIComponent(eqName)}&start=${start} 00:00:00&end=${end} 23:59:59`;
    if (recipeVal && recipeVal !== 'all') {
        url += `&recipe=${encodeURIComponent(recipeVal)}`;
    }

    fetch(url)
        .then(res => res.json())
        .then(data => {
            hideLoading();

            if (!data.timestamps || data.timestamps.length === 0) {
                acvChartInstance?.dispose();
                acvChartInstance = null;
                document.getElementById("cycle-counter").textContent = '';

                // ✅ 선택한 날짜(범위) 기준으로 메시지 출력
                const dateLabel = (start === end) ? start : `${start} ~ ${end}`;
                showAnalysisText(`${dateLabel} 데이터가 없습니다.`, "#acv-chart-container");
                return;
            }

            cycleChunks = splitIntoCycles(data);
            cvLabelChunks = data.cv_labels || [];
            
            recipeData = data.recipe || null; // 마지막(최신) 레시피
            recipeList = Array.isArray(data.recipe_list) ? data.recipe_list : [];
            currentCycleIndex = 0;

            if (cycleChunks.length === 0) {
                acvChartInstance?.dispose();
                acvChartInstance = null;
                document.getElementById("cycle-counter").textContent = '';
                showAnalysisText("사이클이 정상적으로 진행된 데이터가 없습니다.", "#acv-chart-container");
                return;
            }

            acvChartInstance?.dispose();
            acvChartInstance = echarts.init(container);
            renderCycleChart(cycleChunks[currentCycleIndex]);
            populateCycleSelector();
        })
        .catch(err => {
            hideLoading();
            console.error("❌ fetch 실패:", err);
            acvChartInstance?.dispose();
            acvChartInstance = null;
            showAnalysisText("데이터를 불러오는 데 실패했습니다.", "#acv-chart-container");
        });
}

function splitIntoCycles(data) {
    const chunks = [];
    let currentCycle = [];
    let inCycle = false;

    for (let i = 0; i < data.process_step.length; i++) {
        const step = data.process_step[i];

        if (step === 12 && !inCycle) {
            currentCycle = [];
            inCycle = true;
        }

        if (inCycle) {
            currentCycle.push({
                timestamp: formatTimestamp(data.timestamps[i]),
                cur_temp: data.cur_temp[i],
                cur_press: data.cur_press[i],
                process_step: step,
                pack_off_timeover: data.pack_off_timeover[i],
                pack_on_timeover: data.pack_on_timeover[i],
                micro_press: data.micro_press[i],  
                cb_cv1: data.cb_cv1[i],
                cb_cv2: data.cb_cv2[i],
                cb_cv3: data.cb_cv3[i],
                cb_cv4: data.cb_cv4[i]
            });
        }

        if (step === 25 && inCycle) {
            chunks.push(currentCycle);
            currentCycle = [];
            inCycle = false;
        }
    }

    return chunks;
}

function renderCycleChart(cycleData) {
    
    const option = {
        title: { text: `사이클 ${currentCycleIndex + 1}`, left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: {
            data: ['Cur_Temp', 'Cur_Press', 'Process_Step', 'Pack_OFF_Timeover', 'Pack_On_Timeover', 'Micro_Press'],
            top: '10%'
        },
        grid: {
            left: 20,
            right: 20,
            top: '20%',
            bottom: 10,
            containLabel: true
        },
        dataZoom: [
            {
                type: 'slider',
                show: true,
                xAxisIndex: 0,
                height: 20,
                bottom: 10  // ✅ 위치 조정
            },
            {
                type: 'inside',
                xAxisIndex: 0
            }
        ],
        xAxis: {
            type: 'category',
            data: cycleData.map(d => d.timestamp),
            axisLabel: { rotate: 45 }
        },
        yAxis: [
            { type: 'value', name: '값', minInterval: 1 },
            {
                type: 'value',
                name: 'Binary',
                min: 0,
                max: 1,
                interval: 1,
                axisLabel: { formatter: v => v.toFixed(0) }
            }
        ],
        series: [
            { name: 'Cur_Temp', type: 'line', data: cycleData.map(d => d.cur_temp), smooth: true },
            { name: 'Cur_Press', type: 'line', data: cycleData.map(d => d.cur_press), smooth: true },
            { name: 'Process_Step', type: 'line', data: cycleData.map(d => d.process_step), step: 'middle' },
            { name: 'Pack_OFF_Timeover', type: 'line', data: cycleData.map(d => d.pack_off_timeover), step: 'middle' },
            { name: 'Pack_On_Timeover', type: 'line', data: cycleData.map(d => d.pack_on_timeover), step: 'middle',lineStyle: { color: '#FF8339' },itemStyle: { color: '#FF8339' } },
            { name: 'Micro_Press', type: 'bar', data: cycleData.map(d => d.micro_press), yAxisIndex: 1, itemStyle: {color: '#6EC6E8'} },
        ]
    };

    acvChartInstance.setOption(option);
    const recipeForCycle = recipeList[currentCycleIndex] || recipeData || null;

    updateInfoTable(cvLabelChunks[currentCycleIndex],cycleChunks[currentCycleIndex],recipeForCycle);

    document.getElementById("cycle-counter").textContent =
        `사이클 ${currentCycleIndex + 1} / ${cycleChunks.length}`;
}

// ✅ 1. boost_on 구간 분석 함수 (함수 외부)
function findMicroPressRanges(data) {
    const ranges = [];
    let inOne = false;
    let startIdx = -1;

    for (let i = 0; i < data.length; i++) {
        if (data[i].micro_press === 1) {
            if (!inOne) {
                startIdx = i;
                inOne = true;
            }
        } else {
            if (inOne) {
                ranges.push({ start: startIdx, end: i - 1 });
                inOne = false;
            }
        }
    }
    if (inOne) {
        ranges.push({ start: startIdx, end: data.length - 1 });
    }

    return ranges;
}

function updateInfoTable(labelData, cycleData, recipeForCycle)  {
    const infoTbody = document.querySelector("#acv-lot-info-table tbody");
    const statsTbody = document.querySelector("#acv-cycle-info-table tbody");
    infoTbody.innerHTML = "";
    statsTbody.innerHTML = "";

    // ✅ 시간 계산용 stepMap 구성
    const stepMap = {};
    cycleData.forEach(d => {
        if (!stepMap[d.process_step]) {
            stepMap[d.process_step] = new Date(d.timestamp);
        }
    });

    const pressStart = stepMap[15];
    const pressEnd = stepMap[19];
    const exhaustStart = stepMap[21];
    const exhaustEnd = stepMap[25];
    const runStart = stepMap[12];

    const trackInTime = runStart ? formatTimestamp(runStart) : "-";
    const trackOutTime = exhaustEnd ? formatTimestamp(exhaustEnd) : "-";

    // ✅ 투입 LOT 수
    const lotCount = [labelData?.cv1, labelData?.cv2, labelData?.cv3, labelData?.cv4]
        .filter(v => v && v !== "-").length;

    const getDiffSeconds = (start, end) =>
        (start && end) ? Math.round((end - start) / 1000) : "-";

    const pressTime = getDiffSeconds(pressStart, pressEnd);
    const exhaustTime = getDiffSeconds(exhaustStart, exhaustEnd);

    // ✅ boost_on 관련 계산 - STEP 19 시작 이후 30초부터 집계
    const boostRanges = (() => {
        if (!pressEnd) return [];

        const thresholdTime = new Date(pressEnd.getTime() + 30 * 1000); // 30초 이후 기준
        const filteredData = cycleData.filter(d => new Date(d.timestamp) >= thresholdTime);

        return findMicroPressRanges (filteredData);
    })();

    // 최소 Leak Time
    let minLeakTime = "-";
    if (boostRanges.length >= 2) {
        let min = Infinity;
        for (let i = 1; i < boostRanges.length; i++) {
            const prevEndTime = new Date(cycleData[boostRanges[i - 1].end].timestamp);
            const nextStartTime = new Date(cycleData[boostRanges[i].start].timestamp);
            const diffSec = Math.round((nextStartTime - prevEndTime) / 1000);
            if (diffSec < min) min = diffSec;
        }
        minLeakTime = min !== Infinity ? min : "-";
    }

    // 미세가압 Count
    const finePressCount = boostRanges.length;

    // ✅ boost_on 유지 시간 계산
    let maxBoostOnTime = "-";
    if (cycleData.length > 1) {
        let maxCount = 0;
        let currentCount = 0;

        for (let i = 1; i < cycleData.length; i++) {
            if (cycleData[i].micro_press === 1) {
                if (cycleData[i - 1].micro_press === 1) {
                    const prevTime = new Date(cycleData[i - 1].timestamp);
                    const currTime = new Date(cycleData[i].timestamp);
                    const diffSec = Math.round((currTime - prevTime) / 1000);
                    currentCount += diffSec;
                } else {
                    currentCount = 0;
                }
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                }
            } else {
                currentCount = 0;
            }
        }

        maxBoostOnTime = maxCount > 0 ? maxCount : "-";
    }

    // ✅ LOT 테이블 (Track + CV)
    const lotRows = [
        {
            label: "Recipe",
            value: recipeForCycle
                ? `${recipeForCycle.press}bar _ ${recipeForCycle.temp}º _ ${recipeForCycle.time}min`
                : "-"
        },
        { label: "Track In", value: trackInTime },
        { label: "Track Out", value: trackOutTime },
        { label: "CV1", value: labelData?.cv1 || "-" },
        { label: "CV2", value: labelData?.cv2 || "-" },
        { label: "CV3", value: labelData?.cv3 || "-" },
        { label: "CV4", value: labelData?.cv4 || "-" }
    ];
    lotRows.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.label}</td><td>${row.value}</td>`;
        infoTbody.appendChild(tr);
    });

    // ✅ 상세 항목 테이블
    const stats = [
        { label: "투입 LOT (Lot)", value: lotCount },
        { label: "미세가압 Count (회)", value: finePressCount },
        { label: "미세가압 Time (초)", value: maxBoostOnTime },
        { label: "미세가압 Interval (초)", value: minLeakTime },
        { label: "가압 시간 (초)", value: pressTime },
        { label: "배기 시간 (초)", value: exhaustTime },
        {
            label: "패킹 On (초)",
            value: (() => {
                const maxVal = Math.max(...cycleData.map(d => d.pack_on_timeover || 0));
                return isFinite(maxVal) ? maxVal.toFixed(1) : "-";
            })()
        },
        {
            label: "패킹 Off (초)",
            value: (() => {
                const maxVal = Math.max(...cycleData.map(d => d.pack_off_timeover || 0));
                return isFinite(maxVal) ? maxVal.toFixed(1) : "-";
            })()
        }
    ];
    stats.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.label}</td><td>${row.value}</td>`;
        statsTbody.appendChild(tr);
    });
}


function formatTimestamp(date) {
    if (typeof date === 'string') {
        date = new Date(date);
    }

    if (!(date instanceof Date) || isNaN(date)) {
        console.warn("⚠️ Invalid date passed to formatTimestamp:", date);
        return '-';
    }

    const yy = String(date.getFullYear()); // ✅ 전체 연도 출력
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const hh = String(date.getHours()).padStart(2, '0');
    const mi = String(date.getMinutes()).padStart(2, '0');
    const ss = String(date.getSeconds()).padStart(2, '0');
    return `${yy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function populateCycleSelector() {
    const selector = document.getElementById("cycle-selector");
    selector.innerHTML = `<option value="">Track In 시간 선택</option>`; // 초기화

    cycleChunks.forEach((cycle, index) => {
        const step12 = cycle.find(d => d.process_step === 12);
        const label = step12 ? formatTimestamp(step12.timestamp) : `사이클 ${index + 1}`;

        const option = document.createElement("option");
        option.value = index;
        option.textContent = label;
        selector.appendChild(option);
    });
}

// 좌우 버튼

document.getElementById("prev-cycle").addEventListener("click", () => {
    if (currentCycleIndex > 0) {
        currentCycleIndex--;
        renderCycleChart(cycleChunks[currentCycleIndex]);
    }
});

document.getElementById("next-cycle").addEventListener("click", () => {
    if (currentCycleIndex < cycleChunks.length - 1) {
        currentCycleIndex++;
        renderCycleChart(cycleChunks[currentCycleIndex]);
    }
});

document.getElementById("cycle-selector").addEventListener("change", (e) => {
    const idx = parseInt(e.target.value);
    if (!isNaN(idx) && cycleChunks[idx]) {
        currentCycleIndex = idx;
        renderCycleChart(cycleChunks[idx]);
    }
});
