/* =========================
   motor chart.js
   - Chart 분석 탭 제거
   - CUSUM 기능 제거
   - 모니터링 탭 전용
   - 라인 차트 전용
========================= */

let chartInstances = [];
const chartRenderVersions = new WeakMap();

const DEFAULT_DAYS_RANGE = 30;

let idleThresholdMap = null;
let isRunFilterEnabled = false;

/* =========================
   idle threshold 관련
========================= */

async function loadIdleThresholdMap() {
    if (idleThresholdMap) return idleThresholdMap;

    try {
        const res = await fetch('js/07_mon_load/idle_threshold.json');
        if (!res.ok) {
            throw new Error(`idle_threshold.json load failed: ${res.status}`);
        }
        idleThresholdMap = await res.json();
        return idleThresholdMap;
    } catch (err) {
        console.error('idle_threshold.json 로드 실패:', err);
        idleThresholdMap = {};
        return idleThresholdMap;
    }
}

// 화면 표시용 eqp -> json 키용 eqp 변환
function convertDisplayEqpToThresholdEqp(displayEqp) {
    const eqp = String(displayEqp || '').trim().toUpperCase();

    if (eqp === 'A0G') return 'PLM#160G';
    if (eqp === 'A1G') return 'PLM#161G';
    if (eqp === 'A2G') return 'PLM#162G';

    // 35G -> PLM#35G
    if (/^\d{2,3}G$/.test(eqp)) {
        return `PLM#${eqp}`;
    }

    // 혹시 이미 PLM# 형태면 그대로 사용
    if (/^PLM#\d{2,3}G$/.test(eqp)) {
        return eqp;
    }

    return eqp;
}

function getIdleThreshold(displayEqp, motorKey) {
    if (!idleThresholdMap) return null;

    const thresholdEqp = convertDisplayEqpToThresholdEqp(displayEqp);
    const key = `${thresholdEqp}|${motorKey}`;
    const value = idleThresholdMap[key];

    return Number.isFinite(Number(value)) ? Number(value) : null;
}

function filterRunSeriesData(displayEqp, seriesList = []) {
    return (seriesList || [])
        .map(series => {
            const threshold = getIdleThreshold(displayEqp, series.name);

            // threshold 없으면 원본 유지
            if (!Number.isFinite(threshold)) {
                return {
                    ...series,
                    data: [...(series.data || [])]
                };
            }

            return {
                ...series,
                data: (series.data || []).filter(([time, val]) => Number(val) >= threshold)
            };
        })
        .filter(series => Array.isArray(series.data) && series.data.length > 0);
}

/* =========================
   공통 유틸
========================= */

function registerChart(instance, containerId = null) {
    chartInstances.push({ instance, containerId });
}

function disposeChartsByContainer(targetContainer = null) {
    if (!targetContainer?.length) return;

    const containerId = targetContainer.attr('id');
    if (!containerId) return;

    try { echarts.disconnect('linkedZoomGroup'); } catch (e) {}

    const remain = [];

    for (const item of chartInstances) {
        if (item.containerId === containerId) {
            try { item.instance?.dispose(); } catch (e) {}
        } else {
            remain.push(item);
        }
    }

    chartInstances = remain;
}

function disposeAllCharts() {
    try { echarts.disconnect('linkedZoomGroup'); } catch (e) {}

    for (const item of chartInstances) {
        try { item.instance?.dispose(); } catch (e) {}
    }

    chartInstances = [];
}

function removeCharts(targetContainer = null) {
    
    const $container = targetContainer || monitoringChartsContainer || null;
    if (!$container?.length) return;

    // 차트 제거/선택 변경 시 진행 중인 이전 조회도 무효화한다.
    const containerEl = $container[0];
    chartRenderVersions.set(containerEl, (chartRenderVersions.get(containerEl) || 0) + 1);
    disposeChartsByContainer($container);
    $container.empty();
    $container.removeClass('with-border');
}

function isValidDateStr(s) {
    if (!s || /null|undefined/i.test(String(s))) return false;
    const t = new Date(s);
    return !Number.isNaN(t.getTime());
}

function toYMDLocal(d) {
    return new Date(d.getTime() - d.getTimezoneOffset() * 60000)
        .toISOString()
        .slice(0, 10);
}

function normalizeSeriesData(rawSeries = []) {
    return rawSeries
        .map(series => ({
            ...series,
            data: (series.data || [])
                .map(p => {
                    const time = p[0];
                    const value = Number(p[1]);
                    return [time, value];
                })
                .filter(p => p[0] && Number.isFinite(p[1]))
                .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
        }))
        .filter(series => series.data.length > 0);
}

function getContainerSelector($container) {
    const containerId = $container?.attr?.('id');
    return containerId ? `#${containerId}` : null;
}

function safeShowAnalysisText(message, targetSelector) {
    if (!targetSelector) return;
    if (typeof showAnalysisText === 'function') {
        showAnalysisText(message, targetSelector);
    }
}

function getDefaultQueryRange() {
    const today = new Date();
    const from = new Date(today);
    from.setDate(from.getDate() - DEFAULT_DAYS_RANGE + 1);

    return {
        startDate: `${toYMDLocal(from)} 00:00:00`,
        endDate: `${toYMDLocal(today)} 23:59:59`
    };
}

function buildQueryParams(eqp, startDate, endDate) {
    const queryParams = { eqp };

    if (isValidDateStr(startDate) && isValidDateStr(endDate)) {
        queryParams.startDate = startDate;
        queryParams.endDate = endDate;
    } else {
        Object.assign(queryParams, getDefaultQueryRange());
    }

    return queryParams;
}

/* =========================
   메인 차트 생성
========================= */

async function createMotorCharts(
    eqp,
    selectedAxis = 'ALL',
    axisType,
    startDate = null,
    endDate = null,
    targetContainer = null
) {
    const $container = targetContainer || monitoringChartsContainer || null;

    console.log('[createMotorCharts] start', {
        eqp,
        selectedAxis,
        axisType,
        startDate,
        endDate,
        hasTargetContainer: !!targetContainer,
        containerId: $container?.attr?.('id')
    });

    if (!$container || !$container.length) {
        console.warn('차트를 그릴 컨테이너가 없습니다.');
        return;
    }

    const targetSel = getContainerSelector($container);

    if (!eqp) {
        removeCharts($container);
        safeShowAnalysisText('설비를 선택하세요.', targetSel);
        return;
    }

    if (!axisType || axisType === 'Select Type') {
        removeCharts($container);
        safeShowAnalysisText('타입을 선택하세요.', targetSel);
        return;
    }

    removeCharts($container);
    const containerEl = $container[0];
    const renderVersion = chartRenderVersions.get(containerEl);
    const isCurrentRender = () => chartRenderVersions.get(containerEl) === renderVersion;
    await loadIdleThresholdMap();
    if (!isCurrentRender()) return;

    const sourceGroups = axisType === 'Ball Screw' ? ballScrewGroups : autoStageGroups;
    const groups = sourceGroups.map(group => ({ ...group, data: [] }));
    const queryParams = buildQueryParams(eqp, startDate, endDate);

    console.log('[createMotorCharts] queryParams', queryParams);
    console.log('[createMotorCharts] groups', groups.map(g => ({
        name: g.name,
        indices: g.indices
    })));

    if (
        (typeof selectedStartDate === 'undefined' || !selectedStartDate) &&
        (typeof selectedEndDate === 'undefined' || !selectedEndDate)
    ) {
        $('#date-range').val(
            `${queryParams.startDate.slice(0, 10)} ~ ${queryParams.endDate.slice(0, 10)}`
        );
    }

    showLoading();

    $.get('/get-ph2-motor-pick-load-data', queryParams, function (rawData) {
        if (!isCurrentRender()) return;
        try {
            if (!Array.isArray(rawData) || rawData.length === 0) {
                $container.empty().addClass('with-border');
                safeShowAnalysisText('데이터가 없습니다.', targetSel);
                return;
            }

            groups.forEach(group => {
                group.data = [];
            });

            rawData.forEach((item) => {
                const time = item.WriteTime || item.time || item.x;
                if (!time) return;

                Object.keys(item).forEach(key => {
                    if (!key.startsWith('SM')) return;

                    const motorKey = key;
                    const val = Number(item[key]);
                    if (!Number.isFinite(val)) return;

                    if (selectedAxis !== 'ALL' && motorKey !== selectedAxis) return;

                    const smNum = parseInt(motorKey.replace(/^SM/i, ''), 10);
                    if (!Number.isFinite(smNum)) return;

                    const targetGroup = groups.find(
                        g => Array.isArray(g.indices) && g.indices.includes(smNum)
                    );
                    if (!targetGroup) return;

                    let series = targetGroup.data.find(s => s.name === motorKey);
                    if (!series) {
                        series = { name: motorKey, data: [] };
                        targetGroup.data.push(series);
                    }

                    series.data.push([time, val]);
                });
            });

            $container.empty().removeClass('with-border');

            let hasAnyChart = false;
            const safeEqp = String(eqp).replace(/[^a-zA-Z0-9_-]/g, '_');

            // 축 선택 시 해당 축만 1개 차트
            if (selectedAxis && selectedAxis !== 'ALL') {
                let selectedSeries = null;

                for (const group of groups) {
                    let filteredGroupData = normalizeSeriesData(group.data);

                    if (isRunFilterEnabled) {
                        filteredGroupData = filterRunSeriesData(eqp, filteredGroupData);
                    }

                    const found = filteredGroupData.find(s => s.name === selectedAxis);
                    if (found) {
                        selectedSeries = found;
                        break;
                    }
                }

                if (selectedSeries && selectedSeries.data.length > 0) {
                    hasAnyChart = true;

                    const lineChartId =
                        `linechart-${safeEqp}-${selectedAxis}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

                    $container.append(`
                        <div class="chart-pair single-chart">
                            <div id="${lineChartId}" class="linechart-container"></div>
                        </div>
                    `);

                    createLineChart(
                        lineChartId,
                        [selectedSeries],
                        smLabels?.[selectedAxis] || selectedAxis
                    );
                }
            } else {
                // ALL 선택 시 그룹별 차트
                groups.forEach((group, index) => {
                    let filteredGroupData = normalizeSeriesData(group.data);

                    if (isRunFilterEnabled) {
                        filteredGroupData = filterRunSeriesData(eqp, filteredGroupData);
                    }

                    if (!filteredGroupData.length) return;

                    hasAnyChart = true;

                    const lineChartId =
                        `linechart-${safeEqp}-${index + 1}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

                    $container.append(`
                        <div class="chart-pair single-chart">
                            <div id="${lineChartId}" class="linechart-container"></div>
                        </div>
                    `);

                    createLineChart(
                        lineChartId,
                        filteredGroupData,
                        group.name
                    );
                });
            }

            if (!hasAnyChart) {
                $container.empty().addClass('with-border');
                safeShowAnalysisText('데이터가 없습니다.', targetSel);
                return;
            }

            try { echarts.connect('linkedZoomGroup'); } catch (e) {}

            if (axisDropdown?.length) {
                axisDropdown.val(selectedAxis);
            }
        } catch (err) {
            console.error('createMotorCharts render error:', err);
            $container.empty().addClass('with-border');
            safeShowAnalysisText('데이터를 그리는 중 오류가 발생했습니다.', targetSel);
        } finally {
            if (isCurrentRender()) hideLoading();
        }
    }).fail((jqXHR, textStatus, errorThrown) => {
        if (!isCurrentRender()) return;
        console.error('Failed to load motor chart data:', eqp, textStatus, errorThrown);

        $container.empty().addClass('with-border');
        hideLoading();

        if (jqXHR.status === 404) {
            safeShowAnalysisText('데이터가 없습니다.', targetSel);
        } else {
            safeShowAnalysisText('데이터를 불러오는 중 오류가 발생했습니다.', targetSel);
        }
    });
}

/* =========================
   라인 차트
========================= */

function createLineChart(containerId, seriesData, groupLabel) {
    const chartDom = document.getElementById(containerId);
    if (!chartDom) return;

    const normalizedSeries = normalizeSeriesData(seriesData);
    const isLiftGroup = groupLabel === '리프트';

    const allTimestamps = normalizedSeries.flatMap(s =>
        s.data.map(p => new Date(p[0]).getTime()).filter(Number.isFinite)
    );

    if (!allTimestamps.length) return;

    const myChart = echarts.init(chartDom);

    const minTimestamp = Math.min(...allTimestamps);
    const maxTimestamp = Math.max(...allTimestamps);
    const range = Math.max(1, maxTimestamp - minTimestamp);

    const oneMonthAgo = maxTimestamp - (DEFAULT_DAYS_RANGE * 24 * 60 * 60 * 1000);
    const startPercent = Math.max(0, ((oneMonthAgo - minTimestamp) / range) * 100);
    const endPercent = 100;

    const option = {
        title: {
            left: 'left',
            top: 0,
            text: `{l|} {t|${groupLabel}}`,
            textStyle: {
                rich: {
                    l: {
                        backgroundColor: '#007bff',
                        width: 4,
                        height: 25,
                        borderRadius: [2, 0, 0, 2]
                    },
                    t: {
                        color: '#555',
                        fontSize: 18,
                        fontWeight: 'bold',
                        lineHeight: 18,
                        padding: [2, 0, 0, 2]
                    }
                }
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' },
            formatter(params) {
                if (!params || !params.length) return '';

                const time = params[0].axisValueLabel || '';
                let html = `${time}<br/>`;

                params.forEach(p => {
                    const val = Number(p.value?.[1] ?? p.value);
                    html += `${p.marker} ${p.seriesName}: ${
                        Number.isFinite(val) ? val.toFixed(2) : '-'
                    }%<br/>`;
                });

                return html;
            }
        },
        legend: {
            data: normalizedSeries.map(s => (smLabels?.[s.name] || s.name)),
            orient: 'vertical',
            right: 0,
            top: 10,
            padding: 0
        },
        grid: {
            left: 35,
            right: 150,
            bottom: 45,
            top: 80,
            containLabel: true
        },
        xAxis: {
            type: 'time',
            splitLine: { show: false },
            axisLabel: {
                formatter(value) {
                    const d = new Date(value);
                    const y = d.getFullYear();
                    const m = String(d.getMonth() + 1).padStart(2, '0');
                    const day = String(d.getDate()).padStart(2, '0');
                    const hh = String(d.getHours()).padStart(2, '0');
                    const mm = String(d.getMinutes()).padStart(2, '0');

                    if (d.getHours() === 0 && d.getMinutes() === 0) {
                        return `${y}-${m}-${day}`;
                    }
                    return `${hh}:${mm}`;
                }
            }
        },
        yAxis: {
            type: 'value',
            scale: true,
            name: '부하율 (%)',
            nameLocation: 'end',
            nameTextStyle: {
                fontSize: 14,
                padding: [0, 0, 0, 0]
            },
            axisLabel: {
                formatter(value) {
                    return Number(value).toFixed(0);
                }
            },
            splitLine: { show: true }
        },
        dataZoom: [
            // x축 내부 줌
            {
                type: 'inside',
                xAxisIndex: 0,
                filterMode: 'none'
            },
            // x축 슬라이더
            {
                type: 'slider',
                xAxisIndex: 0,
                height: 20, 
                bottom: 15,
                start: startPercent,
                end: endPercent,
                filterMode: 'none'
            },

            // y축 내부 줌
            {
                type: 'inside',
                yAxisIndex: 0,
                filterMode: 'none'
            },
            // y축 슬라이더
            {
                type: 'slider',
                yAxisIndex: 0,
                width: 20, 
                left: 0,
                top: 80,
                bottom: 50,
                start: 0,
                end: 100,
                filterMode: 'none'
            }
        ],
        series: normalizedSeries.map(series => ({
            name: smLabels?.[series.name] || series.name,
            type: 'line',
            showSymbol: false,
            smooth: false,
            connectNulls: false,
            emphasis: { focus: 'series' },
            data: series.data.map(([t, v]) => [
                new Date(t).getTime(),
                isLiftGroup ? v : v / 100
            ])
        })),
        animation: false
    };

    myChart.setOption(option);
    myChart.group = 'linkedZoomGroup';

    const ownerContainer =
        chartDom.closest('.pickloadcharts-container, .monitoring-charts-container');

    registerChart(myChart, ownerContainer ? ownerContainer.id : null);

    window.addEventListener('resize', () => {
        try { myChart.resize(); } catch (e) {}
    });
}

/* =========================
   외부에서 전체 정리용
========================= */

function clearAllMotorCharts() {
    disposeAllCharts();

    if (typeof monitoringChartsContainer !== 'undefined' && monitoringChartsContainer?.length) {
        monitoringChartsContainer.empty().removeClass('with-border');
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    await loadIdleThresholdMap();

    const runCheckbox = document.getElementById('run-filter-checkbox');
    if (runCheckbox) {
        runCheckbox.checked = isRunFilterEnabled;

        runCheckbox.addEventListener('change', function () {
            isRunFilterEnabled = this.checked;

            const eqp = $('#eqp-dropdown').val();
            const axisType = $('#axis-type-dropdown').val();
            const selectedAxis = $('#axis-dropdown').val() || 'ALL';

            if (!eqp || !axisType) return;

            let startDate = null;
            let endDate = null;

            if (typeof selectedStartDate !== 'undefined' && selectedStartDate) {
                startDate = selectedStartDate;
            }
            if (typeof selectedEndDate !== 'undefined' && selectedEndDate) {
                endDate = selectedEndDate;
            }

            createMotorCharts(
                eqp,
                selectedAxis,
                axisType,
                startDate,
                endDate,
                monitoringChartsContainer
            );
        });
    }
});
