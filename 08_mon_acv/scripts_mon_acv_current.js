// 파워몰러 전류 조회, Scatter 차트, Raw 데이터 및 엑셀 저장
const CURRENT_CHANNELS = ['LD1_I', 'LD2_I', 'LD3_I', 'LD4_I',
    'CH1_I', 'CH2_I', 'CH3_I', 'CH4_I', 'ULD1_I', 'ULD2_I', 'ULD3_I', 'ULD4_I'];
const CURRENT_COLUMNS = ['EQ_Name', 'WriteTime', ...CURRENT_CHANNELS];
const CURRENT_PAGE_SIZE = 100;
let currentRows = [];
let currentPage = 0;
let currentChart = null;
let currentRequest = null;

function isCurrentMode() {
    const wrapper = document.querySelector('.acv-mon-current');
    return !!wrapper && wrapper.style.display !== 'none';
}

function cancelCurrentRequest() {
    if (!currentRequest) return;
    currentRequest.abort();
    currentRequest = null;
    hideLoading();
}

function currentEscape(value) {
    return String(value ?? '-').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function renderCurrentRawTable() {
    const table = document.getElementById('acv-current-table');
    table.querySelector('thead').innerHTML = '<tr>' +
        CURRENT_COLUMNS.map(c => '<th>' + currentEscape(c) + '</th>').join('') + '</tr>';
    const start = currentPage * CURRENT_PAGE_SIZE;
    table.querySelector('tbody').innerHTML = currentRows.slice(start, start + CURRENT_PAGE_SIZE).map(row =>
        '<tr>' + CURRENT_COLUMNS.map(c => '<td>' + currentEscape(
            c === 'WriteTime' ? formatTimestamp(row[c]) : row[c]) + '</td>').join('') + '</tr>'
    ).join('');
    document.getElementById('acv-current-page').textContent = currentRows.length
        ? (currentPage + 1) + ' / ' + Math.ceil(currentRows.length / CURRENT_PAGE_SIZE) : '0 / 0';
    document.getElementById('acv-current-prev').disabled = currentPage === 0;
    document.getElementById('acv-current-next').disabled = start + CURRENT_PAGE_SIZE >= currentRows.length;
}

function clearCurrentView(message) {
    currentRows = [];
    currentPage = 0;
    currentChart?.dispose();
    currentChart = null;
    document.getElementById('acv-current-status').textContent = message;
    renderCurrentRawTable();
    showAnalysisText(message, '#acv-current-chart');
}

function renderCurrentScatter(eq) {
    clearAnalysisText('#acv-current-chart');
    currentChart?.dispose();
    currentChart = echarts.init(document.getElementById('acv-current-chart'));
    currentChart.setOption({
        title: { text: eq + ' 파워몰러 전류', left: 'center' },
        animation: false,
        legend: { type: 'scroll', top: 32, data: CURRENT_CHANNELS },
        tooltip: { trigger: 'item' },
        grid: { top: 85, bottom: 65, left: 65, right: 30 },
        xAxis: { type: 'time', name: '시간' },
        yAxis: { type: 'value', name: '전류 (A)', scale: true },
        dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 10 }],
        series: CURRENT_CHANNELS.map(channel => ({
            name: channel,
            type: 'scatter',
            symbolSize: 4,
            progressive: 5000,
            data: currentRows.flatMap(row => {
                const time = new Date(row.WriteTime).getTime();
                const raw = row[channel];
                const value = raw == null || raw === '' ? NaN : Number(raw);
                return Number.isFinite(time) && Number.isFinite(value) ? [[time, value]] : [];
            })
        }))
    });
}

async function loadCurrentAnalysis() {
    if (!isCurrentMode()) return;
    cancelCurrentRequest();
    const eq = document.getElementById('eq-group-select')?.value || 'all';
    const [start, end] = getSelectedRangeYmd();
    if (eq === 'all') {
        clearCurrentView('설비를 선택하세요.');
        return;
    }
    if (!start || !end) {
        clearCurrentView('날짜를 선택하세요.');
        return;
    }

    clearCurrentView('전류 데이터를 불러오는 중…');
    const controller = new AbortController();
    currentRequest = controller;
    showLoading();
    try {
        const params = new URLSearchParams({ eq, start, end });
        const response = await fetch('/get-acv-current-data?' + params, { signal: controller.signal });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || '전류 데이터 조회 실패');
        }
        const rows = await response.json();
        if (currentRequest !== controller || !isCurrentMode()) return;
        if (!Array.isArray(rows)) throw new Error('전류 응답 형식이 올바르지 않습니다.');
        currentRows = rows;
        currentPage = 0;
        if (!rows.length) {
            clearCurrentView('선택한 기간에 전류 데이터가 없습니다.');
            return;
        }
        document.getElementById('acv-current-status').textContent =
            eq + ' | ' + start + ' ~ ' + end + ' | 총 ' + rows.length.toLocaleString() + '건';
        renderCurrentScatter(eq);
        renderCurrentRawTable();
    } catch (error) {
        if (currentRequest === controller && error.name !== 'AbortError') {
            console.error('파워몰러 전류 조회 실패:', error);
            clearCurrentView(error.message || '전류 데이터를 불러오지 못했습니다.');
        }
    } finally {
        if (currentRequest === controller) {
            currentRequest = null;
            hideLoading();
            renderCurrentRawTable();
        }
    }
}

function exportCurrentExcel() {
    if (!currentRows.length) {
        alert('저장할 전류 데이터가 없습니다. 설비와 날짜를 선택하세요.');
        return;
    }
    const sheetRows = [CURRENT_COLUMNS, ...currentRows.map(row =>
        CURRENT_COLUMNS.map(c => c === 'WriteTime' ? formatTimestamp(row[c]) : (row[c] ?? null)))];
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(sheetRows), '파워몰러 전류');
    const timestamp = formatTimestamp(new Date()).replace(/[-: ]/g, '');
    XLSX.writeFile(wb, 'ACV_파워몰러_전류_' + timestamp + '.xlsx');
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('acv-current-prev').addEventListener('click', () => {
        if (currentPage > 0) { currentPage--; renderCurrentRawTable(); }
    });
    document.getElementById('acv-current-next').addEventListener('click', () => {
        if ((currentPage + 1) * CURRENT_PAGE_SIZE < currentRows.length) {
            currentPage++; renderCurrentRawTable();
        }
    });
});
window.addEventListener('resize', () => currentChart?.resize());
