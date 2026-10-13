function escapeEfficiencyCell(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[char]));
}

function resetEfficiencyDetailTable() {
    const table = document.getElementById("effi-detail-data-table");
    table.querySelector("thead").innerHTML = "";
    table.querySelector("tbody").innerHTML = "";
    document.getElementById("effi-detail-message").textContent = "바차트를 클릭하면 상세 정보가 표시됩니다.";
    document.getElementById("export-excel-btn").disabled = true;
}

function showEfficiencyDetailTable(data) {
    const tbody = document.querySelector("#effi-detail-data-table tbody");
    const thead = document.querySelector("#effi-detail-data-table thead");

    const message = document.getElementById("effi-detail-message");
    const exportButton = document.getElementById("export-excel-btn");
    const rowsData = Array.isArray(data) ? data : [];
    message.textContent = rowsData.length ? `상세 정보 ${rowsData.length}건` : "데이터가 없습니다.";
    exportButton.disabled = rowsData.length === 0;

    // ✅ 헤더 동적 구성
    thead.innerHTML = `
        <tr>
            <th>설비</th>
            <th>시간</th>
            <th>CV1</th>
            <th>CV2</th>
            <th>CV3</th>
            <th>CV4</th>
            <th>효율</th>
        </tr>
    `;

    // 클릭한 막대에 해당하는 전체 데이터 출력
    const rows = rowsData.map(row => `
        <tr>
            <td>${escapeEfficiencyCell(row.eq ?? '-')}</td>
            <td>${escapeEfficiencyCell(formatDateTime(row.writeTime))}</td>
            <td>${escapeEfficiencyCell(row.cv1 ?? '-')}</td>
            <td>${escapeEfficiencyCell(row.cv2 ?? '-')}</td>
            <td>${escapeEfficiencyCell(row.cv3 ?? '-')}</td>
            <td>${escapeEfficiencyCell(row.cv4 ?? '-')}</td>
            <td>${escapeEfficiencyCell(row.effi ?? '-')}</td>
        </tr>
    `).join('');

    tbody.innerHTML = rows;

    // ✅ 헤더 클릭 시 정렬 처리
    thead.querySelectorAll("th").forEach((th, index) => {
        let ascending = true;
        th.addEventListener("click", () => {
            const table = document.getElementById("effi-detail-data-table");
            sortTableByColumn(table, index, ascending);

            // 기존 정렬 상태 제거
            thead.querySelectorAll("th").forEach(th => th.classList.remove("sort-active"));
            th.classList.add("sort-active");

            ascending = !ascending;
        });
    });

    document.querySelector("#effi-detail-table .effi-detail-scroll").scrollTop = 0;
}

function sortTableByColumn(table, columnIndex, ascending) {
    const dirModifier = ascending ? 1 : -1;
    const tBody = table.tBodies[0];
    const rows = Array.from(tBody.querySelectorAll("tr"));

    // 숫자/문자 판단해서 정렬
    const getCellValue = (row, index) => {
        const val = row.children[index].innerText;
        return isNaN(val) ? val : parseFloat(val);
    };

    const sortedRows = rows.sort((a, b) => {
        const aVal = getCellValue(a, columnIndex);
        const bVal = getCellValue(b, columnIndex);

        if (aVal < bVal) return -1 * dirModifier;
        if (aVal > bVal) return 1 * dirModifier;
        return 0;
    });

    // 기존 tbody 초기화 후 정렬된 tr 다시 삽입
    tBody.innerHTML = "";
    tBody.append(...sortedRows);
}

document.getElementById("export-excel-btn").addEventListener("click", () => {
    const table = document.getElementById("effi-detail-data-table");
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.table_to_sheet(table);
    XLSX.utils.book_append_sheet(wb, ws, "상세정보");

    const now = new Date();
    const timestamp = `${now.getFullYear()}-${(now.getMonth()+1).toString().padStart(2, '0')}-${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;

    XLSX.writeFile(wb, `ACV_상세정보_${timestamp}.xlsx`);
});