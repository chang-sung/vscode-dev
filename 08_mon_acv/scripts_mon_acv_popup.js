function showPopupTable(data) {
    const popup = document.getElementById("effi-popup");
    const tbody = document.querySelector("#effi-popup-table tbody");
    const thead = document.querySelector("#effi-popup-table thead");

    popup.style.display = "flex";

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

    // ✅ 최신 100개만 출력
    const rows = data.map(row => `
        <tr>
            <td>${row.eq}</td>
            <td>${formatDateTime(row.writeTime)}</td>
            <td>${row.cv1 ?? '-'}</td>
            <td>${row.cv2 ?? '-'}</td>
            <td>${row.cv3 ?? '-'}</td>
            <td>${row.cv4 ?? '-'}</td>
            <td>${row.effi ?? '-'}</td>
        </tr>
    `).join('');

    tbody.innerHTML = rows;

    // ✅ 헤더 클릭 시 정렬 처리
    thead.querySelectorAll("th").forEach((th, index) => {
        let ascending = true;
        th.addEventListener("click", () => {
            const table = document.getElementById("effi-popup-table");
            sortTableByColumn(table, index, ascending);

            // 기존 정렬 상태 제거
            thead.querySelectorAll("th").forEach(th => th.classList.remove("sort-active"));
            th.classList.add("sort-active");

            ascending = !ascending;
        });
    });

    document.querySelector(".effi-popup-close").onclick = () => {
        popup.style.display = "none";
    };
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
    const table = document.getElementById("effi-popup-table");
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.table_to_sheet(table);
    XLSX.utils.book_append_sheet(wb, ws, "상세정보");

    const now = new Date();
    const timestamp = `${now.getFullYear()}-${(now.getMonth()+1).toString().padStart(2, '0')}-${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;

    XLSX.writeFile(wb, `ACV_상세정보_${timestamp}.xlsx`);
});