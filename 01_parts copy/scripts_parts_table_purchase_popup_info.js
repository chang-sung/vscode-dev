// 구매 합의자 정보 팝업 열기
function openManagerInfoWindow() {
    // 기존 팝업이 열려있으면 재사용 (중복 방지)
    if (window.__managerInfoWin && !window.__managerInfoWin.closed) {
        window.__managerInfoWin.focus();
        return;
    }

    const w = 800, h = 320;
    const left = Math.max(0, (window.screenX || window.screenLeft || 0) + 60);
    const top  = Math.max(0, (window.screenY || window.screenTop || 0) + 60);

    const features = `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`;
    const popup = window.open('', 'purchase_manager_info_popup', features);

    if (!popup) {
        alert('팝업이 차단되었습니다. 브라우저 팝업 허용 후 다시 시도해주세요.');
        return;
    }

    window.__managerInfoWin = popup;

    popup.document.write(`
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="utf-8" />
            <title>구매 담당자 정보</title>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <link rel="stylesheet" href="/css/100_utils/styles.css">
            <link rel="stylesheet" href="/css/01_parts/styles_parts.css">
            <style>
                html, body {
                    height: 100%;
                    margin: 0;
                }
                body {
                    font-family: "맑은 고딕", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    background-color: #f7f7f7;
                    display: flex;
                    justify-content: center;
                    align-items: center;   /* 👈 세로 가운데 정렬 */
                }
                .manager-popup-wrapper {
                    background: #fff;
                    box-shadow:0 8px 20px rgba(0,0,0,.08);
                    padding: 15px;
                    width: 720px;
                    min-height: 200px;
                    border-radius: 5px;        
                }
                .manager-popup-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 15px;
                }
                .mgr-section {
                    margin-top: 8px;
                }
                .mgr-sections-row {
                    display: flex;
                    flex-direction: row;
                    justify-content: space-between;
                    gap: 16px;
                }

                .mgr-left, .mgr-right {
                    flex: 1;
                }

                .mgr-section-title {
                    font-size: 14px;
                    font-weight: 600;
                    margin-bottom: 8px;
                }

                table.manager-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                    background: #fff;
                }
                    
                table.manager-table thead tr {
                    background-color: #f0f4ff;
                }
                table.manager-table th,
                table.manager-table td {
                    border: 1px solid #ddd;
                    padding: 6px 8px;
                    text-align: left;
                    white-space: nowrap;
                }
                table.manager-table th {
                    font-weight: 600;
                }
                table.manager-table tbody tr:nth-child(even) {
                    background-color: #fafafa;
                }

                /* 왼쪽 테이블 */
                .mgr-left table.manager-table th:nth-child(1),
                .mgr-left table.manager-table td:nth-child(1) { width: 100px; }

                .mgr-left table.manager-table th:nth-child(2),
                .mgr-left table.manager-table td:nth-child(2) { width: 100px; }

                .mgr-left table.manager-table th:nth-child(3),
                .mgr-left table.manager-table td:nth-child(3) { width: 100px; }

                /* 오른쪽 테이블 */
                .mgr-right table.manager-table th:nth-child(1),
                .mgr-right table.manager-table td:nth-child(1) { width: 100px; }

                .mgr-right table.manager-table th:nth-child(2),
                .mgr-right table.manager-table td:nth-child(2) { width: 100px; }

                .mgr-right table.manager-table th:nth-child(3),
                .mgr-right table.manager-table td:nth-child(3) { width: 100px; }

            </style>
        </head>
        <body>
            <div class="manager-popup-title">구매 담당자 및 합의자 정보</div>
            <div class="manager-popup-wrapper">
                
                <div class="mgr-sections-row">

                    <!-- ✅ 품목별 담당자 -->
                    <div class="mgr-section mgr-left">
                        <div class="mgr-section-title">품목별 담당자</div>
                        <table class="manager-table">

                            <thead>
                                <tr>
                                    <th>품목</th>
                                    <th>이름</th>
                                    <th>아이디</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>P1DD MRO</td>
                                    <td>홍길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>P1DJ MRO</td>
                                    <td>고길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>직거래</td>
                                    <td>강길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>가공품</td>
                                    <td>나길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>폐봉투</td>
                                    <td>다길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- ✅ 결재합의자 -->
                    <div class="mgr-section mgr-right">
                        <div class="mgr-section-title">결재합의자</div>
                        <table class="manager-table">

                            <thead>
                                <tr>
                                    <th>팀</th>
                                    <th>이름</th>
                                    <th>아이디</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>a팀</td>
                                    <td>라길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>b팀</td>
                                    <td>마길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>c팀</td>
                                    <td>바길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                                <tr>
                                    <td>d팀</td>
                                    <td>사길동</td>
                                    <td>aaa.aaa</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>


            </div>
        </body>
        </html>
    `);

    popup.document.close();
    popup.focus();
}

// 메인 화면의 "결재 정보" 버튼에 이벤트 연결
$(function () {
    $('#purchase-manager-inform-btn').on('click', function () {
        openManagerInfoWindow();
    });
});