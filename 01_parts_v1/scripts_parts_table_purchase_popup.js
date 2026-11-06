// 새 창 팝업으로 구매 신청 폼 열기
function openPurchaseRequestWindow(defaults = {}, onSubmitted) {
    // 기존 팝업이 열려있으면 재사용 (중복 방지)
    if (window.__purchaseWin && !window.__purchaseWin.closed) {
        window.__purchaseWin.focus();
        return;
    }

    const w = 550, h = 470;
    const left = Math.max(0, (window.screenX || window.screenLeft || 0) + 60);
    const top  = Math.max(0, (window.screenY || window.screenTop || 0) + 60);

    const features = `popup=yes,width=${w},height=${h},left=${left},top=${top},resizable=yes,scrollbars=yes`;
    const popup = window.open('', 'purchase_request_popup', features);
    if (!popup) {
        alert('팝업이 차단되었습니다. 브라우저 팝업 허용 후 다시 시도해주세요.');
        return;
    }
    window.__purchaseWin = popup;

    // 기본값 (XSS 방지용 간단 이스케이프)
    const esc = (s) => String(s ?? '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    const def = {
        Site: esc(defaults.Site) || 'P1DJ',
        EQP_Div: defaults.EQP_Div ?? 'A3 PLM',
        Q_Code: esc(defaults.Q_Code),
        Parts_Div: esc(defaults.Parts_Div),
        P_N: esc(defaults.P_N),
        Req_Qty: Number.isInteger(defaults.Req_Qty) ? defaults.Req_Qty : 1,
        Req_Comment: esc(defaults.Req_Comment),
        Unit_Price: Number.isFinite(Number(defaults.Unit_Price)) ? Number(defaults.Unit_Price) : 0,
        Total_Price: Number.isFinite(Number(defaults.Total_Price)) ? Number(defaults.Total_Price) : 0
    };

    // HTML 작성
    popup.document.write(`
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8" />
        <title>파츠 구매 신청</title>
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <link rel="stylesheet" href="/css/100_utils/styles.css">
        <link rel="stylesheet" href="/css/01_parts/styles_parts.css">
        <link rel="stylesheet" href="/css/01_parts/styles_parts_popup.css">
    </head>
    <body>
        <div class="purchase-popup-warpper">
            <div class="header">
                <div class="title">파츠 구매 요청</div>
                <div class="hint">필수: 구분, Q-Code, 파츠분류, 업체P/N, 요청 수량</div>
            </div>

            <div class="purchase-popup-container">
                <div class="popup-container-1">
                    
                    <div class ="popup-inner-1">
                        <label>공정 구분</label>
                        <select id="purchase-popup-eqp">
                            <option ${def.EQP_Div==='A3 PLM'?'selected':''} value="A3 PLM">A3 PLM</option>
                            <!-- <option ${def.EQP_Div==='A3 CLC'?'selected':''} value="A3 CLC">A3 CLC</option> -->
                        </select>
                        
                    </div>

                    <div class ="popup-inner-2">
                        <label>SITE</label>
                        <select id="purchase-popup-site">
                            <option ${def.Site==='P1DD'?'selected':''} value="P1DD">P1DD</option>
                            <option ${def.Site==='P1DJ'?'selected':''} value="P1DJ">P1DJ</option>
                        </select>
                    </div>
                    
                    <div class ="popup-inner-3">
                        <label>Q-Code / MRO-Code</label>
                        <div class ="popup-inner3-wrapper">
                            <input id="purchase-popup-qcode" type="text" value="${def.Q_Code}" placeholder="예) Q123-123456">
                            <button id="purchase-popup-qcode-check" type="button" class="purchase-popup-btn" style="margin-left:8px;">확인</button>
                            
                        </div>
                    </div>
                    

                </div>
                <div class="popup-container-2">
                    <small id="purchase-popup-qcode-hint" class="hint" style="display:block;margin-top:4px;color:#666;font-size:14px;"></small>
                    <label> </label>

                </div>

                <div class="popup-container-3">
                    <div class ="popup-inner-4">
                        <label>파츠 분류</label>
                        <input id="purchase-popup-parts" type="text" value="${def.Parts_Div}" placeholder="예) 볼스크류">
                        
                    </div>

                    <div class ="popup-inner-6">
                        <label>업체P/N</label>
                        <input id="purchase-popup-pn" type="text" value="${def.P_N}" placeholder="예) RC-105-X30SS ST.900">
                    </div>
                </div>

                <div class="popup-container-4">

                    <div class ="popup-inner-7">
                        <label>요청 수량</label>
                        <input id="purchase-popup-qty" type="number" min="1" step="1" value="${def.Req_Qty}">
                    </div>

                    <div class ="popup-inner-8">
                        <label>단가</label>
                        <input id="purchase-popup-unit-price" type="text" value="${def.Unit_Price}">
                    </div>

                    <div class ="popup-inner-9">
                        <label>금액</label>
                        <input id="purchase-popup-total-price" type="text" value="${def.Total_Price}">
                    </div>

                </div>
                
                <div class="popup-container-5">
                    <div class ="popup-inner-10">
                        <label>요청 내용</label>
                        <input id="purchase-popup-comment" type="text" value="${def.Req_Comment}" placeholder="구매 사유를 입력해 주세요.">
                    </div>
                </div>
            </div>

            <div class="purchase-actions">
                <button id="purchase-popup-btn-submit" class="purchase-popup-btn">요청</button>
                <button id="purchase-popup-btn-cancel" class="purchase-popup-btn">취소</button>
            </div>
        </div>

        <div id="loading" class="loading">
            <div class="spinner"></div>
        </div>

        <script src="/js/01_parts/scripts_parts_table_purchase_popup_utils.js"></script>
    </body>
    </html>
    `);

    popup.document.close();
    popup.focus();
}
