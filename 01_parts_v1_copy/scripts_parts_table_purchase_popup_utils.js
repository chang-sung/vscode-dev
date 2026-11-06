const $ = (id) => document.getElementById(id);
let isQcodeChecked = false;
let lastCheckedQcode = '';   // ✅ 마지막으로 "확인 성공"한 Q_Code 저장

function showLoading(v){
    $('loading').classList.toggle('show', !!v);
}

// ==== Q/MRO 코드 마스킹 SITE별 마스크 길이 ====
function getMaskByCode(raw) {
    const s = String(raw || '').toUpperCase();
    if (s.startsWith('Q')) return { kind: 'Q', head: 4, tail: 6 }; // Q###
    return { kind: 'Z', head: 8, tail: 6 }; // Z####### (기본)
}

// 허용 문자: 숫자/영문/한글(완성형)
const ALLOWED_RE = new RegExp(
    '[0-9A-Za-z' +
    '\\u3131-\\u318E' +  // 호환 자모
    '\\u1100-\\u11FF' +  // 초성/중성 자모
    '\\uA960-\\uA97F' +  // 중간 확장 자모
    '\\uD7B0-\\uD7FF' +  // 중간 확장 자모
    '\\uAC00-\\uD7A3' +  // 완성형 한글 (가~힣)
    ']', 'g'
);

function formatCodeByRule(raw) {
    const s = String(raw || '').toUpperCase();
    const { head, tail } = getMaskByCode(s);
    const arr = s.replace(/-/g, '').match(ALLOWED_RE) || [];
    // ✅ 아무 글자도 없으면 비워둠 (지우기 허용)
    if (arr.length === 0) return '';

    // 대문자화
    const upper = arr.map(ch => ch.toUpperCase());
    // ✅ 최소 1글자 있을 때만 접두 강제 (첫 글자 기준 Q/Z 결정)
    // upper[0] = (upper[0] === 'Q') ? 'Q' : 'Z';

    const a = upper.slice(0, head).join(''); // 접두 + 앞자리
    const b = upper.slice(head, head + tail).join('');
    return b ? (a + '-' + b) : a;
}

function updateQcodePlaceholder() {
    const input = $('purchase-popup-qcode');
    if (!input) return;
    const v = String(input.value || '').toUpperCase();
    // 첫 글자가 Q면 Q형 예시, 아니면 Z형 예시
    input.placeholder = v.startsWith('Q')
        ? '예) Q123-123456'
        : '예) Q123-123456';
}

function reformatQcode() {
    const input = $('purchase-popup-qcode');
    if (!input) return;
    const prev = String(input.value || '').toUpperCase();
    const next = formatCodeByRule(prev);
    if (prev !== next) input.value = next;
    updateQcodePlaceholder();
}

function validateQcodeFormat() {
    const code = String($('purchase-popup-qcode')?.value || '').toUpperCase();

    // ✅ 0) "채번중" 허용 (공백/대소문자 영향 최소화)
    const normalized = code.replace(/\s+/g, '');
    if (normalized === '채번중') {
        return true;
    }

    // 1) 기존 Q-code: Q + 숫자3자리 - 숫자6자리 (예: Q123-123456)
    if (/^Q[A-Z0-9]{3}-[A-Z0-9]{6}$/.test(code)) {
        return true;
    }

    // 2) MRO-code: 알파벳 + 7자리 - 6자리 (예: Z1234567-123456)
    if (/^[A-Z][A-Z0-9]{7}-[A-Z0-9]{6}$/.test(code)) {
        return true;
    }

    // 그 외는 전부 형식 오류
    return false;
}

function toNumberSafe(v, def = 0) {
    const n = Number(String(v ?? '').replace(/,/g, '').trim());
    return Number.isFinite(n) ? n : def;
}

function formatThousand(n) {
    return (Number.isFinite(n) ? Math.round(n) : 0).toLocaleString();
}

function recalcTotal() {
    const qty   = toNumberSafe($('purchase-popup-qty')?.value, 0);
    const price = toNumberSafe($('purchase-popup-unit-price')?.value, 0);
    const totalEl = $('purchase-popup-total-price');

    if (qty > 0 && price >= 0) {
        totalEl.value = formatThousand(qty * price);
    } else {
        totalEl.value = '0';
    }
}

// 숫자만 추출
function digitsOnly(v) {
    return String(v || '').replace(/\D+/g, '');
}

// 3자리 콤마 포맷 (숫자 문자열 가정)
function formatWithCommaDigits(d) {
    if (!d) return '';
    return d.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// 단가 또는 수량 변경 시 자동 갱신
$('purchase-popup-qty')?.addEventListener('input', recalcTotal);
const unitEl = $('purchase-popup-unit-price');
if (unitEl) {
    // 포커스/클릭 시 0 자동 삭제
    const maybeClearZero = () => {
        const raw = digitsOnly(unitEl.value);
        if (raw === '' || Number(raw) === 0) {
        unitEl.value = ''; // 비워서 곧바로 숫자 입력 가능
        }
    };
    unitEl.addEventListener('focus', maybeClearZero);
    unitEl.addEventListener('click', maybeClearZero);

    // 입력 중 즉시 콤마 적용 (커서는 끝으로)
    unitEl.addEventListener('input', (e) => {
        const d = digitsOnly(e.target.value);
        e.target.value = formatWithCommaDigits(d);
        // 커서를 항상 끝으로 (간단/안정)
        const len = e.target.value.length;
        e.target.setSelectionRange(len, len);
        recalcTotal();
    });

    // 블러 시 비어있으면 0으로
    unitEl.addEventListener('blur', () => {
        const d = digitsOnly(unitEl.value);
        unitEl.value = d ? formatWithCommaDigits(d) : '0';
        recalcTotal();
    });
}

// Q-Code 확인 → NA 체크 후, 필요시 대체 파츠로 변경 + P/N/단가 자동 채움
async function checkQCodeAndFillPN() {
    reformatQcode();

    const codeInput = $('purchase-popup-qcode');
    const hint = $('purchase-popup-qcode-hint');
    const code = String(codeInput?.value || '').trim().toUpperCase();

    if (hint) hint.textContent = '';

    // ✅ 빈 값
    if (!code) {
        alert('Q-Code / MRO-Code를 입력해 주세요.');
        codeInput?.focus();
        return false;
    }

    // ✅ 형식 검증 실패면 "확인 성공"이 아님
    if (!validateQcodeFormat()) {
        alert('Q-Code / MRO-Code 형식이 올바르지 않습니다.');
        codeInput?.focus();
        return false;
    }

    try {
        showLoading(true);

        // ---------------------------
        // 1) NA 체크
        // ---------------------------
        try {
            const naRes = await fetch(
                '/parts/purchase_na_parts/by-na-code?code=' + encodeURIComponent(code),
                { credentials: 'same-origin' }
            );

            if (naRes.ok) {
                const naData = await naRes.json();

                if (naData && naData.NA_Q_Code) {

                    alert(
                        `해당 파츠(${code})는 구매 불가품 입니다.\n` +
                        `대체 파츠(${naData.Q_Code || ''})로 자동 변경됩니다.`
                    );

                    if (naData.Q_Code && codeInput) {
                        codeInput.value = String(naData.Q_Code);
                        reformatQcode();
                    }

                    if (naData.Parts_Name) $('purchase-popup-parts').value = String(naData.Parts_Name);
                    if (naData.P_N) $('purchase-popup-pn').value = String(naData.P_N);

                    if (naData.Price != null) {
                        const priceNum = toNumberSafe(naData.Price, 0);
                        $('purchase-popup-unit-price').value = formatThousand(priceNum);
                    } else {
                        $('purchase-popup-unit-price').value = '0';
                    }
                    recalcTotal();

                    if (hint) {
                        hint.innerHTML =
                            `대체 파츠(Q-Code, 파츠 분류, 업체P/N, 단가)로 자동 업데이트 완료.<br>` +
                            `요청 수량, 요청 내용 입력이 필요합니다.`;
                    }

                    // ✅ "확인 성공"
                    lastCheckedQcode = String(codeInput.value || '').trim().toUpperCase();
                    return true;
                }
            }
        } catch (naErr) {
            console.warn('[popup] NA check failed:', naErr);
        }

        // ---------------------------
        //  추가: 중복 구매 체크
        // ---------------------------
        try {
            const dupRes = await fetch(
                '/parts_purchase/check-duplicate?qcode=' + encodeURIComponent(code),
                { credentials: 'same-origin' }
            );

            if (dupRes.ok) {
                const dupData = await dupRes.json();

                // dupData.exists === true 이런 식으로 받는다고 가정
                if (dupData && dupData.exists) {
                    const proceed = confirm(
                        '해당 품목은 현재 구매가 진행 중인 상태입니다.\n(중복 구매로 과잉 재고가 될 수 있습니다.)\n그래도 추가 구매를 진행하시겠습니까?'
                    );

                    if (!proceed) {
                        return false; // ❌ 여기서 중단
                    }
                }
            }
        } catch (dupErr) {
            console.warn('[popup] duplicate check failed:', dupErr);
        }

        // ---------------------------
        // 2) EMS 가격 조회
        // ---------------------------
        const res = await fetch('/parts_ems_price/by-code?code=' + encodeURIComponent(code), {
            credentials: 'same-origin'
        });
        if (!res.ok) throw new Error('조회 실패 (' + res.status + ')');

        const data = await res.json(); // { ... } | null

        if (!data) {
            if (hint) {
                hint.innerHTML =
                    '지난 1년간 구매 이력이 없습니다.<br>' +
                    '파츠 분류, 업체P/N, 요청 수량, 단가, 요청 내용 입력이 필요합니다.';
            }
            $('purchase-popup-unit-price').value = '0';
            recalcTotal();

            // ✅ 형식 OK + 체크 수행 완료 → "확인 성공"으로 볼지 여부
            //    (원하면 true, 원치 않으면 false로 바꾸면 됨)
            lastCheckedQcode = String(codeInput.value || '').trim().toUpperCase();
            return true;
        }

        // 파츠 분류 자동 채움
        if (data.Parts_Div) $('purchase-popup-parts').value = String(data.Parts_Div);

        // P/N 자동 채움
        if (data.P_N) $('purchase-popup-pn').value = String(data.P_N);

        // 단가 자동 채움 + 금액 계산
        if (data.Price != null) {
            const priceNum = toNumberSafe(data.Price, 0);
            $('purchase-popup-unit-price').value = formatThousand(priceNum);
        } else {
            $('purchase-popup-unit-price').value = '0';
        }
        recalcTotal();

        if (hint) {
            hint.innerHTML =
                '파츠 분류(' + String(data.Parts_Div ?? '') + '), 업체P/N(' +
                String(data.P_N ?? '') + '), 단가 자동 업데이트 완료.<br>' +
                '요청 수량, 요청 내용 입력이 필요합니다.';
        }

        // ✅ "확인 성공"
        lastCheckedQcode = String(codeInput.value || '').trim().toUpperCase();
        return true;

    } catch (e) {
        console.error('[popup] checkQCode error:', e);
        alert('Q-Code 확인 중 오류가 발생했습니다.');
        return false;
    } finally {
        showLoading(false);
    }
}

// 버튼 클릭 연결
$('purchase-popup-qcode-check')?.addEventListener('click', async () => {
    isQcodeChecked = false;
    lastCheckedQcode = '';

    const ok = await checkQCodeAndFillPN();
    isQcodeChecked = !!ok;   // ✅ 성공일 때만 true
});

// Q-Code 입력칸에서 Enter 누르면 확인(단, Ctrl/Cmd+Enter는 제출로 남겨둠)
$('purchase-popup-qcode')?.addEventListener('keydown', async (e) => {
    if (e.key === '-') e.preventDefault();
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        isQcodeChecked = false;
        lastCheckedQcode = '';

        const ok = await checkQCodeAndFillPN();
        isQcodeChecked = !!ok;
    }
});

// 입력 시: 대문자 + 규칙 기반 포맷 + 플레이스홀더 갱신
$('purchase-popup-qcode')?.addEventListener('input', (e) => {
    e.target.value = String(e.target.value || '').toUpperCase();
    reformatQcode();

    // ✅ 확인 이후 코드가 바뀌면 무조건 다시 확인하게 만들기
    const now = String(e.target.value || '').trim().toUpperCase();
    if (now !== lastCheckedQcode) {
        isQcodeChecked = false;
    }
});

// ★ 추가: 현재 사용자 이름 가져오기
async function fetchRequester() {
    try {
        const res = await fetch('/check-client-authority-parts');
        if (!res.ok) throw new Error('authority api failed');
        const data = await res.json();
        // data.userName: 서버 코드에서 내려주는 필드
        return data?.userName || 'Unknown';
    } catch (e) {
        console.warn('[popup] fetchRequester failed:', e);
        return 'Unknown';
    }
}

function getPayloadBase(){
    return {
        Site: $('purchase-popup-site').value.trim(),
        EQP_Div: $('purchase-popup-eqp').value.trim(),
        Q_Code: $('purchase-popup-qcode').value.trim(),
        Parts_Div: $('purchase-popup-parts').value.trim(),
        P_N: $('purchase-popup-pn').value.trim(),
        Req_Qty: Number($('purchase-popup-qty').value),
        Req_Comment: $('purchase-popup-comment').value.trim(),
        Unit_Price: toNumberSafe($('purchase-popup-unit-price').value, 0),
        Total_Price: toNumberSafe($('purchase-popup-total-price').value, 0)
    };
}

function validate(p){
    if(!p.EQP_Div || !p.Q_Code || !p.Parts_Div || !p.P_N || !p.Req_Qty){
        alert('필수 항목(구분, Q-Code, 파츠분류, Unit, 업체P/N, 요청 수량)을 입력해주세요.');
        return false;
    }
    if(!Number.isInteger(p.Req_Qty) || p.Req_Qty <= 0){
        alert('요청 수량은 1 이상의 정수여야 합니다.');
        return false;
    }
    return true;
}

$('purchase-popup-btn-cancel').onclick = () => { window.close(); };

$('purchase-popup-btn-submit').onclick = async () => {

    // ✅ 형식 재검증
    if (!validateQcodeFormat()) {
        alert("Q-Code / MRO-Code 확인 후 요청해주세요.");
        $('purchase-popup-qcode')?.focus();
        return;
    }

    // ✅ 확인 선행 + 확인한 코드와 현재 코드가 동일해야 함
    const now = String($('purchase-popup-qcode')?.value || '').trim().toUpperCase();
    if (!isQcodeChecked || now !== lastCheckedQcode) {
        alert("Q-Code 확인 버튼을 눌러주세요.");
        $('purchase-popup-qcode-check')?.focus();
        return;
    }

    const base = getPayloadBase();
    if(!validate(base)) return;

    showLoading(true);
    try{
        // ★ Requester 동기화
        const requester = await fetchRequester();

        const payload = {
            ...base,
            Requester: requester   // ← 구매자 이름을 함께 보냄
        };

        const res = await fetch('/parts_purchase/request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if(!res.ok){
            const txt = await res.text().catch(()=> '');
            throw new Error('신청 실패: ' + (txt || res.status));
        }
        // 부모 테이블 새로고침
        try {
            window.opener && window.opener.PartsBuy && window.opener.PartsBuy.fetch && (await window.opener.PartsBuy.fetch());
        } catch(e){}
        alert('구매 신청이 완료되었습니다.');
        window.close();
    }catch(err){
        console.error(err);
        alert(err.message || '신청 중 오류가 발생했습니다.');
    }finally{
        showLoading(false);
    }
};

// SITE 변경 시 마스킹 영향 없음 (선택적으로 placeholder만 갱신)
$('purchase-popup-site')?.addEventListener('change', () => {
    updateQcodePlaceholder();
});

// 단축키: ESC 닫기 / Ctrl+Enter 제출
document.addEventListener('keydown', (e) => {
    if(e.key === 'Escape') window.close();
    if((e.ctrlKey || e.metaKey) && e.key === 'Enter') $('purchase-popup-btn-submit').click();
});

// 포커스
$('purchase-popup-qcode').focus();