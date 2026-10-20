async function loadAllMotorData() {
    try {
        showLoading(); // 페이지 단위 로딩

        // 병렬 요청으로 속도 개선
        const [res1, res2, res3, res4] = await Promise.all([
            fetch('/api/monitor_motor_setting'),
            fetch('/api/spec'),
            fetch('/api/spec/comments'),
            fetch('/api/unit/type')
        ]);

        motorDataGlobal = await res1.json();
        motorSpecGlobal = await res2.json();

        const commentData = await res3.json();
        eqTypeMap = await res4.json();
        
        commentMap = {};
        commentData.forEach(row => {
            const originalKey = `${row.EQ_Name}_${row.Motor_Num}_${row.POS_Num}`;
            const transformedKey = `${EqNameUtils.transform(row.EQ_Name)}_${row.Motor_Num}_${row.POS_Num}`;
            commentMap[originalKey]   = row.Comment;
            commentMap[transformedKey] = row.Comment;
        });

        // ✅ 필터/그룹 상태에 맞게 테이블 렌더
        applyCombinedFilters();

        // ✅ 테이블 렌더 완료까지 대기 (안전)
        await waitForMotorTableRendered();

    } catch (error) {
        console.error("데이터 로딩 중 오류 발생:", error);
    } finally {
        hideLoading(); // 로딩 해제
    }
}

function waitForMotorTableRendered(timeout = 15000) {
    return new Promise((resolve) => {
        const container = document.querySelector('#motor-table-container');
        if (!container) return resolve();

        // 이미 내용이 있으면 바로 종료
        if (container.children.length > 0 || container.textContent.trim().length > 0) {
            return resolve();
        }

        const observer = new MutationObserver(() => {
            if (container.children.length > 0 || container.textContent.trim().length > 0) {
                observer.disconnect();
                resolve();
            }
        });

        observer.observe(container, { childList: true, subtree: true });

        // 안전 타임아웃
        setTimeout(() => {
            observer.disconnect();
            resolve();
        }, timeout);
    });
}

function showLoading() {
    document.getElementById("loading-overlay").style.display = "flex";

    // 모든 버튼/셀렉트/체크박스 비활성화
    document.querySelectorAll("button, select, input[type='checkbox']").forEach(el => {
        el.disabled = true;
    });
}

function hideLoading() {
    document.getElementById("loading-overlay").style.display = "none";

    document.querySelectorAll("button, select, input[type='checkbox']").forEach(el => {
        el.disabled = false;
    });
}