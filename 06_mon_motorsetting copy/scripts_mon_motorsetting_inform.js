let informMode = false;
let selectedTd = null;

function exitInformMode() {
    informMode = false;
    const table = document.querySelector("#motorTable");
    table.classList.remove("hover-cell", "inform-mode-active");
    table.classList.add("hover-row");

    document.querySelectorAll("#motorTable td").forEach(td => {
        td.classList.remove("inform-hover", "selected-td");
    });

    document.getElementById("inform-btn").innerText = "Inform 등록";
}

document.getElementById("inform-btn").addEventListener("click", () => {
    informMode = !informMode;
    const table = document.querySelector("#motorTable");

    if (informMode) {
        alert("메모(Inform)를 남길 부분을 클릭하세요.");
        table.classList.add("hover-row", "hover-cell", "inform-mode-active");
        table.classList.add("hover-cell", "inform-mode-active");

        document.querySelectorAll("#motorTable td").forEach(td => {
            td.classList.add("inform-hover");
        });

        document.getElementById("inform-btn").innerText = "Inform 취소";
    } else {
        exitInformMode();
    }
});

document.addEventListener("click", function (e) {
    if (!informMode) return;

    const td = e.target.closest("td");
    const motor = td?.dataset?.motor;
    const pos = td?.dataset?.pos;

    if (td && motor && typeof pos === 'string') {
        document.querySelectorAll("#motorTable td").forEach(td => {
            td.classList.remove("inform-hover", "selected-td");
        });

        td.classList.add("selected-td");
        selectedTd = td;
        openCommentPopup(motor, pos);
        informMode = false;
    }
});

function openCommentPopup(motorNum, posNum) {
    const m = motor[motorNum];
    const posName = posNum === "ACC" ? "ACC" : posNum === "DEC" ? "DEC" : m.pos[posNum]?.name || "Unknown";
    const displayEqName = selectedTd.closest("tr").querySelector(".eq-name")?.textContent;


    const eqName = EqNameUtils.reverse(displayEqName);  // ✅ 실제 DB 이름으로 변환

    const actualPos = (posNum === "ACC" || posNum === "DEC") ? "0" : posNum;
    const commentKey = `${eqName}_${motorNum}_${actualPos}`;  // ✅ 항상 POS_Num 기준
    const existingComment = commentMap[commentKey];

    const popup = document.createElement("div");
    popup.className = "popup-overlay";
    popup.innerHTML = `
        <div class="popup-content">
            <h3>Inform 등록</h3>
            <h3 style="font-size: 14px;">${eqName}\n-\nSM${motorNum}_${m.name}_${posName}</h3>
            <div class="popup-row">
                <label>작성자:</label>
                <input type="text" id="comment-name" value="${(window.currentUserInfo?.userName)||''}">
            </div>
            <div class="popup-row">
                <label>백업 예정일:</label>
                <input type="date" id="recovery-date" />
            </div>
            <div class="popup-row">
                <label>코멘트:</label>
                <textarea id="comment-text" rows="3" placeholder="내용을 입력하세요."></textarea>
            </div>
            
            <div class="popup-buttons">
                <button class="popup-submit" id="submit-comment">저장</button>
                <button class="popup-delete" id="delete-comment">삭제</button>
                <button class="popup-cancel" id="cancel-comment">취소</button>
            </div>
        </div>
    `;
    document.body.appendChild(popup);

    if (existingComment) {
        const lines = existingComment.split('\n').map(line => line.trim());
    
        const comment = lines[1] || "";
        const recovery = lines[2]?.replace("백업예정:", "").trim() || "";
        const name = lines[3] || "";
    
        document.getElementById("comment-text").value = comment;
        document.getElementById("recovery-date").value = recovery;
        document.getElementById("comment-name").value = name;
    }

    document.getElementById("submit-comment").addEventListener("click", async () => {
        const name = document.getElementById("comment-name").value;
        const comment = document.getElementById("comment-text").value;
        const recovery = document.getElementById("recovery-date").value;

        const payload = {
            eqName, motor: motorNum, pos: actualPos, name, comment, recovery, posLabel: posName,
            userName: currentUserInfo?.userName || "unknown" // ✅ 추가
        };

        const res = await fetch('/api/spec/comment', {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            alert('저장되었습니다.');
        
            const fullComment = `\n${comment}\n백업예정: ${recovery}\n${name}`;
            commentMap[commentKey] = fullComment;  // ✅ 즉시 최신 데이터로 갱신
        
            // 아이콘 없으면 생성, 이미 있으면 title 갱신
            if (selectedTd) {
                let icon = selectedTd.querySelector("span");
                if (!icon) {
                    icon = document.createElement("span");
                    icon.textContent = "📝";
                    icon.style.marginLeft = "4px";
                    icon.style.cursor = "pointer";
                    selectedTd.appendChild(icon);
                }
                icon.title = fullComment;  // ✅ 이모티콘 title도 최신으로 갱신
            }
        
            exitInformMode();
            popup.remove();
        } else {
            alert("저장 실패");
        }

        popup.remove();
    });

    document.getElementById("delete-comment").addEventListener("click", async () => {
        if (!confirm('해당 EQ/UNIT의 최신 Inform을 삭제할까요?')) return;

        const res = await fetch('/api/spec/comment', {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ eqName, motor: motorNum, pos: actualPos, userName: currentUserInfo?.userName || "unknown" })
        });

        if (res.ok) {
            alert("Inform이 삭제되었습니다.");
            delete commentMap[commentKey];

            if (selectedTd) {
                const icon = selectedTd.querySelector("span");
                if (icon) selectedTd.removeChild(icon);
            }

            popup.remove();
            exitInformMode();
        } else {
            alert("삭제 실패");
        }
    });

    document.getElementById("cancel-comment").addEventListener("click", () => {
        popup.remove();
        exitInformMode();
    });
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && informMode) {
        exitInformMode();
    }
});


document.addEventListener("click", function (e) {
    const td = e.target.closest("td");
    if (!td) return;

    const motor = td.dataset.motor;
    const pos = td.dataset.pos;

    const isCommentIconClicked = e.target.tagName === "SPAN" && td.querySelector("span");

    if ((informMode || isCommentIconClicked) && motor && typeof pos === 'string') {
        document.querySelectorAll("#motorTable td").forEach(td => {
            td.classList.remove("inform-hover", "selected-td");
        });
        td.classList.add("selected-td");
        selectedTd = td;
        openCommentPopup(motor, pos);
        informMode = false;
    }
});