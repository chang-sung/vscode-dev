// scripts_mon_cylinder_inform.js
let __informMode = false;
let __informSelectedTd = null;

function exitInformMode() {
    __informMode = false;
    const table = document.querySelector("#cylinder-table-container");
    table?.classList.remove("inform-mode-active");
    document.querySelectorAll("#cylinder-table-container td").forEach(td=>{
        td.classList.remove("inform-hover", "selected-td");
    });
    const btn = document.getElementById("inform-btn");
    if (btn) btn.innerText = "Inform 등록";
}

// 버튼 토글
document.addEventListener('DOMContentLoaded', ()=>{
    const btn = document.getElementById("inform-btn");
    if (!btn) return;

    btn.addEventListener("click", () => {
        __informMode = !__informMode;
        const table = document.querySelector("#cylinder-table-container");

        if (__informMode) {
        alert("메모(Inform)를 남길 부분을 클릭하세요.");
        table?.classList.add("inform-mode-active");
        document.querySelectorAll("#cylinder-table-container td").forEach(td=>{
            td.classList.add("inform-hover");
        });
        btn.innerText = "Inform 취소";
        } else {
        exitInformMode();
        }
    });
});

// 셀 클릭 → 팝업 열기
document.addEventListener("click", function (e) {
    const td = e.target.closest("#cylinder-table-container td");
    if (!td) return;

    const isIcon = e.target?.dataset?.informIcon === "1";
    if (!__informMode && !isIcon) return;

    const eqName = td.dataset.eq;
    const unit   = td.dataset.unit; // DB Key 그대로 (ATT_ULD_TR_DOWN 등)
    if (!eqName || !unit) return;

    document.querySelectorAll("#cylinder-table-container td").forEach(x=>{
        x.classList.remove("selected-td");
    });
    td.classList.add("selected-td");
    __informSelectedTd = td;

    openInformPopup(eqName, unit, td);
    __informMode = false;
});

// 기존 인폼 불러와서 아이콘 뿌리기
async function loadAndPaintInforms(){
  try {
    const res = await fetch('/api/cylinder_tact/inform');
    if (!res.ok) throw new Error('HTTP '+res.status);
    const list = await res.json(); // [{EQ_Name, Unit, Writer, Backup_Date, Comment, WriteTime}, ...]
    const table = document.getElementById('cylinder-table-container');
    if (!table) return;

    // 모든 셀에서 기존 아이콘 제거 후 다시 그림
    table.querySelectorAll('td span[data-inform-icon="1"]').forEach(s=>s.remove());

    list.forEach(row=>{
      const { EQ_Name, Unit, Comment } = row;
      const td = table.querySelector(`td[data-eq="${cssEsc(EQ_Name)}"][data-unit="${cssEsc(Unit)}"]`);
      if (!td) return;
      paintInformIcon(td, Comment);
    });
  } catch(e){
    console.warn('loadAndPaintInforms failed:', e);
  }
}

function paintInformIcon(td, title){
  let icon = td.querySelector(`span[data-inform-icon="1"]`);
  if (!icon){
    icon = document.createElement('span');
    icon.dataset.informIcon = "1";
    icon.textContent = "📝";
    icon.style.marginLeft = "4px";
    icon.style.cursor = "pointer";
    td.appendChild(icon);
  }
  icon.title = title || '';
}
function cssEsc(s){
  return String(s).replace(/(["\\])/g,'\\$1');
}

// 팝업
function openInformPopup(eqName, unit, td) {
  const popup = document.createElement("div");
  popup.className = "popup-overlay";
  popup.innerHTML = `
    <div class="popup-content">
        <h3>Inform 등록</h3>
        <h3 style="font-size: 14px;">${eqName}\n-\n${unit}</h3>

        <div class="popup-row">
            <label>작성자</label>
            <input type="text" id="inform-writer" value="${(window.currentUserInfo?.userName)||''}">
        </div>
        <div class="popup-row">
            <label>백업 예정일</label>
            <input type="date" id="inform-backup" value="${new Date().toISOString().slice(0,10)}">
        </div>
        <div class="popup-row">
            <label>메모</label>
            <textarea id="inform-comment" rows="3" placeholder="내용을 입력하세요."></textarea>
        </div>

        <div class="popup-buttons">
            <button class="popup-submit" id="inform-save">저장</button>
            <button class="popup-delete" id="inform-delete">삭제</button>
            <button class="popup-cancel" id="inform-cancel">취소</button>
        </div>
    </div>
  `;
  document.body.appendChild(popup);

  // 기존 값 로드(있으면)
  (async ()=>{
    try{
      const res = await fetch(`/api/cylinder_tact/inform/one?eq=${encodeURIComponent(eqName)}&unit=${encodeURIComponent(unit)}`);
      if (res.ok){
        const row = await res.json(); // or null
        if (row){
          document.getElementById('inform-writer').value = row.Writer || (window.currentUserInfo?.userName)||'';
          document.getElementById('inform-backup').value = (row.Backup_Date||'').slice(0,10);
          document.getElementById('inform-comment').value = row.Comment || '';
        }
      }
    }catch(_){}
  })();

  document.getElementById("inform-save").addEventListener("click", async ()=>{
    const writer = document.getElementById('inform-writer').value.trim();
    const backup = document.getElementById('inform-backup').value;
    const comment= document.getElementById('inform-comment').value.trim();

    try{
      const res = await fetch('/api/cylinder_tact/inform', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ eqName, unit, writer, backupDate: backup||null, comment })
      });
      if (!res.ok){
        const t = await res.text().catch(()=> '');
        throw new Error(t||('HTTP '+res.status));
      }
      // 아이콘/타이틀 페인트
      paintInformIcon(td, comment);
      alert('저장되었습니다.');
      popup.remove();
      exitInformMode();
    }catch(e){
      console.error(e);
      alert('저장 실패: '+e.message);
    }
  });

  document.getElementById("inform-delete").addEventListener("click", async ()=>{
    if (!confirm('해당 EQ/UNIT의 최신 Inform을 삭제할까요?')) return;
    try{
      const res = await fetch('/api/cylinder_tact/inform', {
        method:'DELETE',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ eqName, unit })
      });
      if (!res.ok){
        const t = await res.text().catch(()=> '');
        throw new Error(t||('HTTP '+res.status));
      }
      // 아이콘 제거
      const icon = td.querySelector('span[data-inform-icon="1"]');
      if (icon) icon.remove();
      alert('삭제되었습니다.');
      popup.remove();
      exitInformMode();
    }catch(e){
      console.error(e);
      alert('삭제 실패: '+e.message);
    }
  });

  document.getElementById("inform-cancel").addEventListener("click", ()=>{
    popup.remove();
    exitInformMode();
  });
}

// ESC 취소
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && __informMode) exitInformMode();
});

// 테이블 렌더 후 기존 인폼 표시 (renderTable 끝에서 호출해줘)
window.__loadAndPaintCylinderInforms = loadAndPaintInforms;