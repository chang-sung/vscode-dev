let aiDetSequences = [];
let selectedBayGroup = 'all-groups';

$(document).ready(function () {
    $('#bay-division-select').on('change', function () {
        selectedBayGroup = $(this).val();
        renderAiDetImageSequences(getFilteredAiDetSequences());
    });

    loadAiDetImages();
});

function getFilteredAiDetSequences() {
    if (selectedBayGroup === 'all-groups') {
        return aiDetSequences;
    }

    const eqList = AI_DET_BAY_MAP[selectedBayGroup] || [];

    return aiDetSequences.filter(seq => {
        const eqName = String(seq.EQ_name || '').trim();
        return eqList.includes(eqName);
    });
}

async function loadAiDetImages() {
    const $container = $('.inference-image-container');

    if ($container.length === 0) return;

    $container.html('<div class="image-loading">이미지 로딩 중...</div>');

    try {
        const resp = await fetch('/api/ai-det/roller-change-images');

        if (!resp.ok) {
            throw new Error(await resp.text());
        }

        const json = await resp.json();
        const sequences = Array.isArray(json.sequences) ? json.sequences : [];

        aiDetSequences = sequences;
        renderAiDetImageSequences(getFilteredAiDetSequences());

    } catch (err) {
        console.error('[AI DET IMAGE] load error:', err);
        $container.html('<div class="image-error">이미지 데이터를 불러오지 못했습니다.</div>');
    }
}

function renderAiDetImageSequences(sequences) {

    sequences = [...sequences].sort((a, b) => {
        const aTime = a.images?.[0]?.Write_time || '';
        const bTime = b.images?.[0]?.Write_time || '';

        return new Date(bTime) - new Date(aTime); // 최신순
    });

    const $container = $('.inference-image-container');
    $container.empty();

    if (sequences.length === 0) {
        $container.html('<div class="image-empty">조회된 이미지가 없습니다.</div>');
        return;
    }

    sequences.forEach(seq => {
        const images = seq.images || [];

        const imageCount = images.length;

        const statusText =
            imageCount === 4 ? '롤러 자동<br>교체 성공' : '롤러 자동<br>교체 실패';

        const statusClass =
            imageCount === 4 ? 'change-success' : 'change-fail';

        const info = seq.change_info || null;

        const changeInfoHtml = info ? `
            <div class="ai-change-detail">
                <div>에러 유닛 : ${escapeHtml(info.Err_unit ?? '')}</div>
                <div>롤러 사용 : ${escapeHtml(info.Roller_count ?? '')}</div>
                <div>모델 : ${escapeHtml(info.recipe ?? '')}</div>
            </div>
        ` : `
            <div class="ai-change-detail empty">
                변경 정보 없음
            </div>
        `;

        const $row = $(`
            <div class="ai-image-seq-row">

                <div class="ai-image-left">

                    <div class="ai-image-eq-name">
                        ${escapeHtml(seq.EQ_name || '')}
                    </div>

                    <div class="ai-change-status ${statusClass}">
                        ${statusText}
                    </div>

                    ${changeInfoHtml}

                </div>

                <div class="ai-image-grid"></div>

            </div>
        `);

        const $grid = $row.find('.ai-image-grid');

        const inspectionGroups = [
            {
                label: '1차 검사',
                imgs: images.slice(0, 2)
            },
            {
                label: '2차 검사',
                imgs: images.slice(2, 4)
            }
        ];

        inspectionGroups.forEach((group, groupIndex) => {
            const labelClass =
                group.label === '2차 검사'
                    ? 'ai-inspection-label second'
                    : 'ai-inspection-label';

            const $box = $(`
                <div class="ai-inspection-box">
                    <div class="${labelClass}">${group.label}</div>
                    <div class="ai-inspection-images"></div>
                </div>
            `);

            const $imageWrap = $box.find('.ai-inspection-images');

            group.imgs.forEach((img, idx) => {
                const src = `data:image/jpeg;base64,${img.image_base64}`;

                const isDetected = String(img.result).trim() === '1';

                const resultText = isDetected
                    ? '이형지 감지'
                    : '이형지 없음';

                const cardClass = isDetected
                    ? 'ai-image-card detected'
                    : 'ai-image-card';

                const imageLabel = idx === 0 ? 'A' : 'B';

                const $card = $(`
                    <div class="${cardClass}">
                        <div class="ai-image-label">${imageLabel}</div>

                        <img src="${src}" alt="${escapeHtml(img.filename || '')}">

                        <div class="ai-image-info">
                            <div>${formatDateTime(img.Write_time)}</div>
                            <div>${resultText}</div>
                        </div>
                    </div>
                `);

                $imageWrap.append($card);
            });

            for (let i = group.imgs.length; i < 2; i++) {
                $imageWrap.append('<div class="ai-image-card empty"></div>');
            }

            $grid.append($box);

            // 1차 검사와 2차 검사 사이에 화살표 추가
            if (groupIndex === 0) {
                $grid.append(`
                    <div class="ai-inspection-arrow">
                        <img src="/css/02_mon_ai_det/arrow-right.png" alt="arrow">
                    </div>
                `);
            }
        });

        $container.append($row);
    });
}

function formatDateTime(value) {
    if (!value) return '';

    const d = new Date(value);
    if (isNaN(d.getTime())) return value;

    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mi = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');

    return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

