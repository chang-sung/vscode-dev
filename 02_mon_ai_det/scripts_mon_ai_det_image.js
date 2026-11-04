let aiDetSequences = [];
let selectedBayGroup = 'all-groups';

$(document).ready(function () {
    $('#bay-division-select').on('change', function () {
        selectedBayGroup = $(this).val();
        renderAiDetImageSequences(getFilteredAiDetSequences());
    });

    initAiDetImagePreview();
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

    document.dispatchEvent(new Event('ai-det-preview-hide'));
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
    document.dispatchEvent(new Event('ai-det-preview-hide'));
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
                comment: '(박리 알람 발생 검사)',
                imgs: images.slice(0, 2)
            },
            {
                label: '2차 검사',
                comment: '(이형지 자동 폐기 후 검사)',
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
                    <div class="${labelClass}">
                        <div>${group.label}</div>
                        <div class="ai-inspection-comment">${escapeHtml(group.comment)}</div>
                    </div>
                    <div class="ai-inspection-images"></div>
                </div>
            `);

            const $imageWrap = $box.find('.ai-inspection-images');

            group.imgs.forEach((img, idx) => {
                const src = `data:image/jpeg;base64,${img.image_base64}`;

                const isDetected = String(img.result).trim() === '1';

                const resultText = isDetected
                    ? 'AI 추론 결과 : 이형지 감지'
                    : 'AI 추론 결과 : 이형지 없음';

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


/* 원본 배율(1:1) 미리보기. 화면보다 크면 썸네일 위 마우스 위치로 이동합니다. */
function initAiDetImagePreview() {
    const preview = document.createElement('div');
    preview.className = 'ai-det-image-preview';
    preview.hidden = true;
    preview.setAttribute('aria-hidden', 'true');

    const image = document.createElement('img');
    image.alt = '';
    const caption = document.createElement('div');
    caption.className = 'ai-det-image-preview-caption';
    preview.append(image, caption);
    document.body.appendChild(preview);

    let activeImage = null;
    let pointerX = 0;
    let pointerY = 0;

    function hide() {
        activeImage = null;
        preview.hidden = true;
        image.removeAttribute('src');
    }

    function update() {
        if (!activeImage || !activeImage.naturalWidth || !activeImage.naturalHeight) return;

        const naturalWidth = activeImage.naturalWidth;
        const naturalHeight = activeImage.naturalHeight;
        const width = Math.min(naturalWidth, Math.max(1, window.innerWidth - 32));
        const height = Math.min(naturalHeight, Math.max(1, window.innerHeight - 64));
        const rect = activeImage.getBoundingClientRect();

        // object-fit: contain의 검은 여백을 제외한 실제 이미지 영역 기준
        const scale = Math.min(rect.width / naturalWidth, rect.height / naturalHeight);
        const shownWidth = naturalWidth * scale;
        const shownHeight = naturalHeight * scale;
        const x = Math.max(0, Math.min(1,
            (pointerX - rect.left - (rect.width - shownWidth) / 2) / shownWidth));
        const y = Math.max(0, Math.min(1,
            (pointerY - rect.top - (rect.height - shownHeight) / 2) / shownHeight));

        Object.assign(preview.style, {
            width: width + 'px',
            height: height + 'px',
            left: Math.round((window.innerWidth - width) / 2) + 'px',
            top: Math.round((window.innerHeight - height) / 2) + 'px'
        });
        Object.assign(image.style, {
            width: naturalWidth + 'px',
            height: naturalHeight + 'px',
            left: -Math.round((naturalWidth - width) * x) + 'px',
            top: -Math.round((naturalHeight - height) * y) + 'px'
        });
        caption.textContent = '원본 이미지 · ' + naturalWidth + ' × ' + naturalHeight
            + (width < naturalWidth || height < naturalHeight ? ' · 마우스를 움직여 확인' : '');
        preview.hidden = false;
    }

    $('.inference-image-container')
        .on('mouseenter', '.ai-image-card img', function (event) {
            activeImage = this;
            pointerX = event.clientX;
            pointerY = event.clientY;
            image.src = this.currentSrc || this.src;
            update();
        })
        .on('mousemove', '.ai-image-card img', function (event) {
            if (activeImage !== this) return;
            pointerX = event.clientX;
            pointerY = event.clientY;
            update();
        })
        .on('mouseleave', '.ai-image-card img', hide);

    document.addEventListener('load', function (event) {
        if (event.target === activeImage) update();
    }, true);
    document.addEventListener('error', function (event) {
        if (event.target === activeImage || event.target === image) hide();
    }, true);
    document.addEventListener('scroll', hide, true);
    document.addEventListener('ai-det-preview-hide', hide);
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') hide();
    });
    window.addEventListener('resize', hide);
    window.addEventListener('blur', hide);
}
