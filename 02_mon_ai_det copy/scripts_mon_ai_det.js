$(document).ready(function () {
    initAiDetPageTabs();
});

function initAiDetPageTabs() {
    const defaultPage = 'ai_det';

    setAiDetPage(defaultPage);

    $('.bay-group-btn').on('click', function () {
        const page = $(this).data('group');
        setAiDetPage(page);
    });
}

function setAiDetPage(page) {

    // 버튼 선택 상태
    $('.bay-group-btn').removeClass('active');
    $(`.bay-group-btn[data-group="${page}"]`).addClass('active');

    // 모든 페이지 숨김
    $('.ai-det-filter-container').hide();
    $('.mon-ai-det-wrapper').hide();

    switch (page) {

        case 'ai_det':
            $('.ai-det-filter-container').show();
            $('.mon-ai-det-wrapper').show();
            break;

        // 향후 추가
        // case 'ai_summary':
        //     $('.ai-summary-filter-container').show();
        //     $('.mon-ai-summary-wrapper').show();
        //     break;
    }
}