window.smLabels = {
    'SM1': 'SM1 LD PA TR X',
    'SM2': 'SM2 LD PA TR Z',
    'SM8': 'SM8 LD TO TR X',
    'SM9': 'SM9 PA PRE X',
    'SM15': 'SM15 ATT LD TR X',
    'SM38': 'SM38 FILM SPY TR X',
    'SM36': 'SM36 FILM 2SHT X',
    'SM37': 'SM37 FILM LD TR X',
    'SM39': 'SM39 PA DET TR X',
    'SM40': 'SM40 PA DET A Z',
    'SM41': 'SM41 PA DET B Z',
    'SM42': 'SM42 FILM DET TR X',
    'SM43': 'SM43 FILM DET A Z',
    'SM44': 'SM44 FILM DET B Z',
    'SM45': 'SM45 ATT STG 1Y',
    'SM46': 'SM46 SET TBL 1Y',
    'SM47': 'SM47 ATT STG 2Y',
    'SM48': 'SM48 SET TBL 2Y',
    'SM17': 'SM17 ATT ULD TR X',
    'SM18': 'SM18 INSP STG X',
    'SM19': 'SM19 ATT INSP X',
    'SM3': 'SM3 ULD PA TR X',
    'SM4': 'SM3 ULD PA TR Z',
    'SM10': 'SM10 PA PRE AY',
    'SM11': 'SM11 PA PRE AX',
    'SM12': 'SM12 PA PRE BY',
    'SM49': 'SM49 SET 1AX',
    'SM50': 'SM50 SET 1AY',
    'SM52': 'SM52 SET 1AZ',
    'SM53': 'SM53 SET 1BX',
    'SM54': 'SM54 SET 1BY',
    'SM56': 'SM56 SET 1BZ',
    'SM57': 'SM57 SET 2AX',
    'SM58': 'SM58 SET 2AY',
    'SM60': 'SM60 SET 2AZ',
    'SM61': 'SM61 SET 2BX',
    'SM62': 'SM62 SET 2BY',
    'SM64': 'SM64 SET 2BZ',
    'SM20': 'SM20 ATT AY',
    'SM21': 'SM21 ATT BX',
    'SM22': 'SM22 ATT BY',
    'SM106': 'SM106 LD LFT IN Z',
    'SM107': 'SM107 LD LFT OUT Z',
    'SM109': 'SM109 ULD LFT IN Z',
    'SM110': 'SM110 ULD LFT OUT Z'
};

window.ballScrewGroups = [
    { name: '패널공급', indices: [1, 2, 8, 9, 15], data: [] },
    { name: '필름공급', indices: [36, 37, 38], data: [] },
    { name: '박리', indices: [39, 40, 41, 42, 43, 44], data: [] },
    { name: '라미', indices: [45, 46, 47, 48], data: [] },
    { name: '배출', indices: [17, 18, 19, 3, 4], data: [] },
    { name: '리프트', indices: [106, 107, 109, 110], data: [] }
];

window.autoStageGroups = [
    { name: '패널', indices: [10, 11, 12], data: [] },
    { name: '라미1', indices: [49, 50, 52, 53, 54, 56], data: [] },
    { name: '라미2', indices: [57, 58, 60, 61, 62, 64], data: [] },
    { name: '배출', indices: [20, 21, 22], data: [] }
];

window.shiftGroups = {
    a: ['PLM#38G', 'PLM#39G', 'PLM#40G', 'PLM#41G', 'PLM#42G', 'PLM#43G', 'PLM#44G'],
    b: ['PLM#45G', 'PLM#46G', 'PLM#47G', 'PLM#48G', 'PLM#49G', 'PLM#50G', 'PLM#51G', 'PLM#52G', 'PLM#53G'],
    c: ['PLM#54G', 'PLM#55G', 'PLM#56G', 'PLM#57G', 'PLM#58G', 'PLM#59G', 'PLM#60G', 'PLM#61G', 'PLM#62G', 'PLM#63G'],
    d: ['PLM#64G', 'PLM#65G', 'PLM#160G', 'PLM#161G', 'PLM#162G']
};

// EQP 라벨을 UI용으로 축약
window.toShortEqpLabel = function (eqp) {
    if (!eqp) return '';

    const trimmed = String(eqp).trim();
    const label = trimmed.includes('#') ? trimmed.split('#').pop().trim() : trimmed;

    if (label === '160G') return 'A0G';
    if (label === '161G') return 'A1G';
    if (label === '162G') return 'A2G';
    return label;
};
