const cylinder = {

  1: { name: "ATT_LD_TR", group: "ATT_TR", unit_count: 2, pos: { 1: { name: "ATT_LD_TR_DOWN", speed: 0 }, 2: { name: "ATT_LD_TR_UP", speed: 0 } }},
  2: { name: "ATT_ULD_TR", group: "ATT_TR", unit_count: 2, pos: { 1: { name: "ATT_ULD_TR_DOWN", speed: 0 }, 2: { name: "ATT_ULD_TR_UP", speed: 0 } }},

  3: { name: "FILM_SUPPLY_TR_A_HAND", group: "FILM_SUPPLY", unit_count: 2, pos: { 1: { name: "FILM_SUPPLY_TR_A_HAND_UP", speed: 0 }, 2: { name: "FILM_SUPPLY_TR_A_HAND_DN", speed: 0 } }},
  4: { name: "FILM_SUPPLY_TR_B_HAND", group: "FILM_SUPPLY", unit_count: 2, pos: { 1: { name: "FILM_SUPPLY_TR_B_HAND_UP", speed: 0 }, 2: { name: "FILM_SUPPLY_TR_B_HAND_DN", speed: 0 } }},

  5: { name: "LD_TRAY_TR_A", group: "LD_TRAY", unit_count: 4, pos: { 1: { name: "LD_TRAY_TR_A_SHORT_DN", speed: 0 }, 2: { name: "LD_TRAY_TR_A_SHORT_UP", speed: 0 }, 3: { name: "LD_TRAY_TR_A_LONG_DN", speed: 0 }, 4: { name: "LD_TRAY_TR_A_LONG_UP", speed: 0 } }},
  6: { name: "LD_TRAY_TR_B", group: "LD_TRAY", unit_count: 4, pos: { 1: { name: "LD_TRAY_TR_B_SHORT_DN", speed: 0 }, 2: { name: "LD_TRAY_TR_B_SHORT_UP", speed: 0 }, 3: { name: "LD_TRAY_TR_B_LONG_DN", speed: 0 }, 4: { name: "LD_TRAY_TR_B_LONG_UP", speed: 0 } }},
  7: { name: "ULD_TRAY_TR_A", group: "ULD_TRAY", unit_count: 4, pos: { 1: { name: "ULD_TRAY_TR_A_SHORT_DN", speed: 0 }, 2: { name: "ULD_TRAY_TR_A_SHORT_UP", speed: 0 }, 3: { name: "ULD_TRAY_TR_A_LONG_DN", speed: 0 }, 4: { name: "ULD_TRAY_TR_A_LONG_UP", speed: 0 } }},
  8: { name: "ULD_TRAY_TR_B", group: "ULD_TRAY", unit_count: 4, pos: { 1: { name: "ULD_TRAY_TR_B_SHORT_DN", speed: 0 }, 2: { name: "ULD_TRAY_TR_B_SHORT_UP", speed: 0 }, 3: { name: "ULD_TRAY_TR_B_LONG_DN", speed: 0 }, 4: { name: "ULD_TRAY_TR_B_LONG_UP", speed: 0 } }},

  9: { name: "PA_DET_TRASH_R", group: "TRASH", unit_count: 2, pos: { 1: { name: "PA_DET_TRASH_R_FWD", speed: 0 }, 2: { name: "PA_DET_TRASH_R_BWD", speed: 0 } }},
  10: { name: "PA_DET_TRASH_L", group: "TRASH", unit_count: 2, pos: { 1: { name: "PA_DET_TRASH_L_FWD", speed: 0 }, 2: { name: "PA_DET_TRASH_L_BWD", speed: 0 } }},
  
  11: { name: "SET_TBL_1A_ROLLER", group: "LAMI_ROLLER", unit_count: 2, pos: { 1: { name: "SET_TBL_1A_ROLLER_UP", speed: 0 }, 2: { name: "SET_TBL_1A_ROLLER_DN", speed: 0 } }},
  12: { name: "SET_TBL_1B_ROLLER", group: "LAMI_ROLLER", unit_count: 2, pos: { 1: { name: "SET_TBL_1B_ROLLER_UP", speed: 0 }, 2: { name: "SET_TBL_1B_ROLLER_DN", speed: 0 } }},
  13: { name: "SET_TBL_2A_ROLLER", group: "LAMI_ROLLER", unit_count: 2, pos: { 1: { name: "SET_TBL_2A_ROLLER_UP", speed: 0 }, 2: { name: "SET_TBL_2A_ROLLER_DN", speed: 0 } }},
  14: { name: "SET_TBL_2B_ROLLER", group: "LAMI_ROLLER", unit_count: 2, pos: { 1: { name: "SET_TBL_2B_ROLLER_UP", speed: 0 }, 2: { name: "SET_TBL_2B_ROLLER_DN", speed: 0 } }},
  15: { name: "LD_PA_TO_TR", group: "PA_TO_TR", unit_count: 2, pos: { 1: { name: "LD_PA_TO_TR_UP", speed: 0 }, 2: { name: "LD_PA_TO_TR_DN", speed: 0 } }},
  16: { name: "ULD_PA_TO_TR", group: "PA_TO_TR", unit_count: 2, pos: { 1: { name: "ULD_PA_TO_TR_UP", speed: 0 }, 2: { name: "ULD_PA_TO_TR_DN", speed: 0 } }},
};

const shiftGroups = {
    a : ['PLM#38G','PLM#39G','PLM#40G','PLM#41G','PLM#42G','PLM#43G', 'PLM#44G'],
    b : ['PLM#45G','PLM#46G','PLM#47G','PLM#48G','PLM#49G','PLM#50G', 'PLM#51G','PLM#52G','PLM#53G'],
    c : ['PLM#54G','PLM#55G','PLM#56G','PLM#57G', 'PLM#58G','PLM#59G','PLM#60G','PLM#61G','PLM#62G','PLM#63G'],
    d : ['PLM#64G','PLM#65G','PLM#A0G','PLM#A1G','PLM#A2G']
};

const eqpList = [
  'PLM#38G','PLM#39G','PLM#40G','PLM#41G','PLM#42G','PLM#43G', 'PLM#44G',
  'PLM#45G','PLM#46G','PLM#47G','PLM#48G','PLM#49G','PLM#50G', 'PLM#51G','PLM#52G','PLM#53G',
  'PLM#54G','PLM#55G','PLM#56G','PLM#57G', 'PLM#58G','PLM#59G','PLM#60G','PLM#61G','PLM#62G','PLM#63G',
  'PLM#64G','PLM#65G','PLM#160G','PLM#161G','PLM#162G'
];

const CRITICAL_UNITS = [
    'ATT_ULD_TR_DOWN',
    "FILM_SUPPLY_TRA_HAND_DN",
    "FILM_SUPPLY_TRB_HAND_DN"
];

const FDC_UNITS = [
    'ATT_ULD_TR_DOWN',
    "FILM_SUPPLY_TRA_HAND_DN",
    "FILM_SUPPLY_TRB_HAND_DN"
];


// COLUMN_MAPPING: DB → FRONT
const COLUMN_MAPPING = {
    // === LD_TRAY_TR_A
    "LD_TRAY_TR_A_SHORT_DOWN": "LD_TRAY_TR_A_SHORT_DN",
    "LD_TRAY_TR_A_SHORT_UP": "LD_TRAY_TR_A_SHORT_UP",
    "LD_TRAY_TR_A_LONG_DOWN": "LD_TRAY_TR_A_LONG_DN",
    "LD_TRAY_TR_A_LONG_UP": "LD_TRAY_TR_A_LONG_UP",

    // === LD_TRAY_TR_B
    "LD_TRAY_TR_B_SHORT_DOWN": "LD_TRAY_TR_B_SHORT_DN",
    "LD_TRAY_TR_B_SHORT_UP": "LD_TRAY_TR_B_SHORT_UP",
    "LD_TRAY_TR_B_LONG_DOWN": "LD_TRAY_TR_B_LONG_DN",
    "LD_TRAY_TR_B_LONG_UP": "LD_TRAY_TR_B_LONG_UP",

    // === ULD_TRAY_TR_A
    "ULD_TRAY_TR_A_SHORT_DOWN": "ULD_TRAY_TR_A_SHORT_DN",
    "ULD_TRAY_TR_A_SHORT_UP": "ULD_TRAY_TR_A_SHORT_UP",
    "ULD_TRAY_TR_A_LONG_DOWN": "ULD_TRAY_TR_A_LONG_DN",
    "ULD_TRAY_TR_A_LONG_UP": "ULD_TRAY_TR_A_LONG_UP",

    // === ULD_TRAY_TR_B
    "ULD_TRAY_TR_B_SHORT_DOWN": "ULD_TRAY_TR_B_SHORT_DN",
    "ULD_TRAY_TR_B_SHORT_UP": "ULD_TRAY_TR_B_SHORT_UP",
    "ULD_TRAY_TR_B_LONG_DOWN": "ULD_TRAY_TR_B_LONG_DN",
    "ULD_TRAY_TR_B_LONG_UP": "ULD_TRAY_TR_B_LONG_UP",

    // === ATT
    "ATT_LD_TR_DOWN": "ATT_LD_TR_DOWN",
    "ATT_LD_TR_UP": "ATT_LD_TR_UP",
    "ATT_ULD_TR_DOWN": "ATT_ULD_TR_DOWN",
    "ATT_ULD_TR_UP": "ATT_ULD_TR_UP",

    // === PA_DET
    "PA_DET_TRASH_R_FWD": "PA_DET_TRASH_R_FWD",
    "PA_DET_TRASH_R_BWD": "PA_DET_TRASH_R_BWD",
    "PA_DET_TRASH_L_FWD": "PA_DET_TRASH_L_FWD",
    "PA_DET_TRASH_L_BWD": "PA_DET_TRASH_L_BWD",

    // === FILM_SUPPLY
    "FILM_SUPPLY_TRA_HAND_UP": "FILM_SUPPLY_TR_A_HAND_UP",
    "FILM_SUPPLY_TRA_HAND_DN": "FILM_SUPPLY_TR_A_HAND_DN",
    "FILM_SUPPLY_TRB_HAND_UP": "FILM_SUPPLY_TR_B_HAND_UP",
    "FILM_SUPPLY_TRB_HAND_DN": "FILM_SUPPLY_TR_B_HAND_DN",

    // === SET_TBL
    "SET_TBL_1A_ROLLER_UP": "SET_TBL_1A_ROLLER_UP",
    "SET_TBL_1A_ROLLER_DN": "SET_TBL_1A_ROLLER_DN",
    "SET_TBL_1B_ROLLER_UP": "SET_TBL_1B_ROLLER_UP",
    "SET_TBL_1B_ROLLER_DN": "SET_TBL_1B_ROLLER_DN",
    "SET_TBL_2A_ROLLER_UP": "SET_TBL_2A_ROLLER_UP",
    "SET_TBL_2A_ROLLER_DN": "SET_TBL_2A_ROLLER_DN",
    "SET_TBL_2B_ROLLER_UP": "SET_TBL_2B_ROLLER_UP",
    "SET_TBL_2B_ROLLER_DN": "SET_TBL_2B_ROLLER_DN",

    // === PA_TO_TR
    "LD_PA_TO_TR_UP": "LD_PA_TO_TR_UP",
    "LD_PA_TO_TR_DN": "LD_PA_TO_TR_DN",
    "ULD_PA_TO_TR_UP": "ULD_PA_TO_TR_UP",
    "ULD_PA_TO_TR_DN": "ULD_PA_TO_TR_DN"
};


const EqNameUtils = {
    transform: function(eqName) {
        const match = eqName.match(/^PLM#(\d{3})G$/);
        if (!match) return eqName;

        const num = parseInt(match[1]);
        if (num >= 160 && num <= 162) {
            return `PLM#A${num - 160}G`;
        }

        return eqName;
    },

    reverse: function(displayName) {
        const match = displayName.match(/^PLM#A(\d)G$/);
        if (!match) return displayName;

        const num = parseInt(match[1]);
        return `PLM#${160 + num}G`;
    }
};

const displayEqpList = eqpList.map(eq => EqNameUtils.transform(eq));

const foldable = new Set([
    'PLM#41G', 'PLM#43G', 'PLM#54G', 'PLM#55G', 'PLM#60G', 'PLM#70G',
]);

function getEqSpecType(eqNameAny) {
    // eqNameAny가 화면표시명(PLM#A0G) 일 수 있으니 원복
    const raw = (window.EqNameUtils?.reverse)
        ? EqNameUtils.reverse(eqNameAny)
        : eqNameAny;
    
    return foldable.has(raw) ? "foldable" : "phone";
}

/* ===== 전역 역매핑: FRONT→DB (다른 파일에서도 재사용) ===== */
window.cylinder = cylinder;
window.shiftGroups = shiftGroups;
window.eqpList = displayEqpList;
window.rawEqpList = eqpList;
window.__SPEC_LOWER_DISPLAY = {};
window.__SPEC_UPPER_DISPLAY = {};
window.CRITICAL_UNITS = CRITICAL_UNITS;
window.EqNameUtils = EqNameUtils;
window.getEqSpecType = getEqSpecType;


window.FRONT_TO_DB = window.FRONT_TO_DB || Object.fromEntries(
    Object.entries(COLUMN_MAPPING).map(([dbKey, frontKey]) => [frontKey, dbKey])
);
