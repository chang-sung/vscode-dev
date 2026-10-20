
const motor = {
    1: { name: "LD PA TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD", speed: 0}, 3: {name: "UNLOAD A", speed: 0}, 4: {name: "UNLOAD B", speed: 0} } }, 
    2: { name: "LD PA TR_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD", speed: 0}, 3: {name: "UNLOAD", speed: 0} } }, 
    3: { name: "ULD PA TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD A", speed: 0}, 3: {name: "LOAD B", speed: 0}, 4: {name: "UNLOAD", speed: 0} } }, 
    4: { name: "ULD PA TR_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD", speed: 0}, 3: {name: "UNLOAD", speed: 0} } }, 
    5: { name: "NG PA TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD A", speed: 0}, 3: {name: "LOAD B", speed: 0}, 4: {name: "UNLOAD", speed: 0} } }, 
    6: { name: "NG PA TR_Y", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "LOAD", speed: 0}, 3: {name: "UNLOAD", speed: 0} } }, 
    8: { name: "PA TO TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "UNLOAD", speed: 0} } }, 
    9: { name: "PA PRE STG_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "ALIGN A", speed: 0}, 3: {name: "ALIGN B", speed: 0}, 4: {name: "READ A", speed: 0}, 5: {name: "READ B", speed: 0}, 6: {name: "UNLOAD", speed: 0}, 7: {name: "CLEAN A ST", speed: 0}, 8: {name: "CLEAN B ST", speed: 0}, 10: {name: "CAL", speed: 0} } },
    10: { name: "PA PRE STGA_Y", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "ALIGN", speed: 0}, 3: {name: "2D", speed: 0}, 4: {name: "UNLOAD", speed: 0}, 10: {name: "CAL", speed: 0} } }, 
    11: { name: "PA PRE STGA_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "ALIGN", speed: 0}, 3: {name: "2D", speed: 0}, 4: {name: "UNLOAD", speed: 0}, 10: {name: "CAL", speed: 0} } }, 
    12: { name: "PA PRE STGB_Y", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "ALIGN", speed: 0}, 3: {name: "2D", speed: 0}, 4: {name: "UNLOAD", speed: 0}, 10: {name: "CAL", speed: 0} } }, 
    14: { name: "LD MCR CAM_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "READY", speed: 0}, 2: {name: "WORK", speed: 0} } }, 
    15: { name: "ATT LD TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "LOAD", speed: 0}, 2: {name: "STG1", speed: 0}, 3: {name: "STG 2", speed: 0} } },

    39: { name: "PA DET TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 9: {name: "CH LOAD", speed: 0}, 10: {name: "CH UNLOAD", speed: 0} } }, 
    40: { name: "PA DETA_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 2: {name: "CLAMP1", speed: 0}, 3: {name: "DET END1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 5: {name: "CLAMP2", speed: 0}, 6: {name: "DET END2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 8: {name: "TRANSFER", speed: 0}, 9: {name: "CHANGE", speed: 0} } }, 
    41: { name: "PA DETB_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 2: {name: "CLAMP1", speed: 0}, 3: {name: "DET END1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 5: {name: "CLAMP2", speed: 0}, 6: {name: "DET END2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 8: {name: "TRANSFER", speed: 0}, 9: {name: "CHANGE", speed: 0} } }, 
    42: { name: "FILM DET TR_X", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 2: {name: "CLAMP1", speed: 0}, 3: {name: "DET END1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 5: {name: "CLAMP2", speed: 0}, 6: {name: "DET END2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 9: {name: "CH LOAD", speed: 0}, 10: {name: "CH UNLOAD", speed: 0} } }, 
    43: { name: "FILM DETA_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 2: {name: "CLAMP1", speed: 0}, 3: {name: "DET END1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 5: {name: "CLAMP2", speed: 0}, 6: {name: "DET END2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 8: {name: "TRANSFER", speed: 0}, 9: {name: "CHANGE", speed: 0} } }, 
    44: { name: "FILM DETB_Z", group: "", sort: 0, acc: 0, dec: 0, pos : {1: {name: "DET ST1", speed: 0}, 2: {name: "CLAMP1", speed: 0}, 3: {name: "DET END1", speed: 0}, 4: {name: "DET ST2", speed: 0}, 5: {name: "CLAMP2", speed: 0}, 6: {name: "DET END2", speed: 0}, 7: {name: "REMOVE", speed: 0}, 8: {name: "TRANSFER", speed: 0}, 9: {name: "CHANGE", speed: 0} } },

};

motor[1].group = "PA_LOAD";motor[1].sort = 1;
motor[2].group = "PA_LOAD";motor[2].sort = 2;
motor[8].group = "PA_LOAD";motor[8].sort = 3;
motor[9].group = "PA_LOAD";motor[9].sort = 4;
motor[10].group = "PA_LOAD";motor[10].sort = 5;
motor[11].group = "PA_LOAD";motor[11].sort = 6;
motor[12].group = "PA_LOAD";motor[12].sort = 7;
motor[14].group = "PA_LOAD";motor[14].sort = 8;
motor[15].group = "PA_LOAD";motor[15].sort = 9;

motor[39].group = "DET";motor[39].sort = 1;
motor[40].group = "DET";motor[40].sort = 2;
motor[41].group = "DET";motor[41].sort = 3;
motor[42].group = "DET";motor[42].sort = 4;
motor[43].group = "DET";motor[43].sort = 5;
motor[44].group = "DET";motor[44].sort = 6;

const groups = {
    a : ['PLM#38G','PLM#39G','PLM#40G','PLM#41G','PLM#42G','PLM#43G', 'PLM#44G'],
    b : ['PLM#45G','PLM#46G','PLM#47G','PLM#48G','PLM#49G','PLM#50G', 'PLM#51G','PLM#52G','PLM#53G'],
    c : ['PLM#54G','PLM#55G','PLM#56G','PLM#57G', 'PLM#58G','PLM#59G','PLM#60G','PLM#61G','PLM#62G','PLM#63G'],
    d : ['PLM#64G','PLM#65G','PLM#160G','PLM#161G','PLM#162G']
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

window.eqTypeMap = window.eqTypeMap || {};

function getEqSpecType(eqNameAny)
{
    const raw = EqNameUtils.reverse(eqNameAny);

    return eqTypeMap[raw] || "phone";
}

// 객체 전역 노출
window.motor = motor;
window.groups = groups;
window.EqNameUtils = EqNameUtils;

window.getEqSpecType = getEqSpecType;