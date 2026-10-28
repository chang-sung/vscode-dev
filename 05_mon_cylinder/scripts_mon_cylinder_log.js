// dbKey -> frontKey
function toFrontKey(dbKey) {
  if (!dbKey) return dbKey;
  if (window.COLUMN_MAPPING) {
    // COLUMN_MAPPING: DB→FRONT 이므로 역탐색 필요
    for (const [db, front] of Object.entries(COLUMN_MAPPING)) {
      if (db === dbKey) return front;
    }
  }
  // 혹시 이미 front 형태면 그대로
  return dbKey;
}

// FRONT 키를 사람이 읽기 쉬운 "UNIT + 방향"으로
function prettyUnit(frontKey) {
  if (!frontKey) return frontKey;
  // 예) LD_TRAY_TR_A_SHORT_DN → ["LD","TRAY","TR","A","SHORT","DN"]
  const parts = frontKey.split('_');
  const last = parts[parts.length - 1].toUpperCase();

  const dirMap = { DN: 'DOWN', DOWN: 'DOWN', UP: 'UP', FWD: 'FWD', BWD: 'BWD' };
  const dir = dirMap[last] || last;

  // SHORT/LONG/… 등은 그대로 두고, 마지막 토큰만 방향 라벨로 노출
  // 예: "LD_TRAY_TR_A_SHORT DOWN"
  const base = parts.slice(0, -1).join('_').replaceAll('_', ' ');
  return `${base} ${dir}`;
}

function boundKoLabel(bound) {
  return bound === 'upper' ? '상한Limit' : '하한Limit';
}