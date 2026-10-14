// ---------------------------------------------------------------------------- 모터 부하율 모니터링

app.get('/get-ph2-motor-pick-load-data', (req, res) => {
    const { eqp, startDate, endDate } = req.query;

    // 날짜 유효성 체크 함수
    const isValidDate = (s) => {
        if (!s) return false;
        const t = new Date(s);
        return !Number.isNaN(t.getTime());
    };

    // 기본 쿼리 + 조건 배열
    let query = `
        SELECT EQ_Name, WriteTime, SM1, SM2, SM3, SM8, SM9, SM10, SM11, SM12, SM15, SM17, SM18, SM19, SM20, SM21, SM22,
               SM36, SM37, SM38, SM39, SM40, SM41, SM42, SM43, SM44, SM45, SM46, SM47, SM48, SM49, SM50, SM52, SM53, SM54, SM56, SM57, SM58, SM60, SM61, SM62, SM64,
               SM106, SM107, SM109, SM110
        FROM plc_log.ph2_motor_pick_load
        WHERE EQ_Name != 'SPEC'
    `;
    const params = [];

    if (eqp) {
        query += ` AND EQ_Name = ?`;
        params.push(eqp);
    }

    // startDate/endDate가 유효하면 그대로 사용, 아니면 최근 6개월 자동
    if (isValidDate(startDate) && isValidDate(endDate)) {
        query += ` AND WriteTime BETWEEN ? AND ?`;
        params.push(startDate, endDate);
    } else {
        // 서버/DB 기준 최근 6개월
        query += ` AND WriteTime BETWEEN (NOW() - INTERVAL 6 MONTH) AND NOW()`;
    }

    query += ` ORDER BY WriteTime ASC`;

    db.query(query, params, (err, results) => {
        if (err) {
            console.error('Database query error:', err);
            return res.status(500).json({ error: 'Database query failed', details: err.message });
        }

        // ✅ 빈 결과도 200으로 빈 배열 반환
        return res.json(Array.isArray(results) ? results : []);
    });
});


app.get('/get-eqname-list', (req, res) => {
    // EQ_Name을 문자열로 변환한 후 DISTINCT로 중복 제거
    db.query(
        `SELECT DISTINCT CAST(EQ_Name AS CHAR) AS EQ_Name
         FROM plc_log.ph2_motor_pick_load
         WHERE EQ_Name NOT LIKE '%SPEC%'`, 
        (err, results) => {
            if (err) return res.status(500).send(err);

            const eqNameList = results.map(row => row.EQ_Name);
            res.json(eqNameList);
        }
    );
});

function toFullEqName(eqp) {
    if (!eqp) return eqp;
    const s = String(eqp).trim().toUpperCase();
    if (s.includes('#')) return s; // 이미 풀 라벨

    // 특수 매핑
    const special = { A0G: 'PLM#160G', A1G: 'PLM#161G', A2G: 'PLM#162G' };
    if (special[s]) return special[s];

    // 일반 케이스: 40G 같은 형태
    if (/^\d+G$/.test(s)) return `PLM#${s}`;

    return s; // 그 외는 그대로
}

function toShortEqName(eqp) {
    if (!eqp) return eqp;
    const s = String(eqp).trim().toUpperCase();

    // 특수 매핑
    const special = { 'PLM#160G': 'A0G', 'PLM#161G': 'A1G', 'PLM#162G': 'A2G' };
    if (special[s]) return special[s];

    // 일반 케이스: "PLM#40G" → "40G"
    if (s.startsWith('PLM#')) return s.replace('PLM#', '');

    return s; // 그 외는 그대로
}

app.get('/api/motor-board/summary', (req, res) => {
    const sql = `
        SELECT
            Write_time,
            Last_modify_time,
            EQ_Name,
            Type,
            Axis,
            Lamp_color,
            Outlier_count
        FROM plc_log.board_ph2_plm_motor_pick_load
    `;

    db.query(sql, [], (err, rows) => {
        if (err) {
            console.error('[motor-board-summary] DB error:', err);
            return res.status(500).json({ ok: false, error: 'db failed' });
        }

        const mappedRows = (rows || []).map(row => ({
            ...row,
            EQ_Name: toShortEqName(row.EQ_Name)
        }));

        res.json({ ok: true, data: mappedRows });
    });
});
