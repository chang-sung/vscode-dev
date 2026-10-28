
//================================================================ mon_cylinder

// 최신 스냅샷 + 스펙(min/max) 조회
app.get('/api/cylinder_tact', (req, res) => {
    const dataQuery = `
        SELECT *
        FROM ph2_plm_cylinder_tact AS t1
        WHERE t1.WriteTime = (
        SELECT MAX(t2.WriteTime)
        FROM ph2_plm_cylinder_tact AS t2
        WHERE t2.EQ_Name = t1.EQ_Name
        )
        ORDER BY CAST(SUBSTRING_INDEX(t1.EQ_Name, '#', -1) AS UNSIGNED)
    `;

    const specQuery = `
        SELECT *
        FROM ph2_plm_cylinder_tact
        WHERE EQ_Name = 'spec'
        ORDER BY WriteTime DESC
        LIMIT 1
    `;

    const specMaxQuery = `
        SELECT *
        FROM ph2_plm_cylinder_tact
        WHERE EQ_Name = 'spec_Max'
        ORDER BY WriteTime DESC
        LIMIT 1
    `;

    db_plms.query(dataQuery, (err, dataResults) => {
        if (err) {
        console.error('Failed to fetch data:', err);
        return res.status(500).json({ success: false, error: 'DATA_QUERY_ERROR' });
        }

        db_plms.query(specQuery, (specErr, specResults) => {
            if (specErr) {
                console.error('Failed to fetch specs:', specErr);
                return res.status(500).json({ success: false, error: 'SPEC_QUERY_ERROR' });
            }

            db_plms.query(specMaxQuery, (specMaxErr, specMaxResults) => {
                if (specMaxErr) {
                console.error('Failed to fetch max specs:', specMaxErr);
                return res.status(500).json({ success: false, error: 'SPEC_MAX_QUERY_ERROR' });
                }

                const specs = specResults?.[0] ?? null;
                const specs_Max = specMaxResults?.[0] ?? null;

                return res.json({
                success: true,
                data: dataResults ?? [],
                specs,
                specs_Max,
                });
            });
        });
    });
});


app.put('/api/cylinder_tact/specs', (req, res) => {
  const { specs, specs_Max, logs = [] } = req.body || {};

  if ((!specs || Object.keys(specs).length === 0) &&
      (!specs_Max || Object.keys(specs_Max).length === 0)) {
    return res.status(400).json({ success: false, error: 'EMPTY_PAYLOAD' });
  }

  // === 기존과 유사한 UPDATE 쿼리 빌더 ===
  const buildUpdate = (eqName, colsObj) => {
    const cols = Object.keys(colsObj || {});
    if (!cols.length) return null;
    const setClause = cols.map(col => `\`${col}\` = ?`).join(', ');
    const sql = `
      UPDATE ph2_plm_cylinder_tact
      SET ${setClause}, WriteTime = NOW()
      WHERE EQ_Name = ?
      ORDER BY WriteTime DESC
      LIMIT 1
    `;
    const params = [...cols.map(c => colsObj[c]), eqName];
    return { sql, params };
  };

  // === 로그 벌크 인서트 빌더 (있을 때만 실행) ===
  const buildInsertLogs = (logArr) => {
    if (!Array.isArray(logArr) || logArr.length === 0) return null;

    const valuesClause = logArr.map(() => '(NOW(),?,?,?,?,?)').join(', ');
    const sql = `
      INSERT INTO plm_s.ph2_plm_cylinder_tact_state_log
        (WriteTime, EQ_Name, Unit, Previous_spec, Next_spec, Comment)
      VALUES ${valuesClause}
    `;
    const params = [];
    logArr.forEach(l => {
      params.push(
        (l?.EQ_Name ?? 'ALL'),
        (l?.Unit ?? null),
        (l?.Previous_spec ?? null),
        (l?.Next_spec ?? null),
        (l?.Comment ?? null)
      );
    });
    return { sql, params };
  };

  const qSpec    = buildUpdate('spec', specs);
  const qSpecMax = buildUpdate('spec_Max', specs_Max);
  const qLogs    = buildInsertLogs(logs);

  const runQuery = (q) => new Promise((resolve, reject) => {
    if (!q) return resolve({ skipped: true });
    db_plms.query(q.sql, q.params, (err, result) => err ? reject(err) : resolve(result));
  });

  (async () => {
    try {
      if (qSpec)    await runQuery(qSpec);
      if (qSpecMax) await runQuery(qSpecMax);
      if (qLogs)    await runQuery(qLogs);     // 트랜잭션 없이 단순 실행

      return res.json({ success: true });
    } catch (e) {
      console.error('[PUT /api/cylinder_tact/specs] ERROR:', e);
      return res.status(500).json({ success: false, error: 'UPDATE_FAILED' });
    }
  })();
});

// 전체(또는 최신) 목록 — 프론트는 최신만 쓰면 충분
app.get('/api/cylinder_tact/inform', async (req, res) => {
  try {
    // 각 EQ/UNIT별 최신 1건
    const sql = `
      SELECT t.*
      FROM plm_s.ph2_plm_cylinder_tact_state t
      JOIN (
        SELECT EQ_Name, Unit, MAX(WriteTime) AS mw
        FROM plm_s.ph2_plm_cylinder_tact_state
        GROUP BY EQ_Name, Unit
      ) x
      ON t.EQ_Name = x.EQ_Name AND t.Unit = x.Unit AND t.WriteTime = x.mw
      ORDER BY t.EQ_Name, t.Unit
    `;
    const [rows] = await db_plms.promise().query(sql);
    res.json(rows);
  } catch (e) {
    console.error('GET /api/cylinder_tact/inform error:', e);
    res.status(500).json({ error:'INFORM_QUERY_FAILED' });
  }
});

// 특정 EQ/UNIT 최신 1건 (팝업 오픈 시 프리필)
app.get('/api/cylinder_tact/inform/one', async (req, res) => {
  const { eq, unit } = req.query;
  if (!eq || !unit) return res.json(null);
  try{
    const sql = `
      SELECT *
      FROM plm_s.ph2_plm_cylinder_tact_state
      WHERE EQ_Name = ? AND Unit = ?
      ORDER BY WriteTime DESC
      LIMIT 1
    `;
    const [rows] = await db_plms.promise().query(sql, [eq, unit]);
    res.json(rows[0] || null);
  }catch(e){
    console.error('GET one inform error:', e);
    res.status(500).json({ error:'INFORM_ONE_FAILED' });
  }
});

// 등록/수정: Insert-Only(히스토리형)
app.post('/api/cylinder_tact/inform', async (req, res) => {
  const { eqName, unit, writer, backupDate, comment } = req.body || {};
  if (!eqName || !unit) {
    return res.status(400).json({ error: 'MISSING_EQ_OR_UNIT' });
  }
  try{
    const sql = `
      INSERT INTO plm_s.ph2_plm_cylinder_tact_state
        (WriteTime, EQ_Name, Unit, Writer, Backup_Date, Comment)
      VALUES (NOW(), ?, ?, ?, ?, ?)
    `;
    const params = [eqName, unit, writer || null, backupDate || null, comment || null];
    await db_plms.promise().query(sql, params);
    res.json({ success:true });
  }catch(e){
    console.error('POST /api/cylinder_tact/inform error:', e);
    res.status(500).json({ error:'INFORM_INSERT_FAILED', message: e?.sqlMessage || e?.message });
  }
});

// 삭제: 최신 1건 물리 삭제(간단) — 필요 시 논리삭제로 변경
app.delete('/api/cylinder_tact/inform', async (req, res) => {
  const { eqName, unit } = req.body || {};
  if (!eqName || !unit) {
    return res.status(400).json({ error: 'MISSING_EQ_OR_UNIT' });
  }
  try{
    // 최신 1건만 삭제
    const sql = `
      DELETE FROM plm_s.ph2_plm_cylinder_tact_state
      WHERE EQ_Name = ? AND Unit = ?
      ORDER BY WriteTime DESC
      LIMIT 1
    `;
    await db_plms.promise().query(sql, [eqName, unit]);
    res.json({ success:true });
  }catch(e){
    console.error('DELETE /api/cylinder_tact/inform error:', e);
    res.status(500).json({ error:'INFORM_DELETE_FAILED', message: e?.sqlMessage || e?.message });
  }
});