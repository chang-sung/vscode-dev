const mysql = require('mysql2');
const path = require('path');
const express = require('express');
const app = express();
const schedule = require('node-schedule');
let db, db_motor, db_plms; // 🔹 새 DB 객체 추가

// 서버 시작
const port = 3001;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});

app.use(express.static(path.join(__dirname, 'public')));

// JSON 형식의 데이터를 받기 위한 미들웨어
app.use(express.json());

// 기본 경로로 접속했을 때 parts.html 반환
app.get(['/',''], (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '00_main.html'));
});

app.get('/mon_parts', (req, res) => {
    return res.redirect(303, '/01_mon_parts.html?tab=purchase');
});

app.get('/mon_ai_det', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '02_mon_ai_det.html'));
});

app.get('/roller_repair', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '04_roller_repair.html'));
});

app.get('/mon_cylinder', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '05_mon_cylinder.html'));
});

app.get('/mon_motorsetting', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '06_mon_motorsetting.html'));
});

app.get('/mon_pickload', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '07_mon_load.html'));
});

app.get('/mon_acv', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '08_mon_acv.html'));
});

app.get('/pad', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '09_mon_pad_pm.html'));
});

app.get('/pcmon_notice', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', '99_pcmon_notice.html'));
});

// ================================================================ 바로가기 엔드포인트====================
app.get('/mon_pickload/:eqp/:type/:axis', (req, res) => {
    const rawEqp = decodeURIComponent(req.params.eqp);
    const type   = decodeURIComponent(req.params.type);
    const axis   = decodeURIComponent(req.params.axis);

    // ✅ 타입 정규화
    const normType = (() => {
        const t = (type || '').toLowerCase().replace(/[-_]/g, ' ');
        if (t.includes('ball')) return 'Ball Screw';
        if (t.includes('auto')) return 'Auto Stage';
        return type;
    })();

    // ✅ EQP 정규화 (A0G/A1G/A2G → PLM#160G/161G/162G, "PLM#A0G"도 포함)
    const fullEqp = toFullEqName(rawEqp);

    const target = `/07_mon_load.html?eqp=${encodeURIComponent(fullEqp)}&type=${encodeURIComponent(normType)}&axis=${encodeURIComponent(axis)}`;
    return res.redirect(303, target);
});

app.get('/mon_motor/:shift', (req, res) => {
    const shift  = decodeURIComponent(req.params.shift);

    const target = `/06_mon_motorsetting.html?shift=${encodeURIComponent(shift)}`;
    return res.redirect(303, target); // ✅ 절대경로 + 303
});

app.get('/mon_cylinder/:shift', (req, res) => {
    const shift  = decodeURIComponent(req.params.shift);

    const target = `/05_mon_cylinder.html?shift=${encodeURIComponent(shift)}`;
    return res.redirect(303, target); // ✅ 절대경로 + 303
});

app.get('/mon_parts/:status', (req, res) => {
  const status = decodeURIComponent(req.params.status);

  const target = `/01_mon_parts.html?status=${encodeURIComponent(status)}`;
  return res.redirect(303, target);
});

app.get('/EqpRepair', (req, res) => {
    const target = `/01_mon_parts.html?tab=history`;
    return res.redirect(303, target);
});

// ====================================================================================

function connectToDatabase() {
    db = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '1234',
        database: 'plc_log'
    });
   
    db.connect(err => {
        if (err) {
            console.error('데이터베이스 연결 오류:', err);
            setTimeout(connectToDatabase, 2000); // 연결 실패시 2초 후 재시도
        }
    });

    db.on('error', err => {
        console.error('데이터베이스 연결 오류:', err);
        if (err.code === 'PROTOCOL_CONNECTION_LOST') {
            console.log('데이터베이스 연결이 끊어졌습니다. 재연결을 시도합니다.');
            connectToDatabase(); // 연결이 끊어지면 다시 연결
        } else {
            throw err;
        }
    });
}

// PLM_S 데이터베이스 연결
function connectToPlmsDatabase() {
    db_plms = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '1234',
        database: 'plm_s'
    });

    db_plms.connect(err => {
        if (err) {
            console.error('PLM_S 데이터베이스 연결 오류:', err);
            setTimeout(connectToPlmsDatabase, 2000);
        }
    });

    db_plms.on('error', err => {
        console.error('PLM_S 데이터베이스 연결 오류:', err);
        if (err.code === 'PROTOCOL_CONNECTION_LOST') {
            console.log('PLM_S 데이터베이스 재연결 중...');
            connectToPlmsDatabase();
        } else {
            throw err;
        }
    });
}

// PLM_PLC_PC_DATA 데이터베이스 연결
function connectToMotorDatabase() {
    db_motor = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '1234',
        database: 'plm_plc_pc_data'
    });

    db_motor.connect(err => {
        if (err) {
            console.error('PLM_PLC_PC_DATA 데이터베이스 연결 오류:', err);
            setTimeout(connectToMotorDatabase, 2000); // 연결 실패시 2초 후 재시도
        }
    });

    db_motor.on('error', err => {
        console.error('PLM_PLC_PC_DATA 데이터베이스 연결 오류:', err);
        if (err.code === 'PROTOCOL_CONNECTION_LOST') {
            console.log('PLM_PLC_PC_DATA 데이터베이스 재연결 중...');
            connectToMotorDatabase(); // 연결이 끊어지면 다시 연결
        } else {
            throw err;
        }
    });
}

// 두 개의 데이터베이스 연결 시작
connectToDatabase();
connectToPlmsDatabase()
connectToMotorDatabase();

// ---------------------------------------------------------------------------- ip 확인 영역

app.get('/check-client-authority', (req, res) => {
    const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress;

    // IPv6 형식에서 IPv4 주소 추출
    const ipv4 = clientIp.includes('::ffff:') ? clientIp.split('::ffff:')[1] :
                 clientIp === '::1' ? '127.0.0.1' : clientIp;

    // 데이터베이스에서 IP 조회
    const query = 'SELECT user_name, authority FROM web_user_authority WHERE ip = ?';
    db.query(query, [ipv4], (error, results) => {
        if (error) {
            console.error('Error querying database:', error);
            res.status(500).send('Internal Server Error');
        } else if (results.length > 0) {
            const user = results[0];
            // authority가 '관리자'일 경우만 isAdmin을 true로 설정
            const isAdmin = user.authority === '관리자';
            res.send({
                isAdmin,
                userName: user.user_name,
                authority: user.authority, // authority 값 반환
            });
        } else {
            res.send({ isAdmin: false }); // 관리자 권한 없음
        }
    });
});

app.get('/check-client-authority-parts', (req, res) => {
    try {
        const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress;

        // IPv6에서 IPv4 추출
        const ipv4 =
        clientIp?.includes('::ffff:') ? clientIp.split('::ffff:')[1]
        : clientIp === '::1' ? '127.0.0.1'
        : clientIp;

        // parts 컬럼으로 권한 판별
        const sql = 'SELECT user_name, parts AS parts_authority FROM web_user_authority WHERE ip = ?';
        db.query(sql, [ipv4], (err, rows) => {
        if (err) {
            console.error('[check-client-authority-parts] DB error:', err);
            return res.status(500).send('Internal Server Error');
        }
        if (!rows || rows.length === 0) {
            // 기본: 비로그/미등록 사용자는 '사용자' 취급
            return res.json({
            isAdmin: false,
            isBuyer: false,
            authority: '사용자',
            userName: 'Unknown'
            });
        }

        const { user_name, parts_authority } = rows[0] || {};
        const role = parts_authority || '사용자';

        const isAdmin = role === '관리자';
        const isBuyer = role === '구매담당자';

        return res.json({
            isAdmin,
            isBuyer,
            authority: role,     // '관리자' | '사용자' | '구매담당자'
            userName: user_name
        });
        });
    } catch (e) {
        console.error('[check-client-authority-parts] error:', e);
        res.status(500).send('Internal Server Error');
    }
});

// ---------------------------------------------------------------------------- parts_search.html 엔드포인트

// 최신 EMS 재고 업데이트 시간을 가져오는 엔드포인트 추가
app.get('/parts/ems-latest-update', (req, res) => {
  const scope = String(req.query.scope || 'stock').toLowerCase();

  // ✅ 타이포 주의: 테이블명은 purchase 가 맞습니다 (puchase 아님)
  if (scope === 'purchase') {
    // 방법 1) GREATEST (간단/가독성 높음)
    const sql = `
      SELECT GREATEST(
        IFNULL((SELECT MAX(WriteTime) FROM parts_ems_purchase_status_list),       '1970-01-01 00:00:00'),
        IFNULL((SELECT MAX(WriteTime) FROM parts_ems_purchase_status_mro_list),   '1970-01-01 00:00:00')
      ) AS EMS_update_time
    `;
    db.query(sql, (err, result) => {
      if (err) {
        console.error('쿼리 오류:', err);
        res.status(500).send('서버 오류');
        return;
      }
      const row = result && result[0];
      res.json({ EMS_update_time: row ? row.EMS_update_time : null });
    });
  } else {
    // 기존 동작 (stock)
    const sql = `
      SELECT MAX(EMS_update_time) AS EMS_update_time
      FROM parts_essential_list
    `;
    db.query(sql, (err, result) => {
      if (err) {
        console.error('쿼리 오류:', err);
        res.status(500).send('서버 오류');
        return;
      }
      const row = result && result[0];
      res.json({ EMS_update_time: row ? row.EMS_update_time : null });
    });
  }
});

// 교체 이력 카운트: Q-Code 리스트를 받아 Parts_Code별 count 반환
app.post('/parts/repair_counts', async (req, res) => {
  try {
    const codes = (req.body?.codes || [])
      .filter(v => typeof v === 'string' && v.trim().length > 0);

    if (codes.length === 0) return res.json({ counts: {} });

    // MySQL IN 절 안전 포맷
    // mysql2 사용 시: db.format('IN (?)', [codes]) 가능
    const placeholders = codes.map(() => '?').join(',');
    const sql = `
        SELECT Parts_Code, COUNT(*) AS cnt
        FROM plm_eqp_repair_web_log
        WHERE Parts_Code IN (${placeholders})
            AND WriteTime >= DATE_SUB(NOW(), INTERVAL 300 DAY)
        GROUP BY Parts_Code
    `;

    db.query(sql, codes, (err, rows) => {
      if (err) {
        console.error('[repair_counts] query error:', err);
        return res.status(500).json({ error: 'QUERY_FAILED' });
      }
      const map = {};
      (rows || []).forEach(r => { map[r.Parts_Code] = Number(r.cnt) || 0; });
      res.json({ counts: map });
    });
  } catch (e) {
    console.error('[repair_counts] error:', e);
    res.status(500).json({ error: 'SERVER_ERROR' });
  }
});

app.get('/parts/essential_list/:shelfId', (req, res) => {
    const shelfId = req.params.shelfId;
    const sql = `
        SELECT
            pu.EQ_Name,
            pu.Parts_Name,
            pu.Q_Code,
            pu.SUB_Unit,
            pu.Spec,
            pu.Maker,
            IFNULL(pll.Count, 0) AS 라인재고,
            IFNULL(pel.Count_Now, 0) AS EMS재고
        FROM parts_use_list pu
        LEFT JOIN parts_line_list pll
            ON pu.Q_Code = pll.Q_Code

        LEFT JOIN parts_essential_list pel
            ON pu.Q_Code = pel.Q_Code

        WHERE pu.EQ_Unit = ?`;
       
    db.query(sql, [shelfId], (err, result) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }

        res.json(result); // 결과를 배열 형태로 반환
    });
});


// 🔎 전역 검색: Q-Code 또는 업체P/N(=Spec)로 전체 Unit 검색
// body: { type: 'qcode' | 'pn', term: string, eq_name?: string, limit?: number }
app.post('/parts/use_search', (req, res) => {
  try {
    const { type, term, eq_name, limit } = req.body || {};
    if (!type || !term) {
      return res.status(400).json({ error: 'type, term are required' });
    }

    // 안전 파라미터 구성
    const where = [];
    const params = [];

    // 검색 조건
    if (type === 'qcode') {
      where.push('UPPER(TRIM(pu.Q_Code)) = UPPER(TRIM(?))');
      params.push(term);
    } else if (type === 'pn') {
      // 업체P/N → Spec 부분 검색
      where.push('UPPER(pu.Spec) LIKE CONCAT("%", UPPER(TRIM(?)), "%")');
      params.push(term);
    } else {
      return res.status(400).json({ error: 'type must be qcode or pn' });
    }

    // 과도한 응답 방지
    const safeLimit = Number.isFinite(limit) ? Math.max(1, Math.min(Number(limit), 2000)) : 1000;

    // ✅ 기존 단건 API와 동일한 컬럼 셋(조인 포함)으로 전역 검색
    //    - Unit 컬럼 = pu.EQ_Unit (프론트에서 유닛 구분/순환에 사용 가능)
    const sql = `
      SELECT
        pu.EQ_Unit  AS Unit,
        pu.SUB_Unit,
        pu.Parts_Name,
        pu.Q_Code,
        pu.Spec,
        pu.Maker,
        IFNULL(pll.Count, 0)     AS 라인재고,
        IFNULL(pel.Count_Now, 0) AS EMS재고
      FROM parts_use_list pu
      LEFT JOIN parts_line_list       pll ON pu.Q_Code = pll.Q_Code
      LEFT JOIN parts_essential_list  pel ON pu.Q_Code = pel.Q_Code
      ${where.length ? 'WHERE ' + where.join(' AND ') : ''}
      ORDER BY pu.EQ_Unit, pu.SUB_Unit, pu.Parts_Name
      LIMIT ${safeLimit}
    `;

    db.query(sql, params, (err, rows) => {
      if (err) {
        console.error('[use_search] query error:', err);
        return res.status(500).json({ error: 'QUERY_FAILED' });
      }
       
      // rows: [{ Unit, SUB_Unit, Parts_Name, Q_Code, Spec, Maker, 라인재고, EMS재고 }]
      res.json({ rows });
    });
  } catch (e) {
    console.error('[use_search] error:', e);
    res.status(500).json({ error: 'SERVER_ERROR' });
  }
});

// ========== INSERT: parts_use_list ==========
app.post('/parts_use_list', (req, res) => {
    try {
        const { SUB_Unit, Parts_Name, Q_Code, Spec, Maker, EQ_Name, EQ_Unit } = req.body || {};
        if (!SUB_Unit || !Parts_Name || !Q_Code || !Spec || !Maker || !EQ_Name || !EQ_Unit) {
        return res.status(400).json({ error: 'MISSING_FIELDS' });
        }

        // 1) 존재여부 확인
        const sqlSel = `SELECT 1 FROM parts_use_list WHERE Q_Code=? AND Spec=? AND EQ_Unit=? LIMIT 1`;
        db.query(sqlSel, [Q_Code, Spec, EQ_Unit], (errSel, rowsSel) => {
        if (errSel) return res.status(500).json({ error: 'QUERY_FAILED' });

        if (rowsSel && rowsSel.length > 0) {
            // 2) 중복이면 409
            return res.status(409).json({ code: 'DUPLICATE_KEY', message: 'Duplicate Q_Code, Spec, EQ_Unit' });
        }

        // 3) 삽입
        const sqlIns = `
            INSERT INTO parts_use_list
            (SUB_Unit, Parts_Name, Q_Code, Spec, Maker, EQ_Name, EQ_Unit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        `;
        const params = [SUB_Unit, Parts_Name, Q_Code, Spec, Maker, EQ_Name, EQ_Unit];

        db.query(sqlIns, params, (errIns, result) => {
            if (errIns) return res.status(500).json({ error: 'QUERY_FAILED' });
            return res.status(201).json({ ok: true, insertedId: result.insertId });
        });
        });
    } catch (e) {
        console.error('[parts_use_list][insert] server error:', e);
        res.status(500).json({ error: 'SERVER_ERROR' });
    }
});

// ========== UPDATE: parts_use_list ==========
app.put('/parts_use_list/update', (req, res) => {
  try {
    const b = req.body || {};
    const where = b.where || {};
    const data  = b.data  || {};

    // 기본 키(현 구현): (Q_Code, Spec)
    const wQ = (where.Q_Code || '').trim();
    const wS = (where.Spec || '').trim();
    if (!wQ || !wS) {
      return res.status(400).json({ error: 'MISSING_WHERE', required: ['where.Q_Code', 'where.Spec'] });
    }

    const fields = [];
    const params = [];

    const allow = ['SUB_Unit','Parts_Name','Q_Code','Spec','Maker','EQ_Name','EQ_Unit'];
    allow.forEach(k => {
      if (k in data) { fields.push(`${k} = ?`); params.push(data[k]); }
    });

    if (!fields.length) return res.status(400).json({ error: 'NO_FIELDS_TO_UPDATE' });

    const sql = `
      UPDATE parts_use_list
      SET ${fields.join(', ')}
      WHERE Q_Code = ? AND Spec = ?
    `;
    params.push(wQ, wS);

    db.query(sql, params, (err, result) => {
      if (err) {
        console.error('[parts_use_list][update] query error:', err);
        return res.status(500).json({ error: 'QUERY_FAILED' });
      }
      res.json({ ok: true, affectedRows: result.affectedRows });
    });
  } catch (e) {
    console.error('[parts_use_list][update] server error:', e);
    res.status(500).json({ error: 'SERVER_ERROR' });
  }
});

// ========== BULK DELETE: parts_use_list ==========
/*
  body: { rows: [{Q_Code, Spec}, ...] }
*/
app.post('/parts_use_list/delete', (req, res) => {
    try {
        const rowsRaw = Array.isArray(req.body?.rows) ? req.body.rows : [];

        // 공백/널 정리 + 유효성
        const rows = rowsRaw
        .map(r => ({
            Q_Code:  (r?.Q_Code  ?? '').toString().trim(),
            Spec:    (r?.Spec    ?? '').toString().trim(),
            EQ_Unit: (r?.EQ_Unit ?? '').toString().trim(),
        }))
        .filter(r => r.Q_Code && r.Spec && r.EQ_Unit);

        if (!rows.length) {
        return res.status(400).json({ error: 'NO_ROWS' });
        }

        // 1) parts_use_list에서 (Q_Code, Spec, EQ_Unit) 기준으로 삭제
        const placeholders = rows.map(() => '(?, ?, ?)').join(',');
        const delParams = [];
        rows.forEach(r => { delParams.push(r.Q_Code, r.Spec, r.EQ_Unit); });

        const sqlDelete = `
        DELETE FROM parts_use_list
        WHERE (Q_Code, Spec, EQ_Unit) IN (${placeholders})
        `;

        db.query(sqlDelete, delParams, (errDel, resultDel) => {
        if (errDel) {
            console.error('[parts_use_list][delete] query error:', errDel);
            return res.status(500).json({ error: 'QUERY_FAILED' });
        }

        // 2) 고아 essential 정리: 이번 요청에서 영향을 받은 Q_Code들만 대상으로,
        //    parts_use_list에 더 이상 하나도 남지 않은 Q_Code의 essential만 삭제
        const affectedQ = [...new Set(rows.map(r => r.Q_Code))];
        if (affectedQ.length === 0) {
            return res.json({ ok: true, affectedRows: resultDel.affectedRows, cleaned: 0 });
        }

        const qMarks = affectedQ.map(() => '?').join(',');
        const sqlCleanup = `
            DELETE FROM parts_essential_list pe
            WHERE pe.Q_Code IN (${qMarks})
            AND NOT EXISTS (
                SELECT 1
                FROM parts_use_list pu
                WHERE pu.Q_Code = pe.Q_Code
            )
        `;

        db.query(sqlCleanup, affectedQ, (errCln, resultCln) => {
            if (errCln) {
            // 정리 실패는 치명적 에러로 보지 않고, 삭제 결과만 반환 + 로그
            console.error('[cleanup essentials] query error:', errCln);
            return res.json({ ok: true, affectedRows: resultDel.affectedRows, cleaned: 0, cleanupError: true });
            }
            return res.json({
            ok: true,
            affectedRows: resultDel.affectedRows,
            cleaned: resultCln?.affectedRows || 0
            });
        });
        });
    } catch (e) {
        console.error('[parts_use_list][delete] server error:', e);
        res.status(500).json({ error: 'SERVER_ERROR' });
    }
});

app.post('/parts_essential_list/exists', (req, res) => {
    try {
        const qcodes = (req.body?.qcodes || [])
        .filter(v => typeof v === 'string' && v.trim().length > 0);
        if (qcodes.length === 0) return res.json({ exists: {} });

        const placeholders = qcodes.map(() => '?').join(',');
        const sql = `SELECT Q_Code FROM parts_essential_list WHERE Q_Code IN (${placeholders})`;

        db.query(sql, qcodes, (err, rows) => {
        if (err) {
            console.error('[parts_essential_list/exists] query error:', err);
            return res.status(500).json({ error: 'QUERY_FAILED' });
        }
        const exists = {};
        (rows || []).forEach(r => { exists[r.Q_Code] = true; });
        res.json({ exists });
        });
    } catch (e) {
        console.error('[parts_essential_list/exists] error:', e);
        res.status(500).json({ error: 'SERVER_ERROR' });
    }
});

// ---------------------------------------------------------------------------- parts_essential.html 엔드포인트

/** 날짜 포맷 유틸 (필요시 사용) */
function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// INSERT
app.post('/parts_essential_list', (req, res) => {
    const { EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Etc } = req.body || {};
    if (!EQ_Name || !Parts_Name || !Q_Code || !EQ_Unit || !P_N) {
        return res.status(400).json({ error: 'MISSING_FIELDS' });
    }

    const sql = `
        INSERT INTO parts_essential_list
        (EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Etc)
        VALUES (?, ?, ?, ?, ?, ?)
    `;
    db.query(sql, [EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, (Etc ?? null)], (err, result) => {
        if (err) {
        if (err.code === 'ER_DUP_ENTRY') {
            // 🔴 중복은 409로 명확히 전달
            return res.status(409).json({ ok:false, code:'DUPLICATE', message:'duplicate key' });
        }
        console.error('[parts_essential_list][insert] query error:', err);
        return res.status(500).json({ ok:false, code:'QUERY_FAILED' });
        }
        res.status(201).json({ ok:true, insertedId: result.insertId });
    });
});

// SELECT (조인 포함)
app.get('/parts_essential_list', (req, res) => {
  const query = `
    SELECT
      pe.EQ_Name,
      pe.Parts_Name,
      pe.Q_Code,
      pe.EQ_Unit,
      pe.P_N,
      pe.Count       AS 안전재고,        
      pe.Count_Now   AS EMS재고,
      IFNULL(pl.Count, 0) AS 라인재고,
      pe.Etc,
      pe.Purchase_status,
      pe.Memo
    FROM parts_essential_list pe
    LEFT JOIN parts_line_list pl ON pe.Q_Code = pl.Q_Code
  `;
  db.query(query, (err, results) => {
    if (err) {
      console.error('쿼리 오류:', err);
      return res.status(500).send('서버 오류');
    }
    res.json(results);
  });
});

// BULK DELETE: 키 (Q_Code, EQ_Unit, P_N)
app.post('/parts_essential_list/delete', (req, res) => {
  const { rows } = req.body || {};
  if (!Array.isArray(rows) || rows.length === 0) {
    return res.status(400).send('rows is required');
  }

  const tuples = rows
    .map(r => [r.Q_Code, r.EQ_Unit, r.P_N])
    .filter(t => t[0] && t[1] && t[2]);

  if (tuples.length === 0) return res.status(400).send('valid keys required');

  const placeholders = tuples.map(() => '(?,?,?)').join(',');
  const sql = `
    DELETE FROM parts_essential_list
    WHERE (Q_Code, EQ_Unit, P_N) IN (${placeholders})
  `;
  db.query(sql, tuples.flat(), (err, result) => {
    if (err) {
      console.error('delete error:', err);
      return res.status(500).send('delete failed');
    }
    res.json({ ok: true, affected: result.affectedRows });
  });
});

// UPDATE: where = (Q_Code, EQ_Unit, P_N)
app.put('/parts_essential_list/update', (req, res) => {
    const { where, data } = req.body || {};
    if (!where || !where.Q_Code || !where.EQ_Unit || !where.P_N) {
        return res.status(400).send('where(Q_Code, EQ_Unit, P_N) is required');
    }
    if (!data || typeof data !== 'object') {
        return res.status(400).send('data is required');
    }

    const allowed = ['EQ_Name', 'Parts_Name', 'Q_Code', 'EQ_Unit', 'P_N', 'Etc', 'Purchase_status', 'Memo'];
    const fields = [];
    const params = [];
    for (const k of allowed) {
        if (k in data) {
        fields.push(`${k} = ?`);
        params.push(data[k]);
        }
    }
    if (fields.length === 0) return res.status(400).send('no updatable fields');

    const sql = `
        UPDATE parts_essential_list
        SET ${fields.join(', ')}
        WHERE Q_Code = ? AND EQ_Unit = ? AND P_N = ?
    `;
    params.push(where.Q_Code, where.EQ_Unit, where.P_N);

    db.query(sql, params, (err, result) => {
        if (err) {
        console.error('update error:', err);
        return res.status(500).send('update failed');
        }
        res.json({ ok: true, affected: result.affectedRows });
    });
});


// ---------------------------------------------------------------------------- parts_line.html 엔드포인트
// 라인 파츠 드롭다운 list
app.get('/parts_line/distinct/meta', (req, res) => {
  const sqlUnits = `
    SELECT DISTINCT EQ_Unit AS val
    FROM parts_line_list
    WHERE EQ_Unit IS NOT NULL AND EQ_Unit <> ''
    ORDER BY val
  `;
  const sqlParts = `
    SELECT DISTINCT Parts_Name AS val
    FROM parts_line_list
    WHERE Parts_Name IS NOT NULL AND Parts_Name <> ''
    ORDER BY val
  `;

  db.query(sqlUnits, (errU, rowsU) => {
    if (errU) return res.status(500).send('DB error (units)');
    db.query(sqlParts, (errP, rowsP) => {
      if (errP) return res.status(500).send('DB error (parts)');
      res.json({
        units:      (rowsU || []).map(r => r.val).filter(Boolean),
        partsNames: (rowsP || []).map(r => r.val).filter(Boolean),
      });
    });
  });
});

// 주요 파츠 드롭다운 list
app.get('/parts_essential_list/distinct/meta', (req, res) => {
  const sqlUnits = `
    SELECT DISTINCT EQ_Unit AS val
    FROM parts_essential_list
    WHERE EQ_Unit IS NOT NULL AND EQ_Unit <> ''
    ORDER BY val
  `;
  const sqlParts = `
    SELECT DISTINCT Parts_Name AS val
    FROM parts_essential_list
    WHERE Parts_Name IS NOT NULL AND Parts_Name <> ''
    ORDER BY val
  `;

  db.query(sqlUnits, (errU, rowsU) => {
    if (errU) return res.status(500).send('DB error (units)');
    db.query(sqlParts, (errP, rowsP) => {
      if (errP) return res.status(500).send('DB error (parts)');
      res.json({
        units:      (rowsU || []).map(r => r.val).filter(Boolean),
        partsNames: (rowsP || []).map(r => r.val).filter(Boolean),
      });
    });
  });
});

app.get('/parts', (req, res) => {
    db.query('SELECT * FROM parts_line_list', (err, results) => {
        if (err) {
            console.error('쿼리 오류:', err);
            res.status(500).send('서버 오류');
            return;
        } else {
            res.json(results);
        }
    });
});


app.post('/parts', (req, res) => {
    const { EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Count, Stoker, Etc } = req.body;

    db.query('INSERT INTO parts_line_list (EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Count, Stoker, Etc) VALUES (?,?,?,?,?,?,?,?)',
        [EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Count, Stoker, Etc], (err, result) => {
            if (err) {
                res.status(500).send(err);
            } else {
                res.status(201).json({ message: 'Item added' });
            }
        }
    );
});


// Q_Code, EQ_Unit, P_N 세 컬럼 모두 일치 시 삭제
app.post('/parts_line/delete', (req, res) => {
    const { rows = [] } = req.body || {};
    if (!Array.isArray(rows) || rows.length === 0) {
        return res.status(400).send('rows required');
    }

    const COLS = ['Q_Code', 'EQ_Unit', 'P_N'];

    // 유효성 검사
    const cleaned = [];
    for (let i = 0; i < rows.length; i++) {
        const r = rows[i] || {};
        for (const c of COLS) {
        if (r[c] == null || r[c] === '') {
            return res.status(400).send(`row[${i}] missing column: ${c}`);
        }
        }
        cleaned.push({
        Q_Code: r.Q_Code,
        EQ_Unit: r.EQ_Unit,
        P_N: r.P_N
        });
    }

    // WHERE (Q_Code=? AND EQ_Unit=? AND P_N=?) OR ( ... )
    const clauses = cleaned.map(() => '(Q_Code=? AND EQ_Unit=? AND P_N=?)');
    const vals = cleaned.flatMap(r => [r.Q_Code, r.EQ_Unit, r.P_N]);

    const sql = `DELETE FROM parts_line_list WHERE ${clauses.join(' OR ')}`;

    db.query(sql, vals, (err, result) => {
        if (err) return res.status(500).send(err.sqlMessage || 'DB delete error');
        res.json({ deleted: result.affectedRows });
    });
});


// 대량 삭제처럼 POST 사용
app.post('/parts_line/update', (req, res) => {
    const { where, data } = req.body || {};
    if (!where || !data) return res.status(400).send('where/data required');

    // where: 수정 전 키(수정 전 값)
    const { Q_Code: wQ, EQ_Unit: wU, P_N: wP } = where;
    if (!wQ || !wU || !wP) return res.status(400).send('where keys required');

    // data: 수정 후 최종 값
    const {
        EQ_Name = '', Parts_Name = '', Q_Code = '', EQ_Unit = '',
        P_N = '', Count = 0, Stoker = '', Etc = ''
    } = data;

    const sql = `
        UPDATE parts_line_list
        SET EQ_Name=?, Parts_Name=?, Q_Code=?, EQ_Unit=?, P_N=?, Count=?, Stoker=?, Etc=?
        WHERE Q_Code=? AND EQ_Unit=? AND P_N=?
    `;

    const params = [
        EQ_Name, Parts_Name, Q_Code, EQ_Unit, P_N, Count, Stoker, Etc,
        wQ, wU, wP
    ];

    db.query(sql, params, (err, result) => {
        if (err) return res.status(500).send('DB error (update)');
        if (result.affectedRows === 0) return res.status(404).send('Row not found');
        res.json({ ok: true, affected: result.affectedRows });
    });
});

// ---------------------------------------------------------------------------- parts_purchase.html 엔드포인트
// 구매 탭 드롭다운 list
app.get('/parts_purchase/distinct/meta', (req, res) => {

    const sqlParts = `
        SELECT DISTINCT Parts_Div AS val
        FROM parts_purchase_list
        WHERE Parts_Div IS NOT NULL AND Parts_Div <> ''
        ORDER BY val
    `;

    db.query(sqlParts, (errP, rowsP) => {
        if (errP) return res.status(500).send('DB error (parts)');
        res.json({
           
            partsNames: (rowsP || []).map(r => r.val).filter(Boolean),
        });
    });
});

app.get('/parts_purchase_list', (req, res) => {
    const isDate = (s) => /^\d{4}-\d{2}-\d{2}$/.test(s);

    const today = new Date();
    const y = today.getFullYear();
    const m = today.getMonth();
    const firstDay3Months = new Date(y, m - 2, 1);
    const pad = (n) => String(n).padStart(2, '0');
    const defStart = `${firstDay3Months.getFullYear()}-${pad(firstDay3Months.getMonth() + 1)}-01`;
    const defEnd   = `${y}-${pad(m + 1)}-${pad(today.getDate())}`;

    const start = isDate(req.query.start) ? req.query.start : defStart;
    const end   = isDate(req.query.end)   ? req.query.end   : defEnd;

    const searchType = String(req.query.searchType || '').trim();
    const keyword    = String(req.query.keyword || '').trim();

    let sql = `
        SELECT
            p.No, p.Req_Date, p.EQP_Div, p.Site, p.Q_Code, p.Parts_Div, p.Unit, p.P_N,
            p.Requester, p.Req_Comment, p.Purchase_Status, p.Req_Qty, p.Order_Qty,
            p.Unit_Price, p.Total_Price, p.Manager_Comment, p.Avg_LeadTime,
            p.Purchase_Catg, p.Req_Num, p.ERP_PR, p.Q_Code_2, p.Order_Num, p.Order_Date, p.Arrival_Date, p.PO_Reject,
            ems.EMS재고, ln.라인재고
        FROM parts_purchase_list AS p
        LEFT JOIN (
            SELECT EQ_Name, Q_Code, SUM(COALESCE(Count_Now, 0)) AS EMS재고
            FROM parts_essential_list
            GROUP BY EQ_Name, Q_Code
        ) AS ems
          ON ems.EQ_Name = p.EQP_Div
         AND ems.Q_Code = p.Q_Code
        LEFT JOIN (
            SELECT EQ_Name, Q_Code, SUM(COALESCE(Count, 0)) AS 라인재고
            FROM parts_line_list
            GROUP BY EQ_Name, Q_Code
        ) AS ln
          ON ln.EQ_Name = p.EQP_Div
         AND ln.Q_Code = p.Q_Code
        WHERE 1=1
    `;

    const params = [];
    const hasSearch = !!searchType && !!keyword;

    if (hasSearch) {
        const kw = `%${keyword}%`;

        switch (searchType) {
            case 'all-search':
                sql += `
                    AND (
                        p.Q_Code LIKE ?
                        OR p.P_N LIKE ?
                        OR p.Requester LIKE ?
                        OR p.Parts_Div LIKE ?
                        OR p.Req_Comment LIKE ?
                    )
                `;
                params.push(kw, kw, kw, kw, kw);
                break;

            case 'q-code-search':
                sql += ` AND p.Q_Code LIKE ? `;
                params.push(kw);
                break;

            case 'pn-search':
                sql += ` AND p.P_N LIKE ? `;
                params.push(kw);
                break;

            case 'requester-search':
                sql += ` AND p.Requester LIKE ? `;
                params.push(kw);
                break;

            case 'parts_div-search':
                sql += ` AND p.Parts_Div LIKE ? `;
                params.push(kw);
                break;

            default:
                sql += `
                    AND p.Req_Date >= ?
                    AND p.Req_Date < DATE_ADD(?, INTERVAL 1 DAY)
                `;
                params.push(start, end);
                break;
        }
    } else {
        sql += `
            AND p.Req_Date >= ?
            AND p.Req_Date < DATE_ADD(?, INTERVAL 1 DAY)
        `;
        params.push(start, end);
    }

    sql += ` ORDER BY p.No DESC `;

    db.query(sql, params, (err, rows) => {
        if (err) {
            console.error('GET /parts_purchase_list error:', err);
            return res.status(500).send('DB error (list)');
        }
        res.json(rows || []);
    });
});

// 구매 신청 INSERT (Avg_LeadTime 계산 포함)
app.post('/parts_purchase/request', (req, res) => {
    const {
        EQP_Div, Site, Q_Code, Parts_Div, Unit, P_N,
        Req_Qty, Req_Comment, Requester,
        Unit_Price,
        Total_Price
    } = req.body || {};

    const sql = `
        INSERT INTO parts_purchase_list
            (Req_Date, EQP_Div, Site, Q_Code, Parts_Div, Unit, P_N,
            Req_Qty, Req_Comment, Requester, Purchase_Status,
            Avg_LeadTime, Unit_Price, Total_Price)
        SELECT
        NOW(), ?, ?, ?, ?, ?, ?, ?, ?, ?, '구매 요청',
        (
            SELECT CAST(ROUND(AVG(diff)) AS SIGNED)
            FROM (
                SELECT DATEDIFF(Arrival_Date, Order_Date) AS diff
                FROM parts_ems_purchase_status_list
                WHERE Q_Code = ? AND Arrival_Date IS NOT NULL AND Order_Date IS NOT NULL
                UNION ALL
                SELECT DATEDIFF(Arrival_Date, Order_Date) AS diff
                FROM parts_ems_purchase_status_mro_list
                WHERE MRO_Code = ? AND Arrival_Date IS NOT NULL AND Order_Date IS NOT NULL
            ) AS t
        ) AS Avg_LeadTime,
        ?, ?   -- 팝업에서 넘어온 단가, 금액 그대로 저장
        FROM (SELECT 1) d;
    `;

    const params = [
        EQP_Div, Site, Q_Code, Parts_Div, Unit, P_N,
        Req_Qty, Req_Comment, (Requester || 'Unknown'),
        Q_Code, Q_Code, // Avg_LeadTime 계산용
        Unit_Price, Total_Price
    ];

    db.query(sql, params, (err, result) => {
        if (err) {
        console.error('POST /parts_purchase/request error:', err);
        return res.status(500).send('DB error (insert request)');
        }
        res.json({ ok: true, id: result.insertId });
    });
});

function insertPurchaseApprovalLogs(db, nos, approverName) {
    return new Promise((resolve, reject) => {
        if (!Array.isArray(nos) || nos.length === 0) {
        return resolve({ logged: 0 });
        }

        const approver = String(approverName || '').trim() || 'Unknown';

        const sql = `
        INSERT INTO parts_purchase_log (
            WriteTime,
            Approval_Date,
            Req_Date, Site, EQP_Div, Q_Code, Parts_Div, Unit, P_N,
            EMS_Stock,
            Line_Stock,
            Requester, Req_Comment, Purchase_Status,
            Req_Qty,
            Unit_Price, Total_Price,
            Avg_LeadTime,
            Approver
        )
        SELECT
            p.Req_Date           AS WriteTime,
            NOW()                AS Approval_Date,
            p.Req_Date,
            p.Site,
            p.EQP_Div,
            p.Q_Code,
            p.Parts_Div,
            p.Unit,
            p.P_N,
            ems.EMS재고          AS EMS_Stock,
            ln.라인재고          AS Line_Stock,
            p.Requester,
            p.Req_Comment,
            p.Purchase_Status,
            p.Req_Qty,
            p.Unit_Price,
            p.Total_Price,
            p.Avg_LeadTime,
            ? AS Approver
        FROM parts_purchase_list AS p
        LEFT JOIN (
            SELECT EQ_Name, Q_Code, SUM(COALESCE(Count_Now, 0)) AS EMS재고
            FROM parts_essential_list
            GROUP BY EQ_Name, Q_Code
        ) AS ems
            ON ems.EQ_Name = p.EQP_Div
        AND ems.Q_Code  = p.Q_Code
        LEFT JOIN (
            SELECT EQ_Name, Q_Code, SUM(COALESCE(Count, 0)) AS 라인재고
            FROM parts_line_list
            GROUP BY EQ_Name, Q_Code
        ) AS ln
            ON ln.EQ_Name = p.EQP_Div
        AND ln.Q_Code  = p.Q_Code
        WHERE p.No IN (?)
        `;

        // ✅ 디버그용: 원본 rows를 같이 확인할 수 있게 SELECT 먼저 해도 좋음(선택)
        db.query(sql, [approver, nos], (err, result) => {
        if (err) return reject(err); // 여기서 reject
        resolve({ logged: result?.affectedRows || 0 });
        });
    });
}

app.post('/parts_purchase/status-chg', async (req, res) => {
    const { action, rows, approver } = req.body || {};

    if (!['approve','reject'].includes(action)) {
        return res.status(400).send('invalid action');
    }
    if (!Array.isArray(rows) || rows.length === 0) {
        return res.status(400).send('rows required');
    }

    const nextStatus = (action === 'approve') ? '구매 대기' : 'Drop';

    // ✅ 1) 기존 업데이트 로직 (최소 변경)
    const tasks = rows.map(r => new Promise(resolve => {
        const no = Number(r?.No);
        if (!Number.isFinite(no)) return resolve({ ok:false, No:null, updated:0, error:'No required' });

        const sql = `
        UPDATE parts_purchase_list
        SET Purchase_Status = ?
        WHERE No = ?
            AND Purchase_Status = '구매 요청'
            AND Req_Num IS NULL
        LIMIT 1
        `;
        db.query(sql, [nextStatus, no], (err, result) => {
        if (err) return resolve({ ok:false, No:no, updated:0, error:String(err.sqlMessage || err) });
        resolve({ ok:true, No:no, updated: result.affectedRows || 0 });
        });
    }));

    const results = await Promise.all(tasks);
    const updated = results.reduce((s, r) => s + (r.updated || 0), 0);

    // ✅ 2) approve일 때만: "업데이트 성공한 No들"만 로그 저장
    let logged = 0;
    let logError = null;

    if (action === 'approve') {
        const successNos = results
        .filter(r => r.ok && r.updated === 1 && Number.isFinite(r.No))
        .map(r => r.No);

        try {
        const logRes = await insertPurchaseApprovalLogs(db, successNos, approver);
        logged = logRes.logged || 0;
        } catch (e) {
        // 로그 실패를 "전체 실패"로 볼지, "부분 실패"로 볼지 선택 가능
        console.error('[status-chg] log insert failed:', e);
        logError = String(e.sqlMessage || e.message || e);
        }
    }

    // ✅ 3) 응답 (프론트에서 logged 표시 가능)
    return res.json({ ok:true, updated, logged, logError, results });
});

app.post('/parts_purchase/bulk-update', (req,res) => {
    const { rows } = req.body || {};
    if (!Array.isArray(rows) || !rows.length) {
        return res.status(400).json({ error:'rows required' });
    }

    const mapKey = (k) => ({
        PO_Qty:  'Order_Qty',
        PO_No:   'Order_Num',
        PO_Date: 'Order_Date',
    }[k] || k);

    const NUM_COLS  = new Set(['Req_Qty', 'Order_Qty', 'Unit_Price', 'Total_Price']);
    const DATE_COLS = new Set(['Order_Date', 'Arrival_Date']);

    const normalizeVal = (col, v) => {
        if (v === '' || v === undefined) return null;
        if (NUM_COLS.has(col)) {
        const n = Number(v);
        return Number.isFinite(n) ? n : null;
        }
        if (DATE_COLS.has(col)) {
        const s = String(v).trim();
        return s ? s : null;
        }
        return v;
    };

    const tasks = rows.map(r => new Promise(resolve => {
        const raw = r.updates || {};
        if ('Total_Price' in raw) delete raw.Total_Price; // 서버 계산 우선

        const sets = [];
        const params = [];

        // 어떤 값이 바뀌었는지 추적
        let touchedReqQty   = false;
        let touchedOrderQty = false;
        let touchedUnit     = false;
        let touchedQCode    = false;
        let newQCodeValue   = undefined;

        // 1) 일반 SET 구성 (Q_Code는 따로 처리할 거라 건너뜀)
        for (const [k, v0] of Object.entries(raw)) {
            const col = mapKey(k);
            if (col === 'Q_Code') { // ← Q_Code는 나중에 별도 SET
                touchedQCode  = true;
                newQCodeValue = normalizeVal('Q_Code', v0);
                continue;
            }
            const v = normalizeVal(col, v0);

            if (col === 'Req_Qty')   touchedReqQty   = true;
            if (col === 'Order_Qty') touchedOrderQty = true;
            if (col === 'Unit_Price')touchedUnit     = true;

            sets.push(`${col} = ?`);
            params.push(v);
        }

        // 2) Q_Code가 바뀌면: Q_Code만 SET + Unit_Price는 가격 테이블로 동기화
        if (touchedQCode) {
            // Q_Code 자체 업데이트
            sets.push(`Q_Code = ?`);
            params.push(newQCodeValue);

            // 가격 테이블에 있으면 그 가격, 없으면 NULL로 설정
            sets.push(`
                Unit_Price = (
                SELECT Price
                FROM parts_ems_price_list
                WHERE Q_Code = ?
                LIMIT 1
                )
            `);
            params.push(newQCodeValue);

            // Unit_Price가 바뀌었으므로 합계 재계산 트리거
            touchedUnit = true;
        }

        if (!sets.length) return resolve({ updated: 0 });

        // 3) 합계 자동 계산: Q-Code, Req_Qty, Unit_Price 변경 시만 재계산
        if (touchedReqQty || touchedUnit) {
            sets.push(`
                Total_Price = CASE
                WHEN Unit_Price IS NULL OR Req_Qty IS NULL THEN NULL
                ELSE ROUND(Req_Qty * Unit_Price, 2)
                END
            `);
        }

        // 4) WHERE 구성 (No 우선, 없으면 6-key)
        let where = 'No = ?';
        if (r.No != null) {
            params.push(r.No);
        } else {
            where = `
                (Q_Code  <=> ?) AND (Unit <=> ?) AND (P_N <=> ?) AND
                (EQP_Div <=> ?) AND (Parts_Div <=> ?) AND (Requester <=> ?) AND
                (Site <=> ?)  
            `;
            const w = r.where || {};
            params.push(
                w.Q_Code ?? null, w.Unit ?? null, w.P_N ?? null,
                w.EQP_Div ?? null, w.Parts_Div ?? null, w.Requester ?? null,
                w.Site ?? null  
            );
        }

        const sql = `UPDATE parts_purchase_list SET ${sets.join(', ')} WHERE ${where} LIMIT 1`;
        db.query(sql, params, (err, result) => {
        if (err) {
            console.error('bulk-update err:', err);
            return resolve({ updated: 0, error: String(err.sqlMessage || err) });
        }
        resolve({ updated: result.affectedRows || 0 });
        });
    }));

    Promise.all(tasks).then(rs => {
        const updated = rs.reduce((s, r) => s + (r.updated || 0), 0);
        res.json({ ok: true, updated, results: rs });
    });
});

app.post('/parts_purchase/delete', async (req, res) => {
    try {
        const { rows, force } = req.body || {};
        if (!Array.isArray(rows) || rows.length === 0) {
        return res.status(400).send('rows required');
        }

        // ★ Site 포함 7-key NULL-safe where
        const where7 = `
        ((? IS NULL AND Q_Code    IS NULL) OR Q_Code    = ?) AND
        ((? IS NULL AND Unit      IS NULL) OR Unit      = ?) AND
        ((? IS NULL AND P_N       IS NULL) OR P_N       = ?) AND
        ((? IS NULL AND EQP_Div   IS NULL) OR EQP_Div   = ?) AND
        ((? IS NULL AND Parts_Div IS NULL) OR Parts_Div = ?) AND
        ((? IS NULL AND Requester IS NULL) OR Requester = ?) AND
        ((? IS NULL AND Site      IS NULL) OR Site      = ?)
        `;
        const buildParams7 = (w) => ([
        w.Q_Code ?? null,    w.Q_Code ?? null,
        w.Unit ?? null,      w.Unit ?? null,
        w.P_N ?? null,       w.P_N ?? null,
        w.EQP_Div ?? null,   w.EQP_Div ?? null,
        w.Parts_Div ?? null, w.Parts_Div ?? null,
        w.Requester ?? null, w.Requester ?? null,
        w.Site ?? null,      w.Site ?? null,
        ]);

        const statusGuard = force ? '' : `AND Purchase_Status IN ('구매 요청', '구매 대기')`;

        const deleteOne = (r) => new Promise((resolve) => {
        let sql, params;
        if (r.No != null) {
            sql = `DELETE FROM parts_purchase_list WHERE No = ? ${statusGuard}`;
            params = [r.No];
        } else if (r.where) {
            sql = `DELETE FROM parts_purchase_list WHERE ${where7} ${statusGuard}`;
            params = buildParams7(r.where);
        } else {
            return resolve({ ok:false, deleted:0, error:'bad row' });
        }

        db.query(sql, params, (err, result) => {
            if (err) return resolve({ ok:false, deleted:0, error:err });
            resolve({ ok:true, deleted: result.affectedRows || 0 });
        });
        });

        const results = [];
        for (const r of rows) results.push(await deleteOne(r));
        const deleted = results.reduce((s, r) => s + (r.deleted || 0), 0);

        res.json({ ok:true, deleted });
    } catch (e) {
        console.error('POST /parts_purchase/delete error:', e);
        res.status(500).send('DB error (delete)');
    }
});

// GET /parts_ems_price/by-code?code=Q310-123456
app.get('/parts_ems_price/by-code', (req, res) => {
    const code = String(req.query.code || '').trim();
    if (!code) return res.status(400).json({ error: 'code required' });

    const sql = `
        SELECT Q_Code, P_N, Parts_Div, Price
        FROM parts_ems_price_list
        WHERE Q_Code = ?
        LIMIT 1
    `;
    db.query(sql, [code], (err, rows) => {
        if (err) {
            console.error('GET /parts_ems_price/by-code error:', err);
            return res.status(500).json({ error: 'DB error' });
        }
        if (!rows || rows.length === 0) return res.json(null);
        res.json(rows[0]); // { Q_Code, P_N, Parts_Div, Price }
    });
});

// 드래그앤드롭 결과를 반영해서 No 재배치
app.post('/parts_purchase/reorder', (req, res) => {
    const { orderedNos } = req.body || {};


    if (!Array.isArray(orderedNos) || orderedNos.length === 0) {
        return res.status(400).json({ ok:false, error: 'orderedNos required' });
    }

    db.beginTransaction(async (err) => {
        if (err) return res.status(500).json({ ok:false, error: 'tx begin failed' });

        try {
        // 1) 나머지 행(orderedNos에 없는 것들) 뽑기
        const placeholders = orderedNos.map(() => '?').join(',');
        const notInSql = `
            SELECT No
            FROM parts_purchase_list
            ${orderedNos.length ? `WHERE No NOT IN (${placeholders})` : ''}
            ORDER BY No ASC
        `;
        const [restRows] = await new Promise((resolve, reject) => {
            db.query(notInSql, orderedNos, (e, rows) => e ? reject(e) : resolve([rows]));
        });

        // 2) 최종 순서: orderedNos(드래그 순서) + 나머지
        const finalOrder = [...orderedNos, ...restRows.map(r => r.No)];

        // old_no → new_no 매핑
        const total = finalOrder.length;
        const mapping = finalOrder.map((oldNo, idx) => ({
            old_no: Number(oldNo),
            new_no: total - idx   // ← 기존 idx+1 대신 역순 번호
        }));

        if (mapping.length === 0) {
            await new Promise((resolve) => db.rollback(() => resolve()));
            return res.json({ ok:true, updated: 0 });
        }

        // 3) 임시 음수로 부여
        const valuesSql1 = mapping
            .map(() => 'SELECT ? AS old_no, ? AS new_no')
            .join(' UNION ALL ');
        const params1 = mapping.flatMap(m => [m.old_no, m.new_no]);

        const step1Sql = `
            UPDATE parts_purchase_list p
            JOIN (
            ${valuesSql1}
            ) AS m ON p.No = m.old_no
            SET p.No = -m.new_no
        `;
        const step1Res = await new Promise((resolve, reject) => {
            db.query(step1Sql, params1, (e, r) => e ? reject(e) : resolve(r));
        });


        // 4) 음수를 양수로 복원
        const step2Sql = `UPDATE parts_purchase_list SET No = -No WHERE No < 0`;
        const step2Res = await new Promise((resolve, reject) => {
            db.query(step2Sql, [], (e, r) => e ? reject(e) : resolve(r));
        });


        // 5) AUTO_INCREMENT 보정
        const [aiRow] = await new Promise((resolve, reject) => {
            db.query(`SELECT COALESCE(MAX(No),0)+1 AS nextAi FROM parts_purchase_list`, [], (e, rows) => {
            if (e) return reject(e);
            resolve([rows[0] || { nextAi: 1 }]);
            });
        });
        const nextAi = aiRow?.nextAi || 1;
        await new Promise((resolve, reject) => {
            db.query(`ALTER TABLE parts_purchase_list AUTO_INCREMENT = ?`, [nextAi], (e) => e ? reject(e) : resolve());
        });

        await new Promise((resolve, reject) => {
            db.commit(e => e ? reject(e) : resolve());
        });

        return res.json({ ok:true, updated: mapping.length });

        } catch (e) {
        console.error('[reorder] error:', e);
        db.rollback(() => {
            res.status(500).json({ ok:false, error: String(e.sqlMessage || e.message || e) });
        });
        }
    });
});

app.get('/parts/purchase_na_parts', (req, res) => {
    const sql = `
        SELECT
            EQ_Name,
            NA_Q_Code,
            NA_Parts_Name,
            NA_P_N,
            NA_Price,
            Q_Code,
            Parts_Name,
            P_N,
            Price,
            Memo
        FROM parts_purchase_na_parts_list
        ORDER BY No ASC
    `;

    db.query(sql, (err, rows) => {
        if (err) {
            console.error('[purchase_na_parts] DB error:', err);
            return res.status(500).json({ error: 'DB_ERROR' });
        }
        res.json(rows || []);
    });
});

// NA 구매 제한 리스트 전체 저장
app.post('/parts/purchase_na_parts/save', (req, res) => {
    const rows = req.body.rows;

    if (!Array.isArray(rows)) {
        return res.status(400).json({ error: 'INVALID_PAYLOAD' });
    }

    // 값 정리 (undefined/null 방지, 숫자 처리)
    const values = rows.map(r => ([
        r.EQ_Name       || '',
        r.NA_Q_Code     || '',
        r.NA_Parts_Name || '',
        r.NA_P_N        || '',
        (r.NA_Price === '' || r.NA_Price == null || isNaN(Number(r.NA_Price)))
            ? null
            : Number(r.NA_Price),
        r.Q_Code        || '',
        r.Parts_Name    || '',
        r.P_N           || '',
        (r.Price === '' || r.Price == null || isNaN(Number(r.Price)))
            ? null
            : Number(r.Price),
        r.Memo          || ''
    ]));

    const insertSql = `
        INSERT INTO parts_purchase_na_parts_list
            (EQ_Name, NA_Q_Code, NA_Parts_Name, NA_P_N, NA_Price,
             Q_Code, Parts_Name, P_N, Price, Memo)
        VALUES ?
    `;

    // 🔹 1) TRUNCATE로 테이블 비우고 AUTO_INCREMENT = 1로 리셋
    const truncateSql = 'TRUNCATE TABLE parts_purchase_na_parts_list';

    db.query(truncateSql, (err) => {
        if (err) {
            console.error('[purchase_na_parts/save] TRUNCATE error:', err);
            return res.status(500).json({ error: 'DB_TRUNCATE_ERROR' });
        }

        // 비운 후 저장할 데이터가 없으면 바로 종료
        if (values.length === 0) {
            return res.json({ ok: true, affectedRows: 0 });
        }

        // 🔹 2) 현재 rows 순서대로 INSERT
        //    → AUTO_INCREMENT가 1부터 차례대로 부여됨
        db.query(insertSql, [values], (err2, result) => {
            if (err2) {
                console.error('[purchase_na_parts/save] INSERT error:', err2);
                return res.status(500).json({ error: 'DB_INSERT_ERROR' });
            }

            res.json({
                ok: true,
                affectedRows: result.affectedRows || 0
            });
        });
    });
});

// NA_Q_Code 기준으로 NA 구매 불가품 1건 조회
app.get('/parts/purchase_na_parts/by-na-code', (req, res) => {
    const code = (req.query.code || '').trim();
    if (!code) {
        return res.status(400).json({ error: 'NO_CODE' });
    }

    const sql = `
        SELECT
            EQ_Name,
            NA_Q_Code,
            NA_Parts_Name,
            NA_P_N,
            NA_Price,
            Q_Code,
            Parts_Name,
            P_N,
            Price,
            Memo
        FROM parts_purchase_na_parts_list
        WHERE NA_Q_Code = ?
        LIMIT 1
    `;

    db.query(sql, [code], (err, rows) => {
        if (err) {
            console.error('[purchase_na_parts/by-na-code] DB error:', err);
            return res.status(500).json({ error: 'DB_ERROR' });
        }
        // 없으면 null, 있으면 1건 반환
        res.json(rows && rows[0] ? rows[0] : null);
    });
});

// 주기교체 파츠 리스트 조회
app.get('/parts/purchase_periodic_parts', (req, res) => {
    const sql = `
        SELECT
            Q_Code,
            Parts_Div,
            P_N,
            Price,
            Avg_LeadTime,
            Memo
        FROM parts_purchase_periodic_list
        ORDER BY No ASC
    `;

    db.query(sql, (err, rows) => {
        if (err) {
            console.error('[purchase_periodic_parts] DB error:', err);
            return res.status(500).json({ error: 'DB_ERROR' });
        }

        res.json(rows || []);
    });
});


// 주기교체 파츠 리스트 전체 저장
app.post('/parts/purchase_periodic_parts/save', (req, res) => {
    const rows = req.body.rows;

    if (!Array.isArray(rows)) {
        return res.status(400).json({ error: 'INVALID_PAYLOAD' });
    }

    const values = rows.map(r => ([
        r.Q_Code || '',
        r.Parts_Div || '',
        r.P_N || '',
        (r.Price === '' || r.Price == null || isNaN(Number(r.Price)))
            ? null
            : Number(r.Price),
        (r.Avg_LeadTime === '' || r.Avg_LeadTime == null || isNaN(Number(r.Avg_LeadTime)))
            ? null
            : Number(r.Avg_LeadTime),
        r.Memo || ''
    ]));

    const truncateSql = 'TRUNCATE TABLE parts_purchase_periodic_list';

    const insertSql = `
        INSERT INTO parts_purchase_periodic_list
            (Q_Code, Parts_Div, P_N, Price, Avg_LeadTime, Memo)
        VALUES ?
    `;

    db.query(truncateSql, (err) => {
        if (err) {
            console.error('[purchase_periodic_parts/save] TRUNCATE error:', err);
            return res.status(500).json({ error: 'DB_TRUNCATE_ERROR' });
        }

        if (values.length === 0) {
            return res.json({ ok: true, affectedRows: 0 });
        }

        db.query(insertSql, [values], (err2, result) => {
            if (err2) {
                console.error('[purchase_periodic_parts/save] INSERT error:', err2);
                return res.status(500).json({ error: 'DB_INSERT_ERROR' });
            }

            res.json({
                ok: true,
                affectedRows: result.affectedRows || 0
            });
        });
    });
});


// Q_Code 기준으로 주기교체 파츠 1건 조회
app.get('/parts/purchase_periodic_parts/by-code', (req, res) => {
    const code = (req.query.code || '').trim();

    if (!code) {
        return res.status(400).json({ error: 'NO_CODE' });
    }

    const sql = `
        SELECT
            Q_Code,
            Parts_Div,
            P_N,
            Price,
            Avg_LeadTime,
            Memo
        FROM parts_purchase_periodic_list
        WHERE Q_Code = ?
        LIMIT 1
    `;

    db.query(sql, [code], (err, rows) => {
        if (err) {
            console.error('[purchase_periodic_parts/by-code] DB error:', err);
            return res.status(500).json({ error: 'DB_ERROR' });
        }

        res.json(rows && rows[0] ? rows[0] : null);
    });
});

app.get('/parts_purchase/summary', (req, res) => {
    const sql = `
        SELECT Purchase_Status AS st, COUNT(*) AS cnt
        FROM parts_purchase_list
        WHERE Purchase_Status IN ('구매 요청','구매 대기','구매 중')
        GROUP BY Purchase_Status
    `;

    db.query(sql, (err, rows) => {
        if (err) {
        console.error('[parts_purchase/summary] DB error:', err);
        return res.status(500).send('summary error');
        }

        const out = { req: 0, wait: 0, doing: 0 };
        for (const r of (rows || [])) {
        if (r.st === '구매 요청') out.req = Number(r.cnt) || 0;
        if (r.st === '구매 대기') out.wait = Number(r.cnt) || 0;
        if (r.st === '구매 중')  out.doing = Number(r.cnt) || 0;
        }
        res.json(out);
    });
});

app.get('/parts_purchase/check-duplicate', async (req, res) => {
    const { qcode } = req.query;

    const sql = `
        SELECT COUNT(*) AS cnt
        FROM parts_purchase_list
        WHERE Q_Code = ?
        AND Purchase_Status IN ('구매 요청', '구매 대기', '구매 중')
    `;

    db.query(sql, [qcode], (err, result) => {
        if (err) return res.status(500).send(err);

        res.json({
            exists: result[0].cnt > 0
        });
    });
});


// ----------------------------------- 파츠 교체 / 점검 이력 API ------------------------------
// table: plc_log.plm_eqp_repair_web_log
// --------------------------------------------------------------------------------------------

// 공통: 현재 시각 문자열
function getNowDateTime() {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${y}-${m}-${day} ${hh}:${mm}:${ss}`;
}

// Unit + Sub_Unit → Axis 매칭
const axis_sm_matching = {
    'ULD PA TR|X': 'SM2',
    'ATT LD TR|X': 'SM15',
    // 예시 추가
    // 'LD_PA_TR|X': 'SM1',
    // 'LD_PA_TR|Y': 'SM2',
};

// Unit/Sub_Unit 문자열 정규화
function normalizeAxisKeyValue(v) {
    return String(v || '')
        .trim()
        .toUpperCase();
}

function getAxisByUnitSubUnit(unit, subUnit) {
    const key = `${normalizeAxisKeyValue(unit)}|${normalizeAxisKeyValue(subUnit)}`;
    return axis_sm_matching[key] || '';
}

function deleteMotorBoardByRepairLog(rowData) {
    return new Promise((resolve, reject) => {
        const eqName = String(rowData.EQ_Name || '').trim().toUpperCase();
        const unit = String(rowData.Unit || '').trim();
        const subUnit = String(rowData.Sub_Unit || '').trim();
        const partsGroup = String(rowData.Parts_Group || '').trim();
        const eqpState = String(rowData.Eqp_State || '').trim();

        // ✅ 삭제 조건
        const targetParts = ['볼스크류', '모터'];

        if (!eqName || !unit || !subUnit) return resolve({ skipped: true });
        if (!targetParts.includes(partsGroup)) return resolve({ skipped: true });
        if (eqpState !== '완료') return resolve({ skipped: true });

        const axis = getAxisByUnitSubUnit(unit, subUnit);
        if (!axis) return resolve({ skipped: true, reason: 'axis not matched' });

        const boardEqName = eqName.startsWith('PLM#') ? eqName : `PLM#${eqName}`;

        const sql = `
            DELETE FROM plc_log.board_ph2_plm_motor_pick_load
            WHERE EQ_Name = ?
              AND Axis = ?
        `;

        db.query(sql, [boardEqName, axis], (err, result) => {
            if (err) {
     
                return reject(err);
            }

            resolve({
                skipped: false,
                affectedRows: result.affectedRows || 0,
                EQ_Name: boardEqName,
                Axis: axis
            });
        });
    });
}

// 조회
app.get('/parts_replacement_history', (req, res) => {
    const {
        eqGroup = '',
        eqType = '',
        qcode = '',
        pn = '',
        parts = '',
        unit = ''
    } = req.query || {};

    let sql = `
        SELECT
            DATE_FORMAT(WriteTime, '%Y-%m-%d %H:%i:%s') AS WriteTime,
            DATE_FORMAT(Work_Time, '%Y-%m-%d %H:%i:%s') AS Work_Time,
            Site,
            EQ_Group,
            EQ_Name,
            Unit,
            Sub_Unit,
            Parts_Group,
            Parts_Name,
            Parts_Code,
            Serial_Number,
            Parts_Grade,
            Eqp_State,
            Comment,
            Work_Name,
            Write_IP
        FROM plc_log.plm_eqp_repair_web_log
        WHERE 1=1
    `;

    const params = [];

    if (eqGroup) {
        sql += ` AND EQ_Group LIKE ? `;
        params.push(`%${eqGroup}%`);
    }

    if (eqType && eqType !== 'ALL') {
        sql += ` AND EQ_Name LIKE ? `;
        params.push(`%${eqType}%`);
    }

    if (unit) {
        sql += ` AND (
            Unit LIKE ?
            OR Sub_Unit LIKE ?
        ) `;
        params.push(`%${unit}%`, `%${unit}%`);
    }

    if (parts) {
        sql += ` AND (
            Parts_Group LIKE ?
            OR Parts_Name LIKE ?
        ) `;
        params.push(`%${parts}%`, `%${parts}%`);
    }

    if (qcode) {
        sql += ` AND Parts_Code LIKE ? `;
        params.push(`%${qcode}%`);
    }

    if (pn) {
        sql += ` AND Parts_Name LIKE ? `;
        params.push(`%${pn}%`);
    }

    sql += ` ORDER BY Work_Time DESC, WriteTime DESC `;

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('[parts_replacement_history][GET] query error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB query failed'
            });
        }

        res.json({
            ok: true,
            items: results || []
        });
    });
});

// 등록
app.post('/parts_replacement_history', (req, res) => {
    const {
        WriteTime,
        Work_Time,
        Site,
        EQ_Group,
        EQ_Name,
        Unit,
        Sub_Unit,
        Parts_Group,
        Parts_Name,
        Parts_Code,
        Serial_Number,
        Parts_Grade,
        Eqp_State,
        Comment,
        Work_Name,
        Write_IP
    } = req.body || {};

    const writeTimeFinal = String(WriteTime || '').trim() || getNowDateTime();

    const requiredMap = {
        WriteTime: writeTimeFinal,
        Work_Time,
        Site,
        EQ_Group,
        EQ_Name,
        Unit,
        Sub_Unit,
        Parts_Group,
        Parts_Name,
        Parts_Code,
        Parts_Grade,
        Eqp_State,
        Work_Name
    };

    if (String(Parts_Group || '').trim() === '라미 롤러') {
        requiredMap.Serial_Number = Serial_Number;
    }

    const missing = Object.entries(requiredMap)
        .filter(([_, v]) => !String(v ?? '').trim())
        .map(([k]) => k);

    if (missing.length) {
        return res.status(400).json({
            ok: false,
            message: `필수 항목 누락: ${missing.join(', ')}`
        });
    }

    const sql = `
        INSERT INTO plc_log.plm_eqp_repair_web_log
        (
            WriteTime,
            Work_Time,
            Site,
            EQ_Group,
            EQ_Name,
            Unit,
            Sub_Unit,
            Parts_Group,
            Parts_Name,
            Parts_Code,
            Serial_Number,
            Parts_Grade,
            Eqp_State,
            Comment,
            Work_Name,
            Write_IP
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    const params = [
        writeTimeFinal,
        String(Work_Time || '').trim(),
        String(Site || '').trim(),
        String(EQ_Group || '').trim(),
        String(EQ_Name || '').trim().toUpperCase(),
        String(Unit || '').trim(),
        String(Sub_Unit || '').trim(),
        String(Parts_Group || '').trim(),
        String(Parts_Name || '').trim(),
        String(Parts_Code || '').trim().toUpperCase(),
        String(Serial_Number || '').trim(),
        String(Parts_Grade || '').trim(),
        String(Eqp_State || '').trim(),
        String(Comment || '').trim(),
        String(Work_Name || '').trim(),
        String(Write_IP || '').trim()
    ];

    db.query(sql, params, (err, result) => {
        if (err) {
            console.error('[parts_replacement_history][POST] insert error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB insert failed'
            });
        }

        const savedRow = {
            EQ_Name: String(EQ_Name || '').trim().toUpperCase(),
            Unit: String(Unit || '').trim(),
            Sub_Unit: String(Sub_Unit || '').trim(),
            Parts_Group: String(Parts_Group || '').trim(),
            Eqp_State: String(Eqp_State || '').trim()
        };

        deleteMotorBoardByRepairLog(savedRow)
            .then(boardDeleteResult => {
                res.json({
                    ok: true,
                    affectedRows: result.affectedRows || 0,
                    insertId: result.insertId || null,
                    boardDelete: boardDeleteResult
                });
            })
            .catch(boardErr => {
                console.error('[parts_replacement_history][POST] board delete failed:', boardErr);

                // 이력 저장은 성공했으므로 ok는 true 유지
                res.json({
                    ok: true,
                    affectedRows: result.affectedRows || 0,
                    insertId: result.insertId || null,
                    boardDelete: {
                        ok: false,
                        error: String(boardErr.sqlMessage || boardErr.message || boardErr)
                    }
                });
            });
    });
});


// 수정
app.put('/parts_replacement_history/update', (req, res) => {
    const { where, data } = req.body || {};

    if (!where || !data) {
        return res.status(400).json({
            ok: false,
            message: 'where and data are required'
        });
    }

    const {
        WriteTime: whereWriteTime,
        Work_Time: whereWorkTime,
        EQ_Name: whereEQName,
        Unit: whereUnit,
        Parts_Code: wherePartsCode,
        Work_Name: whereWorkName
    } = where;

    if (
        !String(whereWriteTime ?? '').trim() ||
        !String(whereWorkTime ?? '').trim() ||
        !String(whereEQName ?? '').trim() ||
        !String(whereUnit ?? '').trim() ||
        !String(wherePartsCode ?? '').trim() ||
        !String(whereWorkName ?? '').trim()
    ) {
        return res.status(400).json({
            ok: false,
            message: 'invalid where keys'
        });
    }

    const allowedCols = [
        'Work_Time',
        'Site',
        'EQ_Group',
        'EQ_Name',
        'Unit',
        'Sub_Unit',
        'Parts_Group',
        'Parts_Name',
        'Parts_Code',
        'Serial_Number',
        'Parts_Grade',
        'Eqp_State',
        'Comment',
        'Work_Name',
        'Write_IP'
    ];

    const finalPartsGroup = String(data.Parts_Group ?? '').trim();
    const finalSerialNumber = String(data.Serial_Number ?? '').trim();

    if (finalPartsGroup === '라미 롤러' && !finalSerialNumber) {
        return res.status(400).json({
            ok: false,
            message: '라미 롤러는 Serial_Number가 필수입니다.'
        });
    }

    const sets = [];
    const params = [];

    for (const col of allowedCols) {
        if (Object.prototype.hasOwnProperty.call(data, col)) {
            let val = data[col];

            if (col === 'EQ_Name' || col === 'Parts_Code') {
                val = String(val ?? '').trim().toUpperCase();
            } else {
                val = String(val ?? '').trim();
            }

            sets.push(`${col} = ?`);
            params.push(val);
        }
    }

    if (!sets.length) {
        return res.status(400).json({
            ok: false,
            message: 'no updatable fields'
        });
    }

    const sql = `
        UPDATE plc_log.plm_eqp_repair_web_log
        SET ${sets.join(', ')}
        WHERE WriteTime = ?
          AND Work_Time = ?
          AND EQ_Name = ?
          AND Unit = ?
          AND Parts_Code = ?
          AND Work_Name = ?
        LIMIT 1
    `;

    params.push(
        String(whereWriteTime).trim(),
        String(whereWorkTime).trim(),
        String(whereEQName).trim().toUpperCase(),
        String(whereUnit).trim(),
        String(wherePartsCode).trim().toUpperCase(),
        String(whereWorkName).trim()
    );

    db.query(sql, params, (err, result) => {
        if (err) {
            console.error('[parts_replacement_history][PUT] update error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB update failed'
            });
        }

        const savedRow = {
            EQ_Name: String(data.EQ_Name || '').trim().toUpperCase(),
            Unit: String(data.Unit || '').trim(),
            Sub_Unit: String(data.Sub_Unit || '').trim(),
            Parts_Group: String(data.Parts_Group || '').trim(),
            Eqp_State: String(data.Eqp_State || '').trim()
        };

        deleteMotorBoardByRepairLog(savedRow)
            .then(boardDeleteResult => {
                res.json({
                    ok: true,
                    affectedRows: result.affectedRows || 0,
                    boardDelete: boardDeleteResult
                });
            })
            .catch(boardErr => {
                console.error('[parts_replacement_history][PUT] board delete failed:', boardErr);

                res.json({
                    ok: true,
                    affectedRows: result.affectedRows || 0,
                    boardDelete: {
                        ok: false,
                        error: String(boardErr.sqlMessage || boardErr.message || boardErr)
                    }
                });
            });
    });
});


// 삭제
app.post('/parts_replacement_history/delete', async (req, res) => {
    try {
        const { rows } = req.body || {};

        if (!Array.isArray(rows) || rows.length === 0) {
            return res.status(400).json({
                ok: false,
                message: 'rows required'
            });
        }

        const deleteOne = (r) => new Promise((resolve) => {
            const {
                WriteTime,
                Work_Time,
                EQ_Name,
                Unit,
                Parts_Code,
                Work_Name
            } = r || {};

            if (
                !String(WriteTime ?? '').trim() ||
                !String(Work_Time ?? '').trim() ||
                !String(EQ_Name ?? '').trim() ||
                !String(Unit ?? '').trim() ||
                !String(Parts_Code ?? '').trim() ||
                !String(Work_Name ?? '').trim()
            ) {
                return resolve({
                    ok: false,
                    deleted: 0,
                    error: 'invalid key'
                });
            }

            const sql = `
                DELETE FROM plc_log.plm_eqp_repair_web_log
                WHERE WriteTime = ?
                  AND Work_Time = ?
                  AND EQ_Name = ?
                  AND Unit = ?
                  AND Parts_Code = ?
                  AND Work_Name = ?
                LIMIT 1
            `;

            const params = [
                String(WriteTime).trim(),
                String(Work_Time).trim(),
                String(EQ_Name).trim().toUpperCase(),
                String(Unit).trim(),
                String(Parts_Code).trim().toUpperCase(),
                String(Work_Name).trim()
            ];

            db.query(sql, params, (err, result) => {
                if (err) {
                    console.error('[parts_replacement_history][DELETE] delete error:', err);
                    return resolve({
                        ok: false,
                        deleted: 0,
                        error: String(err.sqlMessage || err.message || err)
                    });
                }

                resolve({
                    ok: true,
                    deleted: result.affectedRows || 0
                });
            });
        });

        const results = [];
        for (const r of rows) {
            results.push(await deleteOne(r));
        }

        const deleted = results.reduce((sum, r) => sum + (r.deleted || 0), 0);

        res.json({
            ok: true,
            deleted,
            results
        });
    } catch (e) {
        console.error('[parts_replacement_history/delete] error:', e);
        res.status(500).json({
            ok: false,
            message: 'DB delete failed'
        });
    }
});

// ----------------------------------- parts_use_list lookup API -------

function normalizePartsUseUnit(unit = '') {
    const v = String(unit || '').trim().toUpperCase();

    if (v === 'SET TBL1' || v === 'SET TBL2') return 'SET TBL';
    if (v === 'ATT STG1' || v === 'ATT STG2') return 'ATT STG';

    return String(unit || '').trim();
}

app.get('/parts_use_lookup', (req, res) => {
    const {
        unit = '',
        subUnit = '',
        partsGroup = ''
    } = req.query || {};

    const unitRaw = String(unit || '').trim();
    const unitVal = normalizePartsUseUnit(unitRaw);
    const subUnitVal = String(subUnit || '').trim();
    const partsGroupVal = String(partsGroup || '').trim();

    if (!unitVal || !partsGroupVal) {
        return res.status(400).json({
            ok: false,
            message: 'unit and partsGroup are required'
        });
    }

    // 프론트 값에서 "-" 존재 여부만 판단
    // "_"로 들어와도 동일하게 취급하도록 "_" -> "-" 치환 후 검사
    const normalizedSubUnit = subUnitVal.replace(/_/g, '-');
    const hasDash = normalizedSubUnit.includes('-') ? 1 : 0;

    const sql = `
        SELECT
            Parts_Name,
            Q_Code,
            EQ_Unit,
            SUB_Unit,
            Spec
        FROM plc_log.parts_use_list
        WHERE EQ_Unit = ?
          AND Parts_Name = ?
          AND (
                CASE
                    WHEN INSTR(REPLACE(IFNULL(SUB_Unit, ''), '_', '-'), '-') > 0 THEN 1
                    ELSE 0
                END
              ) = ?
        ORDER BY
            CASE
                WHEN REPLACE(IFNULL(SUB_Unit, ''), '_', '-') = ? THEN 0
                ELSE 1
            END,
            Q_Code ASC
        LIMIT 1
    `;

    const params = [
        unitVal,
        partsGroupVal,
        hasDash,
        normalizedSubUnit
    ];

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('[parts_use_lookup][GET] query error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB query failed'
            });
        }

        const row = Array.isArray(results) && results.length ? results[0] : null;

        return res.json({
            ok: true,
            item: row
                ? {
                    Parts_Name: row.Parts_Name || '',
                    Q_Code: row.Q_Code || '',
                    EQ_Unit: row.EQ_Unit || '',
                    SUB_Unit: row.SUB_Unit || '',
                    Spec: row.Spec || ''
                }
                : null
        });
    });
});

// ----------------------------------- PAD 전용 파츠 교체 / 점검 이력 API ------------------------------
// table: plc_log.plm_eqp_repair_web_log
// Parts_Group = 'PAD' 고정
// --------------------------------------------------------------------------------------------

app.get('/pad_replacement_history', (req, res) => {
    const {
        eqGroup = '',
        eqType = '',
        qcode = '',
        pn = '',
        unit = ''
    } = req.query || {};

    let sql = `
        SELECT
            DATE_FORMAT(WriteTime, '%Y-%m-%d %H:%i:%s') AS WriteTime,
            DATE_FORMAT(Work_Time, '%Y-%m-%d %H:%i:%s') AS Work_Time,
            Site,
            EQ_Group,
            EQ_Name,
            Unit,
            Sub_Unit,
            Parts_Group,
            Parts_Name,
            Parts_Code,
            Serial_Number,
            Parts_Grade,
            Eqp_State,
            Comment,
            Work_Name,
            Write_IP
        FROM plc_log.plm_eqp_repair_web_log
        WHERE 1=1
          AND UPPER(TRIM(Parts_Group)) = 'PAD'
    `;

    const params = [];

    if (eqGroup && eqGroup !== 'ALL') {
        sql += ` AND EQ_Group LIKE ? `;
        params.push(`%${eqGroup}%`);
    }

    if (eqType && eqType !== 'ALL') {
        sql += ` AND EQ_Name LIKE ? `;
        params.push(`%${eqType}%`);
    }

    if (unit && unit !== 'ALL') {
        sql += ` AND (
            Unit LIKE ?
            OR Sub_Unit LIKE ?
        ) `;
        params.push(`%${unit}%`, `%${unit}%`);
    }

    if (qcode) {
        sql += ` AND Parts_Code LIKE ? `;
        params.push(`%${qcode}%`);
    }

    if (pn) {
        sql += ` AND Parts_Name LIKE ? `;
        params.push(`%${pn}%`);
    }

    sql += ` ORDER BY Work_Time DESC, WriteTime DESC `;

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('[pad_replacement_history][GET] query error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB query failed'
            });
        }

        res.json({
            ok: true,
            items: results || []
        });
    });
});

app.get('/parts_use_lookup_popup', (req, res) => {
    const {
        unit = '',
        partsGroup = 'PAD'
    } = req.query || {};

    const unitVal = normalizePartsUseUnit(unit);
    const partsGroupVal = String(partsGroup || '').trim();

    if (!unitVal || !partsGroupVal) {
        return res.status(400).json({
            ok: false,
            message: 'unit and partsGroup are required'
        });
    }

    const sql = `
        SELECT
            Parts_Name,
            Q_Code,
            EQ_Unit,
            SUB_Unit,
            Spec
        FROM plc_log.parts_use_list
        WHERE EQ_Unit = ?
          AND Parts_Name = ?
        ORDER BY Q_Code ASC
    `;

    const params = [unitVal, partsGroupVal];

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('[parts_use_lookup_popup][GET] query error:', err);
            return res.status(500).json({
                ok: false,
                message: 'DB query failed'
            });
        }

        const items = Array.isArray(results) ? results.map(row => ({
            Parts_Name: row.Parts_Name || '',
            Q_Code: row.Q_Code || '',
            EQ_Unit: row.EQ_Unit || '',
            SUB_Unit: row.SUB_Unit || '',
            Spec: row.Spec || ''
        })) : [];

        return res.json({
            ok: true,
            items
        });

    });
});



// ----------------------------------- 품질 이력 API ------------------------------
// table: plc_log.plm_quality_history_web_log
// ------------------------------------------------------------------------------

// 등록 
app.get('/quality_history', (req, res) => {
    const sql = `
        SELECT
            DATE_FORMAT(WriteTime, '%Y-%m-%d %H:%i:%s') AS WriteTime,
            DATE_FORMAT(WorkTime, '%Y-%m-%d %H:%i:%s') AS WorkTime,
            EQ_Name,
            Model,
            Lot_id,
            Type,
            Category,
            Detail,
            Comment,
            Write_Name,
            Feedback,
            Work_Name,
            Process_status
        FROM plc_log.plm_quality_history_web_log
        ORDER BY WorkTime DESC, WriteTime DESC
    `;

    db.query(sql, [], (err, results) => {
        if (err) {
            console.error('[quality_history][GET] error:', err);
            return res.status(500).json({ ok: false, message: 'DB query failed' });
        }

        res.json({ ok: true, items: results || [] });
    });
});

app.post('/quality_history', (req, res) => {
    const {
        WriteTime,
        WorkTime,
        EQ_Name,
        Model,
        Lot_id,
        Type,
        Category,
        Detail,
        Comment,
        Write_Name,
        Feedback,
        Work_Name,
        Process_status
    } = req.body || {};

    const writeTimeFinal = String(WriteTime || '').trim() || getNowDateTime();

    const required = {
        WorkTime,
        EQ_Name,
        Model,
        Lot_id,
        Type,
        Category,
        Detail,
        Write_Name
    };

    const missing = Object.entries(required)
        .filter(([_, v]) => !String(v ?? '').trim())
        .map(([k]) => k);

    if (missing.length) {
        return res.status(400).json({
            ok: false,
            message: `필수 항목 누락: ${missing.join(', ')}`
        });
    }

    const sql = `
        INSERT INTO plc_log.plm_quality_history_web_log
        (
            WriteTime,
            WorkTime,
            EQ_Name,
            Model,
            Lot_id,
            Type,
            Category,
            Detail,
            Comment,
            Write_Name,
            Feedback,
            Work_Name,
            Process_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    const params = [
        writeTimeFinal,
        String(WorkTime || '').trim(),
        String(EQ_Name || '').trim().toUpperCase(),
        String(Model || '').trim(),
        String(Lot_id || '').trim(),
        String(Type || '').trim(),
        String(Category || '').trim(),
        String(Detail || '').trim(),
        String(Comment || '').trim(),
        String(Write_Name || '').trim(),
        String(Feedback || '').trim(),
        String(Work_Name || '').trim(),
        String(Process_status || '').trim()
    ];

    db.query(sql, params, (err, result) => {
        if (err) {
            console.error('[quality_history][POST] error:', err);
            return res.status(500).json({ ok: false, message: 'DB insert failed' });
        }

        res.json({
            ok: true,
            affectedRows: result.affectedRows || 0,
            insertId: result.insertId || null
        });
    });
});

// 품질 이력 수정
app.put('/quality_history/update', (req, res) => {
    const { where, data } = req.body || {};

    if (!where || !data) {
        return res.status(400).json({ ok: false, message: 'where/data 누락' });
    }

    const sql = `
        UPDATE plc_log.plm_quality_history_web_log
        SET
            WriteTime = ?,
            WorkTime = ?,
            EQ_Name = ?,
            Model = ?,
            Lot_id = ?,
            Type = ?,
            Category = ?,
            Detail = ?,
            Comment = ?,
            Write_Name = ?,
            Feedback = ?,
            Work_Name = ?,
            Process_status = ?
        WHERE
            WriteTime = ?
            AND WorkTime = ?
            AND EQ_Name = ?
            AND Lot_id = ?
            AND Type = ?
            AND Write_Name = ?
    `;

    const params = [
        data.WriteTime,
        data.WorkTime,
        String(data.EQ_Name || '').trim().toUpperCase(),
        data.Model,
        data.Lot_id,
        data.Type,
        data.Category,
        data.Detail,
        data.Comment,
        data.Write_Name,
        data.Feedback,
        data.Work_Name,
        data.Process_status,

        where.WriteTime,
        where.WorkTime,
        where.EQ_Name,
        where.Lot_id,
        where.Type,
        where.Write_Name
    ];

    db.query(sql, params, (err, result) => {
        if (err) {
            console.error('[quality_history][UPDATE] error:', err);
            return res.status(500).json({ ok: false, message: 'DB update failed' });
        }

        res.json({ ok: true, affectedRows: result.affectedRows || 0 });
    });
});

// 품질 이력 삭제
app.post('/quality_history/delete', (req, res) => {
    const { rows } = req.body || {};

    if (!Array.isArray(rows) || rows.length === 0) {
        return res.status(400).json({ ok: false, message: '삭제할 rows 없음' });
    }

    const conditions = rows.map(() => `
        (
            WriteTime = ?
            AND WorkTime = ?
            AND EQ_Name = ?
            AND Lot_id = ?
            AND Type = ?
            AND Write_Name = ?
        )
    `).join(' OR ');

    const params = [];

    rows.forEach(r => {
        params.push(
            r.WriteTime,
            r.WorkTime,
            r.EQ_Name,
            r.Lot_id,
            r.Type,
            r.Write_Name
        );
    });

    const sql = `
        DELETE FROM plc_log.plm_quality_history_web_log
        WHERE ${conditions}
    `;

    db.query(sql, params, (err, result) => {
        if (err) {
            console.error('[quality_history][DELETE] error:', err);
            return res.status(500).json({ ok: false, message: 'DB delete failed' });
        }

        res.json({ ok: true, affectedRows: result.affectedRows || 0 });
    });
});
