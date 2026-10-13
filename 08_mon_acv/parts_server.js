
// ---------------------------------------------------------------------------- ACV 모니터링 시스템
app.get('/get-acv-data', (req, res) => {
    const eqName    = req.query.eq;
    const startDate = req.query.start;
    const endDate   = req.query.end;
    const recipeStr = (req.query.recipe || '').trim(); // ⬅️ 추가

    if (!eqName || !startDate || !endDate) {
        return res.status(400).json({ error: '필수 파라미터 누락' });
    }

    // ⬇️ "8bar _ 50º _ 10min" 같은 문자열을 {press:8, temp:50, time:10}로 파싱
    function parseRecipe(str) {
        // 공백/구분자 유연 대응
        // 예) "8bar _ 50º _ 10min", "8bar_50º_10min", "8 bar 50 º 10 min"
        const cleaned = str.replace(/\s+/g, ' ').replace(/_/g, ' ').trim();
        const m = cleaned.match(/(\d+(?:\.\d+)?)\s*bar.*?(\d+(?:\.\d+)?)\s*º.*?(\d+)\s*min/i);
        if (!m) return null;
        const press = Number(m[1]);  // bar
        const temp  = Number(m[2]);  // ºC
        const time  = Number(m[3]);  // min
        if (!isFinite(press) || !isFinite(temp) || !isFinite(time)) return null;
        return { press, temp, time };
    }

    const parsed = recipeStr && recipeStr.toLowerCase() !== 'all' ? parseRecipe(recipeStr) : null;

    // 기본 SQL + 조건
    let sql = `
        SELECT WriteTime, Cur_Temp, Cur_Press, Process_Step,
            Pack_OFF_Timeover, Pack_On_Timeover, Boost_ON, OUT_Door_Open,
            CB_CV1_Data, CB_CV2_Data, CB_CV3_Data, CB_CV4_Data,
            Recipe_Temp, Recipe_Press, Recipe_Time
        FROM acv_run_cycle_log
        WHERE EQ_Name = ?
        AND WriteTime BETWEEN ? AND ?
        AND Process_Step BETWEEN 12 AND 25
    `;
    const params = [eqName, startDate, endDate];

    // ⬇️ 레시피가 선택된 경우(=parsed 존재) 숫자 필터 추가
    if (parsed) {
        // DB는 10배 스케일(Temp/Press) → 정수로 맞춰 비교
        sql += ` AND Recipe_Press = ? AND Recipe_Temp = ? AND Recipe_Time = ?`;
        params.push(Math.round(parsed.press * 10), Math.round(parsed.temp * 10), parsed.time);
    }

    sql += ` ORDER BY WriteTime ASC`;

    db_plms.query(sql, params, (err, results) => {
        if (err) {
        console.error('❌ DB 오류:', err);
        return res.status(500).json({ error: 'DB 조회 실패' });
        }

        if (!results || results.length === 0) {
        return res.json({
            timestamps: [],
            cur_temp: [],
            cur_press: [],
            process_step: [],
            pack_off_timeover: [],
            pack_on_timeover: [],
            micro_press: [],
            cb_cv1: [],
            cb_cv2: [],
            cb_cv3: [],
            cb_cv4: [],
            cv_labels: [],
            recipe: null,
            recipe_list: []
        });
        }

        const timestamps        = results.map(r => new Date(r.WriteTime).toISOString());
        const cur_temp          = results.map(r => r.Cur_Temp / 10);
        const cur_press         = results.map(r => r.Cur_Press);
        const process_step      = results.map(r => r.Process_Step);
        const pack_on_timeover  = results.map(r => r.Pack_On_Timeover);
        const pack_off_timeover = results.map(r => r.Pack_OFF_Timeover);
        const micro_press       = results.map(r => r.Boost_ON);
        const cb_cv1            = results.map(r => r.CB_CV1_Data);
        const cb_cv2            = results.map(r => r.CB_CV2_Data);
        const cb_cv3            = results.map(r => r.CB_CV3_Data);
        const cb_cv4            = results.map(r => r.CB_CV4_Data);

        const cv_labels   = [];
        const recipe_list = [];
        let recipeInfo    = null;
        let inCycle       = false;

        for (let i = 0; i < results.length; i++) {
        const row  = results[i];
        const step = row.Process_Step;

        if (step === 12 && !inCycle) {
            inCycle = true;

            cv_labels.push({
            cv1: row.CB_CV1_Data,
            cv2: row.CB_CV2_Data,
            cv3: row.CB_CV3_Data,
            cv4: row.CB_CV4_Data
            });

            recipeInfo = {
            temp:  row.Recipe_Temp  != null ? row.Recipe_Temp  / 10 : null,
            press: row.Recipe_Press != null ? row.Recipe_Press / 10 : null,
            time:  row.Recipe_Time  ?? null
            };
            recipe_list.push(recipeInfo);
        }

        if (step === 25 && inCycle) {
            inCycle = false;
        }
        }

        res.json({
        timestamps,
        cur_temp,
        cur_press,
        process_step,
        pack_off_timeover,
        pack_on_timeover,
        micro_press,
        cb_cv1,
        cb_cv2,
        cb_cv3,
        cb_cv4,
        cv_labels,
        recipe: recipeInfo,
        recipe_list
        });
    });
});

app.get('/get-first-cv-data', async (req, res) => {
    let { eq, start, end } = req.query;

    // 기본: 최근 30일
    if (!start || !end) {
        const now = new Date();
        const thirtyDaysAgo = new Date(now);
        thirtyDaysAgo.setDate(now.getDate() - 30);

        start = thirtyDaysAgo.toISOString().slice(0, 10) + ' 00:00:00';
        end = now.toISOString().slice(0, 10) + ' 23:59:59';
    }

    const sql = `
        SELECT EQ_Name AS eq, WriteTime AS writeTime, Week,
               Avg_Effi AS effi,
               CB_CV1_Data AS cv1, CB_CV2_Data AS cv2,
               CB_CV3_Data AS cv3, CB_CV4_Data AS cv4
        FROM acv_effi_summary
        WHERE WriteTime BETWEEN ? AND ?
        ${eq && eq !== 'all' ? 'AND EQ_Name = ?' : ''}
        ORDER BY WriteTime ASC
    `;

    const params = eq && eq !== 'all' ? [start, end, eq] : [start, end];

    db_plms.query(sql, params, (err, results) => {
        if (err) {
            console.error("❌ 효율 테이블 조회 오류:", err);
            return res.status(500).json({ error: 'DB 오류' });
        }

        res.json(results || []);
    });
});


// ✅ 전체 EQ 미세가압 카운트 목록 반환
app.get('/get-acv-summary', (req, res) => {
    const sql = `
        SELECT 
        EQ_Name,
        TRIM(recipe) AS recipe,
        MAX(micro_press_count)  AS max_micro_press_cnt,
        MAX(micro_press_time)   AS max_micro_press_time,
        MIN(leak_interval_time) AS min_leak_interval,
        MAX(packing_on_time)    AS max_pack_on,
        MAX(packing_off_time)   AS max_pack_off,
        MAX(press_time)         AS max_press_time,
        MAX(exhaust_time)       AS max_exhaust_time
        FROM acv_run_summary
        WHERE track_in >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY EQ_Name, TRIM(recipe)
        ORDER BY EQ_Name ASC, max_micro_press_cnt DESC
    `;

    db_plms.query(sql, (err, results) => {
        if (err) {
        console.error('❌ ACV summary 조회 오류:', err);
        return res.status(500).json({ error: 'DB 오류' });
        }
        res.json(results || []);
    });
});

// ✅ Recipe 목록 조회 API
app.get('/get-acv-recipes', (req, res) => {
    const sql = `
        SELECT DISTINCT TRIM(recipe) AS recipe
        FROM acv_run_summary
        WHERE recipe IS NOT NULL 
            AND recipe <> ''
            AND track_in >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        ORDER BY recipe ASC
    `;

    db_plms.query(sql, (err, results) => {
        if (err) {
            console.error('❌ Recipe 목록 조회 오류:', err);
            return res.status(500).json({ error: 'DB 오류' });
        }

        // 결과를 배열로 변환
        const recipeList = results.map(row => row.recipe);
        res.json(recipeList);
    });
});

// ✅ 고장 차트용: acv_run_summary 타임시리즈 (레시피 필터 추가)
app.get('/get-acv-trend-chart', (req, res) => {
    let { eq = 'all', start, end, recipe = 'all' } = req.query;

    // 기본 기간: 최근 300일
    if (!start || !end) {
        const now = new Date();
        const d30 = new Date(now);
        d30.setDate(now.getDate() - 30);
        start = `${d30.toISOString().slice(0,10)} 00:00:00`;
        end   = `${now.toISOString().slice(0,10)} 23:59:59`;
    }

    const sel = `
        SELECT
            EQ_Name,
            track_in, recipe,
            lot_count                           AS lot_count,
            micro_press_count                   AS micro_press_count,
            micro_press_time                    AS micro_press_time,
            leak_interval_time                  AS leak_interval_time,
            press_time                          AS press_time,
            exhaust_time                        AS exhaust_time,
            packing_on_time                     AS packing_on_time,
            packing_off_time                    AS packing_off_time,
            CV1 AS cv1, CV2 AS cv2, CV3 AS cv3, CV4 AS cv4
        FROM acv_run_summary
    `;

    let where = `WHERE track_in BETWEEN ? AND ?`;
    const params = [start, end];

    if (eq && eq !== 'all') {
        where += ` AND EQ_Name = ?`;
        params.push(eq);
    }

    if (recipe && recipe !== 'all') {
        where += ` AND recipe = ?`;
        params.push(recipe);
    }

    const sql = `${sel} ${where} ORDER BY track_in ASC`;

    db_plms.query(sql, params, (err, rows) => {
        if (err) {
            console.error('❌ breakdown 조회 오류:', err);
            return res.status(500).json({ error: 'DB 오류' });
        }
        res.json(rows || []);
    });
});


// ✅ 설비-레시피 조합의 최근 '일자' 반환 API
app.get('/get-latest-eq-recipe-day', (req, res) => {
    let { eq, recipe } = req.query;
    if (!eq || !recipe || recipe === 'all') return res.json({ day: null });

    eq = String(eq).trim();
    recipe = String(recipe).trim();

    const sql = `
        SELECT DATE_FORMAT(MAX(track_in), '%Y-%m-%d') AS day
        FROM acv_run_summary
        WHERE EQ_Name = ? AND TRIM(recipe) = ? AND recipe IS NOT NULL AND recipe <> ''
    `;

    db_plms.query(sql, [eq, recipe], (err, rows) => {
        if (err) {
            console.error('❌ latest-day(summary) 조회 오류:', err);
            return res.status(500).json({ day: null });
        }
        const day = rows?.[0]?.day || null; // 이미 'YYYY-MM-DD' 문자열
        return res.json({ day });
    });
});

// ✅ EQ별 최신 레시피 맵
app.get('/get-latest-recipe-map', (req, res) => {
    const sql = `
        SELECT s.EQ_Name AS eq, TRIM(s.recipe) AS recipe
        FROM acv_run_summary s
        JOIN (
        SELECT EQ_Name, MAX(track_in) AS latest
        FROM acv_run_summary
        WHERE track_in >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY EQ_Name
        ) t ON t.EQ_Name = s.EQ_Name AND s.track_in = t.latest
        ORDER BY s.EQ_Name
    `;
    db_plms.query(sql, (err, rows) => {
        if (err) {
            console.error('❌ 최신 레시피 맵 조회 오류:', err);
            return res.status(500).json({ error: 'DB 오류' });
        }

        res.json(rows || []);
    });
});

// 파워몰러 전류 원본 조회: Recipe는 전류 로그에 없어 적용하지 않음
app.get('/get-acv-current-data', (req, res) => {
    const eq = typeof req.query.eq === 'string' ? req.query.eq.trim() : '';
    const start = req.query.start;
    const end = req.query.end;
    function validDay(value) {
        if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
        const day = new Date(value + 'T00:00:00Z');
        return Number.isFinite(day.getTime()) && day.toISOString().slice(0, 10) === value;
    }
    if (!eq || eq === 'all' || !validDay(start) || !validDay(end) || start > end) {
        return res.status(400).json({ error: '설비와 올바른 시작/종료 날짜를 선택하세요.' });
    }
    const nextDay = new Date(end + 'T00:00:00Z');
    nextDay.setUTCDate(nextDay.getUTCDate() + 1);
    const endExclusive = nextDay.toISOString().slice(0, 10) + ' 00:00:00';
    const sql = `
        SELECT WriteTime, EQ_Name,
            LD1_I, LD2_I, LD3_I, LD4_I,
            CH1_I, CH2_I, CH3_I, CH4_I,
            ULD1_I, ULD2_I, ULD3_I, ULD4_I
        FROM plm_s.acv_powermoller_volt_current_log
        WHERE EQ_Name = ?
          AND WriteTime >= ?
          AND WriteTime < ?
        ORDER BY WriteTime ASC
    `;
    db_plms.query(sql, [eq, start + ' 00:00:00', endExclusive], (err, rows) => {
        if (err) {
            console.error('파워몰러 전류 DB 조회 오류:', err);
            return res.status(500).json({ error: '전류 데이터 DB 조회 실패' });
        }
        res.json(rows || []);
    });
});
