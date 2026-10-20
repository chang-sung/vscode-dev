
// ---------------------------------------------------------------------------- motor_mon_setting.html 엔드포인트
app.get('/api/monitor_motor_setting', (req, res) => {
    const sql = 'SELECT * FROM ph2_motor_state ORDER BY EQ_Name, Motor_Num, POS_Num';

    db.query(sql, (err, result) => {
        if (err) {
            console.error('Database query error:', err);
            return res.status(500).json({ error: 'Database query error' });
        }

        const groupedData = {};

        result.forEach(row => {
            const EqName = row['EQ_Name'];
            const SM = row['Motor_Num'];
            const POS = row['POS_Num'];

            if (!groupedData[EqName]) {
                groupedData[EqName] = {};
            }

            if (!groupedData[EqName][SM]) {
                groupedData[EqName][SM] = [];
            }

            groupedData[EqName][SM][POS] = {
                MODEL: row['MODEL'],
                FIX: row['FIX'],
                OFFSET: row['OFFSET'],
                TARGET: row['TARGET'],
                SPEED: row['SPEED'],
                WriteTime: row['WriteTime']  // ← 여기 추가
            };
        });

        res.json(groupedData);
    });
});

app.get("/api/spec", (req, res) => {
    const sql = "SELECT * FROM ph2_motor_state_spec";

    db.query(sql, (err, results) => {
        if (err) return res.status(500).json({ error: "DB 오류" });

        const specMap = {};

        results.forEach(row => {
        const key  = `${row.Motor_Num}_${row.POS_Num}`;
        const type = row.Type || "phone";   // Phone / Foldable

        if (!specMap[key]) specMap[key] = {};
        if (!specMap[key][type]) specMap[key][type] = {};

        if (row.POS_Num === 0) {
            specMap[key][type][`MODEL_${row.Division}`] = row.MODEL;
            specMap[key][type][`FIX_${row.Division}`]   = row.FIX;
        } else {
            specMap[key][type][`SPEED_${row.Division}`] = row.SPEED;
        }
        });

        res.json(specMap);
    });
});


app.post("/api/spec/update", async (req, res) => {
    const { Motor_Num, POS_Num, field, value, division, type } = req.body;

    // ✅ 클라이언트 IP 가져오기
    const clientIp =
        req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
        req.socket?.remoteAddress ||
        req.ip ||
        "unknown";

    const sqlSelect = `
        SELECT ?? AS oldValue
        FROM ph2_motor_state_spec
        WHERE Motor_Num = ? AND POS_Num = ? AND Division = ? AND Type = ?
        LIMIT 1
    `;

    const sqlUpdate = `
        UPDATE ph2_motor_state_spec
        SET ?? = ?
        WHERE Motor_Num = ? AND POS_Num = ? AND Division = ? AND Type = ?
    `;

    const sqlLogInsert = `
        INSERT INTO ph2_motor_state_log
        (EQ_Name, Motor_Num, POS_Num, MODEL, FIX, OFFSET, TARGET, SPEED, WriteTime, Comment)
        VALUES
        (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NOW(),
        CONCAT('SPEC 변경: ', ?, '(', ?, ' → ', ?, ') [', ?, '] - IP: ', ?))
    `;

    try {
        const [rows] = await db.promise().query(sqlSelect, [field, Motor_Num, POS_Num, division, type]);
        const oldValue = rows[0]?.oldValue ?? 'N/A';

        await db.promise().query(sqlUpdate, [field, value, Motor_Num, POS_Num, division, type]);

        const fieldLabel = `${field}_${division}`;

        await db.promise().query(sqlLogInsert, [
            "SPEC",
            Motor_Num,
            POS_Num,
            fieldLabel,
            oldValue,
            value,
            type,
            clientIp
        ]);

        res.send("성공");
    } catch (err) {
        console.error("DB 오류:", err);
        res.status(500).json({ error: "Update 실패" });
    }
});

app.get('/api/spec/comments', async (req, res) => {
    try {
        const [rows] = await db.promise().query(`
            SELECT EQ_Name, Motor_Num, POS_Num, Comment
            FROM ph2_motor_state
            WHERE Comment IS NOT NULL AND Comment != ''
        `);
        res.json(rows);
    } catch (error) {
        console.error('Error fetching comments:', error);
        res.status(500).json({ error: 'Failed to load comments' });
    }
});

app.post("/api/spec/comment", (req, res) => {
    const { eqName, motor, pos, name, date, comment, recovery, userName } = req.body;
    const fullComment = ` ${date}\n${comment}\n백업예정: ${recovery}\n${name}`;

    const updateSql = `
        UPDATE ph2_motor_state
        SET Comment = ?
        WHERE EQ_Name = ? AND Motor_Num = ? AND POS_Num = ?
    `;

    const insertLogSql = `
        INSERT INTO ph2_motor_state_log
        (EQ_Name, Motor_Num, POS_Num, MODEL, FIX, OFFSET, TARGET, SPEED, WriteTime, Comment)
        SELECT EQ_Name, Motor_Num, POS_Num, MODEL, FIX, OFFSET, TARGET, SPEED, NOW(),
               CONCAT('Inform 등록: ', ?, ' - by ', ?)
        FROM ph2_motor_state
        WHERE EQ_Name = ? AND Motor_Num = ? AND POS_Num = ?
    `;

    db.query(updateSql, [fullComment, eqName, motor, pos], (err) => {
        if (err) {
            console.error("DB 오류 (comment):", err);
            return res.status(500).json({ error: "코멘트 저장 실패" });
        }

        // ✅ 파라미터 개수: comment, userName, eqName, motor, pos
        db.query(insertLogSql, [comment, userName || "unknown", eqName, motor, pos], (logErr) => {
            if (logErr) {
                console.error("로그 저장 실패:", logErr);
            }
            res.send("성공");
        });
    });
});

app.delete("/api/spec/comment", (req, res) => {
    const { eqName, motor, pos, userName } = req.body;

    const updateSql = `
        UPDATE ph2_motor_state
        SET Comment = NULL
        WHERE EQ_Name = ? AND Motor_Num = ? AND POS_Num = ?
    `;

    const insertLogSql = `
        INSERT INTO ph2_motor_state_log
        (EQ_Name, Motor_Num, POS_Num, MODEL, FIX, OFFSET, TARGET, SPEED, WriteTime, Comment)
        SELECT EQ_Name, Motor_Num, POS_Num, MODEL, FIX, OFFSET, TARGET, SPEED, NOW(),
               CONCAT('Inform 삭제 - by ', ?)
        FROM ph2_motor_state
        WHERE EQ_Name = ? AND Motor_Num = ? AND POS_Num = ?
    `;

    db.query(updateSql, [eqName, motor, pos], (err) => {
        if (err) {
            console.error("DB 오류 (comment 삭제):", err);
            return res.status(500).json({ error: "코멘트 삭제 실패" });
        }

        db.query(insertLogSql, [userName || "unknown", eqName, motor, pos], (logErr) => {
            if (logErr) {
                console.error("로그 저장 실패:", logErr);
            }
            res.send("삭제 성공");
        });
    });
});

app.get("/api/unit/type", async (req, res) => {

    const sql = `
        SELECT EQ_Name, TraySIZE
        FROM ph2_unit_monitoring
    `;

    try {

        const [rows] = await db.promise().query(sql);

        const result = {};

        rows.forEach(row => {

            let type = "phone";

            if(row.TraySIZE === "2x1") {

                type = "foldable";

            }
            else {

                const size = row.TraySIZE?.split("x");

                if(size && size.length === 2) {

                    const x = parseInt(size[0]);
                    const y = parseInt(size[1]);

                    if(x >= 4 && y >= 4) {
                        type = "watch";
                    }
                }
            }


            result[row.EQ_Name] = type;

        });


        res.json(result);

    }
    catch(err){

        console.error(err);

        res.status(500).json({
            error:"DB Error"
        });

    }

});