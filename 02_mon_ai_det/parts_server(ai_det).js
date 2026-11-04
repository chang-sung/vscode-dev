
// ----------------------------------------------------------------------------
// AI DET 이미지 시퀀스 조회 API
// GET /api/ai-det/roller-change-images
// 기본 : 최근 3일
// 옵션 : ?start=YYYY-MM-DD&end=YYYY-MM-DD&eq=PLM#45G,PLM#47G

function normalizeAiDetEqName(eqName) {
    // PLM#45G -> 45
    const m = String(eqName || '').match(/PLM#(\d+)G/i);
    if (m) return m[1];

    return String(eqName || '').trim();
}

function extractRecipe(recipe) {
    const parts = String(recipe || '').split('_');

    // AA_AAA_AA2_AA_A 기준 → AA2
    return parts[2] || '';
}

app.get('/api/ai-det/roller-change-images', (req, res) => {
    try {
        const { start, end, eq } = req.query || {};

        let imageWhere = '1=1';
        let changeWhere = '1=1';

        const imageParams = [];
        const changeParams = [];

        // ==========================
        // 기본 최근 3일
        // ==========================
        if (!start && !end) {
            imageWhere += ` AND Write_time >= DATE_SUB(NOW(), INTERVAL 3 DAY)`;
            changeWhere += ` AND Write_time >= DATE_SUB(NOW(), INTERVAL 3 DAY)`;
        }

        // ==========================
        // 사용자 날짜 선택
        // ==========================
        

        if (start) {
            imageWhere += ' AND Write_time >= ?';
            changeWhere += ' AND Write_time >= ?';
            imageParams.push(`${start} 00:00:00`);
            changeParams.push(`${start} 00:00:00`);
        }

        if (end) {
            imageWhere += ' AND Write_time <= ?';
            changeWhere += ' AND Write_time <= ?';
            imageParams.push(`${end} 23:59:59`);
            changeParams.push(`${end} 23:59:59`);
        }

        // ==========================
        // EQ 필터
        // ==========================
        if (eq) {
            const imageEqList = String(eq)
                .split(',')
                .map(v => v.trim())
                .filter(Boolean);

            const changeEqList = imageEqList
                .map(v => normalizeAiDetEqName(v))
                .filter(Boolean);

            if (imageEqList.length > 0) {
                imageWhere += `
                    AND EQ_name IN (${imageEqList.map(() => '?').join(',')})
                `;
                imageParams.push(...imageEqList);
            }

            if (changeEqList.length > 0) {
                changeWhere += `
                    AND EQ_name IN (${changeEqList.map(() => '?').join(',')})
                `;
                changeParams.push(...changeEqList);
            }
        }

        const sql = `
            SELECT
                EQ_name,
                Write_time,
                filename,
                result,
                TO_BASE64(image) AS image_base64
            FROM plc_log.ph2_log_ai_auto_roller_change_image
            WHERE ${imageWhere}
            ORDER BY EQ_name ASC, Write_time ASC
            LIMIT 20000
        `;

        db.query(sql, imageParams, (err, rows) => {
            if (err) {
                console.error('[AI-DET-IMAGE] Query error:', err);
                return res.status(500).json({
                    error: 'QUERY_FAILED',
                    message: err.message
                });
            }

            const sequences = [];
            let currentSeq = null;

            rows.forEach(row => {
                const eqName = row.EQ_name;
                const currentTime = new Date(row.Write_time).getTime();

                if (
                    !currentSeq ||
                    currentSeq.EQ_name !== eqName ||
                    (currentTime - currentSeq.lastTimeMs) > 30000
                ) {
                    currentSeq = {
                        EQ_name: eqName,
                        start_time: row.Write_time,
                        lastTimeMs: currentTime,
                        images: []
                    };

                    sequences.push(currentSeq);
                }

                currentSeq.images.push({
                    Write_time: row.Write_time,
                    filename: row.filename,
                    result: row.result,
                    image_base64: row.image_base64
                });

                currentSeq.lastTimeMs = currentTime;
            });

            const change2Sql = `
                SELECT
                    EQ_name,
                    Write_time,
                    Err_unit,
                    Roller_count,
                    recipe
                FROM plc_log.ph2_log_ai_auto_roller_change2
                WHERE ${changeWhere}
                ORDER BY EQ_name ASC, Write_time ASC
                LIMIT 20000
            `;

            db.query(change2Sql, changeParams, (err2, changeRows) => {
                if (err2) {
                    console.error('[AI-DET-CHANGE2] Query error:', err2);
                    return res.status(500).json({
                        error: 'QUERY_FAILED',
                        message: err2.message
                    });
                }

                const MATCH_TIME_MS = 5000;

                sequences.forEach(seq => {
                    const images = seq.images || [];
                    if (images.length === 0) return;

                    const lastImg = images[images.length - 1];
                    const lastImageTimeMs = new Date(lastImg.Write_time).getTime();
                    const imageEqNo = normalizeAiDetEqName(seq.EQ_name);

                    let best = null;
                    let bestDiff = Infinity;

                    changeRows.forEach(row => {
                        const changeEqNo = normalizeAiDetEqName(row.EQ_name);
                        if (changeEqNo !== imageEqNo) return;

                        const changeTimeMs = new Date(row.Write_time).getTime();
                        const diff = Math.abs(changeTimeMs - lastImageTimeMs);

                        if (diff <= MATCH_TIME_MS && diff < bestDiff) {
                            best = row;
                            bestDiff = diff;
                        }
                    });

                    if (best) {
                        seq.change_info = {
                            Write_time: best.Write_time,
                            Err_unit: best.Err_unit,
                            Roller_count: best.Roller_count,
                            recipe: extractRecipe(best.recipe)
                        };
                    }

                    delete seq.lastTimeMs;
                });

                res.json({
                    count: sequences.length,
                    sequences
                });
            });
        });

    } catch (e) {

        console.error('[AI-DET-IMAGE] Error:', e);

        res.status(500).json({
            error: 'INTERNAL_ERROR',
            message: e?.message || ''
        });
    }
});