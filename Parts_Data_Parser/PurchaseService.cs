using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace log_parser
{
    internal sealed class PurchaseService
    {
        private readonly string connectionString;

        public PurchaseService(string connectionString)
        {
            this.connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }


        private void NormalizePurchaseStatuses()
        {
            using (var conn = new MySqlConnection(connectionString))
            {
                conn.Open();
                using (var tx = conn.BeginTransaction())
                {
                    try
                    {
                        // ------------------------ (1) EMS 대기 → 구매 중 ------------------------
                        // 이 조건은 기존 UPDATE의 WHERE 절과 동일해야 함
                        string emsWhere = @"
                            Purchase_Status = '구매 대기'
                            AND Req_Num IS NOT NULL
                            AND TRIM(Req_Num) <> ''";


                        // 상태 바뀔 대상들 상세 로그
                        LogPurchaseStatusTargets(
                            conn, tx,
                            emsWhere,
                            "구매 중"

                        );

                        int emsToMid = 0;
                        using (var cmd = new MySqlCommand($@"
                            UPDATE parts_purchase_list
                               SET Purchase_Status = '구매 중'
                             WHERE {emsWhere}
                        ", conn, tx))
                        {
                            emsToMid = cmd.ExecuteNonQuery();
                        }
                        Logger.Info($"(EMS 대기→중) 변경: {emsToMid}건");

                        // ------------------------ (2) MRO 대기 → 구매 중 ------------------------
                        string mroWhere = @"
                            Purchase_Status = '구매 대기'
                            AND Order_Date IS NOT NULL";

                        LogPurchaseStatusTargets(
                            conn, tx,
                            mroWhere,
                            "구매 중"

                        );

                        int mroToMid = 0;
                        using (var cmd = new MySqlCommand($@"
                            UPDATE parts_purchase_list
                               SET Purchase_Status = '구매 중'
                             WHERE {mroWhere}
                        ", conn, tx))
                        {
                            mroToMid = cmd.ExecuteNonQuery();
                        }
                        Logger.Info($"(MRO 대기→중) 변경: {mroToMid}건");

                        // ------------------------ (3) 대기/중 → 입고 완료 ------------------------
                        string doneWhere = @"
                            Order_Date  IS NOT NULL
                            AND Arrival_Date IS NOT NULL
                            AND Purchase_Status IN ('구매 대기','구매 중')";

                        LogPurchaseStatusTargets(
                            conn, tx,
                            doneWhere,
                            "입고 완료"

                        );

                        int toDone = 0;
                        using (var cmd = new MySqlCommand($@"
                            UPDATE parts_purchase_list
                               SET Purchase_Status = '입고 완료'
                             WHERE {doneWhere}
                        ", conn, tx))
                        {
                            toDone = cmd.ExecuteNonQuery();
                        }
                        Logger.Info($"(구매 중 → 완료) 변경: {toDone}건");

                        // ------------------------ 커밋 & 최종 로그 ------------------------
                        tx.Commit();
                        Logger.Info(" 상태 업데이트(normalizePurchaseStatuses) 완료");
                    }
                    catch (Exception ex)
                    {
                        tx.Rollback();
                        Logger.Info($" 상태 승급 실패: {ex.Message}");
                    }
                }
            }
        }

        public void SyncEmsToPurchaseList()
        {
            using (var conn = new MySqlConnection(connectionString))
            {
                conn.Open();
                using (var tx = conn.BeginTransaction())
                {
                    try
                    {

                        // 임시테이블 정리
                        using (var cmd = new MySqlCommand(@"
                            DROP TEMPORARY TABLE IF EXISTS tmp_p;
                            DROP TEMPORARY TABLE IF EXISTS tmp_e;
                            DROP TEMPORARY TABLE IF EXISTS tmp_e_agg;
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 1) tmp_p: (Q_Code, Req_Qty) 그룹별 No ASC 순번
                        using (var cmd = new MySqlCommand(@"
                            CREATE TEMPORARY TABLE tmp_p AS
                            SELECT
                                p_no,
                                Q_Code,
                                Req_Qty,
                                Req_Date,
                                Parts_Div,
                                P_N,
                                rn
                            FROM (
                                SELECT
                                    p.`No` AS p_no,
                                    p.Q_Code,
                                    p.Req_Qty,
                                    p.Req_Date,
                                    p.Parts_Div,
                                    p.P_N,
                                    @rn := IF(@grp = CONCAT(COALESCE(p.Q_Code,''),'|',COALESCE(p.Req_Qty,0)), @rn + 1, 1) AS rn,
                                    @grp := CONCAT(COALESCE(p.Q_Code,''),'|',COALESCE(p.Req_Qty,0)) AS _g
                                FROM (
                                    SELECT *
                                      FROM parts_purchase_list
                                     WHERE Purchase_Status IN ('구매 대기','구매 중')
                                     ORDER BY Q_Code, Req_Qty, `No` ASC
                                ) p
                                CROSS JOIN (SELECT @rn := 0, @grp := '') vars
                            ) t;
                            ALTER TABLE tmp_p ADD INDEX ix_p (Q_Code, Req_Qty, rn);
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 2-1) EMS 상태 집계 → tmp_e_agg
                        using (var cmd = new MySqlCommand(@"
                            CREATE TEMPORARY TABLE tmp_e_agg AS
                            SELECT
                                MIN(e.`No`)       AS e_no,
                                e.Req_Num,
                                e.Q_Code,
                                SUM(e.Order_Qty)  AS Merged_Qty,
                                MIN(e.P_N)        AS P_N,
                                MIN(e.ERP_PR)     AS ERP_PR,
                                MIN(e.Order_Num)     AS Order_Num,

                                MIN(e.Order_Date) AS Order_Date,
                                MAX(e.Arrival_Date) AS Arrival_Date,
                                MAX(e.PO_Reject)  AS PO_Reject
                            FROM parts_ems_purchase_status_list e
                            WHERE e.Buyer = '장경재'
                                AND e.Req_Num IS NOT NULL
                                AND TRIM(e.Req_Num) <> ''

                                -- 2025-06-30일 후 발주 데이터만 매칭 대상으로 사용
                                AND e.Order_Date >= '2025-06-30'
                               
                                -- ✅ 이미 Purchase에 붙어 있는 EMS 요청(Req_Num+Q_Code)은 제외
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM parts_purchase_list p
                                    WHERE p.Q_Code  = e.Q_Code
                                      AND p.Req_Num = e.Req_Num
                                      AND p.Order_Num = e.Order_Num
                                      AND (p.Order_Date <=> e.Order_Date)
                                      AND (p.Arrival_Date <=> e.Arrival_Date)
                                      AND e.Arrival_Date IS NOT NULL
                                )
                            GROUP BY e.Req_Num, e.Q_Code;
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 2-2) (Q_Code, 합산수량) 그룹 내 순번 → tmp_e
                        using (var cmd = new MySqlCommand(@"
                            CREATE TEMPORARY TABLE tmp_e AS
                            SELECT
                                e_no,
                                Q_Code,
                                Merged_Qty AS Order_Qty,
                                P_N,
                                Req_Num,
                                ERP_PR,
                                Order_Num,
                                Order_Date,
                                Arrival_Date,
                                PO_Reject,
                                rn
                            FROM (
                                SELECT
                                    a.*,
                                    @rn2 := IF(@grp2 = CONCAT(COALESCE(a.Q_Code,''),'|',COALESCE(a.Merged_Qty,0)), @rn2 + 1, 1) AS rn,
                                    @grp2 := CONCAT(COALESCE(a.Q_Code,''),'|',COALESCE(a.Merged_Qty,0)) AS _g
                                FROM (
                                    SELECT *
                                      FROM tmp_e_agg
                                     ORDER BY Q_Code, Merged_Qty, e_no ASC
                                ) a
                                CROSS JOIN (SELECT @rn2 := 0, @grp2 := '') v2
                            ) t;
                            ALTER TABLE tmp_e ADD INDEX ix_e (Q_Code, Order_Qty, rn);
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 3) 매칭 UPDATE
                        int affected;
                        using (var cmd = new MySqlCommand(@"
                            UPDATE parts_purchase_list p
                            JOIN tmp_p pr ON pr.p_no = p.`No`
                            JOIN tmp_e er
                              ON er.Q_Code    = pr.Q_Code
                             AND er.Order_Qty = pr.Req_Qty
                             AND er.rn        = pr.rn
                            SET
                                p.Req_Num      = er.Req_Num,
                                p.ERP_PR       = er.ERP_PR,
                                p.Q_Code_2     = er.Q_Code,
                                p.Order_Num    = er.Order_Num,
                                p.Order_Date   = er.Order_Date,
                                p.Arrival_Date = COALESCE(er.Arrival_Date, p.Arrival_Date),
                                p.PO_Reject    = er.PO_Reject,
                                p.Order_Qty    = er.Order_Qty
                            WHERE p.Purchase_Status IN ('구매 대기','구매 중')
                              AND er.Order_Date IS NOT NULL
                              AND DATE(er.Order_Date) >= DATE(p.Req_Date);
                        ", conn, tx))
                        {
                            affected = cmd.ExecuteNonQuery();
                        }

                        tx.Commit();
                        Logger.Info($"[EMS→Purchase] 업데이트: {affected}건");
                    }
                    catch (Exception ex)
                    {
                        tx.Rollback();
                        Logger.Info($"[EMS→Purchase] 동기화 실패: {ex.Message}");
                        throw;
                    }
                }
            }

            // 동기화 후 상태 승급(별도 트랜잭션)
            try { NormalizePurchaseStatuses(); } catch { /* 이미 로그 출력됨 */ }
        }

        public void SyncMroToPurchaseList()
        {
            using (var conn = new MySqlConnection(connectionString))
            {
                conn.Open();
                using (var tx = conn.BeginTransaction())
                {
                    try
                    {

                        // 임시테이블 정리
                        using (var cmd = new MySqlCommand(@"
                            DROP TEMPORARY TABLE IF EXISTS tmp_p;
                            DROP TEMPORARY TABLE IF EXISTS tmp_m;
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 1) tmp_p: (Q_Code, Req_Qty) 그룹 내 순번
                        using (var cmd = new MySqlCommand(@"
                            CREATE TEMPORARY TABLE tmp_p AS
                            SELECT
                                p_no,
                                Q_Code,
                                Req_Qty,
                                Req_Date,
                                Parts_Div,
                                P_N,
                                rn
                            FROM (
                                SELECT
                                    p.`No`       AS p_no,
                                    p.Q_Code     AS Q_Code,
                                    p.Req_Qty    AS Req_Qty,
                                    p.Req_Date   AS Req_Date,
                                    p.Parts_Div  AS Parts_Div,
                                    p.P_N        AS P_N,
                                    @rn := IF(@grp = CONCAT(p.Q_Code,'|',p.Req_Qty), @rn + 1, 1) AS rn,
                                    @grp := CONCAT(p.Q_Code,'|',p.Req_Qty) AS _g
                                FROM (
                                    SELECT *
                                      FROM parts_purchase_list
                                     WHERE Purchase_Status IN ('구매 대기','구매 중')
                                     ORDER BY Q_Code, Req_Qty, `No` ASC
                                ) p
                                CROSS JOIN (SELECT @rn := 0, @grp := '') vars
                            ) t;
                            ALTER TABLE tmp_p ADD INDEX ix_p (Q_Code, Req_Qty, rn);
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 2) tmp_m: (MRO_Code, Order_Qty) 그룹 내 순번
                        //    이미 parts_purchase_list에 붙어 있는 주문은 제외하고 rn 계산
                        using (var cmd = new MySqlCommand(@"
                            CREATE TEMPORARY TABLE tmp_m AS
                            SELECT
                                m_no,
                                MRO_Code,
                                Order_Qty,
                                P_N,
                                Order_Num,
                                Order_Date,
                                Arrival_Date,
                                rn
                            FROM (
                                SELECT
                                    m.`No`        AS m_no,
                                    m.MRO_Code    AS MRO_Code,
                                    m.Order_Qty   AS Order_Qty,
                                    m.P_N         AS P_N,
                                    m.Order_Num   AS Order_Num,
                                    m.Order_Date  AS Order_Date,
                                    m.Arrival_Date AS Arrival_Date,
                                    @rn2 := IF(@grp2 = CONCAT(m.MRO_Code,'|',m.Order_Qty), @rn2 + 1, 1) AS rn,
                                    @grp2 := CONCAT(m.MRO_Code,'|',m.Order_Qty) AS _g
                                FROM (
                                    SELECT *
                                    FROM parts_ems_purchase_status_mro_list m
                                    WHERE m.Buyer = '장경재'
                                      -- ✅ 이미 Purchase에 매칭된 주문은 제외
                                      AND NOT EXISTS (
                                          SELECT 1
                                          FROM parts_purchase_list p
                                          WHERE p.Q_Code    = m.MRO_Code
                                            AND p.Order_Num = m.Order_Num
                                            AND p.Order_Qty = m.Order_Qty
                                            AND (p.Order_Date <=> m.Order_Date)
                                            AND (p.Arrival_Date <=> m.Arrival_Date)
                                            AND m.Arrival_Date IS NOT NULL
                                      )
                                    ORDER BY m.MRO_Code, m.Order_Qty, m.`No` ASC
                                ) m
                                CROSS JOIN (SELECT @rn2 := 0, @grp2 := '') vars2
                            ) t;
                            ALTER TABLE tmp_m ADD INDEX ix_m (MRO_Code, Order_Qty, rn);
                        ", conn, tx)) cmd.ExecuteNonQuery();

                        // 3) 매칭 UPDATE
                        int affected;
                        using (var cmd = new MySqlCommand(@"
                            UPDATE parts_purchase_list p
                            JOIN tmp_p pr ON pr.p_no = p.`No`
                            JOIN tmp_m mr
                              ON mr.MRO_Code  = pr.Q_Code
                             AND mr.Order_Qty = pr.Req_Qty
                             AND mr.rn        = pr.rn
                            SET
                                p.Q_Code_2     = mr.MRO_Code,
                                p.Order_Num    = mr.Order_Num,
                                p.Order_Date   = mr.Order_Date,
                                p.Arrival_Date = COALESCE(mr.Arrival_Date, p.Arrival_Date),
                                p.Order_Qty    = mr.Order_Qty,
                                /* MRO는 없는 값 보정 */
                                p.Req_Num    = IF(p.Req_Num   IS NULL OR p.Req_Num   = '', '-', p.Req_Num),
                                p.ERP_PR     = IF(p.ERP_PR    IS NULL OR p.ERP_PR    = '', '-', p.ERP_PR),
                                p.PO_Reject  = IF(p.PO_Reject IS NULL OR p.PO_Reject = '', '-', p.PO_Reject)
                            WHERE p.Purchase_Status IN ('구매 대기','구매 중')
                              AND mr.Order_Date IS NOT NULL
                              AND DATE(mr.Order_Date) >= DATE(p.Req_Date);
                        ", conn, tx))
                        {
                            affected = cmd.ExecuteNonQuery();
                        }

                        tx.Commit();
                        Logger.Info($"[MRO→Purchase] 업데이트: {affected}건");
                    }
                    catch (Exception ex)
                    {
                        tx.Rollback();
                        Logger.Info($"[MRO→Purchase] 동기화 실패: {ex.Message}");
                        throw;
                    }
                }
            }

            // 동기화 후 상태 승급(별도 트랜잭션)
            try { NormalizePurchaseStatuses(); } catch { /* 이미 로그 출력됨 */ }
        }

        public void AutoCreatePurchaseRequestBySafetyStock()
        {
            try
            {
                using (var conn = new MySqlConnection(connectionString))
                {
                    conn.Open();

                    using (var tx = conn.BeginTransaction())
                    {
                        string sql = @"
                            INSERT INTO parts_purchase_list
                            (
                                Req_Date,
                                Site,
                                EQP_Div,
                                Q_Code,
                                Parts_Div,
                                P_N,
                                Requester,
                                Req_Comment,
                                Req_Qty,
                                Unit_Price,
                                Total_Price,
                                Purchase_Status
                            )
                            SELECT
                                CURDATE() AS Req_Date,
                                s.Site,
                                s.EQP_Div,
                                s.Q_Code,
                                s.Parts_Div,
                                s.P_N,
                                '자동구매' AS Requester,
                                '안전재고 자동구매' AS Req_Comment,
                                shortage.Req_Qty,
                                s.Unit_Price,
                                shortage.Req_Qty * s.Unit_Price AS Total_Price,
                                '구매 요청' AS Purchase_Status
                            FROM parts_safety_stock_list s
                            JOIN
                            (
                                SELECT
                                    base.Q_Code,
                                    base.Safety_Stock,
                                    base.Count_Now,
                                    base.Open_Order_Qty,
                                    base.Safety_Stock - base.Count_Now - base.Open_Order_Qty AS Req_Qty
                                FROM
                                (
                                    SELECT
                                        s.Q_Code,
                                        COALESCE(s.Safety_Stock, 0) AS Safety_Stock,
                                        COALESCE(stock.Total_Count_Now, 0) AS Count_Now,
                                        COALESCE(open_po.Total_Open_Order_Qty, 0) AS Open_Order_Qty
                                    FROM parts_safety_stock_list s
                                    LEFT JOIN
                                    (
                                        SELECT
                                            Q_Code,
                                            SUM(Count_Now) AS Total_Count_Now
                                        FROM parts_essential_list
                                        GROUP BY Q_Code
                                    ) stock
                                        ON stock.Q_Code = s.Q_Code
                                    LEFT JOIN
                                    (
                                        SELECT
                                            Q_Code,
                                            SUM(COALESCE(Req_Qty, 0)) AS Total_Open_Order_Qty
                                        FROM parts_purchase_list
                                        WHERE Purchase_Status IN ('구매 요청', '구매 대기', '구매 중')
                                        GROUP BY Q_Code
                                    ) open_po
                                        ON open_po.Q_Code = s.Q_Code
                                ) base
                                WHERE base.Safety_Stock > base.Count_Now + base.Open_Order_Qty
                            ) shortage
                                ON shortage.Q_Code = s.Q_Code";

                        using (var cmd = new MySqlCommand(sql, conn, tx))
                        {
                            int inserted = cmd.ExecuteNonQuery();
                            tx.Commit();

                            Logger.Info($"[안전재고 자동구매] 구매 요청 생성: {inserted}건");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Error($"[안전재고 자동구매] 처리 중 오류: {ex.Message}");
            }
        }

        public void LogTodayDropPurchases()
        {
            DateTime today = DateTime.Today;
            try
            {
                var targets = new Dictionary<string, string>();
                using (var conn = new MySqlConnection(connectionString))
                {
                    conn.Open();
                    using (var cmd = new MySqlCommand(@"
                        SELECT Req_Date, Site, EQP_Div, Q_Code, Parts_Div, P_N, Requester, Req_Qty, Order_Qty
                        FROM parts_purchase_list
                        WHERE Req_Date >= @today
                          AND Req_Date < @tomorrow
                          AND Purchase_Status = 'Drop'
                        ORDER BY Req_Date, Site, EQP_Div, Q_Code", conn))
                    {
                        cmd.Parameters.AddWithValue("@today", today);
                        cmd.Parameters.AddWithValue("@tomorrow", today.AddDays(1));
                        using (var reader = cmd.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                string key = CreateDropPurchaseKey(reader);
                                string requester = reader["Requester"].ToString();
                                string requesterMail = RequesterMailRegistry.GetEmail(requester);
                                string line = $"{reader["Q_Code"]}| {reader["Parts_Div"]}| {reader["P_N"]}| {requester}| {reader["Req_Qty"]}| {reader["Order_Qty"]}| Drop| {requesterMail}";
                                // 한 건을 한 줄로 저장하고, 7개 컬럼의 조합으로 재실행 후에도 식별한다.
                                targets[key] = line.Replace("\r", " ").Replace("\n", " ");
                            }
                        }
                    }
                }

                int written = Logger.AppendDropStatusUpdates(today, targets);
                Logger.Info($"[수동 Drop] 오늘 요청 {targets.Count}건, 신규 기록 {written}건, 중복 제외 {targets.Count - written}건");
            }
            catch (Exception ex)
            {
                Logger.Error($"[수동 Drop] 로그 기록 실패: {ex.Message}");
            }
        }

        private static string CreateDropPurchaseKey(System.Data.IDataRecord row)
        {
            var culture = System.Globalization.CultureInfo.InvariantCulture;
            string[] columns = { "Req_Date", "Site", "EQP_Div", "Q_Code", "Parts_Div", "P_N", "Req_Qty" };
            var keyData = new StringBuilder();
            foreach (string column in columns)
            {
                object value = row[column];
                if (value == null || value == DBNull.Value)
                {
                    // NULL과 빈 문자열을 구별한다.
                    keyData.Append("-1:");
                    continue;
                }

                string text;
                if (column == "Req_Date")
                    text = Convert.ToDateTime(value, culture).ToString("yyyy-MM-ddTHH:mm:ss.fffffff", culture);
                else if (column == "Req_Qty")
                    // 같은 수량 1 / 1.0 / 1.00은 같은 값으로 정규화한다.
                    text = Convert.ToDecimal(value, culture).ToString("G29", culture);
                else
                    text = Convert.ToString(value, culture);

                // 길이 접두어로 구분자나 줄바꿈이 값에 있어도 서로 다른 조합이 충돌하지 않는다.
                keyData.Append(text.Length.ToString(culture)).Append(':').Append(text);
            }

            using (var hash = System.Security.Cryptography.SHA256.Create())
            {
                return Convert.ToBase64String(hash.ComputeHash(Encoding.UTF8.GetBytes(keyData.ToString())));
            }
        }

        private void LogPurchaseStatusTargets(
            MySqlConnection conn,
            MySqlTransaction tx,
            string whereClause,
            string newStatus)
            {
                string sql = $@"
                    SELECT Q_Code, Parts_Div, P_N, Requester, Req_Qty, Order_Qty
                    FROM parts_purchase_list
                    WHERE {whereClause}";

                using (var cmd = new MySqlCommand(sql, conn, tx))
                using (var reader = cmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        string qCode = reader["Q_Code"]?.ToString() ?? "";
                        string partsDiv = reader["Parts_Div"]?.ToString() ?? "";
                        string pN = reader["P_N"]?.ToString() ?? "";
                        string requester = reader["Requester"]?.ToString() ?? "";
                        string reqQty = reader["Req_Qty"]?.ToString() ?? "";
                        string ordQty = reader["Order_Qty"]?.ToString() ?? "";
                        string requesterMail = RequesterMailRegistry.GetEmail(requester);


                        string line = $"{qCode}| {partsDiv}| {pN}| {requester}| {reqQty}| {ordQty}| {newStatus}| {requesterMail}";


                        // 화면 로그
                        Logger.Info(line);

                        // 파일 로그 저장
                        Logger.AppendStatusUpdate(line);
                    }
                }
        }
    }
}
