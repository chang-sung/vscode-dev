using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace log_parser
{
    internal sealed class StockService
    {
        private readonly string connectionString;

        public StockService(string connectionString)
        {
            this.connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }

        private List<string> mysqlQCodes = new List<string>();
        private Dictionary<string, string> excelData = new Dictionary<string, string>();

        public event Action<IReadOnlyList<string>> EssentialStockLoaded;
        public event Action<IReadOnlyList<string>> ParsedStockLoaded;
        public event Action<IReadOnlyList<string>> MatchingStockLoaded;

        public void Execute(string csvFilePath)
        {
                  // MySQL에서 Q_Code와 Count_Now 값을 가져와 ListBox2에 출력
                  // 1) DB에서 필수 파츠 목록
            FetchDataFromMySQL();

                  // 2) 엑셀에서 Q_Code/현재잔고 집계 (이미 DB 교집합만 누적됨)
            LoadQCodeAndBalanceFromExcel(csvFilePath);

                  // 3) 교집합 스냅샷만 저장
            var snapshotTime = DateTime.Now;
            SaveStockRowsToDb(snapshotTime); // ← 파라미터/시그니처 변경

                  // 4) 표시 및 Count_Now 반영(교집합만)
            DisplayMatchingQCodesAndBalances();
            UpdateCountNowInDatabase();
        }

        private void FetchDataFromMySQL()
        {
            try
            {
                using (var connection = new MySqlConnection(connectionString))
                {
                    connection.Open();
                    string query = "SELECT Q_Code, Count_Now FROM parts_essential_list";
                    using (var command = new MySqlCommand(query, connection))
                    using (var reader = command.ExecuteReader())
                    {
                        mysqlQCodes.Clear();
                        var lines = new List<string>();
                        while (reader.Read())
                        {
                            string qCode = reader["Q_Code"].ToString();
                            string countNow = reader["Count_Now"].ToString();
                            mysqlQCodes.Add(qCode);
                            lines.Add($"{qCode} - {countNow}");
                        }
                        EssentialStockLoaded?.Invoke(lines);
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Error($"MySQL에서 데이터를 읽는 중 오류가 발생했습니다: {ex.Message}");
            }
        }

        private void LoadQCodeAndBalanceFromExcel(string csvFilePath)
        {
            try
            {
                int rowCount, colCount;
                object[,] data = ImportFileReader.LoadTsvToObjectArray(csvFilePath, out rowCount, out colCount);

                int idxQ = -1, idxBal = -1;

                // 헤더 매핑 (1행) 헤더 인덱스 찾기
                for (int c = 1; c <= colCount; c++)
                {
                    string header = data[1, c]?.ToString()?.Trim() ?? "";
                    if (header == "ITEMNO") idxQ = c;                           // 기존 "부품"
                    else if (header == "CURVAL") idxBal = c;                    // 기존 "현재잔고"
                }

                if (idxQ < 0 || idxBal < 0)
                {
                    Logger.Info("[STOCK] TSV 헤더 누락 (ITEMNO / CURVAL)");
                    return;
                }

                excelData.Clear();
                var mysqlSet = new HashSet<string>(mysqlQCodes, StringComparer.OrdinalIgnoreCase);

                for (int r = 2; r <= rowCount; r++)
                {
                    string qCode = data[r, idxQ]?.ToString()?.Trim() ?? "";
                    if (string.IsNullOrEmpty(qCode)) continue;

                    if (!mysqlSet.Contains(qCode)) continue;

                    if (!decimal.TryParse(data[r, idxBal]?.ToString(), out var balance))
                        continue;

                    if (excelData.ContainsKey(qCode))
                        excelData[qCode] = (decimal.Parse(excelData[qCode]) + balance).ToString();
                    else
                        excelData[qCode] = balance.ToString();
                }

                ParsedStockLoaded?.Invoke(excelData
                    .Select(kv => $"{kv.Key} - 현재잔고 합계: {kv.Value}").ToList());
            }
            catch (Exception ex)
            {
                Logger.Info($"[STOCK] TSV 처리 오류: {ex.Message}");
            }
        }

        private void DisplayMatchingQCodesAndBalances()
        {
            var lines = mysqlQCodes.Intersect(excelData.Keys)
                .Select(qCode => $"{qCode} - 현재잔고: {excelData[qCode]}").ToList();
            MatchingStockLoaded?.Invoke(lines);
        }

        private void UpdateCountNowInDatabase()
        {
            try
            {
                using (var connection = new MySqlConnection(connectionString))
                {
                    connection.Open();

                                 // 트랜잭션으로 묶어서 일관성 확보 + 성능 향상
                    using (var tx = connection.BeginTransaction())
                    {
                        string updateSql = @"
                            UPDATE parts_essential_list
                            SET
                                Count_Now       = @Balance,
                                EMS_update_time = @UpdateTime
                            WHERE Q_Code = @QCode";

                        using (var cmd = new MySqlCommand(updateSql, connection, tx))
                        {
                                                 // 파라미터를 한 번만 정의하고 루프에서 값만 교체
                            var pBal = cmd.Parameters.Add("@Balance", MySqlDbType.Decimal);
                            var pTime = cmd.Parameters.Add("@UpdateTime", MySqlDbType.DateTime);
                            var pCode = cmd.Parameters.Add("@QCode", MySqlDbType.VarChar);

                            DateTime now = DateTime.Now;

                            foreach (var qCode in mysqlQCodes.Intersect(excelData.Keys))
                            {
                                                          // excelData는 string으로 누적되어 있으니 안전하게 변환
                                if (!decimal.TryParse(excelData[qCode], out var balance))
                                    continue;

                                pBal.Value = balance;
                                pTime.Value = now;      // 현재 시간으로 EMS_update_time 갱신
                                pCode.Value = qCode;

                                cmd.ExecuteNonQuery();
                            }
                        }

                        tx.Commit();
                    }

                                    // UI 업데이트
                    Logger.Info("필수 파츠 재고 업데이트 완료!");
                }
            }
            catch (Exception ex)
            {
                Logger.Error($"MySQL 업데이트 중 오류: {ex.Message}");
            }
        }

        private void SaveStockRowsToDb(DateTime writeTime)
        {
            // mysqlQCodes ∩ excelData.Keys (LoadQCodeAndBalanceFromExcel에서 이미 mysqlQCodes 필터링됨)
            var keys = mysqlQCodes.Intersect(excelData.Keys).ToList();
            if (keys.Count == 0)
            {
                Logger.Info("교집합이 없어 parts_ems_stock_list 저장을 스킵합니다.");
                return;
            }

            try
            {
                using (var conn = new MySqlConnection(connectionString))
                {
                    conn.Open();
                    using (var tx = conn.BeginTransaction())
                    {
                        // 테이블 전체를 스냅샷으로 유지하려면 전체 삭제 후 교집합만 채움
                        using (var delAll = new MySqlCommand("DELETE FROM parts_ems_stock_list", conn, tx))
                        {
                            int deleted = delAll.ExecuteNonQuery();
                            Logger.Info($"기존 데이터 삭제 완료: {deleted}행");
                        }

                        // 필요한 컬럼만 넣어도 되지만(WriteTime, Q_Code, Count),
                        // 스키마가 NOT NULL 제약이 있을 수 있어 기존 컬럼셋 유지 + NULL/빈값 사용
                        string sql = @"
                    INSERT INTO parts_ems_stock_list
                    (WriteTime, Q_Code, Parts_Name, P_N, Count, Auto_PR, Site)
                    VALUES (@t, @q, @pnm, @pn, @cnt, @apr, @site)";

                        using (var cmd = new MySqlCommand(sql, conn, tx))
                        {
                            var pT = cmd.Parameters.Add("@t", MySqlDbType.DateTime);
                            var pQ = cmd.Parameters.Add("@q", MySqlDbType.VarChar);
                            var pPNm = cmd.Parameters.Add("@pnm", MySqlDbType.VarChar);
                            var pPN = cmd.Parameters.Add("@pn", MySqlDbType.VarChar);
                            var pCnt = cmd.Parameters.Add("@cnt", MySqlDbType.Decimal);
                            var pApr = cmd.Parameters.Add("@apr", MySqlDbType.VarChar);
                            var pSite = cmd.Parameters.Add("@site", MySqlDbType.VarChar);

                            foreach (var q in keys)
                            {
                                if (!decimal.TryParse(excelData[q], out var cnt)) continue;

                                pT.Value = writeTime;
                                pQ.Value = q;
                                pPNm.Value = DBNull.Value;   // 필요시 ""로
                                pPN.Value = DBNull.Value;   // 필요시 ""로
                                pCnt.Value = cnt;
                                pApr.Value = DBNull.Value;   // 필요시 ""로
                                pSite.Value = DBNull.Value;   // 필요시 ""로

                                cmd.ExecuteNonQuery();
                            }
                        }

                        tx.Commit();
                    }
                }

                Logger.Info($"parts_ems_stock_list(교집합만) 저장 완료: {keys.Count}행 (WriteTime={writeTime:yyyy-MM-dd HH:mm:ss})");
            }
            catch (Exception ex)
            {
                Logger.Info($"parts_ems_stock_list 저장 중 오류: {ex.Message}");
            }
        }
    }
}
