using MySql.Data.MySqlClient;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Excel = Microsoft.Office.Interop.Excel;

namespace log_parser
{
    internal sealed class MroPurchaseImportService
    {
        private readonly string connectionString;

        public MroPurchaseImportService(string connectionString)
        {
            this.connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }

        public void Execute(string excelFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var rows = ParseMroPurchaseStatusExcelToRows(excelFilePath);

            // 2) DB 저장: INSERT-only (No는 AUTO_INCREMENT)
            var writeTime = DateTime.Now;
            SaveMroPurchaseStatusRowsToDb(rows, writeTime);

            Logger.Info("[MRO] 구매 상태 리스트 처리 완료");
        }

        private List<MroPurchaseStatusRow> ParseMroPurchaseStatusExcelToRows(string excelFilePath)
        {
            var rows = new List<MroPurchaseStatusRow>();
            Excel.Application excelApp = new Excel.Application();
            Excel.Workbook workbook = null;
            Excel.Worksheet worksheet = null;

            try
            {
                workbook = excelApp.Workbooks.Open(excelFilePath);
                worksheet = workbook.Sheets[1];
                Excel.Range used = worksheet.UsedRange;

                int idxMroCode = -1, idxPN = -1, idxOrderNum = -1, idxOrderDate = -1,
                    idxArrivalDate = -1, idxOrderQty = -1, idxBuyer = -1, idxStatus = -1;

                // 헤더 매핑 (정확히 "상태")
                for (int c = 1; c <= used.Columns.Count; c++)
                {
                    string header = used.Cells[1, c].Text?.ToString()?.Trim() ?? "";
                    switch (header)
                    {
                        case "BQMS코드": idxMroCode = c; break;
                        case "[제조사] 품명": idxPN = c; break;
                        case "P/O No.": idxOrderNum = c; break;
                        case "P/O일자": idxOrderDate = c; break;
                        case "G/R일": idxArrivalDate = c; break;
                        case "P/O수량": idxOrderQty = c; break;
                        case "구매담당자": idxBuyer = c; break;
                        case "상태": idxStatus = c; break;  // ✅ 여기만 본다
                    }
                }

                if (idxMroCode < 0 || idxOrderNum < 0 || idxOrderQty < 0 || idxBuyer < 0)
                {
                    Logger.Info("[MRO] 엑셀 헤더 누락(BQMS코드/P-O No./P-O수량/구매담당자)");
                    return rows;
                }

                object[,] data = (object[,])used.Value2;

                int skippedByStatus = 0;

                // 1) 필터: 구매담당자=장경재, P/O수량>0, 상태 != (PO 반려, PO 상신 취소)
                for (int r = 2; r <= data.GetLength(0); r++)
                {
                    string mro = data[r, idxMroCode]?.ToString()?.Trim() ?? "";
                    if (string.IsNullOrEmpty(mro)) continue;

                    decimal? qty = ImportFileReader.ParseDecimal(data[r, idxOrderQty]);
                    if (!qty.HasValue || qty.Value <= 0) continue;

                    string buyer = data[r, idxBuyer]?.ToString()?.Trim() ?? "";
                    if (!string.Equals(buyer, "장경재", StringComparison.OrdinalIgnoreCase)) continue;

                    // ✅ 상태 필터
                    if (idxStatus > 0)
                    {
                        string status = data[r, idxStatus]?.ToString()?.Trim() ?? "";
                        if (status == "PO 반려" || status == "PO 상신취소")
                        {
                            skippedByStatus++;
                            continue;
                        }
                    }

                    var row = new MroPurchaseStatusRow
                    {
                        MRO_Code = mro,
                        P_N = (idxPN > 0) ? (data[r, idxPN]?.ToString()?.Trim() ?? "") : "",
                        Order_Num = data[r, idxOrderNum]?.ToString()?.Trim() ?? "",
                        Order_Date = (idxOrderDate > 0) ? ImportFileReader.ParseExcelDate(data[r, idxOrderDate]) : null,
                        Arrival_Date = (idxArrivalDate > 0) ? ImportFileReader.ParseExcelDate(data[r, idxArrivalDate]) : null,
                        Order_Qty = qty,
                        Buyer = buyer,
                        Order_Num_Seq = 0
                    };

                    rows.Add(row);
                }

                // 2) 넘버링: (BQMS코드, P/O No., P/O수량) 그룹 내 1,2,3...
                var counter = new Dictionary<string, int>();
                foreach (var r in rows.OrderBy(x => x.MRO_Code).ThenBy(x => x.Order_Num))
                {
                    bool hasAll = !string.IsNullOrWhiteSpace(r.MRO_Code)
                                  && !string.IsNullOrWhiteSpace(r.Order_Num)
                                  && r.Order_Qty.HasValue;
                    if (!hasAll) { r.Order_Num_Seq = 0; continue; }

                    string key = $"{r.MRO_Code}|{r.Order_Num}|{r.Order_Qty.Value}";
                    if (!counter.TryGetValue(key, out int seq)) seq = 0;
                    r.Order_Num_Seq = ++seq;
                    counter[key] = seq;
                }

                Logger.Info($"[MRO] 엑셀 파싱 완료(상태 제외 {skippedByStatus}건): {rows.Count} 행");
            }
            catch (Exception ex)
            {
                Logger.Info($"[MRO] 엑셀 파싱 오류: {ex.Message}");
            }
            finally
            {
                if (workbook != null) workbook.Close(false);
                if (excelApp != null) excelApp.Quit();
                if (worksheet != null) Marshal.ReleaseComObject(worksheet);
                if (workbook != null) Marshal.ReleaseComObject(workbook);
                if (excelApp != null) Marshal.ReleaseComObject(excelApp);
            }

            return rows;
        }

        private void SaveMroPurchaseStatusRowsToDb(List<MroPurchaseStatusRow> rows, DateTime writeTime)
        {
            if (rows == null || rows.Count == 0)
            {
                Logger.Info("[MRO] 저장 스킵: 입력할 행이 없습니다.");
                return;
            }

            try
            {
                using (var conn = new MySqlConnection(connectionString))
                {
                    conn.Open();
                    using (var tx = conn.BeginTransaction())
                    {
                        string sql = @"
                                INSERT INTO `parts_ems_purchase_status_mro_list`
                                (`WriteTime`,`MRO_Code`,`Order_Num`,`Order_Qty`,`Order_Num_Seq`,
                                 `P_N`,`Order_Date`,`Arrival_Date`,`Buyer`)
                                VALUES
                                (@t,@mro,@ordnum,@qty,@seq,
                                 @pn,@odate,@adate,@buyer)
                                ON DUPLICATE KEY UPDATE
                                  `WriteTime`    = GREATEST(`WriteTime`, VALUES(`WriteTime`)),
                                  `P_N`          = IFNULL(VALUES(`P_N`), `P_N`),
                                  `Order_Date`   = IFNULL(VALUES(`Order_Date`), `Order_Date`),
                                  `Arrival_Date` = IFNULL(VALUES(`Arrival_Date`), `Arrival_Date`),
                                  `Buyer`        = IFNULL(VALUES(`Buyer`), `Buyer`);
                                ";

                        using (var cmd = new MySqlCommand(sql, conn, tx))
                        {
                            var pT = cmd.Parameters.Add("@t", MySqlDbType.DateTime);
                            var pMro = cmd.Parameters.Add("@mro", MySqlDbType.VarChar);
                            var pON = cmd.Parameters.Add("@ordnum", MySqlDbType.VarChar);
                            var pQty = cmd.Parameters.Add("@qty", MySqlDbType.Decimal);
                            var pSeq = cmd.Parameters.Add("@seq", MySqlDbType.Int32);

                            var pPN = cmd.Parameters.Add("@pn", MySqlDbType.VarChar);
                            var pODt = cmd.Parameters.Add("@odate", MySqlDbType.DateTime);
                            var pADt = cmd.Parameters.Add("@adate", MySqlDbType.DateTime);
                            var pBuyer = cmd.Parameters.Add("@buyer", MySqlDbType.VarChar);

                            int inserted = 0;

                            foreach (var r in rows)
                            {
                                // ✅ PK 필수 컬럼 체크
                                if (string.IsNullOrWhiteSpace(r.MRO_Code) ||
                                    string.IsNullOrWhiteSpace(r.Order_Num) ||
                                    !r.Order_Qty.HasValue)
                                {
                                    continue;
                                }

                                pT.Value = writeTime;
                                pMro.Value = r.MRO_Code;
                                pON.Value = r.Order_Num;
                                pQty.Value = r.Order_Qty.Value;
                                pSeq.Value = r.Order_Num_Seq; // 0 또는 1,2,3…

                                pPN.Value = string.IsNullOrWhiteSpace(r.P_N) ? (object)DBNull.Value : r.P_N;
                                pODt.Value = r.Order_Date.HasValue ? (object)r.Order_Date.Value : DBNull.Value;
                                pADt.Value = r.Arrival_Date.HasValue ? (object)r.Arrival_Date.Value : DBNull.Value;
                                pBuyer.Value = string.IsNullOrWhiteSpace(r.Buyer) ? (object)DBNull.Value : r.Buyer;

                                cmd.ExecuteNonQuery();
                                inserted++;
                            }

                            tx.Commit();
                            Logger.Info($"[MRO] UPSERT 완료: {inserted}건 (WriteTime={writeTime:yyyy-MM-dd HH:mm:ss})");
                        }
                    }
                }
            }
            catch (MySqlException ex)
            {
                Logger.Info($"[MRO] SQL Error ({ex.Number}/{ex.SqlState}): {ex.Message}");
                // AddLogToListBox1("SQL = " + sql); // 필요 시 주석 해제
            }
            catch (Exception ex)
            {
                Logger.Info($"[MRO] 저장 중 오류: {ex.Message}");
            }
        }
    }
}
