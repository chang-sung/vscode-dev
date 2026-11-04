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
    internal sealed class PriceImportService
    {
        private readonly string connectionString;

        public PriceImportService(string connectionString)
        {
            this.connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }

        public void Execute(string excelFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var priceRows = ParsePriceExcelToRows(excelFilePath);

            // 2) DB 저장: 기존 데이터 전체 DELETE → INSERT
            var snapshotTime = DateTime.Now;
            SavePriceRowsToDb(priceRows, snapshotTime);

            Logger.Info("가격 리스트 처리 완료");
        }

        private List<PriceRow> ParsePriceExcelToRows(string excelFilePath)
        {
            var rows = new List<PriceRow>();
            Excel.Application excelApp = new Excel.Application();
            Excel.Workbook workbook = null;
            Excel.Worksheet worksheet = null;
            Excel.Range used = null;

            try
            {
                workbook = excelApp.Workbooks.Open(excelFilePath);
                worksheet = workbook.Sheets[1];
                used = worksheet.UsedRange;

                // 1) 헤더 맵
                var headerToCol = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                int colCount = used.Columns.Count;
                for (int c = 1; c <= colCount; c++)
                {
                    string header = used.Cells[1, c].Text?.ToString()?.Trim() ?? "";
                    if (!string.IsNullOrEmpty(header) && !headerToCol.ContainsKey(header))
                        headerToCol[header] = c;
                }

                // 2) 자동판별: BQMS코드가 있으면 MRO, 없으면 EMS
                bool isMro = headerToCol.ContainsKey("BQMS코드");

                object[,] data = (object[,])used.Value2;

                if (isMro)
                {
                    // ================= MRO =================
                    // 필수: BQMS코드, [모델] 사양, 단가
                    if (!headerToCol.TryGetValue("BQMS코드", out int idxQCode) ||
                        !headerToCol.TryGetValue("[모델] 사양", out int idxPN) ||
                        !headerToCol.TryGetValue("단가", out int idxPrice))
                    {
                        Logger.Info("[가격][MRO] 헤더 누락(BQMS코드 / [모델] 사양 / 단가).");
                        return rows;
                    }
                    // ✅ Parts_Div: [제조사] 품명
                    int idxPartsDiv = headerToCol.TryGetValue("[제조사] 품명", out var tmp) ? tmp : -1;

                    for (int r = 2; r <= data.GetLength(0); r++)
                    {
                        string q = data[r, idxQCode]?.ToString()?.Trim() ?? "";
                        if (string.IsNullOrEmpty(q)) continue;

                        string pn = data[r, idxPN]?.ToString()?.Trim() ?? "";
                        string priceText = data[r, idxPrice]?.ToString()?.Trim() ?? "";
                        decimal? price = null;
                        if (decimal.TryParse(priceText, out var d)) price = d;

                        string partsDiv = (idxPartsDiv > 0) ? (data[r, idxPartsDiv]?.ToString()?.Trim() ?? "") : "";

                        rows.Add(new PriceRow
                        {
                            Q_Code = q,
                            P_N = pn,            // [모델] 사양
                            Site = "MRO",
                            Price = price,
                            Parts_Div = partsDiv // ✅ [제조사] 품명
                        });
                    }

                    Logger.Info($"[가격][자동판별:MRO] 파싱 완료: {rows.Count} 행 (Parts_Div='[제조사] 품명')");
                }
                else
                {
                    // ================= EMS =================
                    // 필수: 부품, 업체P/N, 결산단가, 창고구분(=장부만 반영)
                    if (!headerToCol.TryGetValue("부품", out int idxQCode) ||
                        !headerToCol.TryGetValue("업체P/N", out int idxPN) ||
                        !headerToCol.TryGetValue("결산단가", out int idxPrice) ||
                        !headerToCol.TryGetValue("창고구분", out int idxStorageType))
                    {
                        Logger.Info("[가격][EMS] 헤더 누락(부품 / 업체P/N / 결산단가 / 창고구분).");
                        return rows;
                    }

                    // 선택 헤더
                    int idxSite = headerToCol.TryGetValue("출고창고", out var s) ? s : -1;
                    int idxPartsDiv = headerToCol.TryGetValue("부품명", out var p) ? p : -1; // ✅ EMS: 부품명 → Parts_Div

                    for (int r = 2; r <= data.GetLength(0); r++)
                    {
                        string q = data[r, idxQCode]?.ToString()?.Trim() ?? "";
                        if (string.IsNullOrEmpty(q)) continue;

                        string storageType = data[r, idxStorageType]?.ToString()?.Trim() ?? "";
                        if (!string.Equals(storageType, "장부", StringComparison.OrdinalIgnoreCase)) continue;

                        string pn = data[r, idxPN]?.ToString()?.Trim() ?? "";
                        string site = (idxSite > 0) ? (data[r, idxSite]?.ToString()?.Trim() ?? "") : "";
                        string priceText = data[r, idxPrice]?.ToString()?.Trim() ?? "";
                        decimal? price = null;
                        if (decimal.TryParse(priceText, out var d)) price = d;

                        string partsDiv = (idxPartsDiv > 0) ? (data[r, idxPartsDiv]?.ToString()?.Trim() ?? "") : "";

                        rows.Add(new PriceRow
                        {
                            Q_Code = q,          // EMS: 부품
                            P_N = pn,            // EMS: 업체P/N
                            Site = site,         // EMS: 출고창고
                            Price = price,       // EMS: 결산단가
                            Parts_Div = partsDiv // ✅ EMS: 부품명
                        });
                    }

                    Logger.Info($"[가격][자동판별:EMS/창고구분=장부] 파싱 완료: {rows.Count} 행 (Parts_Div='부품명')");
                }
            }
            catch (Exception ex)
            {
                Logger.Info($"가격 엑셀 파싱 오류: {ex.Message}");
            }
            finally
            {
                if (used != null) Marshal.ReleaseComObject(used);
                if (worksheet != null) Marshal.ReleaseComObject(worksheet);
                if (workbook != null) { workbook.Close(false); Marshal.ReleaseComObject(workbook); }
                if (excelApp != null) { excelApp.Quit(); Marshal.ReleaseComObject(excelApp); }
            }

            return rows;
        }

        private void SavePriceRowsToDb(List<PriceRow> rows, DateTime writeTime)
        {
            if (rows == null || rows.Count == 0)
            {
                Logger.Info("[parts_ems_price_list 저장 스킵] 입력할 행이 없습니다.");
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
                             INSERT INTO parts_ems_price_list
                             (WriteTime, Q_Code, P_N, Site, Price, Parts_Div)
                             VALUES (@t, @q, @pn, @site, @price, @parts_div)
                             ON DUPLICATE KEY UPDATE
                                 WriteTime = GREATEST(WriteTime, VALUES(WriteTime)),
                                 P_N       = IFNULL(VALUES(P_N),  P_N),
                                 Site      = IFNULL(VALUES(Site), Site),
                                 Price     = IFNULL(VALUES(Price), Price),
                                 Parts_Div = IFNULL(VALUES(Parts_Div), Parts_Div);
                             ";

                        using (var cmd = new MySqlCommand(sql, conn, tx))
                        {
                            var pT = cmd.Parameters.Add("@t", MySqlDbType.DateTime);
                            var pQ = cmd.Parameters.Add("@q", MySqlDbType.VarChar);
                            var pPN = cmd.Parameters.Add("@pn", MySqlDbType.VarChar);
                            var pSite = cmd.Parameters.Add("@site", MySqlDbType.VarChar);
                            var pPrice = cmd.Parameters.Add("@price", MySqlDbType.Decimal);
                            var pPD = cmd.Parameters.Add("@parts_div", MySqlDbType.VarChar);

                            foreach (var r in rows)
                            {
                                pT.Value = writeTime;
                                pQ.Value = r.Q_Code ?? "";
                                pPN.Value = string.IsNullOrWhiteSpace(r.P_N) ? (object)DBNull.Value : r.P_N;
                                pSite.Value = string.IsNullOrWhiteSpace(r.Site) ? (object)DBNull.Value : r.Site;
                                pPrice.Value = r.Price.HasValue ? (object)r.Price.Value : DBNull.Value;
                                pPD.Value = string.IsNullOrWhiteSpace(r.Parts_Div) ? (object)DBNull.Value : r.Parts_Div;

                                cmd.ExecuteNonQuery();
                            }
                        }

                        tx.Commit();
                    }
                }

                Logger.Info($"parts_ems_price_list UPSERT 완료: {rows.Count}건 (WriteTime={writeTime:yyyy-MM-dd HH:mm:ss})");
            }
            catch (Exception ex)
            {
                Logger.Info($"parts_ems_price_list 저장 중 오류: {ex.Message}");
            }
        }
    }
}
