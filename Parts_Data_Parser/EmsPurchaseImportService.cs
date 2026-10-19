using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Excel = Microsoft.Office.Interop.Excel;

namespace log_parser
{
    internal sealed class EmsPurchaseImportService
    {
        private readonly string connectionString;

        public EmsPurchaseImportService(string connectionString)
        {
            this.connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
        }

        public void Execute(string csvFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var rows = ParseEmsPurchaseStatusExcelToRows(csvFilePath);

            // 2) DB 저장: DELETE 없이 INSERT만 (요청사항)
            var writeTime = DateTime.Now;
            SaveEmsPurchaseStatusRowsToDb(rows, writeTime);

            Logger.Info("[EMS] 구매 상태 리스트 처리 완료");
        }

        private List<PurchaseStatusRow> ParseEmsPurchaseStatusExcelToRows(string csvFilePath)
        {
            var rows = new List<PurchaseStatusRow>();

            try
            {
                // ✅ TSV 로딩 (헤더 포함)
                int rowCount, colCount;
                object[,] data = ImportFileReader.LoadTsvToObjectArray(csvFilePath, out rowCount, out colCount);

                int idxReqNum = -1, idxErpPr = -1, idxQ = -1, idxPn = -1, idxOrderNum = -1;
                int idxOrderDate = -1, idxArrivalDate = -1, idxPoReject = -1, idxOrderQty = -1, idxRequester = -1;

                // ✅ 새로 추가되는 필터 컬럼
                int idxReqCancel = -1;
                int idxErpPrReject = -1;
                int idxPrNum = -1;

                // ✅ 헤더 매핑 (기존 used.Cells 대신 data[1, c])
                for (int c = 1; c <= colCount; c++)
                {
                    string header = data[1, c]?.ToString()?.Trim() ?? "";
                    switch (header)
                    {
                        case "MRNUM": idxReqNum = c; break;                // 의뢰번호
                        case "PRGERPPRNUM": idxErpPr = c; break;           // ERP PR
                        case "ITEMNUM": idxQ = c; break;                   // 부품(Q_Code)
                        case "VENDORPN": idxPn = c; break;                 // 업체P/N
                        case "PONUM": idxOrderNum = c; break;              // 발주번호
                        case "POORDERDATE": idxOrderDate = c; break;       // 발주일자
                        case "MTINDATE": idxArrivalDate = c; break;        // 입고일
                        case "STATUS": idxPoReject = c; break;             // PO Reject(상태)
                        case "MRQTY": idxOrderQty = c; break;              // 의뢰수량
                        case "MRREQUESTBY": idxRequester = c; break;       // 구매요청자

                        // ✅ 필터용
                        case "MRCANCEL": idxReqCancel = c; break;          // 의뢰취소
                        case "PRREJECT": idxErpPrReject = c; break;        // ERP PR Reject
                        case "PRNUM": idxPrNum = c; break;                 // 통합번호
                    }
                }

                if (idxQ < 0 || idxOrderNum < 0 || idxOrderQty < 0 || idxRequester < 0 || idxPrNum < 0 || idxReqNum < 0)
                {
                    Logger.Info("구매 상태 CSV 헤더 누락(ITEMNUM / PONUM / MRQTY / MRREQUESTBY / PRNUM / MRNUM)");
                    return rows;
                }

                int skippedByCancel = 0, skippedByErpPrRej = 0, skippedByPoRej = 0, skippedByDup = 0;
                var seen = new HashSet<string>(); // 중복키 기록

                // 1) 파싱 + 필터: 구매요청자=장경재, 의뢰수량>=0, 의뢰취소='C', ERP PR Reject='N', PO Reject='N'
                for (int r = 2; r <= rowCount; r++)
                {
                    string q = data[r, idxQ]?.ToString()?.Trim() ?? "";
                    if (string.IsNullOrEmpty(q)) continue;

                    decimal? qty = ImportFileReader.ParseDecimal(data[r, idxOrderQty]);
                    if (!qty.HasValue || qty.Value < 0) continue;

                    string requester = data[r, idxRequester]?.ToString()?.Trim() ?? "";
                    if (!string.Equals(requester, "장경재", StringComparison.OrdinalIgnoreCase)) continue;

                    // ✅ 필터 1: 의뢰취소 = 'C'
                    if (idxReqCancel > 0)
                    {
                        string cancel = data[r, idxReqCancel]?.ToString()?.Trim() ?? "";
                        if (cancel != "C") { skippedByCancel++; continue; }
                    }

                    // ✅ 필터 2: ERP PR Reject = 'N'
                    if (idxErpPrReject > 0)
                    {
                        string erpPrRej = data[r, idxErpPrReject]?.ToString()?.Trim() ?? "";
                        if (erpPrRej != "N") { skippedByErpPrRej++; continue; }
                    }

                    // ✅ 필터 3: PO Reject = 'N'
                    if (idxPoReject > 0)
                    {
                        string poRej = data[r, idxPoReject]?.ToString()?.Trim() ?? "";
                        if (poRej != "N") { skippedByPoRej++; continue; }
                    }

                    // ✅ 필터 4: (MRNUM, ITEMNUM, MRQTY, PRNUM) 4개가 동일하면 중복 제거
                    string mrnum = data[r, idxReqNum]?.ToString() ?? "";
                    string itemnum = data[r, idxQ]?.ToString() ?? "";
                    decimal mrqty = qty.Value;
                    string prnum = data[r, idxPrNum]?.ToString() ?? "";

                    // 4개 조합 키
                    string key = $"{mrnum}|{itemnum}|{mrqty}|{prnum}";
                    if (!seen.Add(key)) { skippedByDup++; continue; }

                    rows.Add(new PurchaseStatusRow
                    {
                        Req_Num = (idxReqNum > 0) ? (data[r, idxReqNum]?.ToString()?.Trim() ?? "") : "",
                        ERP_PR = (idxErpPr > 0) ? (data[r, idxErpPr]?.ToString()?.Trim() ?? "") : "",
                        Q_Code = q,
                        P_N = (idxPn > 0) ? (data[r, idxPn]?.ToString()?.Trim() ?? "") : "",
                        Order_Num = data[r, idxOrderNum]?.ToString()?.Trim() ?? "",
                        Order_Date = (idxOrderDate > 0) ? DateTime.TryParse(data[r, idxOrderDate]?.ToString(), out var od) ? od : (DateTime?)null : null,
                        Arrival_Date = (idxArrivalDate > 0) ? DateTime.TryParse(data[r, idxArrivalDate]?.ToString(), out var ad) ? ad : (DateTime?)null : null,
                        PO_Reject = (idxPoReject > 0) ? (data[r, idxPoReject]?.ToString()?.Trim() ?? "") : "",
                        Order_Qty = qty,
                        Requester = requester,
                        Order_Num_Seq = 0
                    });
                }

                // 2) 넘버링: (Q_Code, Order_Num, Order_Qty) 동일 그룹 → 1,2,3...
                var counter = new Dictionary<string, int>();
                foreach (var r in rows)
                {
                    bool hasAll = !string.IsNullOrWhiteSpace(r.Q_Code)
                                  && !string.IsNullOrWhiteSpace(r.Order_Num)
                                  && r.Order_Qty.HasValue;
                    if (!hasAll) { r.Order_Num_Seq = 0; continue; }

                    string key = $"{r.Q_Code}|{r.Order_Num}|{r.Order_Qty.Value}";
                    if (!counter.TryGetValue(key, out int seq)) seq = 0;
                    r.Order_Num_Seq = ++seq;
                    counter[key] = seq;
                }

                Logger.Info(
                    $"구매 상태 CSV 파싱 완료(필터 적용: 의뢰취소≠C {skippedByCancel}건, ERP PR Reject≠N {skippedByErpPrRej}건, PO Reject≠N {skippedByPoRej}건, 중복스킵 {skippedByDup}건): {rows.Count} 행");
            }
            catch (Exception ex)
            {
                Logger.Info($"구매 상태 CSV 파싱 오류: {ex.Message}");
            }

            return rows;
        }

        private void SaveEmsPurchaseStatusRowsToDb(List<PurchaseStatusRow> rows, DateTime writeTime)
        {
            if (rows == null || rows.Count == 0)
            {
                Logger.Info("[parts_ems_purchase_status_list 저장 스킵] 입력할 행이 없습니다.");
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
                                INSERT INTO parts_ems_purchase_status_list
                                (`WriteTime`, `Q_Code`, `Order_Num`, `Order_Qty`, `Order_Num_Seq`,
                                 `Req_Num`, `ERP_PR`, `P_N`, `Order_Date`, `Arrival_Date`, `PO_Reject`, `Buyer`)
                                VALUES
                                (@t, @q, @ordnum, @qty, @seq,
                                 @req, @erp, @pn, @odate, @adate, @reject, @buyer)
                                ON DUPLICATE KEY UPDATE
                                  `WriteTime`    = GREATEST(`WriteTime`, VALUES(`WriteTime`)),
                                  `Req_Num`      = IFNULL(VALUES(`Req_Num`), `Req_Num`),
                                  `ERP_PR`       = IFNULL(VALUES(`ERP_PR`), `ERP_PR`),
                                  `P_N`          = IFNULL(VALUES(`P_N`), `P_N`),
                                  `Order_Date`   = IFNULL(VALUES(`Order_Date`), `Order_Date`),
                                  `Arrival_Date` = IFNULL(VALUES(`Arrival_Date`), `Arrival_Date`),
                                  `PO_Reject`    = IFNULL(VALUES(`PO_Reject`), `PO_Reject`),
                                  `Buyer`        = IFNULL(VALUES(`Buyer`), `Buyer`);

                                ";

                        using (var cmd = new MySqlCommand(sql, conn, tx))
                        {
                            var pT = cmd.Parameters.Add("@t", MySqlDbType.DateTime);
                            var pQ = cmd.Parameters.Add("@q", MySqlDbType.VarChar);
                            var pOrdNo = cmd.Parameters.Add("@ordnum", MySqlDbType.VarChar);
                            var pQty = cmd.Parameters.Add("@qty", MySqlDbType.Decimal);
                            var pSeq = cmd.Parameters.Add("@seq", MySqlDbType.Int32);

                            var pReq = cmd.Parameters.Add("@req", MySqlDbType.VarChar);
                            var pErp = cmd.Parameters.Add("@erp", MySqlDbType.VarChar);
                            var pPn = cmd.Parameters.Add("@pn", MySqlDbType.VarChar);
                            var pODt = cmd.Parameters.Add("@odate", MySqlDbType.DateTime);
                            var pADt = cmd.Parameters.Add("@adate", MySqlDbType.DateTime);
                            var pRej = cmd.Parameters.Add("@reject", MySqlDbType.VarChar);
                            var pBuyer = cmd.Parameters.Add("@buyer", MySqlDbType.VarChar);

                            int inserted = 0;

                            foreach (var r in rows)
                            {
                                // ✅ 키 4개 모두 있어야 함
                                if (string.IsNullOrWhiteSpace(r.Q_Code) ||
                                    string.IsNullOrWhiteSpace(r.Order_Num) ||
                                    !r.Order_Qty.HasValue)
                                {
                                    continue;
                                }

                                pT.Value = writeTime;
                                pQ.Value = r.Q_Code;
                                pOrdNo.Value = r.Order_Num;
                                pQty.Value = r.Order_Qty.Value;
                                pSeq.Value = r.Order_Num_Seq; // 0 또는 1,2,3…

                                pReq.Value = string.IsNullOrWhiteSpace(r.Req_Num) ? (object)DBNull.Value : r.Req_Num;
                                pErp.Value = string.IsNullOrWhiteSpace(r.ERP_PR) ? (object)DBNull.Value : r.ERP_PR;
                                pPn.Value = string.IsNullOrWhiteSpace(r.P_N) ? (object)DBNull.Value : r.P_N;
                                pODt.Value = r.Order_Date.HasValue ? (object)r.Order_Date.Value : DBNull.Value;
                                pADt.Value = r.Arrival_Date.HasValue ? (object)r.Arrival_Date.Value : DBNull.Value;
                                pRej.Value = string.IsNullOrWhiteSpace(r.PO_Reject) ? (object)DBNull.Value : r.PO_Reject;
                                pBuyer.Value = string.IsNullOrWhiteSpace(r.Requester) ? (object)DBNull.Value : r.Requester;


                                cmd.ExecuteNonQuery();
                                inserted++;
                            }

                            tx.Commit();
                            Logger.Info($"parts_ems_purchase_status_list UPSERT 완료: {inserted}건 (WriteTime={writeTime:yyyy-MM-dd HH:mm:ss})");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Info($"parts_ems_purchase_status_list 저장 중 오류: {ex.Message}");
            }
        }
    }
}
