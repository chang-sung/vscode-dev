using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace log_parser
{
    internal sealed class PriceRow
    {
        public string Q_Code { get; set; } = "";  // 부품
        public string P_N { get; set; } = "";     // 업체P/N
        public string Site { get; set; } = "";    // 출고창고
        public decimal? Price { get; set; }       // 결산단가
        public string Parts_Div { get; set; } = "";// EMS=부품명, MRO=[제조사] 품명
    }

    internal sealed class PurchaseStatusRow
    {
        public string Req_Num { get; set; } = "";       // 의뢰번호
        public string ERP_PR { get; set; } = "";        // ERP PR
        public string Q_Code { get; set; } = "";        // 부품
        public string P_N { get; set; } = "";           // 업체P/N
        public string Order_Num { get; set; } = "";     // 발주번호
        public DateTime? Order_Date { get; set; }       // 발주일자
        public DateTime? Arrival_Date { get; set; }     // 입고일
        public string PO_Reject { get; set; } = "";     // PO Reject
        public decimal? Order_Qty { get; set; }    // 의뢰수량
        public string Requester { get; set; } = "";     // 구매요청자

        public int Order_Num_Seq { get; set; } = 0;   // 기본값 0
    }

    internal sealed class MroPurchaseStatusRow
    {
        public string MRO_Code { get; set; } = "";      // 엑셀: BQMS코드
        public string P_N { get; set; } = "";           // 엑셀: [제조사] 품명
        public string Order_Num { get; set; } = "";     // 엑셀: P/O No.
        public DateTime? Order_Date { get; set; }       // 엑셀: P/O일자
        public DateTime? Arrival_Date { get; set; }     // 엑셀: G/R일
        public decimal? Order_Qty { get; set; }         // 엑셀: P/O수량
        public string Buyer { get; set; } = "";         // 엑셀: 구매담당자

        public int Order_Num_Seq { get; set; } = 0;     // 그룹별 순번 (PK 일부)
    }
    
    internal static class ImportFileReader
    {
        internal static DateTime? ParseExcelDate(object v)
        {
            try
            {
                if (v == null) return null;
                          // Interop Value2가 OLE Automation Date(double)일 수 있음
                if (v is double d) return DateTime.FromOADate(d);
                if (v is DateTime dt) return dt;
                if (DateTime.TryParse(v.ToString(), out var parsed)) return parsed;
                return null;
            }
            catch { return null; }
        }

        internal static decimal? ParseDecimal(object v)
        {
            try
            {
                if (v == null) return null;
                if (v is double d) return (decimal)d;
                if (decimal.TryParse(v.ToString(), out var dec)) return dec;
                return null;
            }
            catch { return null; }
        }

        internal static object[,] LoadTsvToObjectArray(
            string filePath,
            out int rowCount,
            out int colCount,
            Encoding encoding = null)
        {
            // C# 7.3 호환: encoding ??= Encoding.UTF8; 대신
            if (encoding == null) encoding = Encoding.UTF8;

            var lines = File.ReadAllLines(filePath, encoding);
            if (lines.Length == 0)
            {
                rowCount = colCount = 0;
                return new object[1, 1];
            }

            var headerCols = ParseTsvLine(lines[0]);
            colCount = headerCols.Count;
            rowCount = lines.Length;

            var data = new object[rowCount + 1, colCount + 1];

            for (int r = 0; r < lines.Length; r++)
            {
                var cols = ParseTsvLine(lines[r]);

                for (int c = 0; c < colCount; c++)
                {
                    string v = (c < cols.Count) ? cols[c] : "";
                    data[r + 1, c + 1] = string.IsNullOrWhiteSpace(v) ? null : (object)v;
                }
            }

            return data;
        }

        internal static List<string> ParseTsvLine(string line)
        {
            var result = new List<string>();
            if (line == null) return result;

            var sb = new StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char ch = line[i];

                if (ch == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                    continue;
                }

                if (ch == '\t' && !inQuotes)
                {
                    result.Add(sb.ToString());
                    sb.Clear();
                    continue;
                }

                sb.Append(ch);
            }

            result.Add(sb.ToString());
            return result;
        }
    }
}
