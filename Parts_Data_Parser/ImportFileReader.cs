using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace log_parser
{
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
