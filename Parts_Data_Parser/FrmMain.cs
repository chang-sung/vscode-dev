using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using Excel = Microsoft.Office.Interop.Excel;

namespace log_parser
{
    public partial class Form1 : Form
    {
        private string connectionString =
            "Server=127.0.0.1;Port=3306;Database=plc_log;User ID=root;Password=1234;" +
            "SslMode=None;AllowPublicKeyRetrieval=True;Connection Timeout=15;Default Command Timeout=30;" +
            "Allow User Variables=true;";
        private List<string> mysqlQCodes = new List<string>(); // MySQL에서 가져온 Q-Code를 저장하는 리스트
        private Dictionary<string, string> excelData = new Dictionary<string, string>(); // Excel에서 가져온 Q-Code와 현재잔고 저장하는 딕셔너리
        private NotifyIcon notifyIcon;
        // 로그 파일 설정
        private readonly string _logDir = @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\Logs";
        private readonly string StatusUpdateLogDir = @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\Logs\Logs_Purchase_Status_Update";

        private readonly string STOCK_PATH =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_stock_list\STOCK.csv";
        private readonly string PRICE_PATH =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_price_list\price.xlsx";
        private readonly string PURCHASE_EMS_PATH =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_purchase_status_list\EMS.csv";
        private readonly string MRO_PATH =
            @"D:\25_Project\02_C#\00_Parts_Log_Parser\EMS_Log\parts_ems_purchase_status_mro_list\MRO.xlsx";

        private void InitializeNotifyIcon()
        {
            notifyIcon = new NotifyIcon
            {
                Icon = SystemIcons.Application, // 기본 시스템 아이콘 사용
                Visible = true,
                Text = "Log Parser" // 트레이 아이콘에 마우스를 올렸을 때 표시할 텍스트
            };

            // 마우스 오른쪽 클릭 메뉴 설정
            ContextMenuStrip contextMenu = new ContextMenuStrip();
            contextMenu.Items.Add("Open", null, OpenApp); // 열기 메뉴
            contextMenu.Items.Add("Exit", null, ExitApp); // 종료 메뉴
            notifyIcon.ContextMenuStrip = contextMenu;

            // 아이콘을 더블 클릭했을 때 창을 다시 열도록 설정
            notifyIcon.DoubleClick += OpenApp;
        }

        private DateTime? ParseExcelDate(object v)
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

        private decimal? ParseDecimal(object v)
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

        // CSV 데이터 추출 공용 함수
        private static object[,] LoadTsvToObjectArray(
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

        // 탭(\t) 구분 + 따옴표 처리
        private static List<string> ParseTsvLine(string line)
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


        private sealed class PriceRow
        {
            public string Q_Code { get; set; } = "";  // 부품
            public string P_N { get; set; } = "";     // 업체P/N
            public string Site { get; set; } = "";    // 출고창고
            public decimal? Price { get; set; }       // 결산단가
            public string Parts_Div { get; set; } = "";// EMS=부품명, MRO=[제조사] 품명
        }

        private sealed class PurchaseStatusRow
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

        private sealed class MroPurchaseStatusRow
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

        public Form1()
        {
            InitializeComponent();

            Logger.Initialize(_logDir, StatusUpdateLogDir);

            button1.Click += new EventHandler(Button1_Click);
            button2.Click += new EventHandler(Button2_Click);

            // NotifyIcon 초기화
            InitializeNotifyIcon();
     

            // 프로그램 시작 시 Button1_Click 메서드 자동 호출
            this.Load += new EventHandler(Form1_Load);
        }

        // 프로그램이 로드될 때 자동으로 시작
        private void Form1_Load(object sender, EventArgs e)
        {
            Button1_Click(sender, e); // 프로그램이 실행될 때 Button1_Click을 호출하여 시작
            this.WindowState = FormWindowState.Minimized; // 창을 최소화
            notifyIcon.Visible = true; // notifyIcon 보이기
            this.ShowInTaskbar = false; // 폼 숨기기
        }

       

        // 창을 다시 열기
        private void OpenApp(object sender, EventArgs e)
        {
            this.Show();
            this.WindowState = FormWindowState.Normal;
        }

        // 종료 메뉴 클릭 시
        private void ExitApp(object sender, EventArgs e)
        {
            notifyIcon.Dispose();
            Application.Exit();
        }


        // ---------- 공용 실행 헬퍼 ----------
        private void ProcessIfExists(string title, string path, Action<string> run)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                Logger.Info($"[{title}] 파일이 없습니다: {path}");
                return;
            }

            try
            {
                Logger.Info($"[{title}] 처리 시작: {path}");
                run(path); // 기존 파싱 함수 실행
                Logger.Info($"[{title}] 처리 완료");
            }
            catch (Exception ex)
            {
                Logger.Info($"[{title}] 처리 중 오류: {ex.Message}");
            }
        }

        // ---------- 시작 버튼 ----------
        private void Button1_Click(object sender, EventArgs e)
        {
            label6.Text = "Running";
            label6.ForeColor = Color.Green;
            Logger.Info("고정 경로 파싱 모드 시작.");

            System.Threading.Tasks.Task.Run(() =>
            {
                // 수동 Drop 상태는 파일 유무 및 동기화 결과와 관계없이 실행 시작 시 기록
                LogTodayDropPurchases();

                // 1) 재고
                ProcessIfExists("재고(EMS Stock)", STOCK_PATH, ExecuteStockProcess);

                // 2) 가격
                ProcessIfExists("가격(Price)", PRICE_PATH, ExecutePriceProcess);

                // 3) 구매상태 EMS (P1DD, P1DJ)
                ProcessIfExists("[EMS] 구매상태", PURCHASE_EMS_PATH, ExecuteEmsPurchaseStatusProcess);

                // 4) 구매상태 MRO
                ProcessIfExists("[MRO] 구매상태", MRO_PATH, ExecuteMroPurchaseStatusProcess);

                // ===================== 추가: 동기화 및 상태승급 수행
                try
                {
                    SyncEmsToPurchaseList();
                    SyncMroToPurchaseList();

                    Logger.Info("[안전재고 자동구매] 검사 시작");

                    // 안전재고 기준 자동 구매요청
                    AutoCreatePurchaseRequestBySafetyStock();
                    Logger.Info("[안전재고 자동구매] 검사 종료");
                }
                catch { /* 각 메서드에서 로그 출력됨 */ }

                // ✅ 모든 파일 처리 완료 후 로그 + 자동 종료
                Logger.Info("모든 파일 + 동기화/상태승급 완료. 프로그램을 종료합니다.");

                System.Threading.Thread.Sleep(2000);

                this.Invoke((MethodInvoker)(() =>
                {
                    notifyIcon?.Dispose();
                    Application.Exit();
                }));
            });
        }


        // ---------- 종료 버튼 ----------
        private void Button2_Click(object sender, EventArgs e)
        {
            Logger.Info("작업 중지됨 (감시자 없음)");
            label6.Text = "Stop";
            label6.ForeColor = Color.Red;
        }


        // button1의 작업을 수행하는 메서드
        private void ExecuteStockProcess(string csvFilePath)
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

        private void ExecutePriceProcess(string excelFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var priceRows = ParsePriceExcelToRows(excelFilePath);

            // 2) DB 저장: 기존 데이터 전체 DELETE → INSERT
            var snapshotTime = DateTime.Now;
            SavePriceRowsToDb(priceRows, snapshotTime);

            Logger.Info("가격 리스트 처리 완료");
        }

        private void ExecuteEmsPurchaseStatusProcess(string csvFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var rows = ParseEmsPurchaseStatusExcelToRows(csvFilePath);

            // 2) DB 저장: DELETE 없이 INSERT만 (요청사항)
            var writeTime = DateTime.Now;
            SaveEmsPurchaseStatusRowsToDb(rows, writeTime);

            Logger.Info("[EMS] 구매 상태 리스트 처리 완료");
        }

        private void ExecuteMroPurchaseStatusProcess(string excelFilePath)
        {
            // 1) 엑셀 파싱 (전체 행)
            var rows = ParseMroPurchaseStatusExcelToRows(excelFilePath);

            // 2) DB 저장: INSERT-only (No는 AUTO_INCREMENT)
            var writeTime = DateTime.Now;
            SaveMroPurchaseStatusRowsToDb(rows, writeTime);

            Logger.Info("[MRO] 구매 상태 리스트 처리 완료");
        }


        // MySQL 데이터베이스에서 데이터 가져오기
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
                        this.Invoke((MethodInvoker)delegate
                        {
                            listBox2.Items.Clear(); // Q_Code 및 Count_Now ListBox 초기화
                        });

                        mysqlQCodes.Clear(); // MySQL Q-Code 리스트 초기화

                        while (reader.Read())
                        {
                            string qCode = reader["Q_Code"].ToString();
                            string countNow = reader["Count_Now"].ToString();
                            mysqlQCodes.Add(qCode);

                            this.Invoke((MethodInvoker)delegate
                            {
                                listBox2.Items.Add($"{qCode} - {countNow}");
                            });
                        }

                                         // Q-Code 갯수를 label4에 출력
                        this.Invoke((MethodInvoker)delegate
                        {
                            label4.Text = $"Q-Code 갯수: {mysqlQCodes.Count}";
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    Logger.Error($"MySQL에서 데이터를 읽는 중 오류가 발생했습니다: {ex.Message}");
                });
            }
        }

        // Excel 파일에서 '부품'과 '현재잔고' 컬럼의 값을 가져와 ListBox3에 출력
        private void LoadQCodeAndBalanceFromExcel(string csvFilePath)
        {
            try
            {
                int rowCount, colCount;
                object[,] data = LoadTsvToObjectArray(csvFilePath, out rowCount, out colCount);

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

                // 부품 값 갯수를 label5에 출력
                this.Invoke((MethodInvoker)(() =>
                {
                    listBox3.Items.Clear();
                    foreach (var kv in excelData)
                        listBox3.Items.Add($"{kv.Key} - 현재잔고 합계: {kv.Value}");

                    label5.Text = $"부품 갯수: {excelData.Count}";                    
                }));
            }
            catch (Exception ex)
            {
                Logger.Info($"[STOCK] TSV 처리 오류: {ex.Message}");
            }
        }

        // MySQL과 Excel의 동일한 Q-Code와 현재잔고를 ListBox4에 출력
        private void DisplayMatchingQCodesAndBalances()
        {
                     // UI 업데이트를 위해 Invoke 사용
            this.Invoke((MethodInvoker)delegate
            {
                listBox4.Items.Clear(); // ListBox4 초기화
            });

            var matchingQCodes = mysqlQCodes.Intersect(excelData.Keys); // MySQL과 Excel의 Q-Code 비교

            foreach (var qCode in matchingQCodes)
            {
                var balance = excelData[qCode];

                             // UI 업데이트를 위해 Invoke 사용
                this.Invoke((MethodInvoker)delegate
                {
                    listBox4.Items.Add($"{qCode} - 현재잔고: {balance}"); // 일치하는 Q-Code와 현재잔고를 ListBox4에 추가
                });
            }
        }

        // MySQL 데이터베이스에서 parts_essential_list 테이블의 Count_Now 컬럼을 현재잔고 값으로 업데이트
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
                    this.Invoke((MethodInvoker)delegate
                    {
                        Logger.Info("필수 파츠 재고 업데이트 완료!");
                    });
                }
            }
            catch (Exception ex)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    Logger.Error($"MySQL 업데이트 중 오류: {ex.Message}");
                });
            }
        }

        // EMS 전체 파츠 LIST 업데이트
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

                this.Invoke((MethodInvoker)(() =>
                {
                    Logger.Info($"parts_ems_stock_list(교집합만) 저장 완료: {keys.Count}행 (WriteTime={writeTime:yyyy-MM-dd HH:mm:ss})");
                }));
            }
            catch (Exception ex)
            {
                Logger.Info($"parts_ems_stock_list 저장 중 오류: {ex.Message}");
            }
        }


        // 파츠 가격 LIST 업데이트
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


        // 파츠 구매 상태 LIST INSERT
        private List<PurchaseStatusRow> ParseEmsPurchaseStatusExcelToRows(string csvFilePath)
        {
            var rows = new List<PurchaseStatusRow>();

            try
            {
                // ✅ TSV 로딩 (헤더 포함)
                int rowCount, colCount;
                object[,] data = LoadTsvToObjectArray(csvFilePath, out rowCount, out colCount);

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

                    decimal? qty = ParseDecimal(data[r, idxOrderQty]);
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

        // MRO 구매 상태 LIST INSERT

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

                    decimal? qty = ParseDecimal(data[r, idxOrderQty]);
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
                        Order_Date = (idxOrderDate > 0) ? ParseExcelDate(data[r, idxOrderDate]) : null,
                        Arrival_Date = (idxArrivalDate > 0) ? ParseExcelDate(data[r, idxArrivalDate]) : null,
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

        // ============================== 구매 진행 현황 업데이트 (상태 업데이트)
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

        // ============================== EMS → Purchase 동기화
        private void SyncEmsToPurchaseList()
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

        // ============================== MRO → Purchase 동기화
        private void SyncMroToPurchaseList()
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

        // 자동 구매 요청 로직
        private void AutoCreatePurchaseRequestBySafetyStock()
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

        // 사람이 수동으로 Drop 처리한 오늘 요청 건만 기록 (DB 상태 변경 없음)
        private void LogTodayDropPurchases()
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

        // No는 수동 변경될 수 있으므로 구매 건의 7개 값만으로 영속 중복 키를 만든다.
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