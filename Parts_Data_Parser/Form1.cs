using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace log_parser
{
    public partial class Form1 : Form
    {
        private string connectionString =
            "Server=127.0.0.1;Port=3306;Database=plc_log;User ID=root;Password=1234;" +
            "SslMode=None;AllowPublicKeyRetrieval=True;Connection Timeout=15;Default Command Timeout=30;" +
            "Allow User Variables=true;";

        private NotifyIcon notifyIcon;
        private readonly PartsImportRunner importRunner;

        public Form1()
        {
            InitializeComponent();

            var settings = new ParserSettings(connectionString);
            Logger.Initialize(settings.LogDirectory, settings.StatusUpdateLogDirectory);

            var stock = new StockService(settings.ConnectionString);
            stock.EssentialStockLoaded += lines => UpdateStockList(listBox2, label4, "Q-Code 갯수", lines);
            stock.ParsedStockLoaded += lines => UpdateStockList(listBox3, label5, "부품 갯수", lines);
            stock.MatchingStockLoaded += lines => UpdateStockList(listBox4, null, null, lines);
            importRunner = new PartsImportRunner(settings, stock);

            button1.Click += new EventHandler(Button1_Click);
            button2.Click += new EventHandler(Button2_Click);
            InitializeNotifyIcon();
            this.Load += new EventHandler(Form1_Load);
        }

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

        private void Form1_Load(object sender, EventArgs e)
        {
            Button1_Click(sender, e); // 프로그램이 실행될 때 Button1_Click을 호출하여 시작
            this.WindowState = FormWindowState.Minimized; // 창을 최소화
            notifyIcon.Visible = true; // notifyIcon 보이기
            this.ShowInTaskbar = false; // 폼 숨기기
        }

        private void OpenApp(object sender, EventArgs e)
        {
            this.Show();
            this.WindowState = FormWindowState.Normal;
        }

        private void ExitApp(object sender, EventArgs e)
        {
            notifyIcon.Dispose();
            Application.Exit();
        }

        private void Button1_Click(object sender, EventArgs e)
        {
            label6.Text = "Running";
            label6.ForeColor = Color.Green;
            Logger.Info("고정 경로 파싱 모드 시작.");

            System.Threading.Tasks.Task.Run(() =>
            {
                importRunner.Run();

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

        private void Button2_Click(object sender, EventArgs e)
        {
            Logger.Info("작업 중지됨 (감시자 없음)");
            label6.Text = "Stop";
            label6.ForeColor = Color.Red;
        }

        // 서비스는 화면 컨트롤을 참조하지 않고, Form1에서만 UI 스레드로 전달한다.
        private void UpdateStockList(ListBox listBox, Label countLabel, string countText, IReadOnlyList<string> lines)
        {
            this.Invoke((MethodInvoker)(() =>
            {
                listBox.Items.Clear();
                foreach (string line in lines)
                    listBox.Items.Add(line);
                if (countLabel != null)
                    countLabel.Text = $"{countText}: {lines.Count}";
            }));
        }
    }
}
