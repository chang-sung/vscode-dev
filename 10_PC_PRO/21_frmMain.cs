using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Helper;

namespace MotionKit
{
    public partial class frmMain : Form
    {
        private IO io;
        private Machine machine;
        private LogHelper log;

        private readonly object logLock = new object();
        private bool initialized;

        private Action<string> showMessage;
    
        public frmMain()
        {
            InitializeComponent();

            showMessage += DisplayMessage;

            // 디자이너에 연결되어 있어도 중복 실행되지 않도록 처리
            Load -= frmMain_Load;
            Load += frmMain_Load;

            btnIO.Click -= btnIO_Click;
            btnIO.Click += btnIO_Click;

            btnHome.Click -= btnHome_Click;
            btnHome.Click += btnHome_Click;

            btnServo.Click -= btnServo_Click;
            btnServo.Click += btnServo_Click;

        }

        private void frmMain_Load(object sender, EventArgs e)
        {
            if (initialized)
                return;

            initialized = true;

            log = new LogHelper(Global.LogPath);

            io = IO.GetInstance();
            io.OnMsgEvent += WriteLog;

            // 기존 프로젝트의 초기화 흐름 유지
            if (io.BoardInit() == false || io.BoardRun() == false)
            {
                // 연결 실패 시 필요한 처리
            }

            io.Start();

            machine = new Machine();

            machine.XAxis.OnMsgEvent += WriteLog;
            machine.YAxis.OnMsgEvent += WriteLog;

            // 프로그램 시작 시 X축, Y축 서보 ON
            machine.XAxis.ServoOn();
            machine.YAxis.ServoOn();
        }

        // 서보 창의 열림 여부와 관계없이 로그 기록
        private void WriteLog(string msg)
        {
            lock (logLock)
            {
                log.Write(msg);
            }

            showMessage?.Invoke(msg);
        }

        private void btnServo_Click(object sender, EventArgs e)
        {
            if (machine == null)
                return;

            if (ActivateOpenForm<frmServo>())
                return;

            frmServo form = new frmServo(machine);
            form.Show(this);
        }

        private void btnIO_Click(object sender, EventArgs e)
        {
            if (ActivateOpenForm<frmIO>())
                return;

            frmIO form = new frmIO();
            form.Show(this);
        }

        private void btnHome_Click(object sender, EventArgs e)
        {
            if (machine == null)
                return;

            // 기존 코드와 동일하게 비동기로 전체 원점복귀
            Func<bool> home = machine.Home;
            home.BeginInvoke(null, null);
        }

        // 같은 종류의 창이 열려 있으면 앞으로 가져오기
        private bool ActivateOpenForm<T>() where T : Form
        {
            foreach (Form form in Application.OpenForms)
            {
                if (form is T)
                {
                    if (form.WindowState == FormWindowState.Minimized)
                    {
                        form.WindowState = FormWindowState.Normal;
                    }

                    form.Activate();
                    return true;
                }
            }

            return false;
        }

        private void DisplayMessage(string msg)
        {
            if (IsDisposed || Disposing || !IsHandleCreated)
                return;

            if (InvokeRequired)
            {
                try
                {
                    BeginInvoke(new Action<string>(DisplayMessage), msg);
                }
                catch (InvalidOperationException)
                {
                    // 폼을 닫는 중이면 표시하지 않음
                }

                return;
            }

            tbMsg.AppendText(
                $"{DateTime.Now:HH:mm:ss} : {msg}\r\n");
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            if (io != null)
            {
                io.OnMsgEvent -= WriteLog;
            }

            if (machine != null)
            {
                machine.XAxis.OnMsgEvent -= WriteLog;
                machine.YAxis.OnMsgEvent -= WriteLog;
            }

            base.OnFormClosed(e);
        }
    }
}