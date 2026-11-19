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

namespace MotionKit {
  public partial class frmMain : Form {
    private IO io;
    private Machine machine;
    private LogHelper log;

    private readonly object logLock = new object();
    private bool initialized;

    private Action<string> showMessage;

    public frmMain() {
      InitializeComponent();

      showMessage += DisplayMessage;

    }

    private void frmMain_Load(object sender, EventArgs e)
    {
        if (initialized)
            return;

        initialized = true;

        log = new LogHelper(Global.LogPath);
        io = IO.GetInstance();
        io.OnMsgEvent += WriteLog;

        if (!io.BoardInit())
            return;

        // 기존 Machine.Create()에 있던 초기화 대기 시간
        System.Threading.Thread.Sleep(500);

        if (!io.BoardRun())
            return;

        machine = new Machine();

        machine.XAxis.OnMsgEvent += WriteLog;
        machine.YAxis.OnMsgEvent += WriteLog;

        machine.Create();

        machine.XAxis.ServoOn();
        machine.YAxis.ServoOn();

        tmUIUpdate.Start();
    }

    private void tmUIUpdate_Tick(object sender, EventArgs e)
    {
        if (machine == null)
            return;

        LampColorUpdate();
        EqpStatusColorUpdate();
    }

    void LampColorUpdate()
    {
        lbLampRed.BackColor = machine.LampRed.IsOn() ? Color.Red : Color.LightGray;
        lbLampYellow.BackColor = machine.LampYellow.IsOn() ? Color.Yellow : Color.LightGray;
        lbLampGreen.BackColor = machine.LampGreen.IsOn() ? Color.Green : Color.LightGray;
    }

    void EqpStatusColorUpdate()
    {
        lbProcessStatus_Value.Text = machine.eqpStatus.ToString();
        switch (machine.eqpStatus)
        {
            case EqpStatus.Ready:
                lbProcessStatus_Value.BackColor = Color.Brown;
                break;
            case EqpStatus.Stop:
                lbProcessStatus_Value.BackColor = Color.Blue;
                break;
            case EqpStatus.Error:
                lbProcessStatus_Value.BackColor = Color.Red;
                break;
            case EqpStatus.Run:
                lbProcessStatus_Value.BackColor = Color.Green;
                break;
            case EqpStatus.Idle:
                lbProcessStatus_Value.BackColor = Color.Yellow;
                break;
        }
    }


    // 서보 창의 열림 여부와 관계없이 로그 기록
    private void WriteLog(string msg) {
      lock (logLock) {
        log.Write(msg);
      }

      showMessage?.Invoke(msg);
    }

    private void btnServo_Click(object sender, EventArgs e) {
      if (machine == null)
        return;

      if (ActivateOpenForm<frmServo>())
        return;

      frmServo form = new frmServo(machine);
      form.Show(this);
    }

    private void btnIO_Click(object sender, EventArgs e) {
      if (ActivateOpenForm<frmIO>())
        return;

      frmIO form = new frmIO();
      form.Show(this);
    }

    private void btnHome_Click(object sender, EventArgs e) {

    }

    private void btnVision_Click(object sender, EventArgs e) {
      foreach (Form item in Application.OpenForms) {

        if (item.Text == "frmVision") {
          return;
        }
      }

      frmVision fm = new frmVision();
      fm.Show();
    }

    // 같은 종류의 창이 열려 있으면 앞으로 가져오기
    private bool ActivateOpenForm<T>() where T : Form {
      foreach (Form form in Application.OpenForms) {
        if (form is T) {
          if (form.WindowState == FormWindowState.Minimized) {
            form.WindowState = FormWindowState.Normal;
          }

          form.Activate();
          return true;
        }
      }

      return false;
    }

    private void DisplayMessage(string msg) {
      if (IsDisposed || Disposing || !IsHandleCreated)
        return;

      if (InvokeRequired) {
        try {
          BeginInvoke(new Action<string>(DisplayMessage), msg);
        }
        catch (InvalidOperationException) {
          // 폼을 닫는 중이면 표시하지 않음
        }

        return;
      }

      tbMsg.AppendText(
          $"{DateTime.Now:HH:mm:ss} : {msg}\r\n");
    }


    protected override void OnFormClosed(FormClosedEventArgs e) {
      if (io != null) {
        io.OnMsgEvent -= WriteLog;
      }

      if (machine != null) {
        machine.XAxis.OnMsgEvent -= WriteLog;
        machine.YAxis.OnMsgEvent -= WriteLog;
      }

      base.OnFormClosed(e);
    }

    private void btnOneCycleRun_Click(object sender, EventArgs e) {
      machine?.CmdOneCycleStart();
    }

    private void btnInit_Click(object sender, EventArgs e) {
      machine?.CmdInit();
    }

 
  }
}