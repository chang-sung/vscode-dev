using Helper;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MotionKit {
  public partial class frmServo : Form {
    private readonly Machine machine;
    private IO io;

    private readonly Timer servoTimer = new Timer();
    private bool eventsSubscribed;

    // WinForms 디자이너용 생성자
    public frmServo() {
      InitializeComponent();
    }

    // 실제 프로그램에서 사용하는 생성자
    public frmServo(Machine machine) : this() {
      if (machine == null) {
        throw new ArgumentNullException("machine");
      }

      this.machine = machine;
      
      CreateAxisMonitor();
      CreateWorkMoveTeachPos();

      servoTimer.Interval = 100;
      servoTimer.Tick += tmUIUpdate_Tick;
      servoTimer.Start();

    }


    protected override void OnLoad(EventArgs e) {
      base.OnLoad(e);

      if (machine == null || eventsSubscribed)
        return;

      io = IO.GetInstance();

      io.OnMsgEvent += Io_OnMsgEvent;
      machine.XAxis.OnMsgEvent += Io_OnMsgEvent;
      machine.YAxis.OnMsgEvent += Io_OnMsgEvent;

      eventsSubscribed = true;
      servoTimer.Start();
      
    }

    // 메시지를 UI 스레드에서 표시
    public void Io_OnMsgEvent(string msg) {
      if (IsDisposed || Disposing || !IsHandleCreated)
        return;

      if (InvokeRequired) {
        try {
          BeginInvoke(new Action<string>(Io_OnMsgEvent), msg);
        }
        catch (InvalidOperationException) {
          // 창을 닫는 순간 들어온 메시지는 표시하지 않음
        }

        return;
      }

      tbMsg.AppendText(
          $"{DateTime.Now:HH:mm:ss} : {msg}\r\n");
    }

    private void CreateAxisMonitor() {
      lvXMonitor.Items.Clear();
      lvYMonitor.Items.Clear();

      for (AxisItems item = 0; item < AxisItems.MAX; item++) {
        ListViewItem xItem = new ListViewItem(item.ToString());
        xItem.SubItems.Add("0.00");
        lvXMonitor.Items.Add(xItem);

        ListViewItem yItem = new ListViewItem(item.ToString());
        yItem.SubItems.Add("0.00");
        lvYMonitor.Items.Add(yItem);
      }

      for (AxisMap item = 0; item < AxisMap.MAX; item++) {
        ListViewItem xItem = new ListViewItem(item.ToString());
        xItem.SubItems.Add("False");
        lvXMonitor.Items.Add(xItem);

        ListViewItem yItem = new ListViewItem(item.ToString());
        yItem.SubItems.Add("False");
        lvYMonitor.Items.Add(yItem);
      }
    }

    private void CreateWorkMoveTeachPos() {
      cbFrom.Items.Clear();
      cbTo.Items.Clear();

      for (TeachList item = 0; item < TeachList.Max; item++) {
        cbFrom.Items.Add(item);
        cbTo.Items.Add(item);
      }

      if (cbFrom.Items.Count > 0) {
        cbFrom.SelectedIndex = 0;
        cbTo.SelectedIndex = 0;
      }
    }

    void UpdateAxisMonitor() {
      //CmdVel, ActVel, CmdPos, ActPos
      lvXMonitor.Items[(int)AxisItems.CmdVel].SubItems[1].Text
        = machine.XAxis.Info.CmdVel.ToString("F2");
      lvXMonitor.Items[(int)AxisItems.ActVel].SubItems[1].Text
        = machine.XAxis.Info.ActVel.ToString("F2");
      lvXMonitor.Items[(int)AxisItems.CmdPos].SubItems[1].Text
        = machine.XAxis.Info.CmdPos.ToString("F2");
      lvXMonitor.Items[(int)AxisItems.ActPos].SubItems[1].Text
        = machine.XAxis.Info.ActPos.ToString("F2");

      lvYMonitor.Items[(int)AxisItems.CmdVel].SubItems[1].Text
        = machine.YAxis.Info.CmdVel.ToString("F2");
      lvYMonitor.Items[(int)AxisItems.ActVel].SubItems[1].Text
        = machine.YAxis.Info.ActVel.ToString("F2");
      lvYMonitor.Items[(int)AxisItems.CmdPos].SubItems[1].Text
        = machine.YAxis.Info.CmdPos.ToString("F2");
      lvYMonitor.Items[(int)AxisItems.ActPos].SubItems[1].Text
        = machine.YAxis.Info.ActPos.ToString("F2");

      int idx = 4;
      for (AxisMap i = 0; i < AxisMap.MAX; i++) {
        lvXMonitor.Items[idx].SubItems[1].Text
          = machine.XAxis.Info[i].ToString();
        lvYMonitor.Items[idx].SubItems[1].Text
          = machine.YAxis.Info[i].ToString();
      }

      cbsXHome.Checked = machine.XAxis.Info[AxisMap.HomeAbsSwitch];
      cbsXNegLimit.Checked = machine.XAxis.Info[AxisMap.LimitSwitchNeg];
      cbsXPosLimit.Checked = machine.XAxis.Info[AxisMap.LimitSwitchPos];
      cbsYHome.Checked = machine.YAxis.Info[AxisMap.HomeAbsSwitch];
      cbsYNegLimit.Checked = machine.YAxis.Info[AxisMap.LimitSwitchNeg];
      cbsYPosLimit.Checked = machine.YAxis.Info[AxisMap.LimitSwitchPos];
    }

    private void tmUIUpdate_Tick(object sender, EventArgs e) {
      UpdateAxisMonitor();
    }
    // Work Move
    private void btnMove_Click(object sender, EventArgs e) {
      TeachList from;
      TeachList to;

      if (!Enum.TryParse<TeachList>(cbFrom.Text, out from) ||
          !Enum.TryParse<TeachList>(cbTo.Text, out to) ||
          !cbFrom.Items.Contains(from) ||
          !cbTo.Items.Contains(to)) {
            MessageBox.Show(this, "출발 위치와 도착 위치를 선택해 주세요.");
            return;
      }

      Func<TeachList, TeachList, bool> move = machine.WorkMove;
      move.BeginInvoke(from, to, null, null);
    }

    private void btnParam_Click(object sender, EventArgs e) {
      if (machine == null)
        return;

      if (ActivateOpenForm<frmParam>())
        return;

      frmParam form = new frmParam(
          machine.XAxis.Param,
          machine.YAxis.Param);

      form.Show(this);
    }

    private void btnTeach_Click(object sender, EventArgs e) {
      if (machine == null)
        return;

      if (ActivateOpenForm<frmTeach>())
        return;

      frmTeach form = new frmTeach(machine);
      form.Show(this);
    }

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

    protected override void OnFormClosed(FormClosedEventArgs e) {
      servoTimer.Stop();

      if (eventsSubscribed) {
        io.OnMsgEvent -= Io_OnMsgEvent;
        machine.XAxis.OnMsgEvent -= Io_OnMsgEvent;
        machine.YAxis.OnMsgEvent -= Io_OnMsgEvent;

        eventsSubscribed = false;
      }

      servoTimer.Tick -= tmUIUpdate_Tick;
      servoTimer.Dispose();

      base.OnFormClosed(e);
    }

    private void btnHome_Click(object sender, EventArgs e) {
      if (machine == null)
        return;

      Func<bool> home = machine.Home;
      home.BeginInvoke(null, null);

    }
    #region X, Y축 제어
    // X축 제어
    private void btnXOn_Click(object sender, EventArgs e) {
      machine.XAxis.ServoOn();
    }

    private void btnXOff_Click(object sender, EventArgs e) {
      machine.XAxis.ServoOff();
    }

    private void btnX_EStop_Click(object sender, EventArgs e) {
      machine.XAxis.ServoStop();
    }

    private void btnXReset_Click(object sender, EventArgs e) {
      machine.XAxis.ServoReset();
    }

    private void btnXHome_Click(object sender, EventArgs e) {
      machine.XAxis.ServoHome();
    }

    // Y축 제어
    private void btnYOn_Click(object sender, EventArgs e) {
      machine.YAxis.ServoOn();
    }

    private void btnYOff_Click(object sender, EventArgs e) {
      machine.YAxis.ServoOff();
    }

    private void btnY_EStop_Click(object sender, EventArgs e) {
      machine.YAxis.ServoStop();
    }

    private void btnYReset_Click(object sender, EventArgs e) {
      machine.YAxis.ServoReset();
    }

    private void btnYHome_Click(object sender, EventArgs e) {
      machine.YAxis.ServoHome();
    }
    #endregion

    #region 실린더, 램프 메뉴얼 점등
    private void btnLoadCylFwd_Click(object sender, EventArgs e) {
      machine.LoadCyl.On();
    }

    private void btnLoadCylBwd_Click(object sender, EventArgs e) {
      machine.LoadCyl.Off();
    }

    private void btnWorkCylFwd_Click(object sender, EventArgs e) {
      machine.WorkCyl.On();
    }

    private void btnWorkCylBwd_Click(object sender, EventArgs e) {
      machine.WorkCyl.Off();
    }

    private void btnMoveCylFwd_Click(object sender, EventArgs e) {
      machine.MoveCyl.On();
    }

    private void btnMoveCylBwd_Click(object sender, EventArgs e) {
      machine.MoveCyl.Off();
    }

    private void btnUnldCylFwd_Click(object sender, EventArgs e) {
      machine.UnldCyl.On();
    }

    private void btnUnldCylBwd_Click(object sender, EventArgs e) {
      machine.UnldCyl.Off();
    }

    private void btnRedOn_Click(object sender, EventArgs e) {
      machine.LampRed.On();
    }

    private void btnRedOff_Click(object sender, EventArgs e) {
      machine.LampRed.Off();
    }

    private void btnYellowOn_Click(object sender, EventArgs e) {
      machine.LampYellow.On();
    }

    private void btnYellowOff_Click(object sender, EventArgs e) {
      machine.LampYellow.Off();
    }

    private void btnGreenOn_Click(object sender, EventArgs e) {
      machine.LampGreen.On();
    }

    private void btnGreenOff_Click(object sender, EventArgs e) {
      machine.LampGreen.Off();
    }

    private void btnDrillOn_Click(object sender, EventArgs e) {
      machine.Drill.On();
    }

    private void btnDrillOff_Click(object sender, EventArgs e) {
      machine.Drill.Off();
    }

    private void btnConvyOn_Click(object sender, EventArgs e) {
      machine.Convy.On();
    }

    private void btnConvyOff_Click(object sender, EventArgs e) {
      machine.Convy.Off();
    }

    #endregion


  }
}