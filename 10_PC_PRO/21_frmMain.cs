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
    private frmVision vision;

    public frmMain() {
      InitializeComponent();
      UpdateVisionStatus(false);
      lbVision_Value.Text = "Disconnect";

      showMessage += DisplayMessage;

    }

    private void frmMain_Load(object sender, EventArgs e) {
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
      vision = new frmVision();
      vision.CameraConnectionChanged += Vision_CameraConnectionChanged;
      machine.ProductUnloadPositionSelected += Machine_ProductUnloadPositionSelected;
      
      machine.SearchColor = vision.Inspection;
      vision.StartLive();                        // 카메라 연결

      machine.InspectionCompleted += DisplayInspectionResult;
      

      machine.XAxis.OnMsgEvent += WriteLog;
      machine.YAxis.OnMsgEvent += WriteLog;

      machine.Create();

      machine.XAxis.ServoOn();
      machine.YAxis.ServoOn();

      tmUIUpdate.Start();
    }

    private void tmUIUpdate_Tick(object sender, EventArgs e) {
      if (machine == null)
        return;

      LampColorUpdate();
      EqpStatusColorUpdate();
      label1.Text = machine.ConvElapsedTime.ToString();
      lbProductOkValue.Text = machine.ProductOkCount.ToString();
      lbProductNgValue.Text = machine.ProductNgCount.ToString();
    }

    private void Vision_CameraConnectionChanged(object sender, EventArgs e)
    {
      if (IsDisposed || Disposing || !IsHandleCreated)
        return;

      if (InvokeRequired)
      {
        BeginInvoke(new Action(() =>
            Vision_CameraConnectionChanged(sender, e)));
        return;
      }

      UpdateVisionStatus(vision.IsCameraConnected);
    }

    private void Machine_ProductUnloadPositionSelected(
    DateTime inputTime,
    string inspectionResult,
    string unloadPositionName)
    {
      if (InvokeRequired)
      {
        BeginInvoke(new Action(() =>
            Machine_ProductUnloadPositionSelected(
                inputTime, inspectionResult, unloadPositionName)));
        return;
      }

      var item = new ListViewItem(inputTime.ToString("yyyy-MM-dd HH_mm_ss"));
      item.SubItems.Add(inspectionResult);
      item.SubItems.Add(unloadPositionName);

      livProcessInfo.Items.Add(item);
    }

    private void DisplayInspectionResult(Bitmap cropImage, bool detected)
    {
      if (IsDisposed || Disposing || !IsHandleCreated)
        return;

      if (InvokeRequired)
      {
        Invoke(new Action<Bitmap, bool>(DisplayInspectionResult),
               cropImage, detected);
        return;
      }

      Image previousImage = pbInspResult.Image;
      pbInspResult.Image = new Bitmap(cropImage);
      previousImage?.Dispose();

      lbInspResult.Text = detected ? "OK" : "NG";
    }

    void LampColorUpdate() {
      lbLampRed.BackColor = machine.LampRed.IsOn() ? Color.Red : Color.LightGray;
      lbLampYellow.BackColor = machine.LampYellow.IsOn() ? Color.Yellow : Color.LightGray;
      lbLampGreen.BackColor = machine.LampGreen.IsOn() ? Color.Green : Color.LightGray;
    }


    void EqpStatusColorUpdate()
    {
      EqpStatus currentStatus = machine.eqpStatus;
      lbProcessStatus_Value.Text = currentStatus.ToString();

      SetRunButtonsEnabled(
          machine.CanStartCycle || currentStatus == EqpStatus.Stop);

      switch (currentStatus)
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

    private void btnRecipe_Click(object sender, EventArgs e)
    {
      if (ActivateOpenForm<frmRecipe>())
        return;

      frmRecipe form = new frmRecipe();
      form.RecipeSelected += ApplySelectedRecipe;
      form.Show(this);
    }


    private void btnVision_Click(object sender, EventArgs e) {
      if (vision == null || vision.IsDisposed) return;

      vision.Show(this);
      vision.Activate();
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

    private void ApplySelectedRecipe(string recipeName)
    {
      if (vision == null || vision.IsDisposed)
      {
        MessageBox.Show(this, "Vision 초기화가 완료되지 않았습니다.");
        return;
      }

      string error;
      if (!vision.TryApplyRecipe(recipeName, out error))
      {
        MessageBox.Show(this, error);
        return;
      }

      lbRecipeValue.Text = recipeName;

      MessageBox.Show(
      this,
      $"{recipeName} 레시피가 선택되었습니다.",
      "레시피 선택",
      MessageBoxButtons.OK,
      MessageBoxIcon.Information);
    }


    protected override void OnFormClosed(FormClosedEventArgs e) {
      if (io != null) {
        io.OnMsgEvent -= WriteLog;
      }

      if (machine != null) {
        machine.XAxis.OnMsgEvent -= WriteLog;
        machine.YAxis.OnMsgEvent -= WriteLog;
        machine.InspectionCompleted -= DisplayInspectionResult;
      }
      Image lastImage = pbInspResult.Image;
      pbInspResult.Image = null;
      lastImage?.Dispose();

      base.OnFormClosed(e);
    }
    private void SetRunButtonsEnabled(bool enabled)
    {
      btnOneCycleRun.Enabled = enabled;
      btnAutoRun.Enabled = enabled;
    }



    private void btnOneCycleRun_Click(object sender, EventArgs e)
    {
      if (machine?.CmdOneCycleStart() == true)
        SetRunButtonsEnabled(false);
    }

    private void btnInit_Click(object sender, EventArgs e) {
      machine?.CmdInit();
    }

    private void btnAutoRun_Click(object sender, EventArgs e)
    {
      if (machine?.CmdContCycleStart() == true)
        SetRunButtonsEnabled(false);
    }

    private void btnCycleStop_Click(object sender, EventArgs e) {
      machine?.CmdCycleStop();

    }

    private void btnStop_Click(object sender, EventArgs e)
    {
      machine?.CmdEStop();
    }

    private void UpdateVisionStatus(bool isConnected)
    {
      lbVision_Value.Text = isConnected ? "Connect" : "Disconnect";
      lbVision_Value.ForeColor = isConnected ? Color.Blue : Color.Red;
    }


  }
}