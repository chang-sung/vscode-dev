using System;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Helper;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using CvRect = OpenCvSharp.Rect;

namespace MotionKit {
  public partial class frmVision : Form {
    // Day6 Form34: Mat, Rectangle, drawing을 이용한 ROI 처리
    private VideoCapture capture;
    private CancellationTokenSource liveCts;
    private Task liveTask;
    private readonly object frameLock = new object();
    private Mat src;                    // 카메라 원본 프레임 (OpenCV Mat)
    private Rectangle rect;             // 원본 이미지 기준 ROI
    private bool drawing = false;
    private bool hasRoi = false;
    private System.Drawing.Point startPoint;
    private int imageWidth, imageHeight;
    private bool closingAfterStop = false;

    public frmVision() {
      InitializeComponent();
      pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
      pictureBox2.SizeMode = PictureBoxSizeMode.Zoom;

      // 버튼 Click 이벤트는 디자이너에서 연결한 경우 다시 연결하지 않습니다.
      // btnRoiSave는 기존 frmVision.Designer.cs에서 Click 이벤트 연결 유지.

      btnLiveStop.Click += btnLiveStop_Click;
      pictureBox1.MouseDown += pictureBox1_MouseDown;
      pictureBox1.MouseMove += pictureBox1_MouseMove;
      pictureBox1.MouseUp += pictureBox1_MouseUp;
      pictureBox1.Paint += pictureBox1_Paint;
      FormClosing += frmVision_FormClosing;

      cbxColor.DropDownStyle = ComboBoxStyle.DropDownList;
      cbxLowUpper.DropDownStyle = ComboBoxStyle.DropDownList;

      cbxColor.DataSource = Enum.GetValues(typeof(ColorChannel));
      cbxLowUpper.DataSource = Enum.GetValues(typeof(ColorLevel));

      InitRoiListView();
      LoadRoiConfig();
      LoadColorConfig();       // Color_Config.txt에서 값 불러오기
      UpdateColorListView();   // 불러온 값을 리스트에 표시

      btnSave.Click += btnSave_Click;
      btnSearch.Click += btnSearch_Click;
    }

    private void InitRoiListView() {
      livRoiParam.View = View.Details;
      livRoiParam.GridLines = true;
      livRoiParam.FullRowSelect = true;
      livRoiParam.Items.Clear();
      string[] names = { "StartY", "EndY", "StartX", "EndX" };
      foreach (string name in names) {
        ListViewItem item = new ListViewItem(name);
        item.SubItems.Add("0");
        livRoiParam.Items.Add(item);
      }
    }

    private void btnLive_Click(object sender, EventArgs e) {
      if (liveTask != null) return;
      btnLive.Enabled = false;
      try {
        capture = new VideoCapture(0, VideoCaptureAPIs.DSHOW);
        if (!capture.IsOpened()) {
          MessageBox.Show("카메라 연결 실패");
          capture.Dispose();
          capture = null;
          return;
        }
        capture.Set(VideoCaptureProperties.FrameWidth, 1600);
        capture.Set(VideoCaptureProperties.FrameHeight, 1200);
        liveCts = new CancellationTokenSource();
        CancellationToken token = liveCts.Token;
        liveTask = Task.Run(() => CameraLoop(token));
      }
      catch (Exception ex) {
        MessageBox.Show(ex.Message);
        capture?.Dispose();
        capture = null;
      }
      finally { btnLive.Enabled = true; }
    }

    // 수업에서 사용한 Mat / ToBitmap()으로 프레임 처리
    private void CameraLoop(CancellationToken token) {
      using (Mat frame = new Mat()) {
        try {
          while (!token.IsCancellationRequested) {
            if (!capture.Read(frame) || frame.Empty()) {
              Thread.Sleep(30);
              continue;
            }
            // ROI Save는 src.Clone()으로 현재 프레임을 안전하게 복사
            lock (frameLock) {
              Mat old = src;
              src = frame.Clone();
              old?.Dispose();
            }
            Bitmap bitmap = frame.ToBitmap();
            try {
              // BeginInvoke 작업이 누적되지 않도록 UI 처리 완료를 기다림
              IAsyncResult result = pictureBox1.BeginInvoke(new Action(() =>
              {
                if (token.IsCancellationRequested || pictureBox1.IsDisposed) {
                  bitmap.Dispose();
                  return;
                }
                imageWidth = bitmap.Width;
                imageHeight = bitmap.Height;
                Image oldImage = pictureBox1.Image;
                pictureBox1.Image = bitmap;
                oldImage?.Dispose();
                pictureBox1.Invalidate();
              }));
              // 작업 취소 시에도 위 콜백이 Bitmap을 정리함
              while (!result.IsCompleted && !token.IsCancellationRequested)
                Thread.Sleep(5);
            }
            catch (InvalidOperationException) {
              bitmap.Dispose();
              break;
            }
            Thread.Sleep(33);
          }
        }
        catch (Exception ex) {
          System.Diagnostics.Debug.WriteLine(ex.Message);
        }
      }
    }

    private async void btnLiveStop_Click(object sender, EventArgs e) {
      await StopCameraAsync();
    }

    private async Task StopCameraAsync() {
      if (liveCts == null) return;
      btnLive.Enabled = false;
      btnLiveStop.Enabled = false;
      liveCts.Cancel();
      if (liveTask != null) await liveTask;
      capture?.Release();
      capture?.Dispose();
      capture = null;
      liveCts.Dispose();
      liveCts = null;
      liveTask = null;
      lock (frameLock) {
        src?.Dispose();
        src = null;
      }
      Image old = pictureBox1.Image;
      pictureBox1.Image = null;
      old?.Dispose();
      btnLive.Enabled = true;
      btnLiveStop.Enabled = true;
    }

    // PictureBox Zoom이 만든 여백을 고려하여 영상 영역 계산
    private RectangleF GetImageRect() {
      if (imageWidth <= 0 || imageHeight <= 0) return RectangleF.Empty;
      float scale = Math.Min((float)pictureBox1.ClientSize.Width / imageWidth,
                             (float)pictureBox1.ClientSize.Height / imageHeight);
      float w = imageWidth * scale;
      float h = imageHeight * scale;
      return new RectangleF((pictureBox1.ClientSize.Width - w) / 2f,
                            (pictureBox1.ClientSize.Height - h) / 2f, w, h);
    }

    private System.Drawing.Point ToImagePoint(System.Drawing.Point p) {
      RectangleF r = GetImageRect();
      if (r.IsEmpty) return System.Drawing.Point.Empty;
      int x = (int)Math.Round((p.X - r.X) * imageWidth / r.Width);
      int y = (int)Math.Round((p.Y - r.Y) * imageHeight / r.Height);
      return new System.Drawing.Point(Math.Max(0, Math.Min(imageWidth, x)),
                                      Math.Max(0, Math.Min(imageHeight, y)));
    }

    private PointF ToScreenPoint(System.Drawing.Point p) {
      RectangleF r = GetImageRect();
      return new PointF(r.X + p.X * r.Width / imageWidth,
                        r.Y + p.Y * r.Height / imageHeight);
    }

    // Day6 Form34: MouseDown / MouseMove / MouseUp / Paint
    private void pictureBox1_MouseDown(object sender, MouseEventArgs e) {
      if (e.Button != MouseButtons.Left || pictureBox1.Image == null) return;
      if (!GetImageRect().Contains(e.Location)) return;
      startPoint = ToImagePoint(e.Location);
      rect = new Rectangle(startPoint.X, startPoint.Y, 0, 0);
      drawing = true;
      hasRoi = false;
      pictureBox1.Capture = true;
    }

    private void pictureBox1_MouseMove(object sender, MouseEventArgs e) {
      if (!drawing) return;
      System.Drawing.Point p = ToImagePoint(e.Location);
      rect = Rectangle.FromLTRB(Math.Min(startPoint.X, p.X),
                                Math.Min(startPoint.Y, p.Y),
                                Math.Max(startPoint.X, p.X),
                                Math.Max(startPoint.Y, p.Y));
      pictureBox1.Invalidate();
    }

    private void pictureBox1_MouseUp(object sender, MouseEventArgs e) {
      if (!drawing) return;
      pictureBox1_MouseMove(sender, e);
      drawing = false;
      pictureBox1.Capture = false;
      hasRoi = rect.Width > 0 && rect.Height > 0;
      if (hasRoi) UpdateRoiListView();
      pictureBox1.Invalidate();
    }

    private void pictureBox1_Paint(object sender, PaintEventArgs e) {
      if ((!drawing && !hasRoi) || imageWidth <= 0 || imageHeight <= 0) return;
      PointF p1 = ToScreenPoint(rect.Location);
      PointF p2 = ToScreenPoint(new System.Drawing.Point(rect.Right, rect.Bottom));
      RectangleF drawRect = RectangleF.FromLTRB(p1.X, p1.Y, p2.X, p2.Y);
      using (Pen pen = new Pen(Color.Red, 2))
        e.Graphics.DrawRectangle(pen, drawRect.X, drawRect.Y,
                                 drawRect.Width, drawRect.Height);
    }

    private void UpdateRoiListView() {
      livRoiParam.Items[0].SubItems[1].Text = rect.Top.ToString();
      livRoiParam.Items[1].SubItems[1].Text = rect.Bottom.ToString();
      livRoiParam.Items[2].SubItems[1].Text = rect.Left.ToString();
      livRoiParam.Items[3].SubItems[1].Text = rect.Right.ToString();
    }

    // Day6 Form34: Mat의 [top, bottom, left, right] ROI 인덱서 사용
    private void btnRoiSave_Click(object sender, EventArgs e) {
      if (!hasRoi || rect.Width < 1 || rect.Height < 1) {
        MessageBox.Show("ROI 영역을 먼저 설정하세요.");
        return;
      }
      Mat snapshot;
      lock (frameLock)
        snapshot = src?.Clone();
      if (snapshot == null || snapshot.Empty()) {
        snapshot?.Dispose();
        MessageBox.Show("카메라 영상이 없습니다.");
        return;
      }
      using (snapshot) {
        try {
          Rectangle imageRect = new Rectangle(0, 0, snapshot.Width, snapshot.Height);
          Rectangle cropRect = Rectangle.Intersect(rect, imageRect);
          if (cropRect.Width <= 0 || cropRect.Height <= 0) {
            MessageBox.Show("ROI 영역이 올바르지 않습니다.");
            return;
          }
          using (Mat roi = snapshot[cropRect.Top, cropRect.Bottom,
                                    cropRect.Left, cropRect.Right]) {
            // 수업 예제의 ToBitmap()을 이용해 PictureBox에 표시
            Image oldImage = pictureBox2.Image;
            pictureBox2.Image = roi.ToBitmap();
            oldImage?.Dispose();
          }
          string folder = Path.GetFullPath(Global.ParamPath);
          Directory.CreateDirectory(folder);
          string path = Path.Combine(folder, Global.ROI_Config_NAME + ".txt");
          string[] lines = {
                        "StartY=" + rect.Top,
                        "EndY=" + rect.Bottom,
                        "StartX=" + rect.Left,
                        "EndX=" + rect.Right
                    };
          File.WriteAllLines(path, lines);
          MessageBox.Show("ROI 저장 및 이미지 크롭 완료");
        }
        catch (Exception ex) {
          MessageBox.Show("ROI 저장 실패 : " + ex.Message);
        }
      }
    }

    private void LoadRoiConfig() {
      string path = Path.GetFullPath(Path.Combine(Global.ParamPath,
                                                 Global.ROI_Config_NAME + ".txt"));
      if (!File.Exists(path)) return;
      try {
        int startY = 0, endY = 0, startX = 0, endX = 0;
        foreach (string line in File.ReadAllLines(path)) {
          string[] values = line.Split('=');
          if (values.Length != 2) continue;
          int value;
          if (!int.TryParse(values[1].Trim(), out value)) continue;
          switch (values[0].Trim()) {
            case "StartY": startY = value; break;
            case "EndY": endY = value; break;
            case "StartX": startX = value; break;
            case "EndX": endX = value; break;
          }
        }
        rect = Rectangle.FromLTRB(startX, startY, endX, endY);
        hasRoi = rect.Width > 0 && rect.Height > 0;
        if (hasRoi) {
          UpdateRoiListView();
          pictureBox1.Invalidate();
        }
      }
      catch (Exception ex) {
        MessageBox.Show("ROI 불러오기 실패 : " + ex.Message);
      }
    }

    private string ColorConfigPath
    {
        get
        {
            return Path.GetFullPath(
                Path.Combine(Global.ParamPath, Global.Color_Config_NAME + ".txt"));
        }
    }

    private readonly string[] colorParamNames =
    {
        "BlueLowLevel",
        "BlueUpperLevel",
        "GreenLowLevel",
        "GreenUpperLevel",
        "RedLowLevel",
        "RedUpperLevel"
    };

    private readonly int[] colorParamValues = { 0, 255, 0, 255, 0, 255 };


    private void UpdateColorListView()
    {
        livColorParam.Items.Clear();

        for (int i = 0; i < colorParamNames.Length; i++)
        {
            var item = new ListViewItem(colorParamNames[i]);
            item.SubItems.Add(colorParamValues[i].ToString());
            livColorParam.Items.Add(item);
        }
    }

    private void LoadColorConfig()
    {
        try
        {
            if (!File.Exists(ColorConfigPath))
                return;

            foreach (string line in File.ReadAllLines(ColorConfigPath))
            {
                string[] parts = line.Split(new[] { '=' }, 2);

                if (parts.Length != 2)
                    continue;

                int index = Array.IndexOf(colorParamNames, parts[0].Trim());
                int value;

                if (index >= 0 && int.TryParse(parts[1].Trim(), out value)
                    && value >= 0 && value <= 255)
                {
                    colorParamValues[index] = value;
                }
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show("색상 설정 불러오기 실패: " + ex.Message);
        }
    }

    private void btnSave_Click(object sender, EventArgs e)
    {
      if (!(cbxColor.SelectedItem is ColorChannel color)
          || !(cbxLowUpper.SelectedItem is ColorLevel level))
          {
            MessageBox.Show("색상과 Low 또는 Upper를 선택하세요.");
            return;
          }

          string paramName = $"{color}{level}Level";

      int index = Array.IndexOf(colorParamNames, paramName);

        if (index < 0)
        {
            MessageBox.Show("색상과 Low 또는 Upper를 선택하세요.");
            return;
        }

        int value;
        if (!int.TryParse(tbRate.Text.Trim(), out value)
            || value < 0 || value > 255)
        {
            MessageBox.Show("0~255 사이의 정수를 입력하세요.");
            tbRate.Focus();
            return;
        }

        try
        {
            string path = ColorConfigPath;
            Directory.CreateDirectory(Path.GetDirectoryName(path));

            var lines = new string[colorParamNames.Length];

            for (int i = 0; i < colorParamNames.Length; i++)
            {
                int savedValue = i == index ? value : colorParamValues[i];
                lines[i] = colorParamNames[i] + "=" + savedValue;
            }

            File.WriteAllLines(path, lines);

            colorParamValues[index] = value;
            livColorParam.Items[index].SubItems[1].Text = value.ToString();
        }
        catch (Exception ex)
        {
            MessageBox.Show("색상 설정 저장 실패: " + ex.Message);
        }
    }

    public bool Inspection()
    {
      Rectangle roiRect = rect;

      if (!hasRoi || roiRect.Width <= 0 || roiRect.Height <= 0)
        throw new InvalidOperationException("ROI 영역을 먼저 설정하세요.");

      Mat snapshot;
      lock (frameLock)
      {
        snapshot = src?.Clone();
      }

      using (snapshot)
      {
        if (snapshot == null || snapshot.Empty())
          throw new InvalidOperationException("카메라 영상이 없습니다. Live를 먼저 실행하세요.");

        var imageRect = new Rectangle(0, 0, snapshot.Width, snapshot.Height);
        Rectangle cropRect = Rectangle.Intersect(roiRect, imageRect);

        if (cropRect.Width <= 0 || cropRect.Height <= 0)
          throw new InvalidOperationException("ROI가 카메라 영상 범위 밖에 있습니다.");

        var lower = new Scalar(
          colorParamValues[0], colorParamValues[2], colorParamValues[4]);
        var upper = new Scalar(
          colorParamValues[1], colorParamValues[3], colorParamValues[5]);

        using (Mat roi = snapshot[cropRect.Top, cropRect.Bottom,
                                 cropRect.Left, cropRect.Right])
        using (var mask = new Mat())
        {
          Cv2.InRange(roi, lower, upper, mask);
          return Cv2.CountNonZero(mask) > 0;
        }
      }
    }

    private void btnSearch_Click(object sender, EventArgs e)
    {
      lbJegment.Text = "-";

      try
      {
        lbJegment.Text = Inspection() ? "검출" : "미검출";
      }
      catch (Exception ex)
      {
        MessageBox.Show("색상 검사 실패: " + ex.Message);
      }
    }


    private async void frmVision_FormClosing(object sender, FormClosingEventArgs e) {
      if (closingAfterStop) return;
      if (liveTask != null) {
        e.Cancel = true;
        await StopCameraAsync();
        closingAfterStop = true;
        Close();
      }
      else {
        Image old = pictureBox2.Image;
        pictureBox2.Image = null;
        old?.Dispose();
      }
    }
  }
}
