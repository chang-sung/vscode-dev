using LinerScan.Imaging;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Newtonsoft.Json;

namespace LinerScan
{
    public partial class RoiForm : Form
    {
        public Rectangle SelectedRoi1 { get; private set; }
        public Rectangle SelectedRoi2 { get; private set; }

        private Rectangle roi1_A = new Rectangle(453, 238, 300, 500);
        private Rectangle roi1_B = new Rectangle(1003, 237, 300, 500);

        private bool isDragging = false;
        private int draggingRoiIndex = 0;
        private System.Drawing.Point dragOffset;

        private bool isSettingROIs = false;
        private Bitmap currentImage;
        private Mat frameMat = new Mat();

        private Timer frameTimer;
        private readonly int roiIndex;
        private int hoverRoiIndex = 0;

        private string roiConfigPath = "config.json";
        private readonly Func<Mat> _getLatestFrame;

        public RoiForm(int index, Func<Mat> getLatestFrame)
        {
            InitializeComponent();
            this.roiIndex = index;
            this._getLatestFrame = getLatestFrame;
            if (index == 2)
            {
                roi1_A = new Rectangle(453, 400, 300, 500);
                roi1_B = new Rectangle(1003, 400, 300, 500);
            }
            // 디자이너에 남아 있는 이전 마크 검색 버튼은 사용하지 않습니다.
            btn_mark_search.Visible = false;
            btn_mark_search.Enabled = false;

            this.Width = 1600;
            this.Height = 1200;
        }

        private void ROISetting_Load(object sender, EventArgs e)
        {
            pb_roi1.Width = 1600;
            pb_roi1.Height = 1200;
            pb_roi1.SizeMode = PictureBoxSizeMode.Normal;
            pb_roi1.SendToBack();

            LoadRoiConfig();
            UpdateRoiLabels();

            frameTimer = new Timer();
            frameTimer.Interval = 100;
            frameTimer.Tick += FrameTimer_Tick;
            frameTimer.Start();
        }

        private void FrameTimer_Tick(object sender, EventArgs e)
        {
            if (_getLatestFrame == null) return;

            using (var latest = _getLatestFrame())
            {
                if (latest == null || latest.Empty()) return;

                frameMat?.Dispose();
                frameMat = latest.Clone();

                currentImage?.Dispose();
                currentImage = BitmapConverter.ToBitmap(frameMat);
                pb_roi1.Invalidate();
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            frameTimer?.Stop();
            base.OnFormClosing(e);
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            frameTimer?.Stop();
            frameTimer?.Dispose();
            frameMat?.Dispose();
            frameMat = null;
            currentImage?.Dispose();
            currentImage = null;

            base.OnFormClosed(e);
        }

        private void btn_roi_set_Click(object sender, EventArgs e)
        {
            if (!isSettingROIs)
            {
                isSettingROIs = true;
                btn_roi_set.Text = "저장";
                pb_roi1.Invalidate();
            }
            else
            {
                isSettingROIs = false;
                btn_roi_set.Text = "ROI 설정";

                SelectedRoi1 = roi1_A;
                SelectedRoi2 = roi1_B;
                SaveRoiConfig();

                MessageBox.Show($"ROI A 저장됨: ({roi1_A.X}, {roi1_A.Y})\nROI B 저장됨: ({roi1_B.X}, {roi1_B.Y})");
            }
        }

        private void pb_roi1_MouseDown(object sender, MouseEventArgs e)
        {
            if (!isSettingROIs || e.Button != MouseButtons.Left) return;

            // 기존 ROI 드래그
            if (roi1_A.Contains(e.Location))
            {
                isDragging = true;
                draggingRoiIndex = 1;
                dragOffset = new System.Drawing.Point(e.X - roi1_A.X, e.Y - roi1_A.Y);
            }
            else if (roi1_B.Contains(e.Location))
            {
                isDragging = true;
                draggingRoiIndex = 2;
                dragOffset = new System.Drawing.Point(e.X - roi1_B.X, e.Y - roi1_B.Y);
            }
        }

        private void pb_roi1_MouseMove(object sender, MouseEventArgs e)
        {
            if (!isSettingROIs) return;

            if (roi1_A.Contains(e.Location))
            {
                hoverRoiIndex = 1;
                pb_roi1.Cursor = Cursors.Hand;
            }
            else if (roi1_B.Contains(e.Location))
            {
                hoverRoiIndex = 2;
                pb_roi1.Cursor = Cursors.Hand;
            }
            else
            {
                hoverRoiIndex = 0;
                pb_roi1.Cursor = Cursors.Default;
            }

            if (isDragging)
            {
                if (draggingRoiIndex == 1)
                {
                    roi1_A.X = Math.Max(0, Math.Min(pb_roi1.Width - roi1_A.Width, e.X - dragOffset.X));
                    roi1_A.Y = Math.Max(0, Math.Min(pb_roi1.Height - roi1_A.Height, e.Y - dragOffset.Y));
                }
                else if (draggingRoiIndex == 2)
                {
                    roi1_B.X = Math.Max(0, Math.Min(pb_roi1.Width - roi1_B.Width, e.X - dragOffset.X));
                    roi1_B.Y = Math.Max(0, Math.Min(pb_roi1.Height - roi1_B.Height, e.Y - dragOffset.Y));
                }

                UpdateRoiLabels();
            }

            pb_roi1.Invalidate();
        }

        private void pb_roi1_MouseUp(object sender, MouseEventArgs e)
        {
            isDragging = false;
            draggingRoiIndex = 0;
        }

        private void pb_roi1_Paint(object sender, PaintEventArgs e)
        {
            if (currentImage != null)
            {
                e.Graphics.DrawImage(currentImage, 0, 0, pb_roi1.Width, pb_roi1.Height);
            }

            using (Font font = new Font("Arial", 12))
            using (Brush brush = new SolidBrush(Color.Red))
            {
                string labelPrefix = roiIndex.ToString();

                using (Pen penA = new Pen(Color.Red, hoverRoiIndex == 1 ? 4 : 2))
                {
                    e.Graphics.DrawRectangle(penA, roi1_A);
                    e.Graphics.DrawString($"{labelPrefix}_A", font, brush, roi1_A.X - 20, roi1_A.Y);
                }

                using (Pen penB = new Pen(Color.Red, hoverRoiIndex == 2 ? 4 : 2))
                {
                    e.Graphics.DrawRectangle(penB, roi1_B);
                    e.Graphics.DrawString($"{labelPrefix}_B", font, brush, roi1_B.X - 20, roi1_B.Y);
                }
            }

        }

        private void LoadRoiConfig()
        {
            lock (RoiConfig.FileSync)
            {
                if (File.Exists(roiConfigPath))
                {
                    var json = File.ReadAllText(roiConfigPath);
                    var config = JsonConvert.DeserializeObject<RoiConfig>(json);
                    if (config == null) return;

                    if (roiIndex == 1)
                    {
                        roi1_A = config.Roi1_A;
                        roi1_B = config.Roi1_B;
                    }
                    else if (roiIndex == 2)
                    {
                        roi1_A = config.Roi2_A;
                        roi1_B = config.Roi2_B;
                    }
                }
            }
        }

        private void SaveRoiConfig()
        {
            lock (RoiConfig.FileSync)
            {
                RoiConfig config;

                if (File.Exists(roiConfigPath))
                    config = JsonConvert.DeserializeObject<RoiConfig>(File.ReadAllText(roiConfigPath)) ?? new RoiConfig();
                else
                    config = new RoiConfig();

                if (roiIndex == 1)
                {
                    config.Roi1_A = roi1_A;
                    config.Roi1_B = roi1_B;
                }
                else if (roiIndex == 2)
                {
                    config.Roi2_A = roi1_A;
                    config.Roi2_B = roi1_B;
                }

                var json = JsonConvert.SerializeObject(config, Formatting.Indented);
                File.WriteAllText(roiConfigPath, json);
            }
        }

        private void UpdateRoiLabels()
        {
            if (roiIndex == 1)
            {
                lb_roi_pos_A.Text = $"ROI 1_A 위치 : - X: {roi1_A.X}, Y: {roi1_A.Y}";
                lb_roi_pos_B.Text = $"ROI 1_B 위치 : - X: {roi1_B.X}, Y: {roi1_B.Y}";
            }
            else
            {
                lb_roi_pos_A.Text = $"ROI 2_A 위치 : - X: {roi1_A.X}, Y: {roi1_A.Y}";
                lb_roi_pos_B.Text = $"ROI 2_B 위치 : - X: {roi1_B.X}, Y: {roi1_B.Y}";
            }
        }

        // 기존 Designer.cs의 이벤트 연결과 호환하기 위해 유지합니다.
        private void btn_mark_search_Click(object sender, EventArgs e)
        {
        }
    }
}
