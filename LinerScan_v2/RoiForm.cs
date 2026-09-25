using LinerScan.Imaging;
using LinerScan.Networking;
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

        private Rectangle roi1_A = new Rectangle(10, 10, 165, 70);
        private Rectangle roi1_B = new Rectangle(30, 30, 165, 70);

        private bool isDragging = false;
        private int draggingRoiIndex = 0;
        private System.Drawing.Point dragOffset;

        private bool isSettingROIs = false;
        private Bitmap currentImage;
        private Mat frameMat = new Mat();

        private Timer frameTimer;
        private readonly int roiIndex;
        private int hoverRoiIndex = 0;

        private bool isDraggingCrop = false;
        private int draggingCropIndex = 0; // 1=A, 2=B
        private System.Drawing.Point cropDragOffset;

        private System.Drawing.Point markCenterA = System.Drawing.Point.Empty;
        private System.Drawing.Point markCenterB = System.Drawing.Point.Empty;

        private int cropOffsetXA = 0;
        private int cropOffsetYA = 0;
        private int cropOffsetXB = 0;
        private int cropOffsetYB = 0;
        private bool isMarkFindMode = false;
        private bool isCropSetMode = false;

        private string roiConfigPath = "config.json";


        private readonly Func<Mat> _getLatestFrame;

        // ✅ mark search 결과 crop 박스(165x75) 표시용
        private Rectangle cropBoxA = Rectangle.Empty;
        private Rectangle cropBoxB = Rectangle.Empty;
        private string cropInfoA = "";
        private string cropInfoB = "";

        // ✅ Cropper 참조 (MainForm에서 전달 or 여기서 생성)
        private Cropper _cropper;
        private OpenCvSharp.Size finalCropSize = new OpenCvSharp.Size(165, 70);

        public RoiForm(int index, Func<Mat> getLatestFrame, Cropper cropper)
        {
            InitializeComponent();
            this.roiIndex = index;
            this._getLatestFrame = getLatestFrame;
            this._cropper = cropper;

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

            btn_mark_search.Click += btn_mark_search_Click;
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
            // Crop A 드래그
            if (!cropBoxA.IsEmpty && cropBoxA.Contains(e.Location))
            {
                isDraggingCrop = true;
                draggingCropIndex = 1;
                cropDragOffset = new System.Drawing.Point(e.X - cropBoxA.X, e.Y - cropBoxA.Y);
                return;
            }

            // Crop B 드래그
            if (!cropBoxB.IsEmpty && cropBoxB.Contains(e.Location))
            {
                isDraggingCrop = true;
                draggingCropIndex = 2;
                cropDragOffset = new System.Drawing.Point(e.X - cropBoxB.X, e.Y - cropBoxB.Y);
                return;
            }

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

            if (isCropSetMode && !cropBoxA.IsEmpty && cropBoxA.Contains(e.Location))
            {
                isDraggingCrop = true;
                draggingCropIndex = 1;
                cropDragOffset = new System.Drawing.Point(e.X - cropBoxA.X, e.Y - cropBoxA.Y);
                return;
            }

            if (isCropSetMode && !cropBoxB.IsEmpty && cropBoxB.Contains(e.Location))
            {
                isDraggingCrop = true;
                draggingCropIndex = 2;
                cropDragOffset = new System.Drawing.Point(e.X - cropBoxB.X, e.Y - cropBoxB.Y);
                return;
            }
        }

        private void pb_roi1_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDraggingCrop)
            {
                if (draggingCropIndex == 1)
                {
                    cropBoxA.X = Math.Max(0, Math.Min(pb_roi1.Width - cropBoxA.Width, e.X - cropDragOffset.X));
                    cropBoxA.Y = Math.Max(0, Math.Min(pb_roi1.Height - cropBoxA.Height, e.Y - cropDragOffset.Y));

                    if (!markCenterA.IsEmpty)
                    {
                        cropOffsetXA = cropBoxA.X - markCenterA.X;
                        cropOffsetYA = cropBoxA.Y - markCenterA.Y;
                        cropInfoA = $"A Offset X:{cropOffsetXA}, Y:{cropOffsetYA}";
                    }
                }
                else if (draggingCropIndex == 2)
                {
                    cropBoxB.X = Math.Max(0, Math.Min(pb_roi1.Width - cropBoxB.Width, e.X - cropDragOffset.X));
                    cropBoxB.Y = Math.Max(0, Math.Min(pb_roi1.Height - cropBoxB.Height, e.Y - cropDragOffset.Y));

                    if (!markCenterB.IsEmpty)
                    {
                        cropOffsetXB = cropBoxB.X - markCenterB.X;
                        cropOffsetYB = cropBoxB.Y - markCenterB.Y;
                        cropInfoB = $"B Offset X:{cropOffsetXB}, Y:{cropOffsetYB}";
                    }
                }

                pb_roi1.Invalidate();
                return;
            }

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

            isDraggingCrop = false;
            draggingCropIndex = 0;
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

            // ✅ 최종 crop 박스(165x75) 표시 (초록)
            using (Pen cropPen = new Pen(Color.Lime, 3))
            using (Font f = new Font("Arial", 12))
            using (Brush b = new SolidBrush(Color.Lime))
            {
                if (!cropBoxA.IsEmpty)
                {
                    e.Graphics.DrawRectangle(cropPen, cropBoxA);
                    e.Graphics.DrawString(cropInfoA, f, b, cropBoxA.X, cropBoxA.Y - 20);
                }

                if (!cropBoxB.IsEmpty)
                {
                    e.Graphics.DrawRectangle(cropPen, cropBoxB);
                    e.Graphics.DrawString(cropInfoB, f, b, cropBoxB.X, cropBoxB.Y - 20);
                }
            }
        }

        private void LoadRoiConfig()
        {
            if (File.Exists(roiConfigPath))
            {
                var json = File.ReadAllText(roiConfigPath);
                var config = JsonConvert.DeserializeObject<RoiConfig>(json);

                if (roiIndex == 1)
                {
                    roi1_A = config.Roi1_A;
                    roi1_B = config.Roi1_B;

                    cropOffsetXA = config.Roi1_A_CropOffsetX;
                    cropOffsetYA = config.Roi1_A_CropOffsetY;
                    cropOffsetXB = config.Roi1_B_CropOffsetX;
                    cropOffsetYB = config.Roi1_B_CropOffsetY;
                }
                else if (roiIndex == 2)
                {
                    roi1_A = config.Roi2_A;
                    roi1_B = config.Roi2_B;

                    cropOffsetXA = config.Roi2_A_CropOffsetX;
                    cropOffsetYA = config.Roi2_A_CropOffsetY;
                    cropOffsetXB = config.Roi2_B_CropOffsetX;
                    cropOffsetYB = config.Roi2_B_CropOffsetY;
                }
            }
        }

        private void SaveRoiConfig()
        {
            RoiConfig config;

            if (File.Exists(roiConfigPath))
                config = JsonConvert.DeserializeObject<RoiConfig>(File.ReadAllText(roiConfigPath));
            else
                config = new RoiConfig();


            if (roiIndex == 1)
            {
                config.Roi1_A = roi1_A;
                config.Roi1_B = roi1_B;

                config.Roi1_A_CropOffsetX = cropOffsetXA;
                config.Roi1_A_CropOffsetY = cropOffsetYA;
                config.Roi1_B_CropOffsetX = cropOffsetXB;
                config.Roi1_B_CropOffsetY = cropOffsetYB;
            }
            else if (roiIndex == 2)
            {
                config.Roi2_A = roi1_A;
                config.Roi2_B = roi1_B;

                config.Roi2_A_CropOffsetX = cropOffsetXA;
                config.Roi2_A_CropOffsetY = cropOffsetYA;
                config.Roi2_B_CropOffsetX = cropOffsetXB;
                config.Roi2_B_CropOffsetY = cropOffsetYB;
            }

            var json = JsonConvert.SerializeObject(config, Formatting.Indented);
            File.WriteAllText(roiConfigPath, json);
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

        private void btn_mark_search_Click(object sender, EventArgs e)
        {
            // cropper/프레임 체크
            if (_cropper == null)
            {
                MessageBox.Show("Cropper가 연결되지 않았습니다. (Form1에서 전달 필요)");
                return;
            }

            if (frameMat == null || frameMat.Empty())
            {
                MessageBox.Show("현재 프레임이 없습니다.");
                return;
            }

            // 현재 프레임 clone(안전)
            Mat frameClone = frameMat.Clone();

            try
            {
                // ROISetting에서는 roi1_A, roi1_B가 현재 화면의 "큰 검색 ROI"
                SearchAndSetCropBoxes(frameClone);

                // 화면 갱신
                pb_roi1.Invalidate();
            }
            finally
            {
                frameClone.Dispose();
            }
        }

        private void SearchAndSetCropBoxes(Mat frame)
        {
            // 결과 초기화
            cropBoxA = Rectangle.Empty;
            cropBoxB = Rectangle.Empty;
            cropInfoA = "";
            cropInfoB = "";

            // A
            SetCropBoxForOneRoi(frame, roi1_A, true);

            // B
            SetCropBoxForOneRoi(frame, roi1_B, false);
        }

        private void SetCropBoxForOneRoi(Mat frame, Rectangle searchRoi, bool isA)
        {
            Rect cropRect;
            string usedMark;
            double score;

            System.Drawing.Point markCenter;

            bool ok = _cropper.TryFindCropRectWithAnyMark(
                frame,
                searchRoi,
                finalCropSize,
                out cropRect,
                out markCenter,
                out usedMark,
                out score,
                null,
                false,
                null,
                null
            );

            if (!ok)
            {
                if (isA)
                    cropInfoA = "A: mark not found";
                else
                    cropInfoB = "B: mark not found";
                return;
            }

            int savedOffsetX = isA ? cropOffsetXA : cropOffsetXB;
            int savedOffsetY = isA ? cropOffsetYA : cropOffsetYB;

            Rectangle r;

            if (savedOffsetX != 0 || savedOffsetY != 0)
            {
                r = new Rectangle(
                    markCenter.X + savedOffsetX,
                    markCenter.Y + savedOffsetY,
                    finalCropSize.Width,
                    finalCropSize.Height
                );
            }
            else
            {
                r = new Rectangle(cropRect.X, cropRect.Y, cropRect.Width, cropRect.Height);
            }

            if (isA)
            {
                cropBoxA = r;
                markCenterA = markCenter;

                cropOffsetXA = cropBoxA.X - markCenterA.X;
                cropOffsetYA = cropBoxA.Y - markCenterA.Y;

                cropInfoA = $"A: {usedMark} Offset X:{cropOffsetXA}, Y:{cropOffsetYA}";
            }
            else
            {
                cropBoxB = r;
                markCenterB = markCenter;

                cropOffsetXB = cropBoxB.X - markCenterB.X;
                cropOffsetYB = cropBoxB.Y - markCenterB.Y;

                cropInfoB = $"B: {usedMark} Offset X:{cropOffsetXB}, Y:{cropOffsetYB}";
            }
        }

        
    }
}
