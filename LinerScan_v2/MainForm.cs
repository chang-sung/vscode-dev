using LinerScan.Imaging;
using LinerScan.Networking;
using System;
using System.IO;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using LinerScan.Cameras;
using LinerScan.Inference;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using ActProgType64Lib;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace LinerScan
{
    public partial class MainForm : Form
    {
        private ImageClassifier classifier;

        private Rectangle roi1_A;
        private Rectangle roi1_B;
        private Rectangle roi2_A;
        private Rectangle roi2_B;
        //public ActProgType PLC;
        private ActProgType64 PLC;

        private CameraManager cameraManager;
        private Cropper _cropper;
        private readonly Timer cameraStatusTimer = new Timer { Interval = 500 };
        public MainForm()
        {
            InitializeComponent();
            TcpImgSender.Logger = LogText;   // ✅ 추가: TcpImgSender 로그가 tb_logbox + 파일로 저장됨

            // ✅ 프로그램 켜질 때 미리 연결
            TcpImgSender.Start("192.168.3.100", 9000);

            string modelPath = Path.Combine(Application.StartupPath, "best_resnet50.onnx");
            classifier = new ImageClassifier(modelPath);
            LoadRoiFromConfig(); // ROI 정보 불러오기

        }

        private void MainForm_Shown(object sender, EventArgs e)
        {
            ConnectPLC();
            InitializeCameras(); // ✅ 카메라 초기화
        }

        private void ConnectPLC()
        {
            try
            {
                PLC = new ActProgType64();
                PLC.ActUnitType = 0x1A;
                PLC.ActProtocolType = 0x05;
                PLC.ActCpuType = 0xD5;
                PLC.ActPortNumber = 0x00;
                PLC.ActDestinationPortNumber = 5002;
                PLC.ActHostAddress = "192.168.3.1";
                PLC.ActNetworkNumber = 6;
                PLC.ActStationNumber = 1;
                PLC.ActSourceNetworkNumber = 6;
                PLC.ActSourceStationNumber = 15;
                PLC.ActTimeOut = 1000;

                int result = PLC.Open();

                if (result == 0)
                {
                    labelStatus.Text = "PLC 연결 성공!";
                    labelStatus.ForeColor = Color.Green;
                    plc_mon_timer.Start();
                }
                else
                {
                    labelStatus.Text = $"PLC 연결 실패! 오류 코드: {result}";
                    labelStatus.ForeColor = Color.Red;

                    string alcode = "0x" + result.ToString("X").PadLeft(8, '0');
                    string alcoment4 = " ";
                    string alcoment5 = " ";
                    PLC_Mxcomponent_AlarmComment.AlarmComments_MX5.TryGetValue(alcode, out alcoment5);
                    PLC_Mxcomponent_AlarmComment.AlarmComments_MX4.TryGetValue(alcode, out alcoment4);


                    textBox_ex.Text = DateTime.Now.ToString("yyyyMMdd_HHmmss") +
                                      "\r\n" + alcode +
                                      " : \r\nMX5 : " + alcoment5 +
                                      "\r\nMX4 : " + alcoment4;
                }
            }
            catch (Exception ex)
            {
                labelStatus.Text = $"예외 발생 : {ex.Message}";
                labelStatus.ForeColor = Color.Red;
            }
        }

        private void plc_mon_timer_Tick(object sender, EventArgs e)
        {
            try
            {
                short INSP_1, INSP_2;
                PLC.GetDevice2("R25010.0", out INSP_1);
                PLC.GetDevice2("R25010.5", out INSP_2);

                if (INSP_1 == 1)
                {
                    LoadRoiFromConfig();
                    RunInference1FromPLC();

                    PLC.SetDevice("R25010.0", 0);
                    PLC.SetDevice("R25010.3", 1); // 1열 복귀

                }
                else if (INSP_2 == 1)
                {
                    LoadRoiFromConfig();
                    RunInference2FromPLC();

                    PLC.SetDevice("R25010.5", 0);
                    PLC.SetDevice("R25010.8", 1); // 2열 복귀
                }
            }
            catch (Exception ex)
            {
                plc_mon_timer.Stop();
                MessageBox.Show($"PLC 감시 중 오류 발생: {ex.Message}");
            }
        }

        private void InitializeCameras()
        {
            _cropper = new Cropper
            {
                DefaultMinScore = 0.70,
                DefaultExtraDown = 5,
                DefaultExtraX = -30
            };
            cameraManager = new CameraManager(LogText);
            try
            {
                string templatesDir = Path.Combine(Application.StartupPath, "templates");
                _cropper.LoadMarks(templatesDir, "mark*.png");
                LogText($"템플릿 마크 로드: {_cropper.MarkNames.Count}개");
                if (_cropper.MarkNames.Count == 0)
                    LogText($"템플릿 마크 없음");

                var devices = new CameraDeviceCatalog().GetDevices();
                var deviceList = string.Join(Environment.NewLine + Environment.NewLine,
                    devices.Select(d => d.Name + Environment.NewLine + d.DevicePath));
                File.WriteAllText(Path.Combine(Application.StartupPath, "camera-devices.txt"), deviceList);
                var store = new CameraConfigurationStore(Path.Combine(Application.StartupPath, "cam.ini"));
                cameraManager.Start(store.Load());
            }
            catch (Exception ex)
            {
                LogText("카메라 초기화 실패: " + ex.Message);
            }
            cameraStatusTimer.Tick += CameraStatusTimer_Tick;
            cameraStatusTimer.Start();
            CameraStatusTimer_Tick(this, EventArgs.Empty);
        }

        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            bool ready1 = cameraManager != null && cameraManager.IsReady(1);
            bool ready2 = cameraManager != null && cameraManager.IsReady(2);
            Label_CAM1_Status.Text = ready1 ? "CAM1 정상 연결" : "CAM1 프레임 없음 / 설정 확인";
            Label_CAM2_Status.Text = ready2 ? "CAM2 정상 연결" : "CAM2 프레임 없음 / 설정 확인";
            Label_CAM1_Status.ForeColor = ready1 ? Color.Green : Color.Red;
            Label_CAM2_Status.ForeColor = ready2 ? Color.Green : Color.Red;
        }

        private Mat GetLatestFrameClone(int cameraNumber)
        {
            return cameraManager?.GetLatestFrameClone(cameraNumber);
        }
        private Bitmap CropByMarksToBitmap(Mat frame, Rectangle searchRoi, string tagForLog, out string usedMark, out double score)
        {
            usedMark = null;
            score = 0;

            if (_cropper == null) return null;

            string mk;
            double sc;

            Mat cropMat = _cropper.TryCropWithAnyMark(
                frame,
                searchRoi,
                new OpenCvSharp.Size(165, 70),
                out mk,
                out sc,
                null,
                false,   // useCanny (필요하면 true)
                null,
                null
            );

            if (cropMat == null)
            {
                LogText("❌ [" + tagForLog + "] mark 매칭 실패");
                return null;
            }

            try
            {
                usedMark = mk;
                score = sc;
                LogText("✅ [" + tagForLog + "] " + usedMark + " score=" + score.ToString("F2"));
                return BitmapConverter.ToBitmap(cropMat);
            }
            finally
            {
                cropMat.Dispose();
            }
        }

        private void LoadRoiFromConfig()
        {
            string roiConfigPath = "config.json";
            if (File.Exists(roiConfigPath))
            {
                var json = File.ReadAllText(roiConfigPath);
                var config = JsonConvert.DeserializeObject<RoiConfig>(json);

                roi1_A = config.Roi1_A;
                roi1_B = config.Roi1_B;
                roi2_A = config.Roi2_A;
                roi2_B = config.Roi2_B;
            }
            else
            {
                roi1_A = new Rectangle(453, 238, 300, 500);
                roi1_B = new Rectangle(1003, 237, 300, 500);
                roi2_A = new Rectangle(453, 400, 300, 500);
                roi2_B = new Rectangle(1003, 400, 300, 500);
            }
        }

        // 기존 RunInference1FromPLC 수정: camera1 사용
        private string RunInference1FromPLC()
        {
            if (cameraManager == null || !cameraManager.IsReady(1))
                return "❌ camera1이 열려있지 않습니다.";

            var cropped1List = new List<Bitmap>();
            var cropped2List = new List<Bitmap>();

            System.Threading.Thread.Sleep(200);

            for (int i = 0; i < 11; i++)
            {
                using (var frameMat = GetLatestFrameClone(1))
                {
                    if (frameMat == null || frameMat.Empty())
                    {
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    // i==1에 원본 저장(원하면 유지)
                    if (i == 1)
                    {
                        using (var bmpFrame = BitmapConverter.ToBitmap(frameMat))
                        {
                            SaveOriginalFrame(bmpFrame, "camera1");
                        }
                    }

                    string mkA, mkB;
                    double scA, scB;

                    Bitmap bmpA = CropByMarksToBitmap(frameMat, roi1_A, "ROI1_A", out mkA, out scA);
                    Bitmap bmpB = CropByMarksToBitmap(frameMat, roi1_B, "ROI1_B", out mkB, out scB);

                    if (bmpA == null || bmpB == null)
                    {
                        if (bmpA != null) bmpA.Dispose();
                        if (bmpB != null) bmpB.Dispose();
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    // 첫 프레임 버리기(원하면)
                    if (i == 0)
                    {
                        bmpA.Dispose();
                        bmpB.Dispose();
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    // 리스트 저장(복사)
                    cropped1List.Add(new Bitmap(bmpA));
                    cropped2List.Add(new Bitmap(bmpB));

                    // PictureBox 표시
                    if (pb_1_A.Image != null) pb_1_A.Image.Dispose();
                    if (pb_1_B.Image != null) pb_1_B.Image.Dispose();
                    pb_1_A.Image = new Bitmap(bmpA);
                    pb_1_B.Image = new Bitmap(bmpB);

                    bmpA.Dispose();
                    bmpB.Dispose();
                }

                System.Threading.Thread.Sleep(200);
            }

            if (cropped1List.Count == 0 || cropped2List.Count == 0)
                return "❌ crop 결과가 없습니다(템플릿 매칭 실패 가능).";

            return RunModelWithPlcControl(cropped1List, cropped2List, 1);
        }

        // RunInference2FromPLC도 camera2를 사용
        private string RunInference2FromPLC()
        {
            if (cameraManager == null || !cameraManager.IsReady(2))
                return "❌ camera2가 열려있지 않습니다.";

            var cropped1List = new List<Bitmap>();
            var cropped2List = new List<Bitmap>();

            System.Threading.Thread.Sleep(200);

            for (int i = 0; i < 11; i++)
            {
                using (var frameMat = GetLatestFrameClone(2))
                {
                    if (frameMat == null || frameMat.Empty())
                    {
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    if (i == 1)
                    {
                        using (var bmpFrame = BitmapConverter.ToBitmap(frameMat))
                        {
                            SaveOriginalFrame(bmpFrame, "camera2");
                        }
                    }

                    string mkA, mkB;
                    double scA, scB;

                    Bitmap bmpA = CropByMarksToBitmap(frameMat, roi2_A, "ROI2_A", out mkA, out scA);
                    Bitmap bmpB = CropByMarksToBitmap(frameMat, roi2_B, "ROI2_B", out mkB, out scB);

                    if (bmpA == null || bmpB == null)
                    {
                        if (bmpA != null) bmpA.Dispose();
                        if (bmpB != null) bmpB.Dispose();
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    if (i == 0)
                    {
                        bmpA.Dispose();
                        bmpB.Dispose();
                        System.Threading.Thread.Sleep(100);
                        continue;
                    }

                    cropped1List.Add(new Bitmap(bmpA));
                    cropped2List.Add(new Bitmap(bmpB));

                    if (pb_2_A.Image != null) pb_2_A.Image.Dispose();
                    if (pb_2_B.Image != null) pb_2_B.Image.Dispose();
                    pb_2_A.Image = new Bitmap(bmpA);
                    pb_2_B.Image = new Bitmap(bmpB);

                    bmpA.Dispose();
                    bmpB.Dispose();
                }

                System.Threading.Thread.Sleep(200);
            }

            if (cropped1List.Count == 0 || cropped2List.Count == 0)
                return "❌ crop 결과가 없습니다(템플릿 매칭 실패 가능).";

            return RunModelWithPlcControl(cropped1List, cropped2List, 2);
        }

        private string RunModelWithPlcControl(List<Bitmap> cropped1List, List<Bitmap> cropped2List, int roiIndex)
        {
            try
            {
                var labels1 = new List<string>();
                var labels2 = new List<string>();

                foreach (var cropped1 in cropped1List)
                    labels1.Add(classifier.Classify(cropped1).Label);
                foreach (var cropped2 in cropped2List)
                    labels2.Add(classifier.Classify(cropped2).Label);
                // 최빈값 계산
                string mode1 = labels1.GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;
                string mode2 = labels2.GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;

                // PLC에 ZR 쓰기: detect=1, none=0
                WriteZR(54074, ModeToInt(mode1)); // mode1 → ZR54074
                WriteZR(54075, ModeToInt(mode2)); // mode2 → ZR54075


                // PLC 제어
                if (roiIndex == 1)
                {
                    if (mode1 == "none")
                    {
                        PLC.SetDevice("R25010.A", 1);
                        LogCellid("ROI1_A", 84000);
                        LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                    }
                    if (mode2 == "none")
                    {
                        PLC.SetDevice("R25010.B", 1);
                        LogCellid("ROI1_B", 84400);
                        LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                    }
                }
                else if (roiIndex == 2)
                {
                    if (mode1 == "none")
                    {
                        PLC.SetDevice("R25010.A", 1);
                        LogCellid("ROI2_A", 84800);
                        LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                    }
                    if (mode2 == "none")
                    {
                        PLC.SetDevice("R25010.B", 1);
                        LogCellid("ROI2_B", 85200);
                        LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                    }
                }

                if (mode1 == "none" && mode2 == "none")
                {
                    PLC.SetDevice("R25010.1", 1);
                    LogText(" 롤러 교체 완료");
                }
                else if (mode1 == "detect" || mode2 == "detect")
                {
                    PLC.SetDevice("R25010.2", 1);
                    LogText("이형지 폐기 시작 (하나라도 detect)");
                }

                // ✅ 최종 예측 결과는 로그 & 이미지 저장 분리
                LogText($"[ROI_{roiIndex}_A] 예측: {mode1}");
                LogText($"[ROI_{roiIndex}_B] 예측: {mode2}");

                // ✅ 최종 이미지 저장 (cap_image)
                var (pathA, pathB) = SaveImagesAndGetPaths(
                    cropped1List.Last(),
                    cropped2List.Last(),
                    roiIndex,
                    mode1,
                    mode2
                );

                // ✅ cap_image에 저장된 파일만 TCP 전송
                try
                {
                    System.Threading.Tasks.Task.Run(() =>
                    {
                        if (!string.IsNullOrEmpty(pathA))
                            TcpImgSender.SendFile(pathA);

                        if (!string.IsNullOrEmpty(pathB))
                            TcpImgSender.SendFile(pathB);
                    });
                }
                catch (Exception ex)
                {
                    LogText("❌ TCP 전송 예외: " + ex.Message);
                }

                // ✅ crop_image(전체 10장) 저장은 기존 유지
                SaveImagesAll(cropped1List, cropped2List, roiIndex);

                // ✅ 결과 문자열 반환
                string resultString = $"[ROI_{roiIndex}_A] 예측: {mode1}\r\n[ROI_{roiIndex}_B] 예측: {mode2}";

                return resultString;

            }
            finally
            {
                // ✅ 여기서 Bitmap 리스트 전부 해제
                if (cropped1List != null)
                    foreach (var b in cropped1List) b?.Dispose();
                if (cropped2List != null)
                    foreach (var b in cropped2List) b?.Dispose();
            }

        }


        private void btn_ROI1_Click(object sender, EventArgs e)
        {
            using (var roiForm = new RoiForm(1, () => GetLatestFrameClone(1), _cropper))
            {
                if (roiForm.ShowDialog() == DialogResult.OK)
                {
                    LoadRoiFromConfig();
                }
            }
        }

        private void btn_ROI2_Click(object sender, EventArgs e)
        {
            using (var roiForm = new RoiForm(2, () => GetLatestFrameClone(2), _cropper))
            {
                if (roiForm.ShowDialog() == DialogResult.OK)
                {
                    LoadRoiFromConfig();
                }
            }
        }

        private void LogText(string message)
        {
            if (tb_logbox.InvokeRequired)
            {
                tb_logbox.BeginInvoke(new Action(() => LogText(message)));
                return;
            }

            // ---- 여기부터는 UI 스레드 ----
            string timeStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

            string baseDirectory = @"D:\LinerScan_Logs\logs";
            string year = DateTime.Now.ToString("yyyy");
            string month = DateTime.Now.ToString("MM");
            string logDirectory = Path.Combine(baseDirectory, year, month);

            string logFileName = DateTime.Now.ToString("yyyyMMdd") + "_log.txt";
            string logFilePath = Path.Combine(logDirectory, logFileName);

            if (!Directory.Exists(logDirectory))
                Directory.CreateDirectory(logDirectory);

            string logLine = $"[{timeStamp}] {message}";

            var lines = tb_logbox.Lines.ToList();
            lines.Add(logLine);

            int maxLines = 100;
            if (lines.Count > maxLines)
                lines.RemoveRange(0, lines.Count - maxLines);

            tb_logbox.Lines = lines.ToArray();
            File.AppendAllText(logFilePath, logLine + Environment.NewLine);

            // ✅ 라벨 갱신도 이제 안전
            if (message.StartsWith("[ROI_1_A] 예측: "))
                Label_1A_Juge.Text = message.Split(new[] { ": " }, StringSplitOptions.None).Last();
            else if (message.StartsWith("[ROI_1_B] 예측: "))
                Label_1B_Juge.Text = message.Split(new[] { ": " }, StringSplitOptions.None).Last();
            else if (message.StartsWith("[ROI_2_A] 예측: "))
                Label_2A_Juge.Text = message.Split(new[] { ": " }, StringSplitOptions.None).Last();
            else if (message.StartsWith("[ROI_2_B] 예측: "))
                Label_2B_Juge.Text = message.Split(new[] { ": " }, StringSplitOptions.None).Last();
        }


        private void SaveImagesAll(List<Bitmap> cropped1List, List<Bitmap> cropped2List, int roiIndex)
        {
            if (roiIndex <= 0) return;

            string baseDir = @"D:\LinerScan_Logs\crop_image";
            string year = DateTime.Now.ToString("yyyy");
            string month = DateTime.Now.ToString("MM");
            string day = DateTime.Now.ToString("dd");

            string imageDir = Path.Combine(baseDir, year, month, day);

            if (!Directory.Exists(imageDir))
                Directory.CreateDirectory(imageDir);

            string timeFilePart = DateTime.Now.ToString("yyMMdd_HHmmss");

            for (int i = 0; i < cropped1List.Count; i++)
            {
                if (cropped1List[i] != null)
                {
                    string imageFileNameA = $"{timeFilePart}_{roiIndex}열_A_{i + 1:00}.jpg";
                    string imagePathA = Path.Combine(imageDir, imageFileNameA);
                    cropped1List[i].Save(imagePathA, System.Drawing.Imaging.ImageFormat.Jpeg);
                }

                if (cropped2List[i] != null)
                {
                    string imageFileNameB = $"{timeFilePart}_{roiIndex}열_B_{i + 1:00}.jpg";
                    string imagePathB = Path.Combine(imageDir, imageFileNameB);
                    cropped2List[i].Save(imagePathB, System.Drawing.Imaging.ImageFormat.Jpeg);
                }
            }
        }

        private void SaveOriginalFrame(Bitmap originalFrame, string cameraName)
        {
            try
            {
                string baseDir = @"D:\LinerScan_Logs\original_frame";
                string year = DateTime.Now.ToString("yyyy");
                string month = DateTime.Now.ToString("MM");
                string day = DateTime.Now.ToString("dd");

                string imageDir = Path.Combine(baseDir, year, month, day);

                if (!Directory.Exists(imageDir))
                    Directory.CreateDirectory(imageDir);

                string timeFilePart = DateTime.Now.ToString("yyMMdd_HHmmss");
                string fileName = $"{timeFilePart}_{cameraName}_원본.jpg";
                string savePath = Path.Combine(imageDir, fileName);

                originalFrame.Save(savePath, System.Drawing.Imaging.ImageFormat.Jpeg);
            }
            catch (Exception ex)
            {
                LogText($"❌ 원본 이미지 저장 실패: {ex.Message}");
            }
        }

        private (string pathA, string pathB) SaveImagesAndGetPaths(Bitmap imageA, Bitmap imageB, int roiIndex, string modeA, string modeB)
        {
            if (roiIndex <= 0) return (null, null);

            string baseDir = @"D:\LinerScan_Logs\cap_image";
            string year = DateTime.Now.ToString("yyyy");
            string month = DateTime.Now.ToString("MM");
            string day = DateTime.Now.ToString("dd");
            string imageDir = Path.Combine(baseDir, year, month, day);
            Directory.CreateDirectory(imageDir);

            string timeFilePart = DateTime.Now.ToString("yyMMdd_HHmmss");

            string pathA = null, pathB = null;

            if (imageA != null)
            {
                string fileA = $"{timeFilePart}_{roiIndex}열_A_{modeA}.jpg";
                pathA = Path.Combine(imageDir, fileA);
                imageA.Save(pathA, System.Drawing.Imaging.ImageFormat.Jpeg);
            }

            if (imageB != null)
            {
                string fileB = $"{timeFilePart}_{roiIndex}열_B_{modeB}.jpg";
                pathB = Path.Combine(imageDir, fileB);
                imageB.Save(pathB, System.Drawing.Imaging.ImageFormat.Jpeg);
            }

            return (pathA, pathB);
        }

        private void LogCellid(string label, int startAddress)
        {
            const int wordLength = 8; // 8워드 = 16바이트
            int[] buffer = new int[wordLength];

            int result = PLC.ReadDeviceBlock($"ZR{startAddress}", wordLength, out buffer[0]);

            if (result == 0)
            {
                byte[] bytes = new byte[wordLength * 2];

                for (int i = 0; i < wordLength; i++)
                {
                    bytes[i * 2] = (byte)(buffer[i] & 0xFF);             // 하위 바이트
                    bytes[i * 2 + 1] = (byte)((buffer[i] >> 8) & 0xFF);   // 상위 바이트
                }

                string asciiStr = System.Text.Encoding.ASCII.GetString(bytes).TrimEnd('\0');
                LogText($"{label} - Cell ID: {asciiStr}");

                // ✅ 해당 라벨에 출력
                switch (label)
                {
                    case "ROI1_A":
                        Label_1A_Cell_id.Text = asciiStr;
                        break;
                    case "ROI1_B":
                        Label_1B_Cell_id.Text = asciiStr;
                        break;
                    case "ROI2_A":
                        Label_2A_Cell_id.Text = asciiStr;
                        break;
                    case "ROI2_B":
                        Label_2B_Cell_id.Text = asciiStr;
                        break;
                }
            }
            else
            {
                LogText($"{label} - ZR{startAddress} ASCII 읽기 실패! (오류 코드: {result})");
            }
        }

        private void LogRollerCount(string label, int address)
        {
            int[] buffer = new int[1];
            int result = PLC.ReadDeviceBlock($"ZR{address}", 1, out buffer[0]);

            if (result == 0)
            {
                LogText($"{label} : {buffer[0]}");  // 롤러 사용횟수 : 297
            }
            else
            {
                LogText($"{label} 읽기 실패! (오류 코드: {result})");
            }
        }

        // ZR 레지스터에 쓰기
        private void WriteZR(int address, int value)
        {
            try
            {
                int[] data = new int[] { value };
                int ret = PLC.WriteDeviceBlock($"ZR{address}", 1, ref data[0]); // ActUtlType 표준 시그니처

                if (ret == 0)
                    LogText($"ZR{address} <= {value} (쓰기 성공)");
                else
                    LogText($"ZR{address} 쓰기 실패 (오류 코드: {ret})");
            }
            catch (Exception ex)
            {
                LogText($"ZR{address} 쓰기 예외: {ex.Message}");
            }
        }

        private int ModeToInt(string mode)
        {
            // detect -> 1, none -> 0 (그 외 입력도 안전하게 0)
            return string.Equals(mode, "detect", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
        }

        // 종료 시 카메라 해제
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            TcpImgSender.Stop();

            plc_mon_timer.Stop();
            cameraStatusTimer.Stop();
            cameraStatusTimer.Dispose();
            cameraManager?.Dispose();
            cameraManager = null;
            _cropper?.Dispose();
            _cropper = null;
            classifier?.Dispose();
            classifier = null;
            base.OnFormClosing(e);
        }
    }

}

