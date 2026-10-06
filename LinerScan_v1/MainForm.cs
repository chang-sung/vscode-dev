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
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Timer = System.Windows.Forms.Timer;

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
        // PLC COM 객체는 생성부터 종료까지 이 전용 STA 스레드만 사용합니다.
        private ActProgType64 PLC;
        private Thread plcThread;
        private readonly ManualResetEventSlim plcStop = new ManualResetEventSlim(false);
        // PLC 감시 주기(ms). 디자이너 타이머와 독립적으로 설정합니다.
        private const int PlcPollIntervalMilliseconds = 1000;
        private volatile bool closing;
        private bool shutdownComplete;

        private CameraManager cameraManager;
        private Cropper _cropper;
        private readonly Timer cameraStatusTimer = new Timer { Interval = 500 };
        private readonly DateTime?[] cameraMissingSince = new DateTime?[3];
        private readonly DateTime[] lastReconnectAttempt = new DateTime[3];
        private readonly bool[] awaitingRecovery = new bool[3];
        private static readonly TimeSpan CameraLostDelay = TimeSpan.Zero;
        private static readonly TimeSpan CameraRetryInterval = TimeSpan.FromSeconds(30);

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
            if (plcThread != null || closing) return;
            InitializeCameras();
            plcThread = new Thread(PlcWorker) { IsBackground = true, Name = "PLC inspection" };
            plcThread.SetApartmentState(ApartmentState.STA);
            plcThread.Start();
        }

        // 종료 중에는 UI 갱신을 예약하지 않습니다.
        private void PostUI(Action action)
        {
            if (closing || IsDisposed || !IsHandleCreated) return;
            try
            {
                if (InvokeRequired)
                    BeginInvoke(new Action(() => { if (!closing && !IsDisposed) action(); }));
                else
                    action();
            }
            catch (InvalidOperationException) when (closing || IsDisposed || !IsHandleCreated) { }
        }

        private void SetPlcStatus(string text, Color color)
        {
            PostUI(() => { labelStatus.Text = text; labelStatus.ForeColor = color; });
        }

        private void PlcWorker()
        {
            try
            {
                if (!ConnectPLC()) return;
                while (!plcStop.IsSet)
                {
                    PollPLC();
                    if (plcStop.Wait(PlcPollIntervalMilliseconds)) break;
                }
            }
            catch (OperationCanceledException) when (plcStop.IsSet) { }
            catch (Exception ex)
            {
                SetPlcStatus("PLC 감시 중 오류: " + ex.Message, Color.Red);
                LogText("PLC 감시 중단: " + ex.Message);
            }
            finally
            {
                if (PLC != null)
                {
                    try { PLC.Close(); }
                    catch (Exception ex) { LogText("PLC 종료 오류: " + ex.Message); }
                    finally
                    {
                        try
                        {
                            if (Marshal.IsComObject(PLC)) Marshal.FinalReleaseComObject(PLC);
                        }
                        catch (Exception ex) { LogText("PLC COM 해제 오류: " + ex.Message); }
                        finally { PLC = null; }
                    }
                }
            }
        }

        private void CheckPlcStop()
        {
            if (plcStop.IsSet) throw new OperationCanceledException();
        }

        private void WaitForCapture(int milliseconds)
        {
            if (plcStop.Wait(milliseconds)) throw new OperationCanceledException();
        }

        private void CheckPlcResult(int result, string operation)
        {
            if (result != 0)
                throw new InvalidOperationException(operation + " 실패: 0x" + result.ToString("X8"));
        }

        private void SetPlcDevice(string device, int value)
        {
            CheckPlcStop();
            CheckPlcResult(PLC.SetDevice(device, value), device + " 쓰기");
        }

        private bool ConnectPLC()
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
                    SetPlcStatus("PLC 연결 성공!", Color.Green);
                    return true;
                }
                else
                {
                    SetPlcStatus($"PLC 연결 실패! 오류 코드: {result}", Color.Red);

                    string alcode = "0x" + result.ToString("X").PadLeft(8, '0');
                    string alcoment4 = " ";
                    string alcoment5 = " ";
                    PLC_Mxcomponent_AlarmComment.AlarmComments_MX5.TryGetValue(alcode, out alcoment5);
                    PLC_Mxcomponent_AlarmComment.AlarmComments_MX4.TryGetValue(alcode, out alcoment4);


                    string errorText = DateTime.Now.ToString("yyyyMMdd_HHmmss") +
                                      "\r\n" + alcode +
                                      " : \r\nMX5 : " + alcoment5 +
                                      "\r\nMX4 : " + alcoment4;
                    PostUI(() => textBox_ex.Text = errorText);
                }
            }
            catch (Exception ex)
            {
                SetPlcStatus($"예외 발생 : {ex.Message}", Color.Red);
            }
            return false;
        }

        private void PollPLC()
        {
            CheckPlcStop();
            short insp1, insp2;
            CheckPlcResult(PLC.GetDevice2("R25010.0", out insp1), "1열 시작 비트 읽기");
            CheckPlcResult(PLC.GetDevice2("R25010.5", out insp2), "2열 시작 비트 읽기");
            int number = insp1 == 1 ? 1 : insp2 == 1 ? 2 : 0;
            if (number == 0 || cameraManager == null || !cameraManager.IsReady(number)) return;

            // ROI는 검사 시작 시 한 번 읽습니다. 설정 창에서 저장한 값은 다음 검사에 반영됩니다.
            LoadRoiFromConfig();
            string result = number == 1 ? RunInference1FromPLC() : RunInference2FromPLC();
            CheckPlcStop();
            if (result.StartsWith("❌"))
                throw new InvalidOperationException(result);

            SetPlcDevice(number == 1 ? "R25010.0" : "R25010.5", 0);
            SetPlcDevice(number == 1 ? "R25010.3" : "R25010.8", 1);
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
                var previous = store.Load();
                var configuration = store.Load(devices);
                if (string.IsNullOrWhiteSpace(previous.Camera1DevicePath))
                    LogText("CAM1 경로 자동 설정: " + configuration.Camera1DevicePath);
                if (string.IsNullOrWhiteSpace(previous.Camera2DevicePath))
                    LogText("CAM2 경로 자동 설정: " + configuration.Camera2DevicePath);
                cameraManager.Start(configuration);
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
            if (closing) return;
            bool ready1 = CheckAndRecoverCamera(1);
            bool ready2 = CheckAndRecoverCamera(2);
            Label_CAM1_Status.Text = ready1 ? "CAM1 정상 연결" : "CAM1 프레임 없음 / 복구 시도 중";
            Label_CAM2_Status.Text = ready2 ? "CAM2 정상 연결" : "CAM2 프레임 없음 / 복구 시도 중";
            Label_CAM1_Status.ForeColor = ready1 ? Color.Green : Color.Red;
            Label_CAM2_Status.ForeColor = ready2 ? Color.Green : Color.Red;
        }

        private bool CheckAndRecoverCamera(int number)
        {
            if (cameraManager == null) return false;

            if (cameraManager.IsReady(number))
            {
                if (awaitingRecovery[number])
                    LogText($"CAM{number} 프레임 수신 복구 완료");
                awaitingRecovery[number] = false;
                cameraMissingSince[number] = null;
                return true;
            }

            DateTime now = DateTime.UtcNow;
            if (!cameraMissingSince[number].HasValue)
            {
                cameraMissingSince[number] = now;
                LogText($"CAM{number} 프레임 수신 중단 감지");
            }

            if (now - cameraMissingSince[number].Value < CameraLostDelay ||
                now - lastReconnectAttempt[number] < CameraRetryInterval)
                return false;

            lastReconnectAttempt[number] = now;
            awaitingRecovery[number] = true;
            LogText($"CAM{number} 카메라 재연결 시도");
            try
            {
                cameraManager.Reconnect(number);
            }
            catch (Exception ex)
            {
                LogText($"CAM{number} 카메라 재연결 오류: {ex.Message}");
            }

            // 새 그래프가 열려도 실제 프레임이 들어오기 전까지는 복구로 판단하지 않습니다.
            return false;
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
            lock (RoiConfig.FileSync)
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
        }

        // 기존 RunInference1FromPLC 수정: camera1 사용
        private string RunInference1FromPLC()
        {
            return CaptureAndRunInference(1, roi1_A, roi1_B);
        }

        // RunInference2FromPLC도 camera2를 사용
        private string RunInference2FromPLC()
        {
            return CaptureAndRunInference(2, roi2_A, roi2_B);
        }

        private string CaptureAndRunInference(int number, Rectangle roiA, Rectangle roiB)
        {
            if (cameraManager == null || !cameraManager.IsReady(number))
                return "❌ camera" + number + "이 열려있지 않습니다.";

            var cropped1List = new List<Bitmap>();
            var cropped2List = new List<Bitmap>();
            try
            {
                WaitForCapture(200);
                for (int i = 0; i < 11; i++)
                {
                    CheckPlcStop();
                    using (var frameMat = GetLatestFrameClone(number))
                    {
                        if (frameMat == null || frameMat.Empty())
                        {
                            WaitForCapture(100);
                            continue;
                        }
                        if (i == 1)
                        {
                            using (var original = BitmapConverter.ToBitmap(frameMat))
                                SaveOriginalFrame(original, "camera" + number);
                        }

                        string markA, markB;
                        double scoreA, scoreB;
                        using (var bmpA = CropByMarksToBitmap(frameMat, roiA, "ROI" + number + "_A", out markA, out scoreA))
                        using (var bmpB = CropByMarksToBitmap(frameMat, roiB, "ROI" + number + "_B", out markB, out scoreB))
                        {
                            if (bmpA == null || bmpB == null || i == 0)
                            {
                                WaitForCapture(100);
                                continue;
                            }
                            cropped1List.Add(new Bitmap(bmpA));
                            cropped2List.Add(new Bitmap(bmpB));
                            ShowCropPreview(number, bmpA, bmpB);
                        }
                    }
                    WaitForCapture(200);
                }
                if (cropped1List.Count == 0 || cropped2List.Count == 0)
                    return "❌ crop 결과가 없습니다(템플릿 매칭 실패 가능).";
                CheckPlcStop();
                return RunModelWithPlcControl(cropped1List, cropped2List, number);
            }
            finally
            {
                foreach (var bitmap in cropped1List) bitmap.Dispose();
                foreach (var bitmap in cropped2List) bitmap.Dispose();
            }
        }

        private void ShowCropPreview(int number, Bitmap imageA, Bitmap imageB)
        {
            if (closing || IsDisposed || !IsHandleCreated) return;
            // Invoke가 끝날 때까지 입력 Bitmap을 유지합니다. 복사와 교체는 UI에서 수행합니다.
            Action update = () =>
            {
                if (closing || IsDisposed) return;
                PictureBox boxA = number == 1 ? pb_1_A : pb_2_A;
                PictureBox boxB = number == 1 ? pb_1_B : pb_2_B;
                Image oldA = boxA.Image;
                Image oldB = boxB.Image;
                boxA.Image = new Bitmap(imageA);
                boxB.Image = new Bitmap(imageB);
                oldA?.Dispose();
                oldB?.Dispose();
            };
            try
            {
                if (InvokeRequired) Invoke(update);
                else update();
            }
            catch (InvalidOperationException) when (closing || IsDisposed || !IsHandleCreated) { }
        }

        private string RunModelWithPlcControl(List<Bitmap> cropped1List, List<Bitmap> cropped2List, int roiIndex)
        {
            var labels1 = new List<string>();
            var labels2 = new List<string>();

            foreach (var cropped1 in cropped1List)
            {
                CheckPlcStop();
                labels1.Add(classifier.Classify(cropped1).Label);
            }
            foreach (var cropped2 in cropped2List)
            {
                CheckPlcStop();
                labels2.Add(classifier.Classify(cropped2).Label);
            }
            CheckPlcStop();
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
                    SetPlcDevice("R25010.A", 1);
                    LogCellid("ROI1_A", 84000);
                    LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                }
                if (mode2 == "none")
                {
                    SetPlcDevice("R25010.B", 1);
                    LogCellid("ROI1_B", 84400);
                    LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                }
            }
            else if (roiIndex == 2)
            {
                if (mode1 == "none")
                {
                    SetPlcDevice("R25010.A", 1);
                    LogCellid("ROI2_A", 84800);
                    LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                }
                if (mode2 == "none")
                {
                    SetPlcDevice("R25010.B", 1);
                    LogCellid("ROI2_B", 85200);
                    LogRollerCount("롤러 사용횟수", 43163);  // ✅ 추가
                }
            }

            if (mode1 == "none" && mode2 == "none")
            {
                SetPlcDevice("R25010.1", 1);
                LogText(" 롤러 교체 완료");
            }
            else if (mode1 == "detect" || mode2 == "detect")
            {
                SetPlcDevice("R25010.2", 1);
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


        private void btn_ROI1_Click(object sender, EventArgs e)
        {
            OpenRoiSettings(1);
        }

        private void btn_ROI2_Click(object sender, EventArgs e)
        {
            OpenRoiSettings(2);
        }

        private void OpenRoiSettings(int number)
        {
            if (closing || _cropper == null) return;
            // 설정용과 검사용의 마크 이미지 및 매칭 잠금을 분리합니다.
            using (var roiCropper = new Cropper
            {
                DefaultMinScore = _cropper.DefaultMinScore,
                DefaultExtraDown = _cropper.DefaultExtraDown,
                DefaultExtraX = _cropper.DefaultExtraX
            })
            {
                roiCropper.LoadMarks(Path.Combine(Application.StartupPath, "templates"), "mark*.png");
                using (var roiForm = new RoiForm(number, () => GetLatestFrameClone(number), roiCropper))
                    roiForm.ShowDialog(this);
            }
        }

        private void LogText(string message)
        {
            if (closing || IsDisposed || !IsHandleCreated) return;
            if (InvokeRequired)
            {
                PostUI(() => LogText(message));
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
                PostUI(() =>
                {
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
                });
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
            CheckPlcStop();
            int[] data = { value };
            CheckPlcResult(PLC.WriteDeviceBlock($"ZR{address}", 1, ref data[0]), $"ZR{address} 쓰기");
            LogText($"ZR{address} <= {value} (쓰기 성공)");
        }

        private int ModeToInt(string mode)
        {
            // detect -> 1, none -> 0 (그 외 입력도 안전하게 0)
            return string.Equals(mode, "detect", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
        }

        // 종료 시 카메라 해제
        protected override async void OnFormClosing(FormClosingEventArgs e)
        {
            if (shutdownComplete)
            {
                base.OnFormClosing(e);
                return;
            }
            if (closing) { e.Cancel = true; return; }
            base.OnFormClosing(e);
            if (e.Cancel) return;
            e.Cancel = true;
            closing = true;
            cameraStatusTimer.Stop();
            plcStop.Set();

            // UI 메시지 루프를 유지해 진행 중인 Invoke와 PLC 호출이 끝날 수 있게 합니다.
            await Task.Yield();
            if (plcThread != null)
                await Task.Run(() => plcThread.Join());

            TcpImgSender.Stop();
            cameraStatusTimer.Dispose();
            cameraManager?.Dispose();
            cameraManager = null;
            _cropper?.Dispose();
            _cropper = null;
            classifier?.Dispose();
            classifier = null;
            foreach (var box in new[] { pb_1_A, pb_1_B, pb_2_A, pb_2_B })
            {
                box.Image?.Dispose();
                box.Image = null;
            }
            plcStop.Dispose();
            shutdownComplete = true;
            Close();
        }
    }

}
