using System;
using System.Runtime.InteropServices;
using DirectShowLib;
using OpenCvSharp;

namespace LinerScan.Cameras
{
    [ComVisible(true), ClassInterface(ClassInterfaceType.None)]
    public sealed class DirectShowCamera : ISampleGrabberCB, IDisposable
    {
        private IFilterGraph2 graph;
        private ICaptureGraphBuilder2 builder;
        private IBaseFilter source;
        private IBaseFilter sink;
        private ISampleGrabber grabber;
        private readonly LatestFrameBuffer frames = new LatestFrameBuffer();
        private int width, height, stride;
        private bool bottomUp;
        private volatile bool accepting;
        private volatile string callbackError;
        private bool disposed;
        public string DevicePath { get; }
        public bool IsReady => accepting && callbackError == null && frames.HasFreshFrame;
        public string LastError => callbackError;

        /// <summary>
        /// 연결 대상 DevicePath를 보관합니다. 실제 장치 연결은 Open에서 수행합니다.
        /// </summary>
        public DirectShowCamera(string devicePath) { DevicePath = devicePath; }

        /// <summary>
        /// DevicePath로 소스 필터를 열고 SampleGrabber와 NullRenderer를 연결하여 캡처를 시작합니다.
        /// 1600×1200 캡처 형식과 RGB24 출력을 설정하고 실제 프레임 크기·방향·stride를 저장합니다.
        /// 실패 시 생성된 자원을 해제하고 예외를 전달합니다. 해제된 인스턴스는 재사용할 수 없습니다.
        /// </summary>
        public void Open()
        {
            if (disposed) throw new ObjectDisposedException(nameof(DirectShowCamera));
            if (graph != null) throw new InvalidOperationException("이미 연결된 카메라입니다.");
            try
            {
                graph = (IFilterGraph2)new FilterGraph();
                builder = (ICaptureGraphBuilder2)new CaptureGraphBuilder2();
                Check(builder.SetFiltergraph(graph));
                source = new CameraDeviceCatalog().Bind(DevicePath);
                Check(graph.AddFilter(source, "Camera"));
                SetCaptureFormat(1600, 1200);
                grabber = (ISampleGrabber)new SampleGrabber();
                var requested = new AMMediaType
                {
                    majorType = MediaType.Video,
                    subType = MediaSubType.RGB24,
                    formatType = DirectShowLib.FormatType.VideoInfo
                };
                try { Check(grabber.SetMediaType(requested)); }
                finally { DsUtils.FreeAMMediaType(requested); }
                Check(graph.AddFilter((IBaseFilter)grabber, "Frames"));
                sink = (IBaseFilter)new NullRenderer();
                Check(graph.AddFilter(sink, "Sink"));
                Check(builder.RenderStream(PinCategory.Capture, MediaType.Video, source, (IBaseFilter)grabber, sink));
                var connected = new AMMediaType();
                try
                {
                    Check(grabber.GetConnectedMediaType(connected));
                    if (connected.formatType != DirectShowLib.FormatType.VideoInfo || connected.subType != MediaSubType.RGB24)
                        throw new NotSupportedException("RGB24 VideoInfo 프레임이 필요합니다.");
                    var info = (VideoInfoHeader)Marshal.PtrToStructure(connected.formatPtr, typeof(VideoInfoHeader));
                    width = info.BmiHeader.Width;
                    height = Math.Abs(info.BmiHeader.Height);
                    bottomUp = info.BmiHeader.Height > 0;
                    if (width <= 0 || height == 0 || info.BmiHeader.BitCount != 24)
                        throw new NotSupportedException("잘못된 RGB24 프레임 형식입니다.");
                    stride = checked((width * 3 + 3) & ~3);
                }
                finally { DsUtils.FreeAMMediaType(connected); }
                Check(grabber.SetOneShot(false));
                Check(grabber.SetBufferSamples(false));
                Check(grabber.SetCallback(this, 1));
                accepting = true;
                Check(((IMediaControl)graph).Run());
            }
            catch { Dispose(); throw; }
        }

        /// <summary>
        /// 장치의 지원 형식을 조회하여 요청한 해상도로 설정합니다.
        /// USB 대역폭을 줄이기 위해 MJPEG를 먼저 시도한 뒤 다른 VideoInfo 형식도 시도합니다.
        /// FPS는 선택된 장치 형식의 값을 유지하며, 적용 가능한 형식이 없으면 예외를 발생시킵니다.
        /// 조회용 비관리 메모리, 미디어 형식 및 COM 참조는 사용 후 해제합니다.
        /// </summary>
        private void SetCaptureFormat(int requestedWidth, int requestedHeight)
        {
            object configObject;
            Check(builder.FindInterface(
                PinCategory.Capture,
                MediaType.Video,
                source,
                typeof(IAMStreamConfig).GUID,
                out configObject));

            var config = (IAMStreamConfig)configObject;
            IntPtr caps = IntPtr.Zero;

            try
            {
                int count, size;
                Check(config.GetNumberOfCapabilities(out count, out size));

                if (size <= 0)
                    throw new NotSupportedException("카메라 형식 정보를 읽을 수 없습니다.");

                caps = Marshal.AllocCoTaskMem(size);

                for (int i = 0; i < count; i++)
                {
                    AMMediaType media;
                    Check(config.GetStreamCaps(i, out media, caps));

                    try
                    {
                        if (media.formatType != DirectShowLib.FormatType.VideoInfo ||
                            media.formatPtr == IntPtr.Zero)
                            continue;

                        var info = (VideoInfoHeader)Marshal.PtrToStructure(
                            media.formatPtr,
                            typeof(VideoInfoHeader));

                        if (info.BmiHeader.Width != requestedWidth ||
                            Math.Abs(info.BmiHeader.Height) != requestedHeight)
                            continue;

                        // 설비 PC에서 프레임 수신을 확인한 YUY2 형식만 선택
                        if (media.subType != MediaSubType.YUY2)
                            continue;

                        Check(config.SetFormat(media));
                        return;
                    }
                    finally
                    {
                        DsUtils.FreeAMMediaType(media);
                    }
                }

                throw new NotSupportedException(
                    $"카메라가 {requestedWidth}x{requestedHeight} YUY2 캡처 형식을 지원하지 않습니다.");
            }
            finally
            {
                if (caps != IntPtr.Zero)
                    Marshal.FreeCoTaskMem(caps);

                Release(configObject);
            }
        }

        /// <summary>
        /// 캡처 상태와 프레임 유효시간을 확인하여 최신 Mat 복사본 또는 null을 반환합니다.
        /// 반환된 Mat의 소유권은 호출자에게 있으며 사용 후 Dispose해야 합니다.
        /// </summary>
        public Mat GetLatestFrameClone() => IsReady ? frames.GetClone() : null;
        /// <summary>
        /// ISampleGrabberCB 인터페이스 구현을 위한 미사용 콜백입니다.
        /// 현재 BufferCB 방식으로 등록하므로 샘플을 처리하지 않고 성공 코드 0을 반환합니다.
        /// </summary>
        public int SampleCB(double sampleTime, IMediaSample sample) => 0;

        /// <summary>
        /// DirectShow 스트리밍 스레드에서 전달된 RGB24 버퍼를 BGR Mat으로 해석합니다.
        /// 버퍼 길이와 행 패딩을 확인하고 상하 방향을 보정한 뒤 검은 프레임을 제외합니다.
        /// ROI 좌표계에 맞춰 1600×1200으로 변환하고 자체 복사본을 저장합니다.
        /// 입력 포인터는 콜백 동안만 사용하며, 예외는 기록하여 네이티브 호출자에게 전파하지 않습니다.
        /// </summary>
        public int BufferCB(double sampleTime, IntPtr buffer, int length)
        {
            if (!accepting) return 0;
            try
            {
                if (buffer == IntPtr.Zero || length < checked(stride * height)) return 0;
                using (var raw = Mat.FromPixelData(height, width, MatType.CV_8UC3, buffer, stride))
                using (var oriented = new Mat())
                using (var normalized = new Mat())
                {
                    if (bottomUp) Cv2.Flip(raw, oriented, FlipMode.X);
                    else raw.CopyTo(oriented);
                    var mean = Cv2.Mean(oriented);
                    if ((mean.Val0 + mean.Val1 + mean.Val2) / 3 < 5) return 0;
                    Cv2.Resize(oriented, normalized, new OpenCvSharp.Size(1600, 1200));
                    frames.Store(normalized);
                    callbackError = null;
                }
            }
            catch (Exception ex) { callbackError = ex.Message; } // Never unwind into a native callback.
            return 0;
        }

        /// <summary>
        /// DirectShow HRESULT가 실패 코드이면 대응하는 예외를 발생시킵니다.
        /// </summary>
        private static void Check(int hr) => DsError.ThrowExceptionForHR(hr);
        /// <summary>
        /// 값이 COM 객체이면 RCW의 참조 횟수를 한 번 감소시킵니다. null이나 일반 객체는 무시합니다.
        /// </summary>
        private static void Release(object value)
        {
            if (value != null && Marshal.IsComObject(value)) Marshal.ReleaseComObject(value);
        }

        /// <summary>
        /// 새 프레임 처리를 차단하고 캡처 정지, 콜백 해제, 프레임 및 COM 자원 해제를 수행합니다.
        /// Stop은 진행 중인 콜백을 기다릴 수 있으므로 프레임 잠금을 잡지 않은 상태에서 호출합니다.
        /// 소유 UI 스레드에서 호출하며, 이미 해제한 인스턴스에 대한 중복 호출은 무시합니다.
        /// </summary>
        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            accepting = false;
            // Stop without holding the frame lock: Stop waits for streaming callbacks.
            try { if (graph != null) ((IMediaControl)graph).Stop(); }
            finally
            {
                try { if (grabber != null) grabber.SetCallback(null, 1); }
                finally
                {
                    frames.Dispose();
                    Release(sink); sink = null;
                    Release(grabber); grabber = null;
                    Release(source); source = null;
                    Release(builder); builder = null;
                    Release(graph); graph = null;
                }
            }
        }
    }
}

