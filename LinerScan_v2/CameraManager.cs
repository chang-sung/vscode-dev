using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace LinerScan.Cameras
{
    // Lifecycle methods are called on the owning UI thread; frame access is thread safe.
    public sealed class CameraManager : IDisposable
    {
        private readonly Dictionary<int, DirectShowCamera> cameras = new Dictionary<int, DirectShowCamera>();
        private readonly Action<string> log;
        /// <summary>
        /// 카메라 연결 결과를 전달할 로그 콜백을 보관합니다. null이면 로그를 전달하지 않습니다.
        /// </summary>
        public CameraManager(Action<string> log) { this.log = log; }

        /// <summary>
        /// 설정을 검증한 뒤 기존 연결을 종료하고 CAM1/CAM2를 지정 경로로 연결합니다.
        /// CAM 번호는 검사 위치를 나타내는 논리 번호이며 OpenCV 장치 Index가 아닙니다.
        /// 생명주기 관리를 위해 소유 UI 스레드에서 호출합니다.
        /// </summary>
        public void Start(CameraConfiguration configuration)
        {
            configuration.Validate();
            Stop();
            Open(1, configuration.Camera1DevicePath);
            Open(2, configuration.Camera2DevicePath);
        }

        /// <summary>
        /// 지정한 논리 번호의 카메라를 생성하고 캡처를 시작하여 관리 목록에 등록합니다.
        /// 연결 실패 시 해당 카메라를 해제하고 오류를 기록하여 다른 카메라의 연결을 계속할 수 있게 합니다.
        /// </summary>
        private void Open(int number, string path)
        {
            var camera = new DirectShowCamera(path);
            try
            {
                camera.Open();
                cameras.Add(number, camera);
                log?.Invoke($"CAM{number} DirectShow 연결: {path}");
            }
            catch (Exception ex)
            {
                camera.Dispose();
                log?.Invoke($"CAM{number} 연결 실패: {ex.Message}");
            }
        }

        /// <summary>
        /// 해당 카메라가 등록되어 있고 최근의 유효한 프레임을 제공할 수 있는지 반환합니다.
        /// </summary>
        public bool IsReady(int number)
        {
            DirectShowCamera camera;
            return cameras.TryGetValue(number, out camera) && camera.IsReady;
        }

        /// <summary>
        /// 해당 카메라의 최신 프레임 복사본을 반환합니다. 카메라나 유효한 프레임이 없으면 null입니다.
        /// 반환된 Mat은 호출자가 사용 후 Dispose해야 합니다.
        /// </summary>
        public Mat GetLatestFrameClone(int number)
        {
            DirectShowCamera camera;
            return cameras.TryGetValue(number, out camera) ? camera.GetLatestFrameClone() : null;
        }

        /// <summary>
        /// 등록된 카메라를 순서대로 해제하고 관리 목록을 비웁니다.
        /// Start 및 프레임 조회와 생명주기 변경이 겹치지 않도록 소유 UI 스레드에서 호출합니다.
        /// </summary>
        public void Stop()
        {
            try { foreach (var camera in cameras.Values) camera.Dispose(); }
            finally { cameras.Clear(); }
        }
        /// <summary>
        /// Stop을 호출하여 관리 중인 카메라 연결과 프레임 자원을 정리합니다.
        /// </summary>
        public void Dispose() => Stop();
    }
}
