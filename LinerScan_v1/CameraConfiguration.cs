using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace LinerScan.Cameras
{
    public sealed class CameraConfiguration
    {
        public string Camera1DevicePath { get; set; }
        public string Camera2DevicePath { get; set; }

        /// <summary>
        /// 두 카메라의 DevicePath가 비어 있지 않고 서로 다른지 검사합니다.
        /// 경로 비교는 대소문자를 무시하며, 잘못된 설정이면 InvalidOperationException을 발생시킵니다.
        /// 실제 장치의 연결 여부는 검사하지 않습니다.
        /// </summary>
        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(Camera1DevicePath) || string.IsNullOrWhiteSpace(Camera2DevicePath))
                throw new InvalidOperationException("cam.ini에 CAM1_DEVICE_PATH와 CAM2_DEVICE_PATH를 설정하세요. camera-devices.txt에서 ELP 장치 경로를 확인할 수 있습니다.");
            if (string.Equals(Camera1DevicePath, Camera2DevicePath, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("CAM1과 CAM2에는 서로 다른 DevicePath를 지정해야 합니다.");
        }
    }

    public sealed class CameraConfigurationStore
    {
        private readonly string path;
        /// <summary>
        /// INI 파일 경로를 절대 경로로 보관합니다. 파일 생성과 읽기는 Load에서 수행합니다.
        /// </summary>
        public CameraConfigurationStore(string path) { this.path = Path.GetFullPath(path); }

        /// <summary>
        /// Windows INI API로 지정한 섹션과 키의 값을 버퍼에 읽습니다.
        /// 키가 없으면 기본값을 사용하며, 반환값은 종료 문자를 제외한 복사 문자 수입니다.
        /// </summary>
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetPrivateProfileString(string section, string key, string defaultValue,
            StringBuilder result, int size, string filePath);

        /// <summary>
        /// INI에서 CAM1/CAM2의 DevicePath를 읽어 설정 객체를 반환합니다.
        /// 파일이 없으면 경로가 비어 있는 기본 파일을 생성합니다.
        /// 설정 검증은 호출자가 Validate로 수행하며, 기존 Index 설정으로 대체하지 않습니다.
        /// </summary>
        public CameraConfiguration Load()
        {
            if (!File.Exists(path))
                File.WriteAllText(path, "[Camera]\r\nCAM1_DEVICE_PATH=\r\nCAM2_DEVICE_PATH=\r\n", Encoding.Unicode);
            return new CameraConfiguration
            {
                Camera1DevicePath = Read("CAM1_DEVICE_PATH"),
                Camera2DevicePath = Read("CAM2_DEVICE_PATH")
            };
        }

        /// <summary>
        /// 빈 경로만 현재 검색된 HD USB Camera 두 대의 DevicePath로 채워 cam.ini에 저장합니다.
        /// 다른 카메라가 섞이거나 장치 수가 불확실하면 자동 배정하지 않습니다.
        /// </summary>
        public CameraConfiguration Load(IReadOnlyList<CameraDeviceInfo> devices)
        {
            CameraConfiguration config = Load();
            bool missing1 = string.IsNullOrWhiteSpace(config.Camera1DevicePath);
            bool missing2 = string.IsNullOrWhiteSpace(config.Camera2DevicePath);

            if (!missing1 && !missing2)
                return config;

            if (devices == null)
                throw new ArgumentNullException(nameof(devices));

            var cameras = devices
                .Where(d => string.Equals(d.Name, "HD USB Camera", StringComparison.OrdinalIgnoreCase)
                    && !string.IsNullOrWhiteSpace(d.DevicePath))
                .OrderBy(d => d.DevicePath, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (cameras.Length != 2 ||
                string.Equals(cameras[0].DevicePath, cameras[1].DevicePath,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    "CAM 경로 자동 설정 실패: HD USB Camera가 서로 다른 경로로 정확히 두 대 검색되어야 합니다. camera-devices.txt를 확인하세요.");

            string path1 = config.Camera1DevicePath;
            string path2 = config.Camera2DevicePath;

            if (missing1 && missing2)
            {
                path1 = cameras[0].DevicePath;
                path2 = cameras[1].DevicePath;
            }
            else
            {
                string existing = missing1 ? path2 : path1;
                if (!cameras.Any(d => string.Equals(d.DevicePath, existing,
                    StringComparison.OrdinalIgnoreCase)))
                    throw new InvalidOperationException(
                        "기존 CAM 경로가 검색된 HD USB Camera 두 대 중 하나와 일치하지 않습니다. cam.ini와 camera-devices.txt를 확인하세요.");

                string remaining = cameras
                    .First(d => !string.Equals(d.DevicePath, existing,
                        StringComparison.OrdinalIgnoreCase)).DevicePath;
                if (missing1)
                    path1 = remaining;
                else
                    path2 = remaining;
            }

            if (missing1)
                Write("CAM1_DEVICE_PATH", path1);
            if (missing2)
                Write("CAM2_DEVICE_PATH", path2);

            config.Camera1DevicePath = path1;
            config.Camera2DevicePath = path2;
            return config;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool WritePrivateProfileString(
            string section, string key, string value, string filePath);

        private void Write(string key, string value)
        {
            if (!WritePrivateProfileString("Camera", key, value, path))
                throw new IOException("cam.ini 카메라 경로 저장 실패: " + key);
        }

        /// <summary>
        /// Camera 섹션의 지정 키를 읽고 앞뒤 공백을 제거합니다. 키가 없으면 빈 문자열을 반환합니다.
        /// </summary>
        private string Read(string key)
        {
            var buffer = new StringBuilder(32768);
            GetPrivateProfileString("Camera", key, "", buffer, buffer.Capacity, path);
            return buffer.ToString().Trim();
        }
    }
}
