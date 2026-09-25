using System;
using System.IO;
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
