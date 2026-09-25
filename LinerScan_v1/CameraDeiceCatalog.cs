using System;
using System.Collections.Generic;
using System.Linq;
using DirectShowLib;

namespace LinerScan.Cameras
{
    public sealed class CameraDeviceInfo
    {
        public string Name { get; }
        public string DevicePath { get; }
        /// <summary>
        /// 장치 표시 이름과 식별 경로를 보관합니다. COM 장치 객체는 소유하지 않습니다.
        /// </summary>
        public CameraDeviceInfo(string name, string devicePath)
        {
            Name = name;
            DevicePath = devicePath;
        }
    }

    public sealed class CameraDeviceCatalog
    {
        /// <summary>
        /// DirectShow 비디오 입력 장치의 이름과 DevicePath 목록을 반환합니다.
        /// 문자열 정보만 복사하고 열거에 사용한 DsDevice는 해제하며, 카메라 캡처는 시작하지 않습니다.
        /// </summary>
        public IReadOnlyList<CameraDeviceInfo> GetDevices()
        {
            var devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            try
            {
                return devices.Select(d => new CameraDeviceInfo(d.Name, d.DevicePath)).ToArray();
            }
            finally
            {
                foreach (var device in devices) device.Dispose();
            }
        }

        /// <summary>
        /// DevicePath가 대소문자 구분 없이 일치하는 장치의 moniker를 소스 필터로 바인딩합니다.
        /// 일치하는 장치가 없으면 예외를 발생시키며, 다른 장치를 대신 선택하지 않습니다.
        /// 반환한 IBaseFilter의 COM 참조는 호출자가 해제해야 합니다.
        /// </summary>
        internal IBaseFilter Bind(string devicePath)
        {
            var devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            try
            {
                var device = devices.FirstOrDefault(d => string.Equals(d.DevicePath, devicePath,
                    StringComparison.OrdinalIgnoreCase));
                if (device == null) throw new InvalidOperationException("카메라를 찾을 수 없습니다: " + devicePath);
                object source;
                var filterId = typeof(IBaseFilter).GUID;
                device.Mon.BindToObject(null, null, ref filterId, out source);
                return (IBaseFilter)source;
            }
            finally
            {
                foreach (var device in devices) device.Dispose();
            }
        }
    }
}
