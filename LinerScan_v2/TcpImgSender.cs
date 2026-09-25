using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace LinerScan.Networking
{
    /// <summary>
    /// 하나의 TCP 연결을 재사용하여 파일을 IMGV 프로토콜로 전송하는 정적 송신기입니다.
    /// 프레임 쓰기는 잠금으로 직렬화하지만 연결 확인과 재연결은 잠금 밖에서 수행합니다.
    /// </summary>
    public static class TcpImgSender
    {
        /// <summary>
        /// 송신 상태를 전달하는 선택적 로그 콜백입니다.
        /// 호출 스레드에서 실행되므로 UI 갱신이 필요하면 콜백에서 UI 스레드로 전환해야 합니다.
        /// </summary>
        public static Action<string> Logger { get; set; }
        /// <summary>
        /// 등록된 로그 콜백과 디버그 출력으로 메시지를 전달합니다. 콜백의 예외는 별도로 처리하지 않습니다.
        /// </summary>
        private static void Log(string msg) { Logger?.Invoke(msg); Debug.WriteLine(msg); }

        private static readonly object _sendLock = new object();
        private static TcpClient _client;
        private static NetworkStream _ns;
        private static string _ip;
        private static int _port;

        /// <summary>
        /// 접속할 IP와 포트를 저장하고 연결을 확인하거나 새로 연결합니다.
        /// 연결 준비 여부를 반환합니다. 이미 유효한 연결이 있으면 새 주소로 즉시 전환하지 않습니다.
        /// 연결 상태를 잠그지 않으므로 송신 시작 전에 호출해야 합니다.
        /// </summary>
        public static bool Start(string ip, int port)
        {
            _ip = ip; _port = port;
            return EnsureConnectedNoLock();
        }

        /// <summary>
        /// 송신 잠금을 획득한 뒤 스트림과 TCP 클라이언트를 닫고 참조를 비웁니다.
        /// 닫기 오류는 무시하며, 저장된 접속 정보는 유지되어 다음 송신에서 재연결할 수 있습니다.
        /// 영구적인 송신 중지 플래그는 설정하지 않습니다.
        /// </summary>
        public static void Stop()
        {
            lock (_sendLock)
            {
                try { _ns?.Close(); } catch { }
                try { _client?.Close(); } catch { }
                _ns = null; _client = null;
            }
        }

        /// <summary>
        /// 소켓이 없거나 읽기 가능 상태에서 수신 데이터가 없는 경우 연결이 끊긴 것으로 판단합니다.
        /// 소켓 검사 중 예외가 발생하면 false를 반환합니다. true라도 이후 송신 성공을 보장하지 않습니다.
        /// </summary>
        private static bool IsSocketAlive(TcpClient c)
        {
            try
            {
                if (c == null || c.Client == null) return false;

                // Poll(SelectRead) && Available==0  => 상대가 정상적으로 종료했거나 연결이 끊김
                if (c.Client.Poll(0, SelectMode.SelectRead) && c.Client.Available == 0)
                    return false;

                return true;
            }
            catch
            {
                return false;
            }
        }


        /// <summary>
        /// 기존 소켓과 스트림이 사용 가능하면 재사용하고, 그렇지 않으면 정리 후 다시 연결합니다.
        /// 접속 완료를 최대 2초 기다리고 성공 시 쓰기 제한 시간을 3초로 설정합니다.
        /// 접속 실패를 기록하고 false를 반환합니다. 내부 잠금이 없어 동시 재연결을 보호하지 않습니다.
        /// </summary>
        private static bool EnsureConnectedNoLock()
        {
            try
            {
                // 이미 살아있으면 OK (Connected는 약하니, 실제 송신 실패 시 재연결로 커버)
                if (_client != null && IsSocketAlive(_client) && _ns != null && _ns.CanWrite)
                    return true;

                try { _ns?.Close(); } catch { }
                try { _client?.Close(); } catch { }
                _ns = null; _client = null;

                _client = new TcpClient();
                _client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);

                var ar = _client.BeginConnect(_ip, _port, null, null);
                if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(2)))
                {
                    try { _client.Close(); } catch { }
                    Log($"❌ [TCP] Connect timeout (2s) | {_ip}:{_port}");
                    return false;
                }
                _client.EndConnect(ar);

                _ns = _client.GetStream();
                _ns.WriteTimeout = 3000;

                Log($"✅ [TCP] Connected (persistent) | {_ip}:{_port}");
                return true;
            }
            catch (Exception ex)
            {
                Log($"❌ [TCP] EnsureConnected fail | {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 파일 전체를 메모리로 읽고 UTF-8 파일명과 함께 전송합니다.
        /// 경로가 없거나 파일이 비어 있으면 false를 반환하며, 파일 읽기 예외는 호출자에게 전달합니다.
        /// 프레임 송신 실패 시 연결을 다시 확인하고 한 번 더 전송을 시도합니다.
        /// true는 로컬 스트림 쓰기 완료를 뜻하며 서버의 수신·저장 확인 응답을 기다리지는 않습니다.
        /// 재시도 시 서버에서 이미 받은 데이터와 중복될 수 있으므로 정확히 한 번 전달을 보장하지 않습니다.
        /// </summary>
        public static bool SendFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
            {
                Log($"❌ [TCP] invalid file: {filePath}");
                return false;
            }

            byte[] imgBytes = File.ReadAllBytes(filePath);
            if (imgBytes.Length == 0)
            {
                Log($"❌ [TCP] file empty: {filePath}");
                return false;
            }

            string fileName = Path.GetFileName(filePath);
            byte[] nameBytes = Encoding.UTF8.GetBytes(fileName);

            // 1) 연결 확인/재연결 (lock 밖)
            if (!EnsureConnectedNoLock())
                return false;

            // 2) 프레임 전체 Write는 lock으로 직렬화
            if (SendFrameLocked(fileName, nameBytes, imgBytes))
                return true;

            // 3) 실패하면 재연결 후 1회 재시도
            Log("🔁 [TCP] retry after reconnect...");
            if (!EnsureConnectedNoLock())
                return false;

            return SendFrameLocked(fileName, nameBytes, imgBytes);
        }

        /// <summary>
        /// 하나의 잠금 구간에서 IMGV 헤더, 이름 길이(4바이트)와 이름, 파일명 길이(4바이트)와 파일명,
        /// 이미지 길이(8바이트)와 이미지 데이터를 순서대로 기록하여 프레임 간 데이터 혼합을 막습니다.
        /// 길이는 BitConverter의 시스템 바이트 순서를 사용하며 현재 Windows 환경에서는 리틀 엔디언입니다.
        /// 현재 호출 경로에서는 이름과 파일명이 동일하지만 프로토콜에 따라 두 필드를 모두 기록합니다.
        /// 송신 실패 시 연결 자원을 정리하고 false를 반환하여 상위 메서드가 재연결할 수 있게 합니다.
        /// </summary>
        private static bool SendFrameLocked(string fileName, byte[] nameBytes, byte[] imgBytes)
        {
            lock (_sendLock)
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    Log($"➡️ [TCP] Send start | {fileName} | {imgBytes.Length:N0} bytes");

                    // filename bytes
                    if (fileName == null) fileName = "";
                    byte[] fileBytes = Encoding.UTF8.GetBytes(fileName);

                    // ✅ 프레임의 시작~끝까지 하나의 lock 구간에서 전송(서버 요구사항)
                    byte[] magic = { (byte)'I', (byte)'M', (byte)'G', (byte)'V' };
                    _ns.Write(magic, 0, 4);

                    // nameLen + name
                    byte[] nameLenBuf = BitConverter.GetBytes(nameBytes.Length);
                    _ns.Write(nameLenBuf, 0, 4);
                    _ns.Write(nameBytes, 0, nameBytes.Length);

                    // fileLen + filename
                    byte[] fileLenBuf = BitConverter.GetBytes(fileBytes.Length);
                    _ns.Write(fileLenBuf, 0, 4);
                    if (fileBytes.Length > 0)
                        _ns.Write(fileBytes, 0, fileBytes.Length);

                    // imgLen(8) + imgBytes
                    long len = imgBytes.LongLength;
                    byte[] lenBuf = BitConverter.GetBytes(len);
                    _ns.Write(lenBuf, 0, 8);

                    _ns.Write(imgBytes, 0, imgBytes.Length);
                    _ns.Flush();

                    sw.Stop();
                    Log($"✅ [TCP] Send done | {fileName} | {sw.ElapsedMilliseconds} ms");
                    return true;
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    Log($"❌ [TCP] Send fail | {fileName} | {sw.ElapsedMilliseconds} ms | {ex.Message}");

                    // 깨졌으면 정리 (다음 EnsureConnected에서 재연결)
                    try { _ns?.Close(); } catch { }
                    try { _client?.Close(); } catch { }
                    _ns = null; _client = null;

                    return false;
                }
            }
        }
    }
}
