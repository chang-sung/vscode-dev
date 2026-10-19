using System;
using System.IO;
using System.Text;

namespace log_parser
{
    internal static class Logger
    {
        private static readonly object _sync = new object();
        private static string _logDir;
        private static string _statusUpdateLogDir;


        public static void Initialize(string logDir, string statusUpdateLogDir)
        {
            _logDir = logDir ?? throw new ArgumentNullException(nameof(logDir));
            _statusUpdateLogDir = statusUpdateLogDir ?? throw new ArgumentNullException(nameof(statusUpdateLogDir));
        }

        // ----------- 일반 로그 -----------

        private static string EnsureLogFilePath()
        {
            if (string.IsNullOrWhiteSpace(_logDir))
                throw new InvalidOperationException("Logger.Initialize(...) 를 먼저 호출해야 합니다.");

            Directory.CreateDirectory(_logDir);
            return Path.Combine(_logDir, $"log_{DateTime.Now:yyyyMMdd}.txt");
        }

        public static void Info(string message) => Write(message);

        public static void Error(string message) => Write("[ERROR] " + message);

        private static void Write(string message)
        {
            if (string.IsNullOrEmpty(message)) return;

            string line = $"{DateTime.Now:yyyy-MM-dd HH:mm} {message}";

            try
            {
                lock (_sync)
                {
                    string path = EnsureLogFilePath();
                    using (var sw = new StreamWriter(path, append: true, encoding: new UTF8Encoding(false)))
                    {
                        sw.WriteLine(line);
                    }
                }
            }
            catch
            {
                // 파일 로그 실패는 조용히 무시 (무한루프 방지)
            }
        }

        // ----------- 구매 상태 상세 로그 -----------

        public static void AppendStatusUpdate(string message)
        {
            if (string.IsNullOrEmpty(message)) return;

            try
            {
                if (string.IsNullOrWhiteSpace(_statusUpdateLogDir))
                    throw new InvalidOperationException("Logger.Initialize(...) 를 먼저 호출해야 합니다.");

                Directory.CreateDirectory(_statusUpdateLogDir);

                string filePath = Path.Combine(
                    _statusUpdateLogDir,
                    $"PurchaseStatus_{DateTime.Now:yyyy-MM-dd}.txt"
                );

                string line = $"{DateTime.Now:yyyy-MM-dd}, {message}{Environment.NewLine}";

                lock (_sync)
                {
                    File.AppendAllText(filePath, line, Encoding.UTF8);
                }
            }
            catch (Exception ex)
            {
                // 상세 로그 실패는 메인 로그에만 남김
                Error($"상세 로그 저장 실패: {ex.Message}");
            }
        }
    }
}
