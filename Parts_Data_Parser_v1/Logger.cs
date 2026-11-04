using System;
using System.Collections.Generic;
using System.Threading;
using System.IO;
using System.Text;

namespace log_parser
{
    internal static class Logger
    {
        private static readonly object _sync = new object();
        private static string _logDir;
        private static string _statusUpdateLogPath;


        public static void Initialize(string logDir, string statusUpdateLogPath)
        {
            _logDir = logDir ?? throw new ArgumentNullException(nameof(logDir));
            _statusUpdateLogPath = statusUpdateLogPath ?? throw new ArgumentNullException(nameof(statusUpdateLogPath));
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
                if (string.IsNullOrWhiteSpace(_statusUpdateLogPath))
                    throw new InvalidOperationException("Logger.Initialize(...) 를 먼저 호출해야 합니다.");

                Directory.CreateDirectory(_statusUpdateLogPath);

                string filePath = Path.Combine(
                    _statusUpdateLogPath,
                    $"PurchaseStatus_{DateTime.Now:yyyy-MM-dd}.txt"
                );

                string line = $"{DateTime.Now:yyyy-MM-dd}| {message}{Environment.NewLine}";

                WithStatusUpdateLock(() => File.AppendAllText(filePath, line, Encoding.UTF8));
            }
            catch (Exception ex)
            {
                // 상세 로그 실패는 메인 로그에만 남김
                Error($"상세 로그 저장 실패: {ex.Message}");
            }
        }

        // 파일 읽기와 추가 쓰기를 같은 잠금 안에서 수행하여 동시 실행도 직렬화한다.
        private static void WithStatusUpdateLock(Action action)
        {
            lock (_sync)
            {
                using (var mutex = new Mutex(false, @"Local\PartsDataParserPurchaseStatusLog"))
                {
                    bool acquired = false;
                    try
                    {
                        try { acquired = mutex.WaitOne(TimeSpan.FromSeconds(30)); }
                        catch (AbandonedMutexException) { acquired = true; }
                        if (!acquired)
                            throw new IOException("구매 상태 로그 잠금 대기 시간이 초과되었습니다.");
                        action();
                    }
                    finally
                    {
                        if (acquired) mutex.ReleaseMutex();
                    }
                }
            }
        }

        // 중복 키는 별도 관리 파일에 저장하고 구매 상태 로그에는 표시하지 않는다.
        // 기존 로그의 DropKey도 읽어 재실행 시 중복 방지를 유지한다.
        public static int AppendDropStatusUpdates(DateTime date, IDictionary<string, string> targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (targets.Count == 0) return 0;
            if (string.IsNullOrWhiteSpace(_statusUpdateLogPath))
                throw new InvalidOperationException("Logger.Initialize(...) 를 먼저 호출해야 합니다.");

            int written = 0;
            WithStatusUpdateLock(() =>
            {
                Directory.CreateDirectory(_statusUpdateLogPath);
                string path = Path.Combine(_statusUpdateLogPath, $"PurchaseStatus_{date:yyyy-MM-dd}.txt");
                string keyDirectory = Path.Combine(_statusUpdateLogPath, "DropStatusKeys");
                Directory.CreateDirectory(keyDirectory);
                string keyPath = Path.Combine(keyDirectory, $"{date:yyyy-MM-dd}.keys");
                const string marker = "| DropKey=";
                var existingKeys = new HashSet<string>(StringComparer.Ordinal);
                if (File.Exists(keyPath))
                {
                    foreach (string key in File.ReadLines(keyPath, Encoding.UTF8))
                    {
                        if (!string.IsNullOrWhiteSpace(key))
                            existingKeys.Add(key.Trim());
                    }
                }

                var migratedKeys = new List<string>();
                if (File.Exists(path))
                {
                    foreach (string line in File.ReadLines(path, Encoding.UTF8))
                    {
                        int markerIndex = line.LastIndexOf(marker, StringComparison.Ordinal);
                        if (markerIndex >= 0)
                        {
                            string key = line.Substring(markerIndex + marker.Length).Trim();
                            if (existingKeys.Add(key))
                                migratedKeys.Add(key);
                        }
                    }
                }

                // 기존 로그에서 읽은 키도 별도 파일로 이전한다.
                File.AppendAllLines(keyPath, migratedKeys, Encoding.UTF8);

                using (var writer = new StreamWriter(path, true, Encoding.UTF8))
                using (var keyWriter = new StreamWriter(keyPath, true, Encoding.UTF8))
                {
                    foreach (var target in targets)
                    {
                        if (!existingKeys.Add(target.Key)) continue;
                        writer.WriteLine($"{date:yyyy-MM-dd}| {target.Value}");
                        writer.Flush();
                        keyWriter.WriteLine(target.Key);
                        keyWriter.Flush();
                        written++;
                    }
                }
            });
            return written;
        }
    }
}
