using System;
using System.Collections.Generic;

namespace log_parser
{
    /// <summary>
    /// 구매 합의자 / 요청자 이름 → 이메일 매핑용 레지스트리
    /// </summary>
    public static class RequesterMailRegistry
    {
        // 이름 → 이메일 매핑
        private static readonly Dictionary<string, string> _emails =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                { "홍길동", "abc@naver.com" },
                { "가길동", "ddd@naver.com" },
                { "나길동", "ttt@naver.com" },
            };

        public static string GetEmail(string requesterName)
        {
            if (string.IsNullOrWhiteSpace(requesterName))
                return string.Empty;

            if (_emails.TryGetValue(requesterName.Trim(), out var email))
                return email;

            // 미등록 이름인 경우 빈 문자열 또는 "N/A" 등으로 처리 가능
            return string.Empty;
        }
    }
}