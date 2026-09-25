using System;
using System.Diagnostics;
using OpenCvSharp;

namespace LinerScan.Cameras
{
    // Owns the frame copy; callers own and dispose returned clones.
    internal sealed class LatestFrameBuffer : IDisposable
    {
        private readonly object gate = new object();
        private Mat frame;
        private long timestamp;
        private bool disposed;
        private const double MaximumAgeSeconds = 2;

        /// <summary>
        /// 입력 Mat을 복제하여 최신 프레임을 교체하고 수신 시각을 기록합니다.
        /// 잠금 안에서 이전 프레임을 해제하므로 조회 중 해제를 방지합니다. 입력 Mat은 호출자 소유입니다.
        /// 버퍼가 이미 해제되었으면 저장하지 않습니다.
        /// </summary>
        public void Store(Mat source)
        {
            lock (gate)
            {
                if (disposed) return;
                var replacement = source.Clone();
                frame?.Dispose();
                frame = replacement;
                timestamp = Stopwatch.GetTimestamp();
            }
        }

        /// <summary>
        /// 잠금으로 보호된 상태에서 유효시간 이내의 프레임이 있는지 확인합니다.
        /// </summary>
        public bool HasFreshFrame
        {
            get { lock (gate) return IsFresh(); }
        }

        /// <summary>
        /// 해제되지 않은 프레임이 있고 마지막 저장 이후 2초 미만인지 판정합니다.
        /// 시스템 시각 변경의 영향을 받지 않는 Stopwatch 시간을 사용하며, gate 잠금 안에서 호출해야 합니다.
        /// </summary>
        private bool IsFresh()
        {
            return !disposed && frame != null &&
                (Stopwatch.GetTimestamp() - timestamp) / (double)Stopwatch.Frequency < MaximumAgeSeconds;
        }

        /// <summary>
        /// 유효한 최신 프레임을 잠금 안에서 복제하여 반환하고, 만료되었거나 없으면 null을 반환합니다.
        /// 반환된 Mat은 내부 버퍼와 독립적이며 호출자가 Dispose해야 합니다.
        /// </summary>
        public Mat GetClone()
        {
            lock (gate) return IsFresh() ? frame.Clone() : null;
        }

        /// <summary>
        /// 잠금 안에서 보관 프레임을 해제하고 이후 저장과 조회를 차단합니다. 중복 호출이 가능합니다.
        /// </summary>
        public void Dispose()
        {
            lock (gate)
            {
                disposed = true;
                frame?.Dispose();
                frame = null;
            }
        }
    }
}
