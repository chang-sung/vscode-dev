using System.Drawing;
using OpenCvSharp;

namespace LinerScan.Imaging
{
    public static class Cropper
    {
        // ROI 전체를 원본 해상도 그대로 복사합니다. 호출자가 반환 Mat을 해제합니다.
        // 범위를 벗어난 ROI는 잘라내거나 이동시키지 않고 검사 실패로 처리합니다.
        public static Mat CropRoi(Mat frame, Rectangle roi)
        {
            if (frame == null || frame.Empty() ||
                roi.X < 0 || roi.Y < 0 || roi.Width <= 0 || roi.Height <= 0 ||
                (long)roi.X + roi.Width > frame.Width ||
                (long)roi.Y + roi.Height > frame.Height)
                return null;

            using (var region = new Mat(frame, new Rect(roi.X, roi.Y, roi.Width, roi.Height)))
                return region.Clone();
        }
    }
}
