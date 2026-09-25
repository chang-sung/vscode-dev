using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace LinerScan.Imaging
{
    public sealed class Cropper : IDisposable
    {
        private readonly List<Mat> _marks = new List<Mat>();
        private readonly List<string> _markNames = new List<string>();
        private readonly object _marksSync = new object();
        private string _templatesDir;
        private string _markPattern = "mark*.png";
        private string _marksSignature;

        public IReadOnlyList<string> MarkNames
        {
            get
            {
                lock (_marksSync)
                    return _markNames.ToArray();
            }
        }

        public double DefaultMinScore { get; set; } = 0.70;
        public int DefaultExtraDown { get; set; } = 5;
        public int DefaultExtraX { get; set; } = 0;

        public void LoadMarks(
            string templatesDir,
            string pattern = "mark*.png")
        {
            if (string.IsNullOrWhiteSpace(templatesDir))
                throw new ArgumentException("템플릿 폴더 경로가 비어 있습니다.", nameof(templatesDir));

            Directory.CreateDirectory(templatesDir);

            lock (_marksSync)
            {
                _templatesDir = templatesDir;
                _markPattern = pattern;
                _marksSignature = null;
                RefreshMarksIfChangedInternal();
            }
        }

        public void ReloadMarks(
            string templatesDir,
            string pattern = "mark*.png")
        {
            LoadMarks(templatesDir, pattern);
        }

        // 매칭 전에 폴더의 추가/수정/삭제를 확인합니다.
        // 복사 중이거나 손상된 PNG가 있으면 기존 마크를 유지하고 다음 매칭 때 재시도합니다.
        public bool RefreshMarksIfChanged()
        {
            lock (_marksSync)
                return RefreshMarksIfChangedInternal();
        }

        private bool RefreshMarksIfChangedInternal()
        {
            if (_templatesDir == null)
                return false;

            var loaded = new List<Mat>();
            try
            {
                string[] files = Directory.Exists(_templatesDir)
                    ? Directory.GetFiles(_templatesDir, _markPattern)
                        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToArray()
                    : new string[0];

                string signature = string.Join("|", files.Select(f =>
                {
                    var info = new FileInfo(f);
                    return f + ":" + info.Length + ":" + info.LastWriteTimeUtc.Ticks;
                }));

                if (signature == _marksSignature)
                    return false;

                var names = new List<string>();
                foreach (string file in files)
                {
                    Mat mark = Cv2.ImRead(file, ImreadModes.Color);
                    if (mark == null || mark.Empty())
                    {
                        mark?.Dispose();
                        return false;
                    }

                    loaded.Add(mark);
                    names.Add(Path.GetFileName(file));
                }

                ClearMarksInternal();
                _marks.AddRange(loaded);
                loaded.Clear();
                _markNames.AddRange(names);
                _marksSignature = signature;
                return true;
            }
            catch (Exception)
            {
                // 파일 복사가 끝나지 않았거나 잠시 접근할 수 없으면 다음 매칭에서 재시도합니다.
                return false;
            }
            finally
            {
                foreach (Mat mark in loaded)
                    mark.Dispose();
            }
        }

        public static Rect ClampRect(Rect rect, int width, int height)
        {
            int x = Math.Max(0, Math.Min(rect.X, width - 1));
            int y = Math.Max(0, Math.Min(rect.Y, height - 1));

            int cropWidth = Math.Max(
                1,
                Math.Min(rect.Width, width - x));

            int cropHeight = Math.Max(
                1,
                Math.Min(rect.Height, height - y));

            return new Rect(x, y, cropWidth, cropHeight);
        }

        public OpenCvSharp.Point MakeOffsetAuto(
            Mat templ,
            OpenCvSharp.Size cropSize,
            int? extraDown = null,
            int? extraX = null)
        {
            int down = extraDown ?? DefaultExtraDown;
            int x = extraX ?? DefaultExtraX;

            int offsetX =
                templ.Width / 2 - cropSize.Width / 2 + x;

            int offsetY =
                templ.Height / 2 + down;

            return new OpenCvSharp.Point(offsetX, offsetY);
        }

        // 특정 템플릿 하나로 매칭하고 크롭한 영상을 반환합니다.
        public static Mat CropByTemplate(
            Mat frame,
            Rectangle searchRect,
            Mat templateMat,
            OpenCvSharp.Size cropSize,
            OpenCvSharp.Point cropOffset,
            out double score,
            double minScore = 0.6,
            bool useCanny = false)
        {
            score = 0;

            if (frame == null || frame.Empty())
                return null;

            if (templateMat == null || templateMat.Empty())
                return null;

            if (searchRect.Width <= 0 || searchRect.Height <= 0)
                return null;

            Rect searchArea = ClampRect(
                new Rect(
                    searchRect.X,
                    searchRect.Y,
                    searchRect.Width,
                    searchRect.Height),
                frame.Width,
                frame.Height);

            if (templateMat.Width > searchArea.Width ||
                templateMat.Height > searchArea.Height)
                return null;

            using (Mat search = new Mat(frame, searchArea))
            using (Mat searchGray = new Mat())
            using (Mat templGray = new Mat())
            using (Mat result = new Mat())
            {
                Cv2.CvtColor(
                    search,
                    searchGray,
                    ColorConversionCodes.BGR2GRAY);

                Cv2.CvtColor(
                    templateMat,
                    templGray,
                    ColorConversionCodes.BGR2GRAY);

                Cv2.GaussianBlur(
                    searchGray,
                    searchGray,
                    new OpenCvSharp.Size(3, 3),
                    0);

                Cv2.GaussianBlur(
                    templGray,
                    templGray,
                    new OpenCvSharp.Size(3, 3),
                    0);

                if (useCanny)
                {
                    using (Mat searchEdge = new Mat())
                    using (Mat templEdge = new Mat())
                    {
                        Cv2.Canny(
                            searchGray,
                            searchEdge,
                            60,
                            180);

                        Cv2.Canny(
                            templGray,
                            templEdge,
                            60,
                            180);

                        Cv2.MatchTemplate(
                            searchEdge,
                            templEdge,
                            result,
                            TemplateMatchModes.CCoeffNormed);
                    }
                }
                else
                {
                    Cv2.MatchTemplate(
                        searchGray,
                        templGray,
                        result,
                        TemplateMatchModes.CCoeffNormed);
                }

                double maxValue;
                OpenCvSharp.Point maxLocation;

                Cv2.MinMaxLoc(
                    result,
                    out _,
                    out maxValue,
                    out _,
                    out maxLocation);

                score = maxValue;

                if (maxValue < minScore)
                    return null;

                int anchorX = searchArea.X + maxLocation.X;
                int anchorY = searchArea.Y + maxLocation.Y;

                Rect cropRect = ClampRect(
                    new Rect(
                        anchorX + cropOffset.X,
                        anchorY + cropOffset.Y,
                        cropSize.Width,
                        cropSize.Height),
                    frame.Width,
                    frame.Height);

                return new Mat(frame, cropRect).Clone();
            }
        }

        // mark1~4 중 매칭 점수가 가장 높은 템플릿으로 크롭합니다.
        public Mat TryCropWithAnyMark(
            Mat frame,
            Rectangle searchRect,
            OpenCvSharp.Size cropSize,
            out string usedMarkName,
            out double usedScore,
            double? minScore = null,
            bool useCanny = false,
            int? extraDown = null,
            int? extraX = null)
        {
            Rect cropRect;
            System.Drawing.Point markCenter;

            bool found = TryFindCropRectWithAnyMark(
                frame,
                searchRect,
                cropSize,
                out cropRect,
                out markCenter,
                out usedMarkName,
                out usedScore,
                minScore,
                useCanny,
                extraDown,
                extraX);

            if (!found)
                return null;

            return new Mat(frame, cropRect).Clone();
        }

        // 기존 호출부에서 사용하던 형식을 유지합니다.
        public bool TryFindCropRectWithAnyMark(
            Mat frame,
            Rectangle searchRect,
            OpenCvSharp.Size cropSize,
            out Rect cropRect,
            out string usedMarkName,
            out double usedScore,
            double? minScore = null,
            bool useCanny = false,
            int? extraDown = null,
            int? extraX = null)
        {
            System.Drawing.Point markCenter;

            return TryFindCropRectWithAnyMark(
                frame,
                searchRect,
                cropSize,
                out cropRect,
                out markCenter,
                out usedMarkName,
                out usedScore,
                minScore,
                useCanny,
                extraDown,
                extraX);
        }

        // RoiForm에서 사용하는 형식: 크롭 영역과 실제 마크 중심을 반환합니다.
        public bool TryFindCropRectWithAnyMark(
            Mat frame,
            Rectangle searchRect,
            OpenCvSharp.Size cropSize,
            out Rect cropRect,
            out System.Drawing.Point markCenter,
            out string usedMarkName,
            out double usedScore,
            double? minScore = null,
            bool useCanny = false,
            int? extraDown = null,
            int? extraX = null)
        {
            cropRect = new Rect();
            markCenter = System.Drawing.Point.Empty;
            usedMarkName = null;
            usedScore = 0;

            if (frame == null || frame.Empty())
                return false;

            lock (_marksSync)
            {
                RefreshMarksIfChangedInternal();
                if (_marks.Count == 0)
                    return false;

                double threshold = minScore ?? DefaultMinScore;
                double bestScore = double.MinValue;
                bool found = false;
    
                for (int i = 0; i < _marks.Count; i++)
                {
                    Mat templ = _marks[i];
    
                    OpenCvSharp.Point offset = MakeOffsetAuto(
                        templ,
                        cropSize,
                        extraDown,
                        extraX);
    
                    Rect candidateRect;
                    System.Drawing.Point candidateCenter;
                    double candidateScore;
    
                    bool ok = TryFindCropRectByTemplate(
                        frame,
                        searchRect,
                        templ,
                        cropSize,
                        offset,
                        out candidateRect,
                        out candidateCenter,
                        out candidateScore,
                        threshold,
                        useCanny);
    
                    if (!ok)
                        continue;
    
                    if (found && candidateScore <= bestScore)
                        continue;
    
                    found = true;
                    bestScore = candidateScore;
                    cropRect = candidateRect;
                    markCenter = candidateCenter;
                    usedMarkName = _markNames[i];
                    usedScore = candidateScore;
                }
    
                return found;
            }
        }

        private static bool TryFindCropRectByTemplate(
            Mat frame,
            Rectangle searchRect,
            Mat templateMat,
            OpenCvSharp.Size cropSize,
            OpenCvSharp.Point cropOffset,
            out Rect cropRect,
            out System.Drawing.Point markCenter,
            out double score,
            double minScore,
            bool useCanny)
        {
            cropRect = new Rect();
            markCenter = System.Drawing.Point.Empty;
            score = 0;

            if (frame == null || frame.Empty())
                return false;

            if (templateMat == null || templateMat.Empty())
                return false;

            if (searchRect.Width <= 0 || searchRect.Height <= 0)
                return false;

            Rect searchArea = ClampRect(
                new Rect(
                    searchRect.X,
                    searchRect.Y,
                    searchRect.Width,
                    searchRect.Height),
                frame.Width,
                frame.Height);

            if (templateMat.Width > searchArea.Width ||
                templateMat.Height > searchArea.Height)
                return false;

            using (Mat search = new Mat(frame, searchArea))
            using (Mat searchGray = new Mat())
            using (Mat templGray = new Mat())
            using (Mat result = new Mat())
            {
                Cv2.CvtColor(
                    search,
                    searchGray,
                    ColorConversionCodes.BGR2GRAY);

                Cv2.CvtColor(
                    templateMat,
                    templGray,
                    ColorConversionCodes.BGR2GRAY);

                Cv2.GaussianBlur(
                    searchGray,
                    searchGray,
                    new OpenCvSharp.Size(3, 3),
                    0);

                Cv2.GaussianBlur(
                    templGray,
                    templGray,
                    new OpenCvSharp.Size(3, 3),
                    0);

                if (useCanny)
                {
                    using (Mat searchEdge = new Mat())
                    using (Mat templEdge = new Mat())
                    {
                        Cv2.Canny(
                            searchGray,
                            searchEdge,
                            60,
                            180);

                        Cv2.Canny(
                            templGray,
                            templEdge,
                            60,
                            180);

                        Cv2.MatchTemplate(
                            searchEdge,
                            templEdge,
                            result,
                            TemplateMatchModes.CCoeffNormed);
                    }
                }
                else
                {
                    Cv2.MatchTemplate(
                        searchGray,
                        templGray,
                        result,
                        TemplateMatchModes.CCoeffNormed);
                }

                double maxValue;
                OpenCvSharp.Point maxLocation;

                Cv2.MinMaxLoc(
                    result,
                    out _,
                    out maxValue,
                    out _,
                    out maxLocation);

                score = maxValue;

                if (maxValue < minScore)
                    return false;

                int anchorX = searchArea.X + maxLocation.X;
                int anchorY = searchArea.Y + maxLocation.Y;

                // 템플릿의 실제 매칭 위치에서 중심을 계산합니다.
                // 아래 ClampRect에 의해 크롭 영역이 바뀌어도 이 값은 유지됩니다.
                markCenter = new System.Drawing.Point(
                    anchorX + templateMat.Width / 2,
                    anchorY + templateMat.Height / 2);

                cropRect = ClampRect(
                    new Rect(
                        anchorX + cropOffset.X,
                        anchorY + cropOffset.Y,
                        cropSize.Width,
                        cropSize.Height),
                    frame.Width,
                    frame.Height);

                return true;
            }
        }

        private void ClearMarksInternal()
        {
            foreach (Mat mark in _marks)
            {
                mark?.Dispose();
            }

            _marks.Clear();
            _markNames.Clear();
        }

        public void Dispose()
        {
            lock (_marksSync)
            {
                _templatesDir = null;
                ClearMarksInternal();
            }
        }
    }
}