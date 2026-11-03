using System;
using System.Drawing;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LinerScan.Inference
{
    /// <summary>
    /// 분류된 라벨과 Softmax 확률을 담는 결과 객체입니다.
    /// </summary>
    public sealed class ClassificationResult
    {
        /// <summary>
        /// 모델 출력 순서에 대응하는 분류 이름입니다: detect 또는 none.
        /// </summary>
        public string Label { get; }
        /// <summary>
        /// 선택된 클래스의 Softmax 확률입니다. 정상적인 모델 출력에서는 0~1 범위입니다.
        /// </summary>
        public float Confidence { get; }
        /// <summary>
        /// 전달받은 라벨과 확률을 저장합니다. 값에 대한 별도 검증이나 보정은 하지 않습니다.
        /// </summary>
        public ClassificationResult(string label, float confidence) { Label = label; Confidence = confidence; }
        /// <summary>
        /// 화면과 로그에 표시할 라벨 및 소수점 한 자리 백분율 문자열을 반환합니다.
        /// </summary>
        public override string ToString() => $"{Label} ({Confidence:P1})";
    }

    /// <summary>
    /// ONNX 모델 세션을 소유하고 이미지 전처리, 추론, 분류 결과 계산을 담당합니다.
    /// </summary>
    public sealed class ImageClassifier : IDisposable
    {
        private readonly InferenceSession session;
        private readonly string[] labels = { "detect", "none" };
        /// <summary>
        /// 지정한 ONNX 모델을 로드하여 추론 세션을 생성합니다.
        /// 파일 또는 모델 로딩 오류는 호출자에게 전달합니다.
        /// </summary>
        public ImageClassifier(string modelPath) { session = new InferenceSession(modelPath); }

        /// <summary>
        /// 이미지의 RGB 값을 0~1로 변환하고 채널별 평균과 표준편차로 정규화합니다.
        /// NCHW 배열 [1, 3, 높이, 너비]를 모델의 input 입력에 전달하여 추론합니다.
        /// 첫 번째 출력의 두 점수를 detect, none 순서로 해석하고 Softmax 확률이 가장 큰 결과를 반환합니다.
        /// 이미지 크기를 변경하지 않으므로 호출자가 모델에 맞는 ROI 이미지를 전달해야 합니다.
        /// 입력 Bitmap은 호출자 소유로 유지하며, 추론 결과의 네이티브 자원은 이 메서드에서 해제합니다.
        /// </summary>
        public ClassificationResult Classify(Bitmap image)
        {
            int width = image.Width, height = image.Height;
            var input = new float[3 * height * width];
            // 학습 시 사용한 RGB 채널별 정규화 기준을 적용합니다.
            float[] mean = { 0.485f, 0.456f, 0.406f };
            float[] std = { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    int offset = y * width + x;
                    input[offset] = (pixel.R / 255f - mean[0]) / std[0];
                    input[height * width + offset] = (pixel.G / 255f - mean[1]) / std[1];
                    input[2 * height * width + offset] = (pixel.B / 255f - mean[2]) / std[2];
                }
            var tensor = new DenseTensor<float>(input, new[] { 1, 3, height, width });
            using (var result = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensor) }))
            {
                var logits = result.First().AsEnumerable<float>().ToArray();
                if (logits.Length != labels.Length) throw new InvalidOperationException("모델 출력 클래스 개수가 다릅니다.");
                // 지수 계산이 지나치게 커지지 않도록 최댓값을 뺀 뒤 Softmax를 계산합니다.
                float max = logits.Max();
                var exp = logits.Select(value => (float)Math.Exp(value - max)).ToArray();
                int prediction = Array.IndexOf(logits, max);
                return new ClassificationResult(labels[prediction], exp[prediction] / exp.Sum());
            }
        }
        /// <summary>
        /// 소유한 ONNX 추론 세션을 해제합니다. 진행 중인 추론이 끝난 뒤 호출해야 합니다.
        /// </summary>
        public void Dispose() => session.Dispose();
    }
}
