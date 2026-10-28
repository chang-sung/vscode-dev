import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['detect', 'none']  # 학습 class_to_idx 순서와 동일해야 함

cropped_image_dir = '/content/drive/MyDrive/Colab Notebooks/film_dataset/cropped_images'
# 10의 기본 체크포인트. 11 결과는 아래 경로로 변경:
# model_path = './shufflenetv2_kfold_checkpoints/best_overall_shufflenetv2.pth'
model_path = 'best_shufflenetv2_film_classifier.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 모델 로드
model = shufflenet_v2_x1_0(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model = model.to(device)
model.eval()

def predict_folder_with_summary(folder_path: str):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])

    total = 0
    counts = {name: 0 for name in class_names}

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        with Image.open(img_path) as img:
            x = transform(img.convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred_idx = logits.argmax(dim=1).item()

        pred_name = class_names[pred_idx]
        print(f"이미지명 : {fname}  추론 결과 : {pred_name}")

        total += 1
        counts[pred_name] += 1

    # ✅ 마지막 요약 출력
    print("\n" + "-" * 60)
    print(f"총 {total}개 | detect {counts.get('detect', 0)}개 | none {counts.get('none', 0)}개")
    print("-" * 60)

# ✅ 실행
predict_folder_with_summary(cropped_image_dir)