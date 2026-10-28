import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['detect', 'none']  # 학습 class_to_idx 순서와 동일해야 함

cropped_image_dir = '/content/drive/MyDrive/Colab Notebooks/film_dataset/cropped_images'
model_path = 'best_resnet50_film_classifier.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 모델 로드
model = resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def predict_folder_with_summary(folder_path: str):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(exts)])

    total = 0
    counts = {name: 0 for name in class_names}

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert('RGB')

        x = transform(img).unsqueeze(0).to(device)

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