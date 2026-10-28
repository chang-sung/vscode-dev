import os
import copy
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision.models import shufflenet_v2_x1_0
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
batch_size = 8
num_epochs = 10
log_interval = 10  # 첫 배치, 이후 10배치마다 진행 상황 출력

# ✅ 전처리 완료된 이미지 폴더
train_dir = r'C:\Users\Home\Desktop\dataset\train'
val_dir = r'C:\Users\Home\Desktop\dataset\valid'

# ✅ transforms 정의 (단순 Normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 데이터셋 & 데이터로더
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=transform),
    'val': datasets.ImageFolder(val_dir, transform=transform)
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
    for x in ['train', 'val']
}

if len(image_datasets['train'].classes) != num_classes:
    raise ValueError(f"학습 폴더는 {num_classes}개 클래스가 필요합니다.")
if image_datasets['train'].class_to_idx != image_datasets['val'].class_to_idx:
    raise ValueError("train/valid의 클래스 폴더 이름과 순서가 일치해야 합니다.")
print("Classes:", image_datasets['train'].class_to_idx, flush=True)
print("선택 장치:", device, flush=True)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
for name in ['train', 'val']:
    print(f"{name}: 이미지 {len(image_datasets[name]):,}장 / "
          f"배치 {len(dataloaders[name]):,}개", flush=True)

# ✅ 로컬 ShuffleNet V2 x1.0 기본 가중치 불러오기
local_weight_path = r'C:\Users\Home\Desktop\transparent_object\shufflenetv2_x1-5666bf0f80.pth'
model = shufflenet_v2_x1_0(weights=None)
state_dict = torch.load(local_weight_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)

# ✅ 마지막 fc 레이어 교체 (2-class 분류)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
print("모델 장치:", next(model.parameters()).device, flush=True)

# ✅ 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ✅ EarlyStopping & ModelCheckpoint 설정
best_acc = -1.0
best_val_loss = float('inf')  # 검증 Loss의 최저값 (최고 Acc와 다른 epoch일 수 있음)
patience = 5
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

train_acc_list, val_acc_list = [], []


def format_duration(seconds):
    return str(timedelta(seconds=max(0, int(seconds))))


def estimate_remaining(epoch, phase, batch_idx, phase_batches, seconds_per_batch):
    """현재 배치 완료 시점부터 최대 epoch까지의 train + val 예상 시간."""
    remaining_batches = {
        name: (num_epochs - epoch - 1) * count
        for name, count in phase_batches.items()
    }
    remaining_batches[phase] += phase_batches[phase] - batch_idx
    if phase == 'train':
        remaining_batches['val'] += phase_batches['val']

    if any(count > 0 and seconds_per_batch[name] is None
           for name, count in remaining_batches.items()):
        return None
    return sum(count * (seconds_per_batch[name] or 0.0)
               for name, count in remaining_batches.items())


progress_width = 0


def update_progress(message, final=False):
    """같은 콘솔 줄을 갱신하고 epoch가 끝날 때만 줄바꿈합니다."""
    global progress_width
    progress_width = max(progress_width, len(message))
    print("\r" + message.ljust(progress_width),
          end="\n" if final else "", flush=True)
    if final:
        progress_width = 0


def show_epoch_progress(epoch, phase, batch_idx, status="", final=False):
    completed = batch_idx + (phase_batches['train'] if phase == 'val' else 0)
    percent = 100 * completed / sum(phase_batches.values())
    best_text = (
        f"Best Loss {best_val_loss:.4f} Acc {best_acc:.4f}"
        if best_acc >= 0 else "Best Loss -- Acc --"
    )
    remaining = estimate_remaining(
        epoch, phase, batch_idx, phase_batches, seconds_per_batch
    )
    if remaining is None:
        time_text = "ETA: valid 측정 대기"
    else:
        finish_at = datetime.now().astimezone() + timedelta(seconds=remaining)
        time_text = (
            f"남음 {format_duration(remaining)} | "
            f"완료 {finish_at:%m-%d %H:%M:%S %Z}"
        )
    update_progress(
        f"Epoch {epoch+1}/{num_epochs} {percent:5.1f}% [{phase}] | "
        f"{best_text} | {time_text}"
        + (f" | {status}" if status else ""),
        final=final
    )

# ETA는 최대 num_epochs의 학습 + epoch별 검증 기준입니다.
# EarlyStopping이면 더 일찍 끝납니다. 최종 평가/그래프 표시 시간은 제외합니다.
phase_batches = {name: len(dataloaders[name]) for name in ['train', 'val']}
seconds_per_batch = {'train': None, 'val': None}
phase_total_seconds = {'train': 0.0, 'val': 0.0}
phase_total_batches = {'train': 0, 'val': 0}
training_started_at = datetime.now().astimezone()
training_start = time.perf_counter()
print(f"학습 시작: {training_started_at:%Y-%m-%d %H:%M:%S %Z}", flush=True)
print(f"완료 예상: 최대 {num_epochs} epoch 기준 / EarlyStopping 시 조기 종료 / "
      "최종 평가·그래프 제외", flush=True)

# ✅ 학습 루프
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()

        running_loss = 0.0
        running_corrects = 0

        phase_start = time.perf_counter()
        processed_images = 0
        show_epoch_progress(epoch, phase, 0, status="로딩 대기")

        for batch_idx, (inputs, labels) in enumerate(dataloaders[phase], start=1):
            if batch_idx == 1:
                show_epoch_progress(
                    epoch, phase, 0,
                    status=f"로딩 {time.perf_counter() - phase_start:.2f}초 / 연산 중"
                )
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            processed_images += inputs.size(0)

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == phase_batches[phase]:
                # loss.item()/corrects.item() 이후라 GPU 연산 완료가 시간에 반영됩니다.
                phase_elapsed = time.perf_counter() - phase_start
                seconds_per_batch[phase] = (
                    (phase_total_seconds[phase] + phase_elapsed)
                    / (phase_total_batches[phase] + batch_idx)
                )
                show_epoch_progress(epoch, phase, batch_idx)

        phase_total_seconds[phase] += time.perf_counter() - phase_start
        phase_total_batches[phase] += phase_batches[phase]

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects / len(image_datasets[phase])


        if phase == 'train':
            train_acc_list.append(epoch_acc)
        else:
            val_acc_list.append(epoch_acc)
            best_val_loss = min(best_val_loss, epoch_loss)

            # Model checkpoint 저장
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_shufflenetv2_film_classifier.pth')

            # EarlyStopping
            else:
                counter += 1


    show_epoch_progress(
        epoch, 'val', phase_batches['val'],
        status="EarlyStopping" if counter >= patience else "완료",
        final=True
    )
    if counter >= patience:
        break

# ✅ Best 모델 적용
model.load_state_dict(best_model_wts)
# 그래프 창을 닫기 전에도 최종 최고 모델 파일이 저장되도록 먼저 저장합니다.
torch.save(model.state_dict(), 'best_shufflenetv2.pth')
training_finished_at = datetime.now().astimezone()
print(
    f"학습 및 최고 모델 저장 완료 | 실제 epoch {epoch+1}/{num_epochs} | "
    f"종료 시각 {training_finished_at:%Y-%m-%d %H:%M:%S %Z} | "
    f"총 학습 소요 {format_duration(time.perf_counter() - training_start)} | "
    f"최고 검증 정확도 {best_acc:.4f}",
    flush=True
)
print("최종 검증 지표·그래프 생성 시작 (위 학습 시간에서 제외)", flush=True)

# ✅ Confusion Matrix 시각화 함수
def plot_roc_curve(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    if len(set(all_labels)) < 2:
        print("ROC 생략: 검증 데이터에 두 클래스가 모두 필요합니다.")
        return

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# ✅ Confusion Matrix 시각화 함수
def plot_confusion_matrix(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=image_datasets['val'].classes,
                yticklabels=image_datasets['val'].classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f"\n Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n Classification Report")
    print(classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=image_datasets['val'].classes, zero_division=0))



# ✅ Confusion Matrix 출력
plot_confusion_matrix(model, dataloaders['val'])

evaluate_model(model, dataloaders['val'])

plot_roc_curve(model, dataloaders['val'])

