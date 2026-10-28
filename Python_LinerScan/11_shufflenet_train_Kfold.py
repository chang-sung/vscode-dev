import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import shufflenet_v2_x1_0

from sklearn.model_selection import StratifiedKFold

# =========================
# 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
num_classes = 2
batch_size = 8
num_epochs = 10

n_splits = 5                 # StratifiedKFold 폴드 수
patience = 5                 # EarlyStopping patience
min_delta = 0.0              # 개선 최소치(필요 시 1e-4 같은 값으로)

# ✅ 전처리 완료된 이미지 폴더 (train 폴더만 사용)
train_dir = r"C:\Users\Home\Desktop\dataset\train"

# ✅ 로컬 ShuffleNet V2 x1.0 기본 가중치
local_weight_path = r"C:\Users\Home\Desktop\transparent_object\shufflenetv2_x1-5666bf0f80.pth"

# ✅ 저장 경로
save_dir = "./shufflenetv2_kfold_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# =========================
# seed 고정
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# =========================
# transforms
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# Dataset 준비 (train_dir 전체)
# =========================
full_dataset = datasets.ImageFolder(train_dir, transform=transform)
targets = np.array(full_dataset.targets)  # stratify를 위한 label 배열
classes = full_dataset.classes

if len(classes) != num_classes:
    raise ValueError(f"학습 폴더는 {num_classes}개 클래스가 필요합니다.")
if np.bincount(targets, minlength=num_classes).min() < n_splits:
    raise ValueError("각 클래스의 이미지 수가 n_splits 이상이어야 합니다.")

print("Classes:", classes)
print("Total images:", len(full_dataset))

# =========================
# 모델 생성 함수
# =========================
def build_model(num_classes: int):
    # 1) 기본 ShuffleNet V2 x1.0 생성
    model = shufflenet_v2_x1_0(weights=None)

    # 2) 로컬 weight 로드 (fc는 1000클래스라서 아직 교체 전 로드해야 함)
    state_dict = torch.load(local_weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    # 3) fc 교체
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)

# =========================
# 1 epoch 학습/평가 함수
# =========================
def run_one_epoch(model, dataloader, criterion, optimizer=None):
    """
    optimizer가 None이면 eval 모드로 동작
    return: loss(float), acc(float)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        running_corrects += (preds == labels).sum().item()
        total += bs

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)
    return epoch_loss, epoch_acc

# =========================
# K-Fold 학습
# =========================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

overall_best_val_acc = -1.0
overall_best_fold = None
overall_best_path = None

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_dataset)), targets), start=1):
    print("\n" + "=" * 60)
    print(f"Fold {fold}/{n_splits}")
    print("=" * 60)

    train_subset = Subset(full_dataset, train_idx.tolist())
    val_subset = Subset(full_dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # fold별 best 저장 (val loss 기준 or val acc 기준 선택 가능)
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_wts = copy.deepcopy(model.state_dict())

    es_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"[Fold {fold}] Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        # ✅ best 판단 기준: (1) val_loss가 줄면 best
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            es_counter = 0

            fold_best_path = os.path.join(save_dir, f"best_shufflenetv2_fold{fold}.pth")
            torch.save(best_wts, fold_best_path)
        else:
            es_counter += 1
            if es_counter >= patience:
                print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

    # fold best 로드
    model.load_state_dict(best_wts)

    # fold 최종 best 기록 출력
    print(f"[Fold {fold}] Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    fold_accuracies.append(best_val_acc)

    # ✅ 전체 folds 중 최고 모델도 저장(Val Acc 기준)
    if best_val_acc > overall_best_val_acc:
        overall_best_val_acc = best_val_acc
        overall_best_fold = fold
        overall_best_path = os.path.join(save_dir, "best_overall_shufflenetv2.pth")
        torch.save(best_wts, overall_best_path)

print("\n" + "#" * 60)
print(f"K-Fold Val Acc: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Overall Best Val Acc: {overall_best_val_acc:.4f} (Fold {overall_best_fold})")
if overall_best_path:
    print(f"Saved overall best model -> {overall_best_path}")
print("#" * 60)