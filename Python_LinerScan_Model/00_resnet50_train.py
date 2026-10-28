import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision.models.resnet import resnet50
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
batch_size = 8
num_epochs = 10

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
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}

# ✅ 로컬 ResNet50 기본 가중치 불러오기
local_weight_path = r'C:\Users\Home\Desktop\transparent_object\resnet50-0676ba61.pth'
model = resnet50(pretrained=False)
state_dict = torch.load(local_weight_path, map_location=device)
model.load_state_dict(state_dict)

# ✅ 마지막 fc 레이어 교체 (2-class 분류)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# ✅ 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ✅ EarlyStopping & ModelCheckpoint 설정
best_acc = 0.0
patience = 5
counter = 0
best_model_wts = model.state_dict()

train_acc_list, val_acc_list = [], []

# ✅ 학습 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
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
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == 'train':
            train_acc_list.append(epoch_acc.item())
        else:
            val_acc_list.append(epoch_acc.item())

            # Model checkpoint 저장
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), 'best_resnet50_film_classifier.pth')

            # EarlyStopping
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    if counter >= patience:
        break

# ✅ Best 모델 적용
model.load_state_dict(best_model_wts)

# ✅ Confusion Matrix 시각화 함수
def plot_confusion_matrix(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1)[:,1]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle ='--')
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

    cm = confusion_matrix(all_labels, all_preds)
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
        for inpus, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"\n Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n Classification Report")
    print(classification_report(all_labels, all_preds, target_names=image_datasets['val'].classes))



# ✅ Confusion Matrix 출력
plot_confusion_matrix(model, dataloaders['val'])

evaluate_model(model, dataloaders['val'])

plot_roc_curve(model, dataloaders['val'])

# ✅ 최종 모델 저장
torch.save(model.state_dict(), 'best_resnet50.pth')
print("✅ 모델 저장 완료!")