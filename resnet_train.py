import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#1. CONFIG
DATASET_PATH = "dataset"
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 20
CLASSES      = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES  = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


#2. DATASET
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class BrainTumorDataset(Dataset):
    def __init__(self, split="Training"):
        self.data      = []
        self.transform = train_transform if split == "Training" else test_transform

        for label_idx, class_name in enumerate(CLASSES):
            folder = os.path.join(DATASET_PATH, split, class_name)
            for filename in os.listdir(folder):
                path = os.path.join(folder, filename)
                self.data.append((path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

train_dataset = BrainTumorDataset("Training")
test_dataset  = BrainTumorDataset("Testing")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Testing  samples : {len(test_dataset)}")


#3. LOAD RESNET18 — PRETRAINED
model = models.resnet18(weights="IMAGENET1K_V1")
print("\nResNet18 loaded with pretrained weights ✅")

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters : {trainable:,} / {total:,}")
print("(Most layers are frozen — only final layer trains)")


#4. TRAINING SETUP
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.0001},
    {"params": model.fc.parameters(),     "lr": 0.001},
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


#5. TRAIN
train_losses     = []
train_accuracies = []

print("\n── Starting Training ──")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    scheduler.step()

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Loss: {epoch_loss:.4f}  "
          f"Accuracy: {epoch_acc:.2f}%")

print("\n── Training Complete ✅ ──")


#6. EVALUATE
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    correct = 0
    total   = 0
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        labels  = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


#7. SAVE MODEL
torch.save(model.state_dict(), "resnet_brain_tumor.pth")
print("Model saved as resnet_brain_tumor.pth ✅")


#8. PLOT RESULTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(range(1, EPOCHS+1), train_accuracies, 'g-o')
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True)

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2])
axes[2].set_title("Confusion Matrix")
axes[2].set_ylabel("Actual")
axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("resnet_results.png")
plt.show()

print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds, target_names=CLASSES))