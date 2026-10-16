import os
import cv2
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#1. CONFIG
DATASET_PATH = "dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#2. DATASET
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0,
                            translate=(0.1, 0.1)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
])

test_transform = None

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

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

train_dataset = BrainTumorDataset("Training")
test_dataset = BrainTumorDataset("Testing")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_dataset)}")
print(f"Testing  samples : {len(test_dataset)}")

#3. CNN MODEL
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        #Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 channel in, 32 filters out
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 → 64x64
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 → 32x32
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 → 16x16
            nn.Dropout(0.25),
        )

        #Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128 * 16 * 16 = 32768 numbers
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),  # 4 outputs
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = BrainTumorCNN().to(DEVICE)
print(f"\nModel created ✅")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

#4. TRAINING SETUP
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#5. TRAIN MODEL
train_losses = []
train_accuracies = []

print("\n── Starting Training ──")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    scheduler.step()

    print(f"Epoch [{epoch + 1:2d}/{EPOCHS}]  "
          f"Loss: {epoch_loss:.4f}  "
          f"Accuracy: {epoch_acc:.2f}%")

print("\n── Training Complete ✅ ──")

#6. EVALUATE ON TEST SET
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

#7. SAVE MODEL
torch.save(model.state_dict(), "brain_tumor_cnn.pth")
print("Model saved as brain_tumor_cnn.pth ✅")

#8. RESULTS

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1 - Training Loss
axes[0].plot(range(1, EPOCHS + 1), train_losses, 'b-o')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# Plot 2 - Training Accuracy
axes[1].plot(range(1, EPOCHS + 1), train_accuracies, 'g-o')
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(True)

# Plot 3 - Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[2])
axes[2].set_title("Confusion Matrix")
axes[2].set_ylabel("Actual")
axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("results.png")
plt.show()

# Detailed report
print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds, target_names=CLASSES))